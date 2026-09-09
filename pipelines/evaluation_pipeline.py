"""SPEC-07: verify frozen finalists, replay validation once, publish aggregate diagnostics."""

import json
import time
from pathlib import Path

import pandas as pd
from threadpoolctl import threadpool_limits

from machine_learning_project.data.development import load_development
from machine_learning_project.data.ingestion import sha256_file
from machine_learning_project.data.preparation import prepare_supervised_data
from machine_learning_project.data.splitting import _atomic_write, _hash_value
from machine_learning_project.features.artifacts import (
    input_contract_fingerprint,
    load_frozen_configs,
)
from machine_learning_project.features.registry import fingerprint
from machine_learning_project.models.baseline import resolve_baseline_config
from machine_learning_project.models.benchmark import (
    compare_candidate,
    load_benchmark,
    prepare_benchmark_partitions,
    require_matching_identity,
)
from machine_learning_project.models.error_analysis import (
    analyze_errors,
    analyze_subgroups,
    client_sensitivity,
    compare_paired_predictions,
    subgroup_memberships,
)
from machine_learning_project.models.evaluation_artifacts import (
    PAYLOADS,
    evaluation_directory,
    load_evaluation,
    payload_path,
)
from machine_learning_project.models.evaluation_decision import LIMITATIONS, decide_evaluation
from machine_learning_project.models.training_artifacts import (
    FAMILIES,
    code_environment,
    load_training_run,
    safe_payload,
    write_json,
)
from machine_learning_project.utils.config import (
    validate_data_config,
    validate_evaluation_config,
    validate_preprocessing_config,
    validate_split_config,
)
from machine_learning_project.utils.exceptions import DataValidationError


def _read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def preflight(
    data,
    training,
    preprocessing,
    evaluation,
    training_report_dir,
    model_dir,
    feature_manifest,
    baseline_manifest,
    baseline_config,
    run_id,
):
    validate_evaluation_config(evaluation, training)
    validate_data_config(data)
    validate_preprocessing_config(preprocessing)
    validate_split_config(training)
    baseline_config = resolve_baseline_config(baseline_config, training)
    report, models = Path(training_report_dir).resolve(), Path(model_dir).resolve()
    protected = {
        Path(data["csv_path"]),
        Path(training["split_manifest_path"]),
        Path(training["split_assignments_path"]),
        Path(feature_manifest),
        Path(baseline_manifest),
        report / "training_manifest.json",
    }
    for path in (
        Path(feature_manifest),
        Path(baseline_manifest),
        report / "training_manifest.json",
    ):
        document = _read(path)
        for key, root in (
            ("artifact_hashes", path.parent),
            ("report_hashes", report),
            ("model_hashes", models),
        ):
            protected.update(safe_payload(root, name) for name in document.get(key, {}))
    output = evaluation_directory(evaluation, run_id, protected)
    for directory in (
        report,
        models,
        Path(feature_manifest).resolve().parent,
        Path(baseline_manifest).resolve().parent,
    ):
        if output.is_relative_to(directory):
            raise DataValidationError(
                "Evaluation output must be isolated from frozen artifact directories"
            )
    snapshots = {path.resolve(): sha256_file(path) for path in protected}
    train, validation, provenance = load_development(data, training)
    train, validation, identity = prepare_benchmark_partitions(
        train, validation, data, training, provenance
    )
    del train
    frozen, feature_hash = load_frozen_configs(
        feature_manifest, data, preprocessing, training, provenance
    )
    benchmark, baseline_hash = load_benchmark(baseline_manifest, identity, baseline_config)
    manifest, predictors = load_training_run(report, models)
    require_matching_identity(identity, manifest["evaluation_identity"])
    resolved, selection, saved = [
        _read(report / name)
        for name in ("resolved_config.json", "selection.json", "validation_metrics.json")
    ]
    require_matching_identity(identity, saved["evaluation_identity"])
    expected_hashes = {
        name: fingerprint(value)
        for name, value in (
            ("data", data),
            ("training", training),
            ("preprocessing", preprocessing),
            ("baseline", baseline_config),
        )
    }
    checks = [
        resolved["original_config_hashes"] == expected_hashes,
        manifest["resolved_config_sha256"] == fingerprint(resolved),
        manifest["selection_sha256"] == fingerprint(selection),
        manifest["fold_manifest_sha256"] == fingerprint(_read(report / "fold_manifest.json")),
        selection["trial_metrics_sha256"] == fingerprint(_read(report / "trial_metrics.json")),
        selection["selection_population"] == "training_grouped_folds",
        selection["validation_used_for_selection"] is False,
        saved["test_accessed"] is False,
        set(saved["finalists"]) == set(FAMILIES),
        set(selection["finalists"]) == set(FAMILIES),
        resolved["feature_manifest_sha256"] == feature_hash,
        resolved["baseline_manifest_sha256"] == baseline_hash,
        resolved["feature_input_contract_sha256"]
        == input_contract_fingerprint(data, preprocessing, training),
        resolved["selected_feature_hashes"] == {f: fingerprint(frozen[f]) for f in FAMILIES},
    ]
    code, environment = code_environment()
    for key in ("python", "numpy", "pandas", "sklearn", "threadpoolctl"):
        checks.append(manifest["environment"][key] == environment[key])
    for family, predictor in predictors.items():
        metadata, pipeline = predictor.metadata, predictor.pipeline
        effective = pipeline.named_steps["model"].get_params()
        checks.extend(
            [
                predictor.data_config == data,
                metadata["selected_model"] == family,
                metadata["selection_sha256"] == fingerprint(selection),
                metadata["feature_manifest_sha256"] == feature_hash,
                metadata["baseline_manifest_sha256"] == baseline_hash,
                metadata["feature_config_sha256"] == fingerprint(frozen[family]),
                metadata["original_model_parameters"] == training["candidates"][family],
                metadata["effective_model_parameters"] == effective,
                metadata["class_order"] == list(map(str, pipeline.classes_)),
                set(pipeline.classes_) == set(data["allowed_target_values"]),
                effective.get("n_jobs", 1) in (None, 1),
                metadata["fitting_population"] == "train",
                all(
                    effective[k.removeprefix("model__")] == v
                    for k, v in selection["finalists"][family]["parameters"].items()
                ),
            ]
        )
    if not all(checks):
        raise DataValidationError("Frozen evaluation provenance/configuration/environment mismatch")
    if selection["preferred_for_development"]["family"] not in FAMILIES:
        raise DataValidationError("Invalid frozen CV reference")
    prepared = prepare_supervised_data(validation, data, feature_contract_version="2.0")
    approved = set(data["numeric_columns"]) | {
        d for d in evaluation["dimensions"] if d in data["categorical_columns"]
    }
    if "previous_impressions_bucket" in evaluation["dimensions"]:
        approved.add("impressions_prev_30d")
    context = prepared.features.loc[:, sorted(approved)].copy()
    subgroup_memberships(context, evaluation, data["numeric_columns"])
    groups = validation[training["group_column"]].to_numpy()
    del validation
    return locals()


def _plots(output, errors, subgroups, config):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_data = {
        "confusion": {f: errors[f]["confusion"] for f in FAMILIES},
        "supported_down_recall": [
            {k: row[k] for k in ("dimension", "group", "family", "down_recall", "support_down")}
            for row in subgroups["records"]
            if row["down_recall_supported"]
        ],
    }
    max_count = max(max(row) for f in FAMILIES for row in errors[f]["confusion"]["counts"])
    for family in FAMILIES:
        c = errors[family]["confusion"]
        for kind, key in (("counts", "counts"), ("rates", "row_normalized")):
            fig, ax = plt.subplots(figsize=(6, 5))
            matrix = np.asarray(c[key], dtype=float)
            im = ax.imshow(
                np.ma.masked_invalid(matrix),
                vmin=0,
                vmax=max_count if kind == "counts" else 1,
                cmap="Blues",
            )
            ax.set(
                xticks=range(len(c["labels"])),
                yticks=range(len(c["labels"])),
                xticklabels=c["labels"],
                yticklabels=c["labels"],
                xlabel="Prediction",
                ylabel="Truth",
                title=f"{family}: validation {kind}",
            )
            for i in range(len(matrix)):
                for j in range(len(matrix)):
                    value = matrix[i, j]
                    ax.text(
                        j,
                        i,
                        "N/A"
                        if np.isnan(value)
                        else str(int(value))
                        if kind == "counts"
                        else f"{value:.2f}",
                        ha="center",
                    )
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(
                payload_path(output, f"figures/{family}_{kind}.png"), dpi=config["plot_dpi"]
            )
            plt.close(fig)
    rows = plot_data["supported_down_recall"]
    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.20)))
    if rows:
        ax.barh(range(len(rows)), [r["down_recall"] for r in rows])
        ax.set_yticks(
            range(len(rows)), [f"{r['dimension']}/{r['group']} / {r['family']}" for r in rows]
        )
    else:
        ax.text(0.5, 0.5, "No supported down-recall slices", ha="center")
    ax.set(xlim=(0, 1), xlabel="Validation down recall (supported slices only)")
    fig.tight_layout()
    fig.savefig(payload_path(output, "figures/supported_down_recall.png"), dpi=config["plot_dpi"])
    plt.close(fig)
    return plot_data


def run_evaluation(
    data_config,
    training_config,
    preprocessing_config,
    evaluation_config,
    training_report_dir,
    model_dir,
    feature_manifest,
    baseline_manifest,
    baseline_config,
    run_id,
    dry_run=False,
):
    started = time.perf_counter()
    s = preflight(
        data_config,
        training_config,
        preprocessing_config,
        evaluation_config,
        training_report_dir,
        model_dir,
        feature_manifest,
        baseline_manifest,
        baseline_config,
        run_id,
    )
    if dry_run:
        return {
            "status": "preflight_passed",
            "planned_fits": 0,
            "planned_predictions": 2,
            "evaluation_identity": s["identity"],
        }
    load_seconds = time.perf_counter() - started
    truth = s["prepared"].target.to_numpy()
    labels = data_config["allowed_target_values"]
    predictions, errors, replay, comparisons, timings = {}, {}, {}, {}, {}
    audit_context = f"content-trend-tuning-v1:{s['provenance']['source_sha256']}"
    for family in FAMILIES:
        before = time.perf_counter()
        with threadpool_limits(limits=1):
            predictions[family] = s["predictors"][family].pipeline.predict(s["prepared"].features)
        timings[family] = time.perf_counter() - before
        errors[family] = analyze_errors(truth, predictions[family], labels)
        digest = _hash_value(fingerprint(list(predictions[family])), audit_context)
        reference = s["saved"]["finalists"][family]
        comparison = compare_candidate(
            errors[family]["metrics"],
            s["identity"],
            s["benchmark"],
            s["baseline_config"],
            training_config,
        )
        if (
            digest != reference["prediction_sha256"]
            or errors[family]["metrics"] != reference["metrics"]
            or comparison != reference["comparison"]
        ):
            raise DataValidationError(
                "Frozen prediction/metric/comparison replay drift; no publication"
            )
        replay[family] = {
            "prediction_sha256": digest,
            "metrics": errors[family]["metrics"],
            "comparison": comparison,
            "exact_replay": True,
        }
        comparisons[family] = comparison
    analysis_start = time.perf_counter()
    paired = compare_paired_predictions(truth, predictions, labels)
    subgroups = analyze_subgroups(
        s["context"],
        truth,
        predictions,
        labels,
        evaluation_config,
        numeric_columns=data_config["numeric_columns"],
        down_recall_guardrail=training_config["down_recall_guardrail"],
    )
    sensitivity = client_sensitivity(
        s["groups"],
        truth,
        predictions,
        labels,
        evaluation_config,
        hash_context=f"content-trend-evaluation-v1:{s['provenance']['source_sha256']}",
    )
    blockers = [
        {"reason": "unresolved_cutoff_evidence"},
        {"reason": "unresolved_final_test_policy"},
        {"reason": "production_runtime_not_certified"},
        *subgroups["alerts"],
    ]
    if sensitivity["ordering_reverses"]:
        blockers.append({"reason": "client_composition_reverses_model_order"})
    decision = decide_evaluation(
        {f: errors[f]["metrics"] for f in FAMILIES},
        comparisons,
        s["selection"]["preferred_for_development"]["family"],
        {"review_flags": blockers},
        evaluation_config,
    )
    references = {
        "training_manifest_sha256": sha256_file(s["report"] / "training_manifest.json"),
        "selection_sha256": s["manifest"]["report_hashes"]["selection.json"],
        "model_hashes": s["manifest"]["model_hashes"],
        "report_hashes": s["manifest"]["report_hashes"],
        "feature_manifest_sha256": s["feature_hash"],
        "baseline_manifest_sha256": s["baseline_hash"],
    }
    decision["frozen_references"] = references
    resolved = {
        "evaluation": {k: v for k, v in evaluation_config.items() if k != "output_root"},
        "inherited_training_thresholds": {
            k: training_config[k] for k in ("minimum_macro_f1", "down_recall_guardrail")
        },
        "minimum_macro_f1_improvement": s["baseline_config"]["minimum_macro_f1_improvement"],
        "original_config_hashes": s["expected_hashes"],
        "numeric_missingness_columns": data_config["numeric_columns"],
        "prediction_audit_convention": "_hash_value(fingerprint(list(predictions)), "
        "'content-trend-tuning-v1:' + source_sha256)",
        "references": references,
    }
    analysis_seconds = time.perf_counter() - analysis_start
    output = s["output"]
    output.mkdir(parents=True, exist_ok=False)
    (output / "figures").mkdir()
    publication_start = time.perf_counter()
    try:
        payloads = {
            "resolved_config.json": resolved,
            "evaluation_metrics.json": {"evaluation_identity": s["identity"], "finalists": replay},
            "error_analysis.json": errors,
            "confusion_matrices.json": {f: errors[f]["confusion"] for f in FAMILIES},
            "paired_comparison.json": paired,
            "subgroup_coverage.json": subgroups["coverage"],
            "client_sensitivity.json": sensitivity,
            "decision.json": decision,
            "plot_data.json": _plots(output, errors, subgroups, evaluation_config),
        }
        for name, value in payloads.items():
            write_json(payload_path(output, name), value)
        class_rows = []
        for family in FAMILIES:
            for row in errors[family]["classes"]:
                record = {"family": family, **row}
                if row["class"] == "down":
                    for key, value in errors[family]["down"].items():
                        if isinstance(value, int):
                            record[f"down_{key}"] = value
                        elif isinstance(value, dict) and "denominator" in value:
                            record.update({f"down_{key}_{k}": v for k, v in value.items()})
                class_rows.append(record)
        for name, rows in (
            ("class_errors.csv", class_rows),
            ("subgroup_metrics.csv", subgroups["records"]),
        ):
            frame = pd.DataFrame(rows)
            if frame.empty:
                frame = pd.DataFrame(
                    columns=["dimension", "group", "family", "row_count", "status"]
                )
            _atomic_write(
                payload_path(output, name), frame.to_csv(index=False, lineterminator="\n")
            )
        report = [
            "# SPEC-07 validation error analysis",
            "",
            decision["recommendation_status"],
            (
                f"Recommended model: {decision['recommended_model']}; development reference: "
                f"{decision['development_reference']}. Production ready: false."
            ),
            "",
            "| Family | Macro F1 | Down recall | Missed down | False alerts |",
            "| --- | ---: | ---: | ---: | ---: |",
        ]
        for f in FAMILIES:
            e = errors[f]
            report.append(
                f"| {f} | {e['metrics']['macro_f1']:.6f} | "
                f"{e['metrics']['per_class']['down']['recall']:.6f} | {e['down']['fn']} | {e['down']['fp']} |"
            )
        report += [
            "",
            (
                f"Paired disagreements: {paired['disagreement']}; "
                f"supported subgroup alerts: {len(subgroups['alerts'])}."
            ),
            (
                f"Client omission scenarios: {sensitivity['scenario_count']}; "
                f"supported ordering reversal: {sensitivity['ordering_reverses']}."
            ),
            "",
            (
                "Slices overlap across dimensions; do not add their error counts. "
                "Category labels are anonymous and ordered by frequency, then value. "
                "Suppressed populations remain in subgroup_coverage.json."
            ),
            "",
            *LIMITATIONS,
            "",
            (
                "SPEC-08 handoff: decision.json binds eligibility and immutable "
                "model references. A null recommendation provides rejection evidence for research."
            ),
            "",
            (
                "Future experiment hypothesis: investigate the largest off-diagonal errors using "
                "separately authorized data and cutoff evidence. No causal benefit is established."
            ),
            "",
        ]
        report += [f"![{f} normalized confusion](figures/{f}_rates.png)" for f in FAMILIES]
        _atomic_write(payload_path(output, "evaluation_report.md"), "\n".join(report) + "\n")
        write_json(
            output / "timings.json",
            {
                "load_seconds": load_seconds,
                "predict_seconds": timings,
                "analysis_seconds": analysis_seconds,
                "publication_seconds": time.perf_counter() - publication_start,
                "total_seconds": time.perf_counter() - started,
                "production_sla_certified": False,
            },
        )
        for path, expected in s["snapshots"].items():
            if sha256_file(path) != expected:
                raise DataValidationError("Protected prerequisite changed; no completion published")
        manifest = {
            "status": "complete",
            "artifact_schema_version": "1.0",
            "evaluation_contract_version": "1.0",
            "metric_contract_version": "1.0",
            "run_id": run_id,
            "evaluation_identity": s["identity"],
            "references": references,
            "code": s["code"],
            "environment": s["environment"],
            "training_environment": s["manifest"]["environment"],
            "input_paths": {
                "training_report_dir": str(s["report"]),
                "model_dir": str(s["models"]),
                "feature_manifest": str(feature_manifest),
                "baseline_manifest": str(baseline_manifest),
            },
            "protected_input_hashes": {str(p): h for p, h in s["snapshots"].items()},
            "payload_hashes": {
                name: sha256_file(payload_path(output, name)) for name in sorted(PAYLOADS)
            },
        }
        write_json(output / "evaluation_manifest.json", manifest)
        load_evaluation(output)
    except BaseException:
        (output / "evaluation_manifest.json").unlink(missing_ok=True)
        raise
    return {"status": "complete", "report_dir": str(output), "decision": decision}
