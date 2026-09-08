"""Bounded SPEC-06 search: immutable prerequisites, training-only selection, two finalists."""

from __future__ import annotations

import json
import time
import warnings
from pathlib import Path

import joblib
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
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
)
from machine_learning_project.models.evaluate import evaluate_predictions
from machine_learning_project.models.train import build_trial
from machine_learning_project.models.training_artifacts import (
    MODEL_FILES,
    REPORT_FILES,
    code_environment,
    load_training_run,
    run_directories,
    safe_payload,
    write_json,
)
from machine_learning_project.models.tune import (
    expand_trials,
    make_training_folds,
    rank_trials,
    search_training,
)
from machine_learning_project.utils.config import (
    validate_data_config,
    validate_preprocessing_config,
    validate_split_config,
    validate_tuning_config,
)
from machine_learning_project.utils.exceptions import DataValidationError


def preflight(
    data,
    preprocessing,
    training,
    tuning,
    feature_manifest,
    baseline_manifest,
    baseline_config,
    run_id,
):
    validate_tuning_config(tuning, training)
    validate_data_config(data)
    validate_preprocessing_config(preprocessing)
    validate_split_config(training)
    baseline_config = resolve_baseline_config(baseline_config, training)
    output, models = run_directories(tuning, run_id)
    if not feature_manifest or not baseline_manifest:
        raise DataValidationError("Tuning requires frozen feature and baseline manifests")
    feature_manifest, baseline_manifest = Path(feature_manifest), Path(baseline_manifest)
    if not feature_manifest.is_file() or not baseline_manifest.is_file():
        raise DataValidationError(
            "Missing frozen manifests; run the explicit feature/baseline workflows"
        )
    protected = {
        Path(data["csv_path"]),
        Path(training["split_manifest_path"]),
        Path(training["split_assignments_path"]),
        feature_manifest,
        baseline_manifest,
    }
    for path in (feature_manifest, baseline_manifest):
        document = json.loads(path.read_text(encoding="utf-8"))
        for name in document.get("artifact_hashes", {}):
            protected.add(safe_payload(path.parent, name))
    snapshots = {path: sha256_file(path) for path in protected}
    train, validation, provenance = load_development(data, training)
    train, validation, identity = prepare_benchmark_partitions(
        train, validation, data, training, provenance
    )
    frozen, feature_hash = load_frozen_configs(
        feature_manifest, data, preprocessing, training, provenance
    )
    benchmark, baseline_hash = load_benchmark(baseline_manifest, identity, baseline_config)
    # Scope audit hashes to this source and protocol, without publishing raw row keys.
    context = f"content-trend-tuning-v1:{provenance['source_sha256']}"
    folds, fold_manifest = make_training_folds(train, data, training, tuning, context)
    trials = expand_trials(tuning)
    if len(trials) != 15 or len(trials) * len(folds) + 2 != 47:
        raise DataValidationError("Expanded search exceeds or disagrees with fixed fit schedule")
    return locals()


def run_tuning(
    data,
    preprocessing,
    training,
    tuning,
    *,
    feature_manifest,
    baseline_manifest,
    baseline_config,
    run_id,
    dry_run=False,
):
    state = preflight(
        data,
        preprocessing,
        training,
        tuning,
        feature_manifest,
        baseline_manifest,
        baseline_config,
        run_id,
    )
    output, models = state["output"], state["models"]
    summary = {
        "status": "preflight_passed",
        "configurations": 15,
        "planned_fits": 47,
        "folds": 3,
        "evaluation_identity": state["identity"],
    }
    if dry_run:
        return summary
    # Exclusive directories are staging until the final manifest is written.
    output.mkdir(parents=True, exist_ok=False)
    models.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    code, environment = code_environment()
    resolved = {
        "tuning": {
            k: v for k, v in tuning.items() if k not in ("output_root", "model_output_root")
        },
        "original_config_hashes": {
            name: fingerprint(config)
            for name, config in (
                ("data", data),
                ("preprocessing", preprocessing),
                ("training", training),
                ("baseline", baseline_config),
            )
        },
        "feature_input_contract_sha256": input_contract_fingerprint(data, preprocessing, training),
        "feature_manifest_sha256": state["feature_hash"],
        "baseline_manifest_sha256": state["baseline_hash"],
        "selected_feature_hashes": {
            family: fingerprint(config) for family, config in state["frozen"].items()
        },
        "trial_schedule": state["trials"],
        "planned_fits": 47,
    }
    write_json(output / "resolved_config.json", resolved)
    write_json(output / "fold_manifest.json", state["fold_manifest"])

    def progress(trials, timings):
        write_json(output / "trial_metrics.json", trials)
        write_json(output / "timings.json", {"folds": timings, "status": "searching"})

    try:
        trials, timings = search_training(
            state["train"],
            data,
            preprocessing,
            training,
            state["frozen"],
            tuning,
            state["folds"],
            state["context"],
            progress,
        )
        guardrail, tolerance = training["down_recall_guardrail"], tuning["selection_tolerance"]
        choices = {
            family: rank_trials([t for t in trials if t["family"] == family], guardrail, tolerance)
            for family in tuning["families"]
        }
        finalists = [
            t for t in trials if t["trial_id"] in {v["trial_id"] for v in choices.values()}
        ]
        preferred = rank_trials(finalists, guardrail, tolerance, cross_family=True)
        selection = {
            "finalists": choices,
            "preferred_for_development": preferred,
            "selection_population": "training_grouped_folds",
            "validation_used_for_selection": False,
            "trial_metrics_sha256": fingerprint(trials),
            "selection_tolerance": tolerance,
        }
        write_json(output / "selection.json", selection)
        # Choices are on disk before validation features or scores reach any estimator.
        train = prepare_supervised_data(state["train"], data, feature_contract_version="2.0")
        validation = prepare_supervised_data(
            state["validation"], data, feature_contract_version="2.0"
        )
        confirmed, refit_timings = {}, []
        for family, choice in choices.items():
            pipeline = build_trial(data, preprocessing, training, state["frozen"][family], choice)
            with warnings.catch_warnings(record=True) as observed:
                warnings.simplefilter("always")
                with threadpool_limits(limits=1):
                    start = time.perf_counter()
                    pipeline.fit(train.features, train.target)
                    fitted = time.perf_counter() - start
                    if any(issubclass(w.category, ConvergenceWarning) for w in observed):
                        raise DataValidationError(
                            "Finalist refit did not converge; no validation retuning"
                        )
                    start = time.perf_counter()
                    predictions = pipeline.predict(validation.features)
                    predicted = time.perf_counter() - start
            metrics = evaluate_predictions(
                validation.target, predictions, data["allowed_target_values"]
            )
            confirmed[family] = {
                "metrics": metrics,
                "prediction_sha256": _hash_value(fingerprint(list(predictions)), state["context"]),
                "warnings": [w.category.__name__ for w in observed],
                "comparison": compare_candidate(
                    metrics, state["identity"], state["benchmark"], baseline_config, training
                ),
            }
            metadata = {
                "model_version": f"spec06-{run_id}-{family}",
                "bundle_schema_version": "2.0",
                "development_only": True,
                "human_review_required": True,
                "fitting_population": "train",
                "selected_model": family,
                "feature_contract_version": "2.0",
                "feature_config_sha256": fingerprint(state["frozen"][family]),
                "feature_manifest_sha256": state["feature_hash"],
                "baseline_manifest_sha256": state["baseline_hash"],
                "selection_sha256": fingerprint(selection),
                "evaluation_identity": state["identity"],
                "original_model_parameters": training["candidates"][family],
                "effective_model_parameters": pipeline.named_steps["model"].get_params(),
                "class_order": list(map(str, pipeline.classes_)),
                "transformed_feature_names": list(
                    map(str, pipeline.named_steps["preprocessor"].get_feature_names_out())
                ),
            }
            bundle = {
                "pipeline": pipeline,
                "metadata": metadata,
                "data_config": data,
                "feature_config": state["frozen"][family],
                "preprocessing_config": preprocessing,
            }
            start = time.perf_counter()
            joblib.dump(bundle, models / f"{family}.joblib")
            serialized = time.perf_counter() - start
            write_json(models / f"{family}.json", metadata)
            refit_timings.append(
                {
                    "family": family,
                    "fit_seconds": fitted,
                    "prediction_seconds": predicted,
                    "serialization_seconds": serialized,
                }
            )
        write_json(
            output / "validation_metrics.json",
            {
                "evaluation_identity": state["identity"],
                "finalists": confirmed,
                "test_accessed": False,
            },
        )
        rows = []
        for trial in trials:
            rows.append(
                {
                    "trial_id": trial["trial_id"],
                    "family": trial["family"],
                    "status": trial["status"],
                    "mean_macro_f1": trial.get("summary", {}).get("macro_f1", {}).get("mean"),
                    "mean_down_recall": trial.get("summary", {}).get("down_recall", {}).get("mean"),
                    "recall_eligible": trial["status"] == "valid"
                    and trial["summary"]["down_recall"]["mean"] >= guardrail,
                    "selected": trial["trial_id"] == choices[trial["family"]]["trial_id"],
                }
            )
        rows.sort(
            key=lambda row: (
                row["status"] != "valid",
                not row["recall_eligible"],
                -(row["mean_macro_f1"] or 0),
                row["trial_id"],
            )
        )
        for rank, row in enumerate(rows, 1):
            row["score_rank"] = rank
        _atomic_write(
            output / "cv_results.csv", pd.DataFrame(rows).to_csv(index=False, lineterminator="\n")
        )
        write_json(
            output / "timings.json",
            {
                "folds": timings,
                "finalists": refit_timings,
                "total_seconds": time.perf_counter() - started,
                "actual_estimator_fits": len(timings) + len(refit_timings),
                "environment": environment,
                "throughput_benchmark": "not_run_optional",
            },
        )
        report = [
            "# SPEC-06 development training",
            "",
            f"CV preference: {preferred['family']}; recall fallback: {preferred['recall_fallback_used']}.",
            "15 configurations, three shared client-grouped training folds, 47 estimator fits.",
            "Mean fold macro F1 determines choices; 0.001 ties use declared complexity and stable IDs.",
            "",
            "| Family | CV macro F1 | Validation macro F1 | Validation down recall | Material improvement | Recall | Target |",
            "| --- | ---: | ---: | ---: | --- | --- | --- |",
        ]
        for family, result in confirmed.items():
            flags, metrics = result["comparison"], result["metrics"]
            report.append(
                f"| {family} | {choices[family]['cv_summary']['macro_f1']['mean']:.6f} | "
                f"{metrics['macro_f1']:.6f} | {metrics['per_class']['down']['recall']:.6f} | "
                f"{flags['material_improvement_met']} | {flags['down_recall_guardrail_met']} | "
                f"{flags['project_macro_f1_target_met']} |"
            )
        report.extend(
            [
                "",
                "Both finalists are fitted on train only. Validation cannot restart the search.",
                "CV uses unweighted fold means and population SD; worst-fold recall is also recorded.",
                "Frozen feature selection used broader training evidence: this is not nested evaluation.",
                "Validation informed earlier feature confirmation; historical test evaluation exists.",
                "These are development results, not independent generalization estimates or promotion decisions.",
                "Historical metadata cutoff evidence remains unresolved. Human review is required.",
                "Timings exclude any production SLA claim; the optional 30,000-row throughput test was not run.",
                "SPEC-07: consume both finalists via load_training_run with the report and model directories.",
                "SPEC-08: production packaging and promotion remain separate.",
                "",
            ]
        )
        _atomic_write(output / "training_report.md", "\n".join(report))
        for path, expected in state["snapshots"].items():
            if sha256_file(path) != expected:
                raise DataValidationError(
                    "Frozen prerequisite changed during training; no completion published"
                )
        manifest = {
            "artifact_schema_version": "1.0",
            "tuning_contract_version": "1.0",
            "status": "complete",
            "run_id": run_id,
            "evaluation_identity": state["identity"],
            "code": code,
            "environment": environment,
            "resolved_config_sha256": fingerprint(resolved),
            "selection_sha256": fingerprint(selection),
            "fold_manifest_sha256": fingerprint(state["fold_manifest"]),
            "report_hashes": {name: sha256_file(output / name) for name in sorted(REPORT_FILES)},
            "model_hashes": {name: sha256_file(models / name) for name in sorted(MODEL_FILES)},
        }
        # Validate staging with the same reader, then expose completion only after round-trip success.
        staging = output / "training_manifest.pending.json"
        write_json(staging, manifest)
        from machine_learning_project.inference.predictor import Predictor

        for family in choices:
            if (
                sha256_file(models / f"{family}.joblib")
                != manifest["model_hashes"][f"{family}.joblib"]
            ):
                raise DataValidationError("Model changed before reload")
            predictor = Predictor.load(models / f"{family}.joblib")
            with threadpool_limits(limits=1):
                reloaded = predictor.predict(state["validation"])
            actual = _hash_value(fingerprint(list(reloaded["predicted_trend"])), state["context"])
            if actual != confirmed[family]["prediction_sha256"]:
                raise DataValidationError("Reloaded finalist predictions differ")
        write_json(output / "training_manifest.json", manifest)
        staging.unlink()
        load_training_run(output, models)
        return {
            "status": "complete",
            "report_dir": str(output),
            "model_dir": str(models),
            "selection": selection,
            "validation": confirmed,
        }
    except BaseException as error:
        # Incomplete runs retain trial progress but cannot be consumed as completed evidence.
        final = output / "training_manifest.json"
        if final.exists():
            final.unlink()
        write_json(
            output / "failure.json",
            {
                "status": "incomplete",
                "error_type": type(error).__name__,
                "reason": "See trial statuses; no automatic retuning or resume",
            },
        )
        raise
