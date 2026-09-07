"""Fixed feature-family study with an explicit development-only interface."""

from __future__ import annotations

import copy
import json
import platform
import subprocess
import time
from pathlib import Path

import pandas as pd
import sklearn
from sklearn.model_selection import StratifiedGroupKFold

from machine_learning_project.data.ingestion import sha256_file
from machine_learning_project.data.preparation import prepare_supervised_data
from machine_learning_project.data.splitting import _atomic_write
from machine_learning_project.features.artifacts import input_contract_fingerprint
from machine_learning_project.features.diagnostics import training_diagnostics
from machine_learning_project.features.engineering import CutoffSafeFeatureEngineer
from machine_learning_project.features.registry import fingerprint, resolve_registry
from machine_learning_project.models.evaluate import evaluate_predictions
from machine_learning_project.models.train import build_candidates
from machine_learning_project.utils.exceptions import DataValidationError


def variant_configs(data_config, config):
    resolve_registry(data_config, config)
    if config.get("variants") != ["A", "B", "C", "D", "E"]:
        raise DataValidationError("The frozen study requires variants A, B, C, D, E")
    if (
        not isinstance(config.get("folds"), int)
        or isinstance(config["folds"], bool)
        or config["folds"] < 2
    ):
        raise DataValidationError("folds must be an integer >= 2")
    for key in ("tie_tolerance", "down_recall_guardrail"):
        value = config.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise DataValidationError(f"Invalid {key}")
    if not isinstance(config.get("random_seed"), int) or isinstance(config["random_seed"], bool):
        raise DataValidationError("random_seed must be an integer")
    result = {}
    for name in config["variants"]:
        candidate = copy.deepcopy(config)
        candidate["families"] = [] if name == "A" else ["ratios"]
        candidate["excluded_inputs"] = []
        if name == "C":
            candidate["families"].append("logs")
        if name == "D":
            candidate["excluded_inputs"] = [
                x
                for x in (
                    "age_tier",
                    "age_tier_order",
                    "freshness_tier",
                    "word_count_tier",
                    "char_count_tier",
                )
                if x
                in data_config.get("numeric_columns", [])
                + data_config.get("categorical_columns", [])
            ]
        if name == "E" and "model_used" in data_config.get("categorical_columns", []):
            candidate["excluded_inputs"] = ["model_used"]
        resolve_registry(data_config, candidate)
        result[name] = candidate
    return result


def choose_variant(summary, tolerance, guardrail):
    eligible = [row for row in summary if row["down_recall"] >= guardrail]
    if not eligible:
        return "A"
    best = max(row["macro_f1"] for row in eligible)
    tied = [row for row in eligible if row["macro_f1"] >= best - tolerance]
    return min(tied, key=lambda row: (row["width"], row["variant"] != "A", row["variant"]))[
        "variant"
    ]


def run_feature_study(
    train,
    validation,
    data_config,
    preprocessing_config,
    training_config,
    feature_config,
    output_dir,
    provenance=None,
    source_path=None,
):
    """No test argument: fold fitting and diagnostics receive training data only."""
    variants = variant_configs(data_config, feature_config)
    group, target = training_config["group_column"], data_config["target_column"]
    if set(train[group]) & set(validation[group]):
        raise DataValidationError("Development partitions share groups")
    labels = list(data_config["allowed_target_values"])
    x = prepare_supervised_data(train, data_config).features
    vx = prepare_supervised_data(validation, data_config).features
    y, vy = train[target].astype(str), validation[target].astype(str)
    if set(y) != set(labels) or set(vy) != set(labels):
        raise DataValidationError("Development partitions must cover every class")
    splitter = StratifiedGroupKFold(
        n_splits=feature_config["folds"], shuffle=True, random_state=feature_config["random_seed"]
    )
    if train[group].nunique() < feature_config["folds"]:
        raise DataValidationError("Insufficient groups for feature folds")
    folds = list(splitter.split(x, y, groups=train[group]))
    for left, right in folds:
        if set(y.iloc[left]) != set(labels) or set(y.iloc[right]) != set(labels):
            raise DataValidationError(
                "Infeasible grouped class coverage; revise protocol explicitly"
            )
        if set(train[group].iloc[left]) & set(train[group].iloc[right]):
            raise DataValidationError("Fold group overlap")
    rows, catalogs, diagnostics = [], {}, {}
    for variant, config in variants.items():
        registry = resolve_registry(data_config, config)
        catalogs[variant] = registry
        engineered = CutoffSafeFeatureEngineer(data_config, config).fit_transform(x)
        diagnostics[variant] = training_diagnostics(engineered, registry)
        for fold, (left, right) in enumerate(folds):
            candidates = build_candidates(
                data_config, preprocessing_config, training_config, config
            )
            for estimator, pipeline in candidates.items():
                print(f"Variant {variant}, fold {fold + 1}, {estimator}", flush=True)
                started = time.perf_counter()
                pipeline.fit(x.iloc[left], y.iloc[left])
                fit_seconds = time.perf_counter() - started
                started = time.perf_counter()
                predictions = pipeline.predict(x.iloc[right])
                prediction_seconds = time.perf_counter() - started
                metrics = evaluate_predictions(y.iloc[right], predictions, labels)
                rows.append(
                    {
                        "variant": variant,
                        "estimator": estimator,
                        "fold": fold,
                        "seed": training_config.get("random_seed", 42),
                        "macro_f1": metrics["macro_f1"],
                        "down_recall": metrics["per_class"]["down"]["recall"],
                        "width": len(pipeline[:-1].get_feature_names_out()),
                        "train_rows": len(left),
                        "validation_rows": len(right),
                        "train_groups": train[group].iloc[left].nunique(),
                        "validation_groups": train[group].iloc[right].nunique(),
                        "fit_seconds": fit_seconds,
                        "prediction_seconds": prediction_seconds,
                    }
                )
    results = pd.DataFrame(rows)
    decisions = {}
    frozen = {}
    names = {}
    for estimator, subset in results.groupby("estimator", sort=True):
        summary = subset.groupby("variant", as_index=False)[
            ["macro_f1", "down_recall", "width"]
        ].mean()
        candidate = choose_variant(
            summary.to_dict("records"),
            feature_config["tie_tolerance"],
            feature_config["down_recall_guardrail"],
        )
        confirmations = {}
        pipelines = {}
        for variant in dict.fromkeys(["A", candidate]):
            pipeline = build_candidates(
                data_config, preprocessing_config, training_config, variants[variant]
            )[estimator]
            pipeline.fit(x, y)
            confirmations[variant] = evaluate_predictions(vy, pipeline.predict(vx), labels)
            pipelines[variant] = pipeline
        reference, proposed = confirmations["A"], confirmations[candidate]
        delta = proposed["macro_f1"] - reference["macro_f1"]
        more_complex = len(pipelines[candidate][:-1].get_feature_names_out()) > len(
            pipelines["A"][:-1].get_feature_names_out()
        )
        eligible = (
            proposed["per_class"]["down"]["recall"] >= feature_config["down_recall_guardrail"]
        )
        promoted = (
            eligible
            and delta >= -feature_config["tie_tolerance"]
            and (not more_complex or delta >= feature_config["tie_tolerance"])
        )
        selected = candidate if promoted else "A"
        decisions[estimator] = {
            "fold_choice": candidate,
            "selected_variant": selected,
            "confirmation": confirmations,
            "promotion_passed": promoted,
            "cv_guardrail_met": bool(
                (summary["down_recall"] >= feature_config["down_recall_guardrail"]).any()
            ),
            "cv_summary": summary.to_dict("records"),
            "cv_macro_f1_std": subset.groupby("variant")["macro_f1"].std().to_dict(),
            "worst_fold_down_recall": subset.groupby("variant")["down_recall"].min().to_dict(),
        }
        frozen[estimator] = variants[selected]
        names[estimator] = list(map(str, pipelines[selected][:-1].get_feature_names_out()))
    output = Path(output_dir)
    payload = {
        "schema_version": "1.0",
        "feature_contract_version": "2.0",
        "provenance": provenance or {},
        "config_sha256": fingerprint(
            {
                "data": data_config,
                "preprocessing": preprocessing_config,
                "training": training_config,
                "features": feature_config,
            }
        ),
        "input_contract_sha256": input_contract_fingerprint(
            data_config, preprocessing_config, training_config
        ),
        "fitting_population": "train; fold selection within train; validation confirmation",
        "test_accessed": False,
        "historical_test_exposure": True,
        "decisions": decisions,
        "selected_configs": frozen,
        "encoded_names": names,
        "registry": catalogs,
        "protocol": {
            "features": feature_config,
            "training_candidates": training_config.get("candidates", {}),
            "model_seed": training_config.get("random_seed", 42),
        },
        "excluded_source_columns": data_config.get("drop_columns", [])
        + data_config.get("id_columns", []),
        "environment": {
            "python": platform.python_version(),
            "sklearn": sklearn.__version__,
            "pandas": pd.__version__,
        },
    }
    try:
        revision = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
        ).stdout.strip()
        payload["code"] = {"revision": revision, "dirty": bool(dirty)}
    except (OSError, subprocess.CalledProcessError):
        payload["code"] = {"revision": "unavailable", "dirty": None}
    project_root = Path(__file__).resolve().parents[1]
    payload["code"]["source_tree_sha256"] = fingerprint(
        {
            str(path.relative_to(project_root)).replace("\\", "/"): sha256_file(path)
            for directory in ("src", "pipelines", "scripts")
            for path in sorted((project_root / directory).rglob("*.py"))
        }
    )
    report = [
        "# Feature selection report",
        "",
        "Development-only fixed family study.",
        "",
        "The existing test holdout was previously evaluated; no test rows were used here.",
        "Snapshot data and unverified metadata timing do not establish future performance.",
        "Undefined ratios remain missing with indicators; empty numeric columns use zero.",
        "",
    ]
    for estimator, decision in decisions.items():
        report.append(
            f"- {estimator}: selected {decision['selected_variant']}; "
            f"grouped-fold choice {decision['fold_choice']}; "
            f"validation promotion passed: {decision['promotion_passed']}."
        )
        report.extend(
            [
                "",
                "| Variant | Mean fold macro F1 | Mean down recall | Worst fold down recall | Mean width |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in decision["cv_summary"]:
            worst = decision["worst_fold_down_recall"][row["variant"]]
            report.append(
                f"| {row['variant']} | {row['macro_f1']:.6f} | {row['down_recall']:.6f} | {worst:.6f} | {row['width']:.1f} |"
            )
        report.extend(
            [
                "",
                "Validation confirmation:",
                "",
                "| Variant | Macro F1 | Down recall |",
                "| --- | ---: | ---: |",
            ]
        )
        for variant, metrics in decision["confirmation"].items():
            report.append(
                f"| {variant} | {metrics['macro_f1']:.6f} | {metrics['per_class']['down']['recall']:.6f} |"
            )
        report.append("")
    report += [
        "",
        "All variants and fold dispersion are recorded in feature_manifest.json.",
        (
            "Variants not selected were rejected by the frozen guardrail, complexity, "
            "or validation confirmation rule. No threshold was relaxed."
        ),
        "",
    ]
    catalog_rows = [
        {**entry, "variant": variant, "dependencies": ",".join(entry["dependencies"])}
        for variant, registry in catalogs.items()
        for entry in registry
    ]
    contents = {
        "ablation_results.csv": results.to_csv(index=False, lineterminator="\n"),
        "feature_catalog.csv": pd.DataFrame(catalog_rows).to_csv(index=False),
        "training_diagnostics.json": json.dumps(diagnostics, indent=2, allow_nan=False),
        "selection_report.md": "\n".join(report),
        "selected_features.json": json.dumps(frozen, indent=2, sort_keys=True),
    }
    if source_path is not None and sha256_file(Path(source_path)) != (provenance or {}).get(
        "source_sha256"
    ):
        raise DataValidationError("Source changed during feature study; outputs not published")
    for name, content in contents.items():
        _atomic_write(output / name, content)
    payload["artifact_hashes"] = {name: sha256_file(output / name) for name in contents}
    _atomic_write(
        output / "feature_manifest.json",
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False),
    )
    return payload
