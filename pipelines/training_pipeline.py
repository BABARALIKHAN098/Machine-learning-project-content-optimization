from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from machine_learning_project.data.development import load_development
from machine_learning_project.data.preparation import prepare_supervised_data
from machine_learning_project.features.artifacts import load_frozen_configs
from machine_learning_project.features.registry import fingerprint
from machine_learning_project.features.selection import configured_feature_columns
from machine_learning_project.models.baseline import evaluate_baselines, resolve_baseline_config
from machine_learning_project.models.benchmark import (
    compare_candidate,
    load_benchmark,
    prepare_benchmark_partitions,
)
from machine_learning_project.models.evaluate import evaluate_predictions
from machine_learning_project.models.train import build_candidates
from machine_learning_project.models.tune import select_candidate
from machine_learning_project.utils.config import (
    validate_data_config,
    validate_preprocessing_config,
    validate_split_config,
)


@dataclass(frozen=True)
class TrainingResult:
    selected_model: str
    artifact_path: Path
    metadata_path: Path
    metrics_path: Path
    metrics: dict[str, Any]


def _write_json(path: str | Path, payload: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return output


def run_training(
    data_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
    training_config: dict[str, Any],
    feature_config: dict[str, Any] | None = None,
    feature_manifest: str | Path | None = None,
    baseline_config: dict[str, Any] | None = None,
    baseline_manifest: str | Path | None = None,
) -> TrainingResult:
    baseline_config = resolve_baseline_config(baseline_config, training_config)
    validate_data_config(data_config)
    validate_preprocessing_config(preprocessing_config)
    validate_split_config(training_config)
    train, validation, provenance = load_development(data_config, training_config)
    train, validation, evaluation_identity = prepare_benchmark_partitions(
        train, validation, data_config, training_config, provenance
    )
    frozen_configs, manifest_sha256 = None, None
    if feature_manifest is not None:
        frozen_configs, manifest_sha256 = load_frozen_configs(
            feature_manifest, data_config, preprocessing_config, training_config, provenance
        )
    target = data_config["target_column"]
    labels = list(data_config["allowed_target_values"])
    contract_version = (feature_config or preprocessing_config)["feature_contract_version"]
    train_prepared = prepare_supervised_data(
        train, data_config, feature_contract_version=contract_version
    )
    validation_prepared = prepare_supervised_data(
        validation, data_config, feature_contract_version=contract_version
    )
    train_x, validation_x = train_prepared.features, validation_prepared.features
    train_y, validation_y = (
        train_prepared.target.astype(str),
        validation_prepared.target.astype(str),
    )

    baseline_manifest_sha256 = None
    if baseline_manifest is not None:
        benchmark, baseline_manifest_sha256 = load_benchmark(
            baseline_manifest, evaluation_identity, baseline_config
        )
    else:
        benchmark = evaluate_baselines(train_y, validation_y, baseline_config, labels)
        benchmark["evaluation_identity"] = evaluation_identity
    baseline_results = benchmark["canonical"]

    candidates = build_candidates(
        data_config, preprocessing_config, training_config, feature_config
    )
    if frozen_configs is not None:
        candidates = {
            name: build_candidates(data_config, preprocessing_config, training_config, config)[name]
            for name, config in frozen_configs.items()
        }
    validation_results: dict[str, Any] = {}
    for name, pipeline in candidates.items():
        started = time.perf_counter()
        pipeline.fit(train_x, train_y)
        predictions = pipeline.predict(validation_x)
        metrics = evaluate_predictions(validation_y, predictions, labels)
        metrics["elapsed_seconds"] = time.perf_counter() - started
        validation_results[name] = metrics

    selected_name = select_candidate(
        validation_results, float(training_config.get("down_recall_guardrail", 0.0))
    )
    selected = candidates[selected_name]
    if frozen_configs is not None:
        feature_config = frozen_configs[selected_name]
        contract_version = feature_config["feature_contract_version"]
    # Retain the train-fitted candidate. Final refit/test evaluation is a separate phase.
    threshold = float(training_config.get("minimum_macro_f1", 0.45))
    metrics_payload = {
        "selected_model": selected_name,
        "primary_metric": training_config.get("primary_metric", "macro_f1"),
        "minimum_macro_f1": threshold,
        "threshold_met": validation_results[selected_name]["macro_f1"] >= threshold,
        "baselines_validation": baseline_results,
        "candidates_validation": validation_results,
        "evaluation_partition": "validation",
        "test_accessed": False,
    }
    metrics_payload["evaluation_identity"] = evaluation_identity
    metrics_payload["baseline_benchmark"] = benchmark
    metrics_payload["baseline_manifest_sha256"] = baseline_manifest_sha256
    metrics_payload["baseline_comparisons"] = {
        name: compare_candidate(
            metrics, evaluation_identity, benchmark, baseline_config, training_config
        )
        for name, metrics in validation_results.items()
    }
    metrics_payload["selection_status"] = "selected_for_development"
    metrics_payload["recall_fallback_used"] = not any(
        metrics["per_class"]["down"]["recall"] >= training_config.get("down_recall_guardrail", 0.5)
        for metrics in validation_results.values()
    )
    metrics_path = _write_json(training_config["metrics_path"], metrics_payload)

    preprocessor = selected.named_steps["preprocessor"]
    metadata = {
        "model_version": "2.0.0" if feature_config else "1.0.0",
        "selected_model": selected_name,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_sha256": provenance["source_sha256"],
        "split_provenance": provenance,
        "fitting_population": "train",
        "bundle_schema_version": "2.0" if feature_config else "1.0",
        "feature_config_sha256": fingerprint(feature_config) if feature_config else None,
        "feature_manifest_sha256": manifest_sha256,
        "baseline_manifest_sha256": baseline_manifest_sha256,
        "metric_contract_version": evaluation_identity["metric_contract_version"],
        "evaluation_identity": evaluation_identity,
        "feature_contract_version": contract_version,
        "feature_columns": configured_feature_columns(data_config),
        "transformed_feature_names": list(map(str, preprocessor.get_feature_names_out())),
        "target_column": target,
        "class_order": list(map(str, selected.named_steps["model"].classes_)),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "metrics_path": str(metrics_path),
        "human_review_required": True,
    }
    metadata_path = _write_json(training_config["metadata_path"], metadata)
    artifact_path = Path(training_config["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": selected,
            "metadata": metadata,
            "data_config": data_config,
            "feature_config": feature_config,
            "preprocessing_config": preprocessing_config,
        },
        artifact_path,
    )
    return TrainingResult(
        selected_name, artifact_path, metadata_path, metrics_path, metrics_payload
    )
