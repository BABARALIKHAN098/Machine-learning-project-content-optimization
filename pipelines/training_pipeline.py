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

from machine_learning_project.data.ingestion import load_csv
from machine_learning_project.data.preparation import prepare_supervised_data
from machine_learning_project.data.splitting import split_by_group, write_split_artifacts
from machine_learning_project.data.validation import require_valid_schema
from machine_learning_project.features.selection import configured_feature_columns
from machine_learning_project.models.baseline import build_baselines
from machine_learning_project.models.evaluate import evaluate_predictions, subgroup_metrics
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
) -> TrainingResult:
    validate_data_config(data_config)
    validate_preprocessing_config(preprocessing_config)
    validate_split_config(training_config)
    loaded = load_csv(data_config)
    dataframe = loaded.dataframe
    require_valid_schema(dataframe, data_config)
    target = data_config["target_column"]
    group = training_config.get("group_column", data_config.get("group_column", "client_id"))
    labels = list(data_config["allowed_target_values"])
    splits = split_by_group(
        dataframe,
        target_column=target,
        group_column=group,
        row_key=training_config["row_key"],
        allowed_labels=labels,
        test_size=float(training_config.get("test_size", 0.2)),
        validation_size=float(training_config.get("validation_size", 0.2)),
        random_seed=int(training_config.get("random_seed", 42)),
        search_attempts=int(training_config["search_attempts"]),
        row_ratio_tolerance=float(training_config["row_ratio_tolerance"]),
        class_ratio_tolerance=float(training_config["class_ratio_tolerance"]),
        size_weight=float(training_config["split_objective_weights"]["size"]),
        class_weight=float(training_config["split_objective_weights"]["class"]),
        source_sha256=loaded.sha256,
        data_schema_version=data_config.get("schema_version", "unversioned"),
        split_contract_version=training_config["split_contract_version"],
        algorithm_version=training_config["split_algorithm_version"],
        alias_context=training_config.get("split_alias_context", "content-trend-split-v1"),
    )
    write_split_artifacts(
        splits,
        training_config["split_manifest_path"],
        training_config["split_assignments_path"],
    )

    contract_version = preprocessing_config.get("feature_contract_version", "1.0")
    train_prepared = prepare_supervised_data(
        splits.train, data_config, feature_contract_version=contract_version
    )
    validation_prepared = prepare_supervised_data(
        splits.validation, data_config, feature_contract_version=contract_version
    )
    test_prepared = prepare_supervised_data(
        splits.test, data_config, feature_contract_version=contract_version
    )
    train_x = train_prepared.features
    validation_x = validation_prepared.features
    test_x = test_prepared.features
    train_y = train_prepared.target.astype(str)
    validation_y = validation_prepared.target.astype(str)
    test_y = test_prepared.target.astype(str)

    baseline_results: dict[str, Any] = {}
    for name, baseline in build_baselines(int(training_config.get("random_seed", 42))).items():
        started = time.perf_counter()
        baseline.fit(train_x, train_y)
        predictions = baseline.predict(validation_x)
        metrics = evaluate_predictions(validation_y, predictions, labels)
        metrics["elapsed_seconds"] = time.perf_counter() - started
        baseline_results[name] = metrics

    candidates = build_candidates(data_config, preprocessing_config, training_config)
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
    development = pd.concat([splits.train, splits.validation], axis=0)
    development_prepared = prepare_supervised_data(
        development, data_config, feature_contract_version=contract_version
    )
    development_x = development_prepared.features
    development_y = development_prepared.target.astype(str)
    fit_started = time.perf_counter()
    selected.fit(development_x, development_y)
    final_fit_seconds = time.perf_counter() - fit_started

    prediction_started = time.perf_counter()
    test_predictions = selected.predict(test_x)
    test_prediction_seconds = time.perf_counter() - prediction_started
    test_metrics = evaluate_predictions(test_y, test_predictions, labels)
    test_metrics["prediction_seconds"] = test_prediction_seconds
    test_metrics["subgroups"] = subgroup_metrics(
        splits.test,
        test_y,
        test_predictions,
        [group, "content_type", "main_intent", "age_tier", "freshness_tier"],
    )

    benchmark_x = prepare_supervised_data(
        dataframe, data_config, feature_contract_version=contract_version
    ).features
    benchmark_started = time.perf_counter()
    selected.predict(benchmark_x)
    benchmark_seconds = time.perf_counter() - benchmark_started
    threshold = float(training_config.get("minimum_macro_f1", 0.45))
    runtime_limit = float(training_config.get("batch_runtime_limit_seconds", 30))
    metrics_payload = {
        "selected_model": selected_name,
        "primary_metric": training_config.get("primary_metric", "macro_f1"),
        "minimum_macro_f1": threshold,
        "threshold_met": test_metrics["macro_f1"] >= threshold,
        "baselines_validation": baseline_results,
        "candidates_validation": validation_results,
        "test": test_metrics,
        "benchmark": {
            "row_count": len(dataframe),
            "prediction_seconds": benchmark_seconds,
            "limit_seconds": runtime_limit,
            "limit_met": benchmark_seconds <= runtime_limit,
        },
    }
    metrics_path = _write_json(training_config["metrics_path"], metrics_payload)

    preprocessor = selected.named_steps["preprocessor"]
    metadata = {
        "model_version": "1.0.0",
        "selected_model": selected_name,
        "created_at_utc": pd.Timestamp.now(tz="UTC").isoformat(),
        "source_sha256": loaded.sha256,
        "feature_contract_version": preprocessing_config.get("feature_contract_version", "1.0"),
        "feature_columns": configured_feature_columns(data_config),
        "transformed_feature_names": list(map(str, preprocessor.get_feature_names_out())),
        "target_column": target,
        "class_order": list(map(str, selected.named_steps["model"].classes_)),
        "final_fit_seconds": final_fit_seconds,
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "scikit_learn_version": sklearn.__version__,
        "metrics_path": str(metrics_path),
        "human_review_required": True,
    }
    metadata_path = _write_json(training_config["metadata_path"], metadata)
    artifact_path = Path(training_config["artifact_path"])
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": selected, "metadata": metadata, "data_config": data_config}, artifact_path)
    return TrainingResult(selected_name, artifact_path, metadata_path, metrics_path, metrics_payload)
