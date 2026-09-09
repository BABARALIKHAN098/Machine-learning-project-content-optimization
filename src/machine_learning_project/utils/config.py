from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import yaml

from .exceptions import DataValidationError


def validate_baseline_config(config: dict[str, Any]) -> None:
    if not isinstance(config, dict):
        raise DataValidationError("Baseline configuration must be a mapping")
    supported = {
        "baseline_contract_version": "1.0",
        "metric_contract_version": "1.0",
        "evaluation_partition": "validation",
        "ordering": "row_key",
    }
    errors = [
        f"Unsupported baselines.{key}"
        for key, value in supported.items()
        if config.get(key) != value
    ]
    if config.get("strategies") != ["most_frequent", "stratified"]:
        errors.append("Baseline strategies must be [most_frequent, stratified]")
    seeds = config.get("repeat_seeds")
    valid_seed = lambda seed: type(seed) is int and 0 <= seed < 2**32
    if not valid_seed(config.get("reference_seed")):
        errors.append("reference_seed must be a uint32 integer")
    if (
        not isinstance(seeds, list)
        or not seeds
        or not all(valid_seed(x) for x in seeds)
        or len(set(seeds)) != len(seeds)
    ):
        errors.append("repeat_seeds must be a nonempty list of unique uint32 integers")
    elif config.get("reference_seed") not in seeds:
        errors.append("repeat_seeds must include reference_seed")
    if config.get("reference_seed") != 42 or seeds != [42, 43, 44, 45, 46]:
        errors.append("Contract 1.0 requires reference seed 42 and repeats [42, 43, 44, 45, 46]")
    margin = config.get("minimum_macro_f1_improvement")
    if type(margin) not in (int, float) or not math.isfinite(margin) or not 0 < margin <= 1:
        errors.append("minimum_macro_f1_improvement must be finite and in (0, 1]")
    if not isinstance(config.get("output_directory"), str) or not config["output_directory"]:
        errors.append("output_directory must be a nonempty string")
    allowed = set(supported) | {
        "strategies",
        "reference_seed",
        "repeat_seeds",
        "minimum_macro_f1_improvement",
        "output_directory",
    }
    if set(config) - allowed:
        errors.append("Unknown baseline configuration fields")
    if errors:
        raise DataValidationError("; ".join(errors))


def validate_data_config(config: dict[str, Any]) -> None:
    errors: list[str] = []
    delimiter = config.get("delimiter", ",")
    if not isinstance(delimiter, str) or not delimiter:
        errors.append("data.delimiter must be a non-empty string")
    encoding = config.get("encoding", "utf-8")
    if not isinstance(encoding, str) or not encoding:
        errors.append("data.encoding must be a non-empty string")
    missing_tokens = config.get("missing_value_tokens", [])
    if not isinstance(missing_tokens, list):
        errors.append("data.missing_value_tokens must be a list")
    for key in ("id_columns", "drop_columns", "numeric_columns", "categorical_columns"):
        values = config.get(key, [])
        if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
            errors.append(f"data.{key} must be a list of strings")
        elif len(values) != len(set(values)):
            errors.append(f"data.{key} contains duplicate entries")
    if not isinstance(config.get("allow_extra_columns", False), bool):
        errors.append("data.allow_extra_columns must be a boolean")
    if errors:
        raise DataValidationError("; ".join(errors))


def load_yaml(path: str | Path) -> dict[str, Any]:
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as handle:
        content = yaml.safe_load(handle) or {}
    if not isinstance(content, dict):
        raise TypeError("Configuration root must be a mapping.")
    return content


def validate_preprocessing_config(config: dict[str, Any]) -> None:
    supported = {
        "numeric_imputation": {"median"},
        "categorical_encoding": {"one_hot"},
        "handle_unknown_categories": {"ignore"},
    }
    errors: list[str] = []
    for key, allowed in supported.items():
        value = config.get(key)
        if value not in allowed:
            errors.append(f"preprocessing.{key} must be one of {sorted(allowed)}, got {value!r}")
    fill_value = config.get("categorical_imputation")
    if not isinstance(fill_value, str) or not fill_value:
        errors.append("preprocessing.categorical_imputation must be a non-empty string")
    if not isinstance(config.get("scale_numeric"), bool):
        errors.append("preprocessing.scale_numeric must be a boolean")
    version = config.get("feature_contract_version")
    if not isinstance(version, str) or not version:
        errors.append("preprocessing.feature_contract_version must be a non-empty string")
    if errors:
        raise DataValidationError("; ".join(errors))


def validate_eda_config(config: dict[str, Any]) -> None:
    errors: list[str] = []
    required_positive = (
        "iqr_multiplier",
        "rare_category_min_count",
        "minimum_group_support",
        "maximum_categories_displayed",
        "plot_dpi",
    )
    for key in required_positive:
        value = config.get(key)
        if not isinstance(value, (int, float)) or value <= 0:
            errors.append(f"eda.{key} must be positive")
    correlation = config.get("high_correlation_threshold")
    if not isinstance(correlation, (int, float)) or not 0 < correlation <= 1:
        errors.append("eda.high_correlation_threshold must be in (0, 1]")
    quantiles = config.get("quantiles")
    if (
        not isinstance(quantiles, list)
        or not quantiles
        or any(not isinstance(value, (int, float)) or not 0 <= value <= 1 for value in quantiles)
        or quantiles != sorted(set(quantiles))
    ):
        errors.append("eda.quantiles must be a sorted unique list in [0, 1]")
    version = config.get("artifact_schema_version")
    if not isinstance(version, str) or not version:
        errors.append("eda.artifact_schema_version must be a non-empty string")
    for key in ("target_cohorts", "priority_numeric_columns"):
        value = config.get(key, [])
        if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
            errors.append(f"eda.{key} must be a list of strings")
    if errors:
        raise DataValidationError("; ".join(errors))


def validate_split_config(config: dict[str, Any]) -> None:
    errors: list[str] = []
    test_size = config.get("test_size")
    validation_size = config.get("validation_size")
    if (
        not isinstance(test_size, (int, float))
        or not isinstance(validation_size, (int, float))
        or not 0 < test_size < 1
        or not 0 < validation_size < 1
        or test_size + validation_size >= 1
    ):
        errors.append("training test_size and validation_size must be positive and sum below 1")
    for key in ("group_column", "row_key", "split_contract_version", "split_algorithm_version"):
        if not isinstance(config.get(key), str) or not config[key]:
            errors.append(f"training.{key} must be a non-empty string")
    attempts = config.get("search_attempts")
    if not isinstance(attempts, int) or attempts <= 0:
        errors.append("training.search_attempts must be a positive integer")
    for key in ("row_ratio_tolerance", "class_ratio_tolerance"):
        value = config.get(key)
        if not isinstance(value, (int, float)) or not 0 <= value < 1:
            errors.append(f"training.{key} must be in [0, 1)")
    weights = config.get("split_objective_weights", {})
    if not isinstance(weights, dict) or any(
        not isinstance(weights.get(key), (int, float)) or weights[key] < 0
        for key in ("size", "class")
    ):
        errors.append("training.split_objective_weights must contain non-negative size/class")
    if errors:
        raise DataValidationError("; ".join(errors))


def validate_packaging_config(config: dict[str, Any]) -> None:
    fixed = {
        "packaging_contract_version": "1.0",
        "package_schema_version": "1.0",
        "inference_contract_version": "2.0",
        "overwrite": False,
        "resume": False,
        "remote_loading": False,
    }
    if not isinstance(config, dict) or set(config) != set(fixed) | {
        "families",
        "package_root",
        "output_root",
    }:
        raise DataValidationError("Invalid packaging configuration fields")
    for key, value in fixed.items():
        if type(config[key]) is not type(value) or config[key] != value:
            raise DataValidationError(f"Unsupported packaging.{key}")
    families = config["families"]
    if (
        not isinstance(families, list)
        or not families
        or any(f not in ("logistic_regression", "random_forest") for f in families)
        or len(set(families)) != len(families)
    ):
        raise DataValidationError("Packaging requires explicit unique frozen families")
    for key in ("package_root", "output_root"):
        value = config[key]
        if (
            not isinstance(value, str)
            or not value.strip()
            or "://" in value
            or value.startswith(("//", "\\\\"))
        ):
            raise DataValidationError("Packaging requires local output paths")


def validate_inference_config(config: dict[str, Any]) -> None:
    fixed = {
        "inference_contract_version": "2.0",
        "include_probabilities": False,
        "probability_tolerance": 1e-9,
        "thread_limit": 1,
        "unknown_categories": "ignore",
        "extra_columns": "reject",
        "numeric_string_coercion": False,
        "encoding": "utf-8",
        "delimiter": ",",
        "missing_tokens": ["", "NA", "N/A", "null", "None", "?"],
    }
    if not isinstance(config, dict) or set(config) != set(fixed) | {
        "maximum_batch_rows",
        "chunk_rows",
    }:
        raise DataValidationError("Invalid inference configuration fields")
    for key, value in fixed.items():
        if type(config[key]) is not type(value) or config[key] != value:
            raise DataValidationError(f"Unsupported inference.{key}")
    if any(type(config[k]) is not int for k in ("maximum_batch_rows", "chunk_rows")) or not (
        1 <= config["chunk_rows"] <= config["maximum_batch_rows"] <= 30000
    ):
        raise DataValidationError("Require 1 <= chunk_rows <= maximum_batch_rows <= 30000")


def validate_evaluation_config(config: dict[str, Any], training: dict[str, Any]) -> None:
    """Closed, validation-only diagnostic protocol, validated before model loading."""
    fixed = {
        "evaluation_contract_version": "1.0",
        "artifact_schema_version": "1.0",
        "metric_contract_version": "1.0",
        "partition": "validation",
        "families": ["logistic_regression", "random_forest"],
        "client_sensitivity": "leave_one_client_out",
        "prediction_n_jobs": 1,
        "row_exports": False,
        "probability_metrics": False,
        "resume": False,
    }
    supports = (
        "minimum_subgroup_rows",
        "minimum_class_support",
        "minimum_down_support",
        "minimum_predicted_down_support",
        "maximum_categories_per_dimension",
        "plot_dpi",
    )
    rates = ("comparison_tolerance", "subgroup_alert_macro_f1_gap")
    dimensions = {
        "content_type",
        "main_intent",
        "age_tier",
        "freshness_tier",
        "previous_impressions_bucket",
        "numeric_missingness_bucket",
    }
    if not isinstance(config, dict) or set(config) != (
        set(fixed) | set(supports) | set(rates) | {"dimensions", "output_root"}
    ):
        raise DataValidationError("Evaluation requires exactly the protocol fields")
    for key, value in fixed.items():
        if type(config[key]) is not type(value) or config[key] != value:
            raise DataValidationError(f"Unsupported evaluation.{key}")
    for key in supports:
        if type(config[key]) is not int or config[key] <= 0:
            raise DataValidationError(f"evaluation.{key} must be a positive integer")
    for key in rates:
        if (
            type(config[key]) not in (int, float)
            or not math.isfinite(config[key])
            or not (0 <= config[key] <= 1)
        ):
            raise DataValidationError(f"Invalid evaluation.{key}")
    selected = config["dimensions"]
    if (
        not isinstance(selected, list)
        or not selected
        or any(not isinstance(item, str) or item not in dimensions for item in selected)
        or len(set(selected)) != len(selected)
    ):
        raise DataValidationError("Unknown, duplicate or missing evaluation dimensions")
    root = config["output_root"]
    if not isinstance(root, str) or not root.strip() or root.startswith(("\\\\", "//")):
        raise DataValidationError("Evaluation output_root must be a local path")
    if training.get("primary_metric") != "macro_f1":
        raise DataValidationError("Evaluation requires the inherited macro_f1 contract")
    for key in ("minimum_macro_f1", "down_recall_guardrail"):
        value = training.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise DataValidationError(f"Invalid training.{key}")


def validate_tuning_config(config: dict[str, Any], training: dict[str, Any]) -> None:
    """Contract 1.0 is a predetermined search, not an adaptive parameter search."""
    import json

    fixed = {
        "tuning_contract_version": "1.0",
        "artifact_schema_version": "1.0",
        "families": ["logistic_regression", "random_forest"],
        "search_method": "fixed_grid",
        "folds": 3,
        "shuffle": True,
        "random_seed": 42,
        "primary_metric": "macro_f1",
        "selection_tolerance": 0.001,
        "search_n_jobs": 1,
        "estimator_n_jobs": 1,
        "max_configurations": 15,
        "max_estimator_fits": 47,
        "convergence_policy": "exclude",
        "evaluation_partition": "validation",
        "require_feature_manifest": True,
        "require_baseline_manifest": True,
        "resume": False,
        "grids": {
            "logistic_regression": {"model__C": [0.1, 1.0, 10.0]},
            "random_forest": {
                "model__min_samples_leaf": [1, 2, 5],
                "model__max_features": ["sqrt", 0.5],
                "model__max_depth": [None, 20],
            },
        },
    }
    if not isinstance(config, dict) or set(config) != set(fixed) | {
        "output_root",
        "model_output_root",
    }:
        raise DataValidationError("Tuning config must contain exactly the contract 1.0 fields")
    try:
        for key, expected in fixed.items():
            if json.dumps(config[key], sort_keys=True, allow_nan=False) != json.dumps(
                expected, sort_keys=True, allow_nan=False
            ):
                raise DataValidationError(f"Unsupported tuning.{key} for fixed protocol 1.0")
    except (TypeError, ValueError) as error:
        raise DataValidationError("Invalid fixed tuning protocol values") from error
    for key in ("output_root", "model_output_root"):
        if not isinstance(config[key], str) or not config[key].strip():
            raise DataValidationError(f"tuning.{key} must be a nonempty local path")
    if training.get("primary_metric") != "macro_f1" or type(training.get("random_seed")) is not int:
        raise DataValidationError("Training metric/seed is incompatible with tuning")
    if training["random_seed"] != 42:
        raise DataValidationError("Tuning requires the original training seed 42")
    for key in ("minimum_macro_f1", "down_recall_guardrail"):
        value = training.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise DataValidationError(f"Invalid training.{key}")
    candidates = training.get("candidates")
    allowed = {
        "logistic_regression": {"C", "max_iter", "class_weight"},
        "random_forest": {
            "n_estimators",
            "min_samples_leaf",
            "max_features",
            "class_weight",
            "n_jobs",
        },
    }
    if not isinstance(candidates, dict) or set(candidates) != set(allowed):
        raise DataValidationError("Both original candidate families are required")
    for family, fields in allowed.items():
        if not isinstance(candidates[family], dict) or set(candidates[family]) != fields:
            raise DataValidationError("Unknown or missing original candidate parameters")
    logistic, forest = candidates["logistic_regression"], candidates["random_forest"]
    for value in (logistic["C"],):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise DataValidationError("Original logistic C must be positive and finite")
    for value in (logistic["max_iter"], forest["n_estimators"], forest["min_samples_leaf"]):
        if type(value) is not int or value <= 0:
            raise DataValidationError("Original estimator counts must be positive integers")
    if (
        logistic["class_weight"] != "balanced"
        or forest["class_weight"] != "balanced_subsample"
        or type(forest["n_jobs"]) is not int
        or forest["n_jobs"] == 0
    ):
        raise DataValidationError("Unsupported original weighting/resource settings")
    value = forest["max_features"]
    if not (value == "sqrt" or type(value) in (int, float) and 0 < value <= 1):
        raise DataValidationError("Unsupported original max_features")
