from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from .exceptions import DataValidationError


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
