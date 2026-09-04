from __future__ import annotations

from typing import Any

import pandas as pd

from ..utils.exceptions import DataValidationError


def _configured_columns(config: dict[str, Any]) -> set[str]:
    keys = ("id_columns", "drop_columns", "numeric_columns", "categorical_columns")
    return {column for key in keys for column in config.get(key, [])}


def validate_schema(dataframe: pd.DataFrame, config: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    target = config.get("target_column")
    required = set(config.get("required_columns", []))

    if not target or target == "TBD":
        errors.append("Set data.target_column in configs/data.yaml.")
    elif target not in dataframe.columns:
        errors.append(f"Target column not found: {target}")

    missing_required = sorted(required.difference(dataframe.columns))
    if missing_required:
        errors.append(f"Required columns not found: {missing_required}")

    if target in dataframe.columns:
        usable_values = dataframe[target].dropna()
        if dataframe[target].isna().any():
            errors.append(f"Target column contains missing values: {target}")
        if usable_values.empty:
            errors.append("Target column contains no usable values.")
        elif usable_values.nunique() < 2:
            errors.append("Target column must contain at least two distinct values.")
        allowed = set(config.get("allowed_target_values", []))
        unknown = sorted(set(usable_values.astype(str)) - allowed) if allowed else []
        if unknown:
            errors.append(f"Unknown target values: {unknown}")

    expected_rows = config.get("expected_row_count")
    if expected_rows is not None and len(dataframe) != expected_rows:
        errors.append(f"Expected {expected_rows} rows, found {len(dataframe)}.")
    expected_columns = config.get("expected_column_count")
    if expected_columns is not None and len(dataframe.columns) != expected_columns:
        errors.append(f"Expected {expected_columns} columns, found {len(dataframe.columns)}.")

    for column in config.get("unique_columns", []):
        if column in dataframe:
            if dataframe[column].isna().any():
                errors.append(f"Unique column contains missing values: {column}")
            if dataframe[column].duplicated().any():
                errors.append(f"Column must be unique: {column}")

    for column in config.get("non_negative_columns", []):
        if column in dataframe:
            numeric = pd.to_numeric(dataframe[column], errors="coerce")
            invalid = dataframe[column].notna() & numeric.isna()
            if invalid.any():
                errors.append(f"Column contains non-numeric values: {column}")
            if (numeric.dropna() < 0).any():
                errors.append(f"Column contains negative values: {column}")

    role_lists = [
        set(config.get("id_columns", [])),
        set(config.get("drop_columns", [])),
        set(config.get("numeric_columns", [])),
        set(config.get("categorical_columns", [])),
    ]
    conflicts: set[str] = set()
    for index, left in enumerate(role_lists):
        for right in role_lists[index + 1 :]:
            conflicts.update(left & right)
    if conflicts:
        errors.append(f"Columns have conflicting roles: {sorted(conflicts)}")

    if any(key in config for key in ("drop_columns", "numeric_columns", "categorical_columns")):
        roles = _configured_columns(config)
        unassigned = set(dataframe.columns) - roles - ({target} if target else set())
        if unassigned and not config.get("allow_extra_columns", False):
            errors.append(f"Columns have no configured role: {sorted(unassigned)}")
        referenced = roles | set(config.get("sensitive_columns", [])) | set(
            config.get("unique_columns", [])
        )
        unknown_references = referenced - set(dataframe.columns)
        if unknown_references:
            errors.append(f"Configured columns not found: {sorted(unknown_references)}")
    return errors


def require_valid_schema(dataframe: pd.DataFrame, config: dict[str, Any]) -> None:
    errors = validate_schema(dataframe, config)
    if errors:
        raise DataValidationError("; ".join(errors))
