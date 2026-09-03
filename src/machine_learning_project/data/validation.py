from __future__ import annotations

from typing import Any

import pandas as pd

from ..utils.exceptions import DataValidationError


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
        if usable_values.empty:
            errors.append("Target column contains no usable values.")
        elif usable_values.nunique() < 2:
            errors.append("Target column must contain at least two distinct values.")
    return errors


def require_valid_schema(dataframe: pd.DataFrame, config: dict[str, Any]) -> None:
    errors = validate_schema(dataframe, config)
    if errors:
        raise DataValidationError("; ".join(errors))
