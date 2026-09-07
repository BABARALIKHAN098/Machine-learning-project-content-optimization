from __future__ import annotations

from typing import Any

import pandas as pd

from ..utils.exceptions import DataValidationError


def configured_feature_columns(config: dict[str, Any]) -> list[str]:
    """Return the approved feature order from the central data contract."""
    numeric = list(config.get("numeric_columns", []))
    categorical = list(config.get("categorical_columns", []))
    overlap = set(numeric) & set(categorical)
    if overlap:
        raise DataValidationError(f"Feature roles overlap: {sorted(overlap)}")
    if len(numeric + categorical) != len(set(numeric + categorical)):
        raise DataValidationError("Duplicate feature names")
    forbidden = (
        set(config.get("id_columns", []))
        | set(config.get("drop_columns", []))
        | set(config.get("sensitive_columns", []))
    )
    target = config.get("target_column")
    if target:
        forbidden.add(target)
    leaked = (set(numeric) | set(categorical)) & forbidden
    if leaked:
        raise DataValidationError(f"Forbidden columns configured as features: {sorted(leaked)}")
    return numeric + categorical


def select_features(dataframe: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    columns = configured_feature_columns(config)
    missing = [column for column in columns if column not in dataframe]
    if missing:
        raise DataValidationError(f"Required feature columns are missing: {missing}")
    selected = dataframe.loc[:, columns].copy()
    for column in config.get("numeric_columns", []):
        original = selected[column]
        converted = pd.to_numeric(original, errors="coerce")
        invalid = original.notna() & converted.isna()
        if invalid.any():
            raise DataValidationError(f"Feature contains non-numeric values: {column}")
        selected[column] = converted
    return selected
