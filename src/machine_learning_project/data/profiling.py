from __future__ import annotations

from typing import Any

import pandas as pd


def build_profile(
    dataframe: pd.DataFrame,
    fingerprint: str,
    config: dict[str, Any] | None = None,
    *,
    feature_contract_version: str = "1.0",
) -> dict[str, Any]:
    config = config or {}
    row_count = len(dataframe)
    target = config.get("target_column")
    ids = set(config.get("id_columns", []))
    dropped = set(config.get("drop_columns", []))
    numeric = set(config.get("numeric_columns", []))
    categorical = set(config.get("categorical_columns", []))
    sensitive = set(config.get("sensitive_columns", []))
    maximum_missing_ratio = float(config.get("maximum_missing_ratio", 1.0))
    high_cardinality_threshold = int(config.get("high_cardinality_threshold", 0))
    columns: dict[str, Any] = {}
    for name in dataframe.columns:
        series = dataframe[name]
        null_count = int(series.isna().sum())
        if name == target:
            role = "target"
        elif name in ids:
            role = "identifier"
        elif name in numeric or name in categorical:
            role = "feature"
        elif name in dropped:
            role = "dropped"
        else:
            role = "unassigned"
        missing_ratio = null_count / row_count
        unique_count = int(series.nunique(dropna=True))
        warnings: list[str] = []
        if missing_ratio > maximum_missing_ratio:
            warnings.append("high_missingness")
        if high_cardinality_threshold and unique_count > high_cardinality_threshold:
            warnings.append("high_cardinality")
        columns[name] = {
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_percentage": round(missing_ratio * 100, 4),
            "unique_count": unique_count,
            "is_constant": bool(series.nunique(dropna=False) <= 1),
            "role": role,
            "is_sensitive": name in sensitive,
            "model_eligible": role == "feature",
            "warnings": warnings,
        }
    return {
        "profile_schema_version": "1.0",
        "data_schema_version": config.get("schema_version", "unversioned"),
        "feature_contract_version": feature_contract_version,
        "row_count": row_count,
        "column_count": len(dataframe.columns),
        "duplicate_row_count": int(dataframe.duplicated().sum()),
        "memory_bytes": int(dataframe.memory_usage(deep=True).sum()),
        "source_sha256": fingerprint,
        "columns": columns,
        "column_disposition_counts": {
            role: sum(details["role"] == role for details in columns.values())
            for role in ("target", "identifier", "feature", "dropped", "unassigned")
        },
    }


def render_profile_markdown(profile: dict[str, Any]) -> str:
    lines = [
        "# Dataset Audit",
        "",
        f"- Rows: {profile['row_count']:,}",
        f"- Columns: {profile['column_count']}",
        f"- Duplicate rows: {profile['duplicate_row_count']:,}",
        f"- Source SHA-256: `{profile['source_sha256']}`",
        "",
        "| Column | Role | Type | Missing | Missing % | Unique | Constant | Warnings |",
        "| --- | --- | --- | ---: | ---: | ---: | --- | --- |",
    ]
    for name, details in profile["columns"].items():
        lines.append(
            f"| {name} | {details.get('role', 'unassigned')} | {details['dtype']} | "
            f"{details['null_count']} | "
            f"{details['null_percentage']:.2f} | {details['unique_count']} | "
            f"{details['is_constant']} | {', '.join(details.get('warnings', [])) or '-'} |"
        )
    return "\n".join(lines) + "\n"
