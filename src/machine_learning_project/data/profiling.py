from __future__ import annotations

from typing import Any

import pandas as pd


def build_profile(dataframe: pd.DataFrame, fingerprint: str) -> dict[str, Any]:
    row_count = len(dataframe)
    columns: dict[str, Any] = {}
    for name in dataframe.columns:
        series = dataframe[name]
        null_count = int(series.isna().sum())
        columns[name] = {
            "dtype": str(series.dtype),
            "null_count": null_count,
            "null_percentage": round((null_count / row_count) * 100, 4),
            "unique_count": int(series.nunique(dropna=True)),
            "is_constant": bool(series.nunique(dropna=False) <= 1),
        }
    return {
        "row_count": row_count,
        "column_count": len(dataframe.columns),
        "duplicate_row_count": int(dataframe.duplicated().sum()),
        "memory_bytes": int(dataframe.memory_usage(deep=True).sum()),
        "source_sha256": fingerprint,
        "columns": columns,
    }
