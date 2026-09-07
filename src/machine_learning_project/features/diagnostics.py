"""Aggregate diagnostics. Call only with the training population."""

import pandas as pd


def training_diagnostics(frame, registry):
    numeric = [entry["name"] for entry in registry if entry["role"] == "numeric"]
    columns = {}
    for name in frame:
        series = frame[name]
        item = {
            "missing_count": int(series.isna().sum()),
            "distinct_count": int(series.nunique()),
            "constant": bool(series.nunique() <= 1),
        }
        if name in numeric:
            item.update(
                {
                    "minimum": None if series.dropna().empty else float(series.min()),
                    "maximum": None if series.dropna().empty else float(series.max()),
                    "empty_numeric_fallback": 0 if series.isna().all() else None,
                }
            )
        columns[name] = item
    correlation = frame[numeric].corr() if numeric else pd.DataFrame()
    pairs = [
        {"left": left, "right": right, "correlation": float(correlation.loc[left, right])}
        for index, left in enumerate(numeric)
        for right in numeric[index + 1 :]
        if pd.notna(correlation.loc[left, right]) and abs(correlation.loc[left, right]) >= 0.95
    ]
    return {
        "fitting_population": "train",
        "row_count": len(frame),
        "columns": columns,
        "high_correlation_pairs": pairs,
    }
