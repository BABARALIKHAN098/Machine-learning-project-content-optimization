from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class EDAResults:
    summary: dict[str, Any]
    numeric_summary: pd.DataFrame
    categorical_summary: pd.DataFrame
    target_crosstabs: pd.DataFrame
    missingness_summary: pd.DataFrame
    associations: pd.DataFrame
    anomalies: pd.DataFrame
    leakage_register: pd.DataFrame


def _cramers_v(left: pd.Series, right: pd.Series) -> float:
    table = pd.crosstab(left, right)
    if table.empty or min(table.shape) < 2:
        return 0.0
    observed = table.to_numpy(dtype=float)
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0)) / observed.sum()
    contributions = np.zeros_like(expected)
    np.divide((observed - expected) ** 2, expected, out=contributions, where=expected > 0)
    chi2 = float(contributions.sum())
    n = observed.sum()
    phi2 = chi2 / n
    rows, columns = observed.shape
    corrected = max(0.0, phi2 - ((columns - 1) * (rows - 1)) / max(n - 1, 1))
    corrected_rows = rows - ((rows - 1) ** 2) / max(n - 1, 1)
    corrected_columns = columns - ((columns - 1) ** 2) / max(n - 1, 1)
    denominator = min(corrected_columns - 1, corrected_rows - 1)
    return math.sqrt(corrected / denominator) if denominator > 0 else 0.0


def _eta_squared(values: pd.Series, target: pd.Series) -> float:
    valid = values.notna() & target.notna()
    numeric = values[valid].astype(float)
    groups = target[valid]
    if numeric.empty or numeric.nunique() <= 1:
        return 0.0
    grand_mean = numeric.mean()
    between = sum(
        len(group) * (group.mean() - grand_mean) ** 2
        for _, group in numeric.groupby(groups, observed=True)
    )
    total = float(((numeric - grand_mean) ** 2).sum())
    return float(between / total) if total else 0.0


def analyze_numeric(dataframe: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    quantiles = config["quantiles"]
    multiplier = float(config["iqr_multiplier"])
    rows: list[dict[str, Any]] = []
    for column in dataframe.select_dtypes(include="number").columns:
        series = pd.to_numeric(dataframe[column], errors="coerce")
        finite = series.replace([np.inf, -np.inf], np.nan).dropna()
        first = finite.quantile(0.25) if not finite.empty else np.nan
        third = finite.quantile(0.75) if not finite.empty else np.nan
        iqr = third - first
        outliers = (
            int(((finite < first - multiplier * iqr) | (finite > third + multiplier * iqr)).sum())
            if not finite.empty
            else 0
        )
        row: dict[str, Any] = {
            "column": column,
            "count": int(finite.size),
            "missing_count": int(series.isna().sum()),
            "missing_percentage": round(float(series.isna().mean() * 100), 4),
            "zero_count": int((finite == 0).sum()),
            "negative_count": int((finite < 0).sum()),
            "non_finite_count": int(np.isinf(series.to_numpy(dtype=float, na_value=np.nan)).sum()),
            "mean": float(finite.mean()) if not finite.empty else np.nan,
            "std": float(finite.std()) if finite.size > 1 else np.nan,
            "min": float(finite.min()) if not finite.empty else np.nan,
            "max": float(finite.max()) if not finite.empty else np.nan,
            "skew": float(finite.skew()) if finite.size > 2 else np.nan,
            "iqr": float(iqr) if not finite.empty else np.nan,
            "outlier_count": outliers,
        }
        for quantile in quantiles:
            row[f"q{int(quantile * 100):02d}"] = (
                float(finite.quantile(quantile)) if not finite.empty else np.nan
            )
        rows.append(row)
    return pd.DataFrame(rows).sort_values("column").reset_index(drop=True)


def analyze_categorical(
    dataframe: pd.DataFrame, config: dict[str, Any], excluded_columns: set[str] | None = None
) -> pd.DataFrame:
    threshold = int(config["rare_category_min_count"])
    rows: list[dict[str, Any]] = []
    excluded_columns = excluded_columns or set()
    categorical = [
        column
        for column in dataframe.select_dtypes(exclude="number").columns
        if column not in excluded_columns
    ]
    for column in categorical:
        values = dataframe[column].astype("string").fillna("__MISSING__")
        counts = values.value_counts(dropna=False)
        cumulative = 0.0
        for level, count in counts.items():
            percentage = float(count / len(dataframe) * 100)
            cumulative += percentage
            rows.append(
                {
                    "column": column,
                    "level": str(level),
                    "count": int(count),
                    "percentage": round(percentage, 4),
                    "cumulative_percentage": round(cumulative, 4),
                    "is_rare": int(count) < threshold,
                }
            )
    return pd.DataFrame(rows)


def analyze_target(
    dataframe: pd.DataFrame, target: str, cohorts: list[str], minimum_support: int
) -> tuple[dict[str, Any], pd.DataFrame]:
    counts = dataframe[target].astype("string").value_counts().sort_index()
    probabilities = counts / counts.sum()
    entropy = float(-(probabilities * np.log2(probabilities)).sum())
    summary = {
        "counts": {str(key): int(value) for key, value in counts.items()},
        "proportions": {str(key): round(float(value), 6) for key, value in probabilities.items()},
        "imbalance_ratio": round(float(counts.max() / counts.min()), 6),
        "entropy_bits": round(entropy, 6),
    }
    rows: list[dict[str, Any]] = []
    for cohort in cohorts:
        if cohort not in dataframe:
            continue
        groups = dataframe[cohort].astype("string").fillna("__MISSING__")
        if cohort == "client_id":
            aliases = {value: f"client_{index:02d}" for index, value in enumerate(sorted(groups.unique()))}
            groups = groups.map(aliases)
        table = pd.crosstab(groups, dataframe[target], dropna=False)
        for group, values in table.iterrows():
            total = int(values.sum())
            for label, count in values.items():
                rows.append(
                    {
                        "cohort": cohort,
                        "group": str(group),
                        "target": str(label),
                        "count": int(count),
                        "group_total": total,
                        "percentage": round(float(count / total * 100), 4),
                        "support_sufficient": total >= minimum_support,
                    }
                )
    return summary, pd.DataFrame(rows)


def analyze_missingness(
    dataframe: pd.DataFrame, target: str, maximum_missing_ratio: float
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for column in dataframe.columns:
        indicator = dataframe[column].isna()
        missing_count = int(indicator.sum())
        rows.append(
            {
                "kind": "column",
                "name": column,
                "count": missing_count,
                "percentage": round(float(indicator.mean() * 100), 4),
                "target_cramers_v": (
                    round(_cramers_v(indicator.astype(str), dataframe[target].astype(str)), 6)
                    if 0 < missing_count < len(dataframe)
                    else 0.0
                ),
                "exceeds_threshold": bool(indicator.mean() > maximum_missing_ratio),
            }
        )
    patterns = dataframe.isna().apply(
        lambda row: "|".join(dataframe.columns[row.to_numpy()]) or "__COMPLETE__", axis=1
    )
    for pattern, count in patterns.value_counts().head(20).items():
        rows.append(
            {
                "kind": "pattern",
                "name": pattern,
                "count": int(count),
                "percentage": round(float(count / len(dataframe) * 100), 4),
                "target_cramers_v": np.nan,
                "exceeds_threshold": False,
            }
        )
    return pd.DataFrame(rows)


def analyze_associations(
    dataframe: pd.DataFrame, target: str, threshold: float, excluded_columns: set[str] | None = None
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric = dataframe.select_dtypes(include="number")
    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    columns = list(numeric.columns)
    for index, left in enumerate(columns):
        for right in columns[index + 1 :]:
            p_value = pearson.loc[left, right]
            s_value = spearman.loc[left, right]
            rows.append(
                {
                    "association_type": "numeric_pair",
                    "left": left,
                    "right": right,
                    "pearson": p_value,
                    "spearman": s_value,
                    "effect_size": np.nan,
                    "support": int(dataframe[[left, right]].dropna().shape[0]),
                    "high_association": bool(
                        (pd.notna(p_value) and abs(p_value) >= threshold)
                        or (pd.notna(s_value) and abs(s_value) >= threshold)
                    ),
                }
            )
    for column in columns:
        rows.append(
            {
                "association_type": "numeric_target",
                "left": column,
                "right": target,
                "pearson": np.nan,
                "spearman": np.nan,
                "effect_size": round(_eta_squared(dataframe[column], dataframe[target]), 6),
                "support": int(dataframe[[column, target]].dropna().shape[0]),
                "high_association": False,
            }
        )
    excluded_columns = excluded_columns or set()
    for column in dataframe.select_dtypes(exclude="number").columns:
        if column == target or column in excluded_columns:
            continue
        support = int(dataframe[[column, target]].dropna().shape[0])
        rows.append(
            {
                "association_type": "categorical_target",
                "left": column,
                "right": target,
                "pearson": np.nan,
                "spearman": np.nan,
                "effect_size": round(
                    _cramers_v(dataframe[column].fillna("__MISSING__"), dataframe[target]), 6
                ),
                "support": support,
                "high_association": False,
            }
        )
    return pd.DataFrame(rows)


def analyze_anomalies(
    dataframe: pd.DataFrame, data_config: dict[str, Any], eda_config: dict[str, Any]
) -> pd.DataFrame:
    rules: list[dict[str, Any]] = []

    def add(rule_id: str, severity: str, checked: int, violations: int, guidance: str) -> None:
        rules.append(
            {
                "rule_id": rule_id,
                "severity": severity,
                "checked_rows": checked,
                "violation_count": violations,
                "violation_percentage": round(violations / checked * 100, 4) if checked else 0.0,
                "guidance": guidance,
            }
        )

    add(
        "ANOM-DUPLICATE-ROWS",
        "warning",
        len(dataframe),
        int(dataframe.duplicated().sum()),
        "Review duplicate records; do not delete automatically.",
    )
    for column in data_config.get("non_negative_columns", []):
        if column in dataframe:
            values = pd.to_numeric(dataframe[column], errors="coerce")
            add(
                f"ANOM-NONNEGATIVE-{column.upper()}",
                "error",
                int(values.notna().sum()),
                int((values < 0).sum()),
                "Correct or quarantine values below zero.",
            )
    for column, limits in eda_config.get("rate_domains", {}).items():
        if column in dataframe:
            values = pd.to_numeric(dataframe[column], errors="coerce")
            low, high = limits
            add(
                f"ANOM-RANGE-{column.upper()}",
                "warning",
                int(values.notna().sum()),
                int(((values < low) | (values > high)).sum()),
                f"Confirm that {column} uses the documented [{low}, {high}] scale.",
            )
    for suffix in ("prev_30d", "last_30d", "90d"):
        clicks, impressions = f"clicks_{suffix}", f"impressions_{suffix}"
        if clicks in dataframe and impressions in dataframe:
            valid = dataframe[[clicks, impressions]].dropna()
            add(
                f"ANOM-CLICKS-GT-IMPRESSIONS-{suffix.upper()}",
                "error",
                len(valid),
                int((valid[clicks] > valid[impressions]).sum()),
                "Verify matched-window click and impression aggregation.",
            )
    return pd.DataFrame(rules)


def build_leakage_register(
    dataframe: pd.DataFrame, data_config: dict[str, Any], eda_config: dict[str, Any]
) -> pd.DataFrame:
    target = data_config["target_column"]
    ids = set(data_config.get("id_columns", []))
    dropped = set(data_config.get("drop_columns", []))
    features = set(data_config.get("numeric_columns", [])) | set(
        data_config.get("categorical_columns", [])
    )
    derived = set(eda_config.get("target_derived_columns", []))
    patterns = eda_config.get("leakage_name_patterns", [])
    rows: list[dict[str, Any]] = []
    for column in dataframe.columns:
        if column == target:
            category, rationale, availability = "target", "Prediction label.", "no"
        elif column in ids:
            category, rationale, availability = "identifier", "Join/group use only.", "yes"
        elif column in derived:
            category = "target_derived"
            rationale = "Derived from or overlaps the outcome period."
            availability = "no"
        elif any(pattern in column for pattern in patterns):
            category = "outcome_window"
            rationale = "Outcome or potentially overlapping aggregation window."
            availability = "no"
        elif column in dropped:
            category = "excluded"
            rationale = "Excluded by the approved data contract."
            availability = "unknown"
        else:
            category = "candidate_feature"
            rationale = "Approved cutoff-safe candidate in the current contract."
            availability = "yes"
        rows.append(
            {
                "column": column,
                "role": (
                    "target"
                    if column == target
                    else "identifier"
                    if column in ids
                    else "feature"
                    if column in features
                    else "dropped"
                ),
                "available_at_cutoff": availability,
                "leakage_category": category,
                "model_eligible": column in features and availability == "yes",
                "rationale": rationale,
                "approval_status": "approved" if column in features or column in dropped else "review",
            }
        )
    return pd.DataFrame(rows)


def run_analyses(
    dataframe: pd.DataFrame,
    fingerprint: str,
    data_config: dict[str, Any],
    eda_config: dict[str, Any],
) -> EDAResults:
    target = data_config["target_column"]
    target_summary, crosstabs = analyze_target(
        dataframe,
        target,
        eda_config.get("target_cohorts", []),
        int(eda_config["minimum_group_support"]),
    )
    numeric = analyze_numeric(dataframe, eda_config)
    identifiers = set(data_config.get("id_columns", []))
    categorical = analyze_categorical(dataframe, eda_config, identifiers)
    missingness = analyze_missingness(
        dataframe, target, float(data_config.get("maximum_missing_ratio", 1.0))
    )
    associations = analyze_associations(
        dataframe,
        target,
        float(eda_config["high_correlation_threshold"]),
        identifiers,
    )
    anomalies = analyze_anomalies(dataframe, data_config, eda_config)
    leakage = build_leakage_register(dataframe, data_config, eda_config)
    summary = {
        "artifact_schema_version": eda_config["artifact_schema_version"],
        "source_sha256": fingerprint,
        "data_schema_version": data_config.get("schema_version", "unversioned"),
        "row_count": len(dataframe),
        "column_count": len(dataframe.columns),
        "target": target_summary,
        "high_correlation_pair_count": int(
            associations.query("association_type == 'numeric_pair'")["high_association"].sum()
        ),
        "anomaly_violation_count": int(anomalies["violation_count"].sum()),
        "leakage_prohibited_column_count": int((~leakage["model_eligible"]).sum()),
    }
    return EDAResults(
        summary,
        numeric,
        categorical,
        crosstabs,
        missingness,
        associations,
        anomalies,
        leakage,
    )
