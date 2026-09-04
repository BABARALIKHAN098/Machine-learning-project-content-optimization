import numpy as np
import pandas as pd
import pytest

from machine_learning_project.data.eda import (
    analyze_anomalies,
    analyze_associations,
    analyze_categorical,
    analyze_missingness,
    analyze_numeric,
    analyze_target,
    build_leakage_register,
)
from machine_learning_project.utils.config import validate_eda_config
from machine_learning_project.utils.exceptions import DataValidationError


def eda_config():
    return {
        "artifact_schema_version": "1.0",
        "quantiles": [0.25, 0.5, 0.75],
        "iqr_multiplier": 1.5,
        "rare_category_min_count": 2,
        "high_correlation_threshold": 0.9,
        "minimum_group_support": 2,
        "maximum_categories_displayed": 10,
        "plot_dpi": 72,
        "target_cohorts": ["client_id", "kind"],
        "priority_numeric_columns": ["value"],
        "leakage_name_patterns": ["_last_30d"],
        "target_derived_columns": ["trend_pct"],
        "rate_domains": {"rate": [0.0, 1.0]},
    }


def test_numeric_and_categorical_summaries_are_exact():
    frame = pd.DataFrame(
        {
            "value": [0.0, 1.0, 2.0, 100.0, np.nan],
            "kind": ["a", "a", "b", None, "rare"],
        }
    )
    numeric = analyze_numeric(frame, eda_config()).set_index("column")
    assert numeric.loc["value", "count"] == 4
    assert numeric.loc["value", "zero_count"] == 1
    assert numeric.loc["value", "outlier_count"] == 1
    categorical = analyze_categorical(frame, eda_config())
    missing = categorical.query("column == 'kind' and level == '__MISSING__'").iloc[0]
    assert missing["count"] == 1
    assert bool(missing["is_rare"])


def test_target_masks_clients_and_reports_distribution():
    frame = pd.DataFrame(
        {
            "client_id": ["secret-a", "secret-a", "secret-b", "secret-b"],
            "target": ["down", "down", "up", "down"],
        }
    )
    summary, crosstabs = analyze_target(frame, "target", ["client_id"], 2)
    assert summary["counts"] == {"down": 3, "up": 1}
    assert not crosstabs["group"].str.contains("secret").any()
    assert crosstabs["support_sufficient"].all()


def test_missingness_and_associations_detect_known_relationships():
    frame = pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, 4.0],
            "b": [2.0, 4.0, 6.0, 8.0],
            "kind": ["x", "x", "y", "y"],
            "target": ["down", "down", "up", "up"],
        }
    )
    missing = analyze_missingness(frame, "target", 0.2)
    a = missing.query("kind == 'column' and name == 'a'").iloc[0]
    assert a["count"] == 1
    assert bool(a["exceeds_threshold"])
    associations = analyze_associations(frame, "target", 0.9)
    pair = associations.query("association_type == 'numeric_pair'").iloc[0]
    assert pair["spearman"] == pytest.approx(1.0)
    assert bool(pair["high_association"])


def test_anomalies_and_leakage_register_are_traceable():
    frame = pd.DataFrame(
        {
            "id": ["a", "b"],
            "clicks_prev_30d": [2, -1],
            "impressions_prev_30d": [1, 4],
            "rate": [0.5, 1.5],
            "metric_last_30d": [10, 20],
            "trend_pct": [-5.0, 2.0],
            "target": ["down", "up"],
        }
    )
    data_config = {
        "target_column": "target",
        "id_columns": ["id"],
        "numeric_columns": ["clicks_prev_30d", "impressions_prev_30d"],
        "categorical_columns": [],
        "drop_columns": ["rate", "metric_last_30d", "trend_pct"],
        "non_negative_columns": ["clicks_prev_30d"],
    }
    anomalies = analyze_anomalies(frame, data_config, eda_config())
    assert anomalies["violation_count"].sum() == 3
    leakage = build_leakage_register(frame, data_config, eda_config())
    assert len(leakage) == len(frame.columns)
    assert not leakage.set_index("column").loc["trend_pct", "model_eligible"]
    assert leakage.set_index("column").loc["trend_pct", "leakage_category"] == "target_derived"


def test_invalid_eda_config_is_rejected():
    config = eda_config()
    config["quantiles"] = [0.5, 0.25]
    with pytest.raises(DataValidationError, match="quantiles"):
        validate_eda_config(config)
