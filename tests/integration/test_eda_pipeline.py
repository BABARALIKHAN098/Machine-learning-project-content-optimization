from pathlib import Path

import pandas as pd

from pipelines.data_pipeline import run_eda


def test_eda_pipeline_creates_artifacts_without_mutating_source(tmp_path):
    source = tmp_path / "data.csv"
    frame = pd.DataFrame(
        {
            "content_id": ["a", "b", "c", "d"],
            "client_id": ["private-a", "private-a", "private-b", "private-b"],
            "value": [1.0, None, 3.0, 4.0],
            "kind": ["x", "x", "y", "y"],
            "trend_pct": [-2.0, -1.0, 1.0, 2.0],
            "trend_direction": ["down", "down", "up", "up"],
        }
    )
    frame.to_csv(source, index=False)
    before = source.read_bytes()
    data_config = {
        "schema_version": "test",
        "csv_path": str(source),
        "target_column": "trend_direction",
        "allowed_target_values": ["down", "up"],
        "required_columns": list(frame.columns),
        "id_columns": ["content_id", "client_id"],
        "numeric_columns": ["value"],
        "categorical_columns": ["kind"],
        "drop_columns": ["trend_pct"],
        "unique_columns": ["content_id"],
        "allow_extra_columns": False,
    }
    eda_config = {
        "artifact_schema_version": "test",
        "quantiles": [0.25, 0.5, 0.75],
        "iqr_multiplier": 1.5,
        "rare_category_min_count": 2,
        "high_correlation_threshold": 0.9,
        "minimum_group_support": 2,
        "maximum_categories_displayed": 10,
        "plot_dpi": 72,
        "target_cohorts": ["client_id", "kind"],
        "priority_numeric_columns": ["value"],
        "output_directory": str(tmp_path / "unused"),
        "figures_directory": str(tmp_path / "unused-figures"),
        "leakage_name_patterns": ["_last_30d"],
        "target_derived_columns": ["trend_pct"],
        "rate_domains": {},
    }
    output = tmp_path / "eda"
    figures = tmp_path / "figures"
    results, artifacts = run_eda(
        data_config,
        eda_config,
        output_directory=str(output),
        figures_directory=str(figures),
    )
    assert source.read_bytes() == before
    assert results.summary["row_count"] == 4
    assert (output / "eda_report.md").is_file()
    assert (figures / "target_distribution.png").is_file()
    assert all(Path(path).is_file() for path in artifacts)
    report = (output / "eda_report.md").read_text(encoding="utf-8")
    crosstabs = (output / "target_crosstabs.csv").read_text(encoding="utf-8")
    assert "private-a" not in report
    assert "private-a" not in crosstabs
