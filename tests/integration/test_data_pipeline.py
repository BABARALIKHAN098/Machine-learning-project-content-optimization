import pandas as pd

from pipelines.data_pipeline import run_data_review


def test_data_pipeline_profiles_valid_csv(tmp_path):
    path = tmp_path / "dataset.csv"
    pd.DataFrame({"content_id": ["a", "b"], "target": ["down", "up"]}).to_csv(path, index=False)
    profile = run_data_review(
        {
            "csv_path": str(path),
            "target_column": "target",
            "required_columns": ["content_id", "target"],
            "id_columns": ["content_id"],
            "allowed_target_values": ["down", "up"],
            "unique_columns": ["content_id"],
        }
    )
    assert profile["row_count"] == 2
    assert len(profile["source_sha256"]) == 64
    assert profile["column_disposition_counts"] == {
        "target": 1,
        "identifier": 1,
        "feature": 0,
        "dropped": 0,
        "unassigned": 0,
    }


def test_data_pipeline_does_not_modify_source(tmp_path):
    path = tmp_path / "dataset.csv"
    path.write_text("content_id,target\na,down\nb,up\n", encoding="utf-8")
    before = path.read_bytes()
    config = {
        "csv_path": str(path),
        "target_column": "target",
        "required_columns": ["content_id", "target"],
        "id_columns": ["content_id"],
        "allowed_target_values": ["down", "up"],
        "unique_columns": ["content_id"],
    }
    first = run_data_review(config)
    second = run_data_review(config)
    assert path.read_bytes() == before
    assert first == second
