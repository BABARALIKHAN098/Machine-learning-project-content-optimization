import pandas as pd
import pytest

from machine_learning_project.data.ingestion import load_csv
from machine_learning_project.utils.exceptions import DataValidationError


def test_load_valid_csv(tmp_path):
    path = tmp_path / "valid.csv"
    pd.DataFrame({"feature": [1, 2], "target": [0, 1]}).to_csv(path, index=False)
    result = load_csv({"csv_path": str(path)})
    assert result.dataframe.shape == (2, 2)
    assert len(result.sha256) == 64


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv({"csv_path": str(tmp_path / "missing.csv")})


def test_empty_and_header_only_csv_are_rejected(tmp_path):
    empty = tmp_path / "empty.csv"
    empty.write_bytes(b"")
    with pytest.raises(ValueError, match="empty"):
        load_csv({"csv_path": str(empty)})

    header_only = tmp_path / "header.csv"
    header_only.write_text("feature,target\n", encoding="utf-8")
    with pytest.raises(ValueError, match="no data rows"):
        load_csv({"csv_path": str(header_only)})


def test_custom_parsing_and_header_collision(tmp_path):
    valid = tmp_path / "custom.csv"
    valid.write_text("feature;target\nNA;up\n", encoding="utf-8")
    result = load_csv(
        {"csv_path": str(valid), "delimiter": ";", "missing_value_tokens": ["NA"]}
    )
    assert pd.isna(result.dataframe.loc[0, "feature"])

    duplicate = tmp_path / "duplicate.csv"
    duplicate.write_text(" name,name \n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate column names"):
        load_csv({"csv_path": str(duplicate)})


def test_undecodable_csv_has_normalized_error(tmp_path):
    path = tmp_path / "invalid.csv"
    path.write_bytes(b"a,b\n\xff,1\n")
    with pytest.raises(DataValidationError, match="Unable to read CSV"):
        load_csv({"csv_path": str(path), "encoding": "utf-8"})
