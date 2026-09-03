import pandas as pd
import pytest

from machine_learning_project.data.ingestion import load_csv


def test_load_valid_csv(tmp_path):
    path = tmp_path / "valid.csv"
    pd.DataFrame({"feature": [1, 2], "target": [0, 1]}).to_csv(path, index=False)
    result = load_csv({"csv_path": str(path)})
    assert result.dataframe.shape == (2, 2)
    assert len(result.sha256) == 64


def test_missing_csv_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv({"csv_path": str(tmp_path / "missing.csv")})
