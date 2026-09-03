import pandas as pd

from machine_learning_project.data.validation import validate_schema


def test_missing_target_is_reported():
    dataframe = pd.DataFrame({"feature": [1, 2]})
    errors = validate_schema(dataframe, {"target_column": "target"})
    assert any("Target column not found" in error for error in errors)


def test_valid_target_has_no_errors():
    dataframe = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    assert validate_schema(dataframe, {"target_column": "target"}) == []
