import pandas as pd
import pytest

from machine_learning_project.data.validation import validate_schema
from machine_learning_project.utils.config import validate_data_config
from machine_learning_project.utils.exceptions import DataValidationError


def test_missing_target_is_reported():
    dataframe = pd.DataFrame({"feature": [1, 2]})
    errors = validate_schema(dataframe, {"target_column": "target"})
    assert any("Target column not found" in error for error in errors)


def test_valid_target_has_no_errors():
    dataframe = pd.DataFrame({"feature": [1, 2], "target": [0, 1]})
    assert validate_schema(dataframe, {"target_column": "target"}) == []


def test_unknown_target_and_duplicate_identifier_are_reported():
    dataframe = pd.DataFrame(
        {"content_id": ["same", "same"], "target": ["down", "mystery"]}
    )
    errors = validate_schema(
        dataframe,
        {
            "target_column": "target",
            "allowed_target_values": ["down", "up"],
            "unique_columns": ["content_id"],
            "id_columns": ["content_id"],
        },
    )
    assert any("Unknown target values" in error for error in errors)
    assert any("must be unique" in error for error in errors)


def test_missing_target_values_and_configured_references_are_reported():
    dataframe = pd.DataFrame({"id": ["a", None], "target": ["down", None]})
    errors = validate_schema(
        dataframe,
        {
            "target_column": "target",
            "id_columns": ["id"],
            "unique_columns": ["id"],
            "numeric_columns": ["missing_feature"],
            "categorical_columns": [],
            "drop_columns": [],
        },
    )
    assert any("Target column contains missing" in error for error in errors)
    assert any("Unique column contains missing" in error for error in errors)
    assert any("Configured columns not found" in error for error in errors)


def test_unassigned_column_is_rejected_by_default():
    dataframe = pd.DataFrame({"feature": [1, 2], "extra": [3, 4], "target": [0, 1]})
    errors = validate_schema(
        dataframe,
        {
            "target_column": "target",
            "numeric_columns": ["feature"],
            "categorical_columns": [],
            "drop_columns": [],
        },
    )
    assert any("Columns have no configured role: ['extra']" in error for error in errors)


def test_data_config_rejects_duplicate_role_entries():
    with pytest.raises(DataValidationError, match="numeric_columns contains duplicate"):
        validate_data_config({"numeric_columns": ["value", "value"]})
