import numpy as np
import pandas as pd
import pytest

from machine_learning_project.data.preparation import prepare_supervised_data
from machine_learning_project.features.preprocessing import (
    build_preprocessor,
    build_preprocessor_from_config,
)
from machine_learning_project.features.selection import select_features
from machine_learning_project.utils.exceptions import DataValidationError


def test_preprocessor_handles_missing_and_unseen_values():
    train = pd.DataFrame(
        {"age": [20.0, np.nan, 40.0], "city": ["A", "B", None]}
    )
    inference = pd.DataFrame({"age": [30.0], "city": ["UNSEEN"]})
    transformer = build_preprocessor(["age"], ["city"])
    transformer.fit(train)
    output = transformer.transform(inference)
    assert output.shape[0] == 1


def test_feature_selection_excludes_ids_target_and_leakage_columns():
    dataframe = pd.DataFrame(
        {
            "content_id": ["a"],
            "feature": ["2"],
            "category": ["x"],
            "trend_pct": [-10.0],
            "target": ["down"],
        }
    )
    config = {
        "target_column": "target",
        "id_columns": ["content_id"],
        "drop_columns": ["trend_pct"],
        "numeric_columns": ["feature"],
        "categorical_columns": ["category"],
    }
    selected = select_features(dataframe, config)
    assert selected.columns.tolist() == ["feature", "category"]
    assert selected["feature"].iloc[0] == 2


def test_feature_contract_rejects_leakage():
    dataframe = pd.DataFrame({"target": ["down"]})
    config = {
        "target_column": "target",
        "numeric_columns": ["target"],
        "categorical_columns": [],
    }
    with pytest.raises(DataValidationError):
        select_features(dataframe, config)


def _preprocessing_config():
    return {
        "feature_contract_version": "1.0",
        "numeric_imputation": "median",
        "categorical_imputation": "__MISSING__",
        "categorical_encoding": "one_hot",
        "scale_numeric": True,
        "handle_unknown_categories": "ignore",
    }


def test_preparation_separates_target_ids_and_dropped_columns_without_mutation():
    dataframe = pd.DataFrame(
        {
            "id": ["a", "b"],
            "number": ["1", "2"],
            "category": ["x", "y"],
            "leak": [10, 20],
            "target": ["down", "up"],
        },
        index=[4, 8],
    )
    original = dataframe.copy(deep=True)
    prepared = prepare_supervised_data(
        dataframe,
        {
            "target_column": "target",
            "id_columns": ["id"],
            "drop_columns": ["leak"],
            "numeric_columns": ["number"],
            "categorical_columns": ["category"],
        },
    )
    assert prepared.features.columns.tolist() == ["number", "category"]
    assert prepared.target.tolist() == ["down", "up"]
    assert prepared.identifiers.columns.tolist() == ["id"]
    assert prepared.disposition["leak"] == "dropped"
    pd.testing.assert_frame_equal(dataframe, original)
    assert prepared.features.index.tolist() == [4, 8]


def test_configured_preprocessor_uses_training_statistics_only():
    train = pd.DataFrame({"value": [1.0, np.nan, 3.0], "kind": ["a", "b", None]})
    validation = pd.DataFrame({"value": [1000.0], "kind": ["unseen"]})
    transformer = build_preprocessor_from_config(
        ["value"], ["kind"], _preprocessing_config(), scale_numeric=False
    )
    transformer.fit(train)
    numeric_imputer = transformer.named_transformers_["numeric"].named_steps["imputer"]
    assert numeric_imputer.statistics_.tolist() == [2.0]
    before_width = transformer.transform(train).shape[1]
    assert transformer.transform(validation).shape == (1, before_width)
    assert len(transformer.get_feature_names_out()) == before_width


def test_configured_preprocessor_supports_single_feature_type():
    transformer = build_preprocessor_from_config(
        ["value"], [], _preprocessing_config(), scale_numeric=False
    )
    assert transformer.fit_transform(pd.DataFrame({"value": [1.0, np.nan]})).shape == (2, 1)


def test_invalid_preprocessing_config_is_rejected():
    config = _preprocessing_config()
    config["numeric_imputation"] = "mean"
    with pytest.raises(DataValidationError, match="numeric_imputation"):
        build_preprocessor_from_config(["value"], [], config)
