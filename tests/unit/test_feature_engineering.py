import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone

from machine_learning_project.features.engineering import CutoffSafeFeatureEngineer
from machine_learning_project.features.preprocessing import build_preprocessor_from_config
from machine_learning_project.utils.exceptions import DataValidationError


def config():
    return {
        "feature_contract_version": "2.0",
        "registry_schema_version": "1.0",
        "engineering_version": "1.0",
        "families": ["ratios", "logs"],
        "log_sources": ["impressions_prev_30d"],
    }


def data_config():
    return {"numeric_columns": ["impressions_prev_30d", "clicks_prev_30d", "sessions_prev_30d"]}


def frame():
    return pd.DataFrame(
        {
            "impressions_prev_30d": [10.0, 10.0, 0.0, np.nan],
            "clicks_prev_30d": [2.0, 0.0, 0.0, 2.0],
            "sessions_prev_30d": [4.0, 1.0, 0.0, np.nan],
        },
        index=[8, 2, 5, 1],
    )


def test_exact_ratios_missingness_logs_and_nonmutation():
    raw = frame()
    original = raw.copy(deep=True)
    transformer = clone(CutoffSafeFeatureEngineer(data_config(), config()))
    result = transformer.fit_transform(raw)
    assert result.previous_ctr.iloc[:2].tolist() == [0.2, 0.0]
    assert result.previous_ctr.iloc[2:].isna().all()
    assert result.previous_ctr_unavailable.tolist() == [0.0, 0.0, 1.0, 1.0]
    assert result.previous_sessions_per_click.iloc[0] == 2
    assert result.log1p_impressions_prev_30d.iloc[2] == 0
    assert result.index.equals(raw.index)
    assert list(transformer.get_feature_names_out()) == list(result)
    pd.testing.assert_frame_equal(raw, original)
    pd.testing.assert_frame_equal(transformer.transform(raw.iloc[::-1]), result.iloc[::-1])


@pytest.mark.parametrize("value", [-1.0, np.inf, -np.inf, "bad"])
def test_invalid_numeric_rejected(value):
    raw = frame().astype(object)
    raw.iloc[0, 0] = value
    with pytest.raises(DataValidationError):
        CutoffSafeFeatureEngineer(data_config(), config()).fit_transform(raw)


def test_missing_dependency_and_overflow_rejected():
    transformer = CutoffSafeFeatureEngineer(data_config(), config())
    with pytest.raises(DataValidationError, match="Missing dependencies"):
        transformer.fit(frame().drop(columns="clicks_prev_30d"))
    raw = frame()
    raw.iloc[0, 0] = 1e-300
    raw.iloc[0, 1] = 1e300
    with pytest.raises(DataValidationError, match="Overflow"):
        transformer.fit(raw)


def test_all_missing_column_preserved_and_validation_does_not_fit():
    preprocessing = {
        "feature_contract_version": "1.0",
        "numeric_imputation": "median",
        "categorical_imputation": "__MISSING__",
        "categorical_encoding": "one_hot",
        "scale_numeric": True,
        "handle_unknown_categories": "ignore",
    }
    raw = pd.DataFrame({"empty": [np.nan, np.nan], "value": [1.0, 3.0], "category": ["a", "b"]})
    processor = build_preprocessor_from_config(["empty", "value"], ["category"], preprocessing)
    processor.fit(raw)
    imputer = processor.named_transformers_["numeric"].named_steps["imputer"]
    assert imputer.statistics_.tolist() == [0.0, 2.0]
    before_names = processor.get_feature_names_out().copy()
    output = processor.transform(
        pd.DataFrame({"empty": [999.0], "value": [1e5], "category": ["z"]})
    )
    assert output.shape[1] == len(before_names)
    assert imputer.statistics_.tolist() == [0.0, 2.0]
    assert processor.named_transformers_["categorical"].named_steps["encoder"].categories_[
        0
    ].tolist() == ["a", "b"]


def test_entirely_missing_categorical_column_and_null_normalization():
    from machine_learning_project.utils.config import load_yaml

    raw = pd.DataFrame({"kind": [None, None]})
    feature_config = config()
    feature_config.update({"families": [], "log_sources": []})
    engineered = CutoffSafeFeatureEngineer(
        {"categorical_columns": ["kind"]}, feature_config
    ).fit_transform(raw)
    pre = load_yaml("configs/preprocessing.yaml")["preprocessing"]
    processor = build_preprocessor_from_config([], ["kind"], pre)
    output = processor.fit_transform(engineered)
    assert output.shape == (2, 1)
    assert processor.get_feature_names_out().tolist() == ["categorical__kind___MISSING__"]
    assert processor.transform(pd.DataFrame({"kind": ["unknown"]})).shape == (1, 1)
