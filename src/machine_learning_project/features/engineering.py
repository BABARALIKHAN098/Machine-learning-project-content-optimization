from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted

from ..utils.exceptions import DataValidationError
from .registry import RATIOS, resolve_registry


class CutoffSafeFeatureEngineer(TransformerMixin, BaseEstimator):
    def __init__(self, data_config, feature_config):
        self.data_config = data_config
        self.feature_config = feature_config

    def fit(self, X, y=None):
        if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any():
            raise DataValidationError("Engineering requires a dataframe with unique columns")
        self.registry_ = resolve_registry(self.data_config, self.feature_config)
        self.feature_names_in_ = np.asarray(X.columns, dtype=object)
        self.n_features_in_ = len(X.columns)
        self.transform(X)
        return self

    def transform(self, X):
        check_is_fitted(self, "registry_")
        if not isinstance(X, pd.DataFrame) or X.columns.duplicated().any():
            raise DataValidationError("Engineering requires a dataframe with unique columns")
        required = {dep for entry in self.registry_ for dep in entry["dependencies"]}
        if required - set(X.columns):
            raise DataValidationError(f"Missing dependencies: {sorted(required - set(X.columns))}")
        working = X.loc[:, sorted(required)].copy()
        for name in set(self.data_config.get("numeric_columns", [])) & required:
            try:
                values = pd.to_numeric(working[name], errors="raise").astype(float)
            except (ValueError, TypeError) as exc:
                raise DataValidationError(f"Non-numeric feature: {name}") from exc
            if np.isinf(values).any() or (values.dropna() < 0).any():
                raise DataValidationError(f"Negative or non-finite feature: {name}")
            working[name] = values
        result = pd.DataFrame(index=X.index)
        for entry in self.registry_:
            name, operation = entry["name"], entry["operation"]
            dependencies = entry["dependencies"]
            if operation == "identity":
                values = working[dependencies[0]]
                result[name] = values.where(values.notna(), np.nan)
            elif operation == "log1p":
                result[name] = np.log1p(working[dependencies[0]])
            else:
                numerator, denominator = (working[d] for d in dependencies)
                unavailable = numerator.isna() | denominator.isna() | denominator.eq(0)
                if operation == "unavailable":
                    result[name] = unavailable.astype(float)
                else:
                    with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                        result[name] = numerator / denominator.where(~unavailable)
                    if np.isinf(result[name]).any():
                        raise DataValidationError(f"Overflow in derived feature: {name}")
        return result

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "registry_")
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise DataValidationError("Input feature names disagree with fitted schema")
        return np.asarray([entry["name"] for entry in self.registry_], dtype=object)


def add_cutoff_safe_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy using v2 missing-ratio semantics and explicit dependencies."""
    sources = list(dict.fromkeys(dep for pair in RATIOS.values() for dep in pair))
    config = {
        "feature_contract_version": "2.0",
        "registry_schema_version": "1.0",
        "engineering_version": "1.0",
        "families": ["ratios"],
    }
    derived = CutoffSafeFeatureEngineer({"numeric_columns": sources}, config).fit_transform(
        dataframe
    )
    result = dataframe.copy()
    for name in derived.columns:
        if name not in sources:
            result[name] = derived[name]
    return result
