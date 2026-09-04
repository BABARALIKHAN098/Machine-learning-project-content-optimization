from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ..features.selection import select_features
from ..utils.exceptions import DataValidationError


@dataclass(frozen=True)
class PreparedData:
    features: pd.DataFrame
    target: pd.Series
    identifiers: pd.DataFrame
    disposition: dict[str, str]
    feature_contract_version: str


def prepare_supervised_data(
    dataframe: pd.DataFrame,
    data_config: dict[str, Any],
    *,
    feature_contract_version: str = "1.0",
) -> PreparedData:
    target_name = data_config.get("target_column")
    if not target_name or target_name not in dataframe:
        raise DataValidationError(f"Target column not found: {target_name}")

    features = select_features(dataframe, data_config)
    target = dataframe[target_name].copy()
    id_columns = list(data_config.get("id_columns", []))
    identifiers = dataframe.loc[:, id_columns].copy()
    selected = set(features.columns)
    ids = set(id_columns)
    dropped = set(data_config.get("drop_columns", []))
    disposition: dict[str, str] = {}
    for column in dataframe.columns:
        if column == target_name:
            disposition[column] = "target"
        elif column in ids:
            disposition[column] = "identifier"
        elif column in selected:
            disposition[column] = "feature"
        elif column in dropped:
            disposition[column] = "dropped"
        else:
            raise DataValidationError(f"Column has no disposition: {column}")
    if len(features) != len(dataframe) or not features.index.equals(dataframe.index):
        raise DataValidationError("Feature preparation changed row count or order.")
    return PreparedData(features, target, identifiers, disposition, feature_contract_version)
