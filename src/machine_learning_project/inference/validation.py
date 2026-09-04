from __future__ import annotations

from typing import Any

import pandas as pd

from ..features.selection import configured_feature_columns
from ..utils.exceptions import DataValidationError


def validate_inference_frame(dataframe: pd.DataFrame, data_config: dict[str, Any]) -> None:
    required = ["content_id", *configured_feature_columns(data_config)]
    missing = [column for column in required if column not in dataframe]
    if missing:
        raise DataValidationError(f"Inference columns are missing: {missing}")
    if dataframe.empty:
        raise DataValidationError("Inference input contains no rows.")
    if dataframe["content_id"].isna().any():
        raise DataValidationError("content_id contains missing values.")
    if dataframe["content_id"].duplicated().any():
        raise DataValidationError("content_id must be unique in an inference batch.")
