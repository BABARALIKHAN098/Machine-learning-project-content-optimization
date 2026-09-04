from __future__ import annotations

from collections.abc import Sequence

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from ..utils.config import validate_preprocessing_config
from ..utils.exceptions import DataValidationError


def build_preprocessor(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    *,
    scale_numeric: bool = True,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median"))
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(numeric_steps)
    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="constant", fill_value="__MISSING__"),
            ),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=True),
            ),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(numeric_columns)),
            ("categorical", categorical_pipeline, list(categorical_columns)),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def build_preprocessor_from_config(
    numeric_columns: Sequence[str],
    categorical_columns: Sequence[str],
    config: dict,
    *,
    scale_numeric: bool | None = None,
) -> ColumnTransformer:
    validate_preprocessing_config(config)
    if not numeric_columns and not categorical_columns:
        raise DataValidationError("At least one approved feature is required.")
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy=config["numeric_imputation"]))
    ]
    should_scale = config["scale_numeric"] if scale_numeric is None else scale_numeric
    if should_scale:
        numeric_steps.append(("scaler", StandardScaler()))
    transformers: list[tuple[str, object, list[str]]] = []
    if numeric_columns:
        transformers.append(("numeric", Pipeline(numeric_steps), list(numeric_columns)))
    if categorical_columns:
        categorical_pipeline = Pipeline(
            [
                (
                    "imputer",
                    SimpleImputer(
                        strategy="constant", fill_value=config["categorical_imputation"]
                    ),
                ),
                (
                    "encoder",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                ),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, list(categorical_columns)))
    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
        verbose_feature_names_out=True,
    )
