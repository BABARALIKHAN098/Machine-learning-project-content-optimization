from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..features.preprocessing import build_preprocessor_from_config


def build_candidates(
    data_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
    training_config: dict[str, Any],
) -> dict[str, Pipeline]:
    numeric = data_config.get("numeric_columns", [])
    categorical = data_config.get("categorical_columns", [])
    scale = bool(preprocessing_config.get("scale_numeric", True))
    seed = int(training_config.get("random_seed", 42))
    settings = training_config.get("candidates", {})
    logistic = settings.get("logistic_regression", {})
    forest = settings.get("random_forest", {})
    return {
        "logistic_regression": Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor_from_config(
                        numeric, categorical, preprocessing_config, scale_numeric=scale
                    ),
                ),
                (
                    "model",
                    LogisticRegression(
                        C=float(logistic.get("C", 1.0)),
                        max_iter=int(logistic.get("max_iter", 1000)),
                        class_weight=logistic.get("class_weight", "balanced"),
                        random_state=seed,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                (
                    "preprocessor",
                    build_preprocessor_from_config(
                        numeric, categorical, preprocessing_config, scale_numeric=False
                    ),
                ),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=int(forest.get("n_estimators", 200)),
                        min_samples_leaf=int(forest.get("min_samples_leaf", 2)),
                        max_features=forest.get("max_features", "sqrt"),
                        class_weight=forest.get("class_weight", "balanced_subsample"),
                        n_jobs=int(forest.get("n_jobs", -1)),
                        random_state=seed,
                    ),
                ),
            ]
        ),
    }
