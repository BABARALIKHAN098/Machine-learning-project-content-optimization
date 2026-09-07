from __future__ import annotations

from typing import Any

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from ..features.engineering import CutoffSafeFeatureEngineer
from ..features.preprocessing import build_preprocessor_from_config
from ..features.registry import resolve_registry
from ..utils.exceptions import DataValidationError


def build_candidates(
    data_config: dict[str, Any],
    preprocessing_config: dict[str, Any],
    training_config: dict[str, Any],
    feature_config: dict[str, Any] | None = None,
) -> dict[str, Pipeline]:
    numeric = data_config.get("numeric_columns", [])
    categorical = data_config.get("categorical_columns", [])
    if feature_config is not None:
        if (
            preprocessing_config["feature_contract_version"]
            != feature_config["feature_contract_version"]
        ):
            raise DataValidationError("Preprocessing and feature contract versions disagree")
        registry = resolve_registry(data_config, feature_config)
        numeric = [entry["name"] for entry in registry if entry["role"] == "numeric"]
        categorical = [entry["name"] for entry in registry if entry["role"] == "categorical"]
    scale = bool(preprocessing_config.get("scale_numeric", True))
    seed = int(training_config.get("random_seed", 42))
    settings = training_config.get("candidates", {})
    logistic = settings.get("logistic_regression", {})
    forest = settings.get("random_forest", {})
    candidates = {
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
    if feature_config is not None:
        for pipeline in candidates.values():
            pipeline.steps.insert(
                0, ("engineering", CutoffSafeFeatureEngineer(data_config, feature_config))
            )
    return candidates
