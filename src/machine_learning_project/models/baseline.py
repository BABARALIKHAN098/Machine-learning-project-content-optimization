from __future__ import annotations

import copy
import time
from typing import Any

import numpy as np
from sklearn.dummy import DummyClassifier

from ..features.registry import fingerprint
from ..utils.config import validate_baseline_config
from ..utils.exceptions import DataValidationError
from .evaluate import evaluate_predictions


def resolve_baseline_config(config=None, training_config=None):
    result = (
        copy.deepcopy(config)
        if config is not None
        else {
            "baseline_contract_version": "1.0",
            "metric_contract_version": "1.0",
            "strategies": ["most_frequent", "stratified"],
            "reference_seed": 42,
            "repeat_seeds": [42, 43, 44, 45, 46],
            "minimum_macro_f1_improvement": 0.01,
            "evaluation_partition": "validation",
            "ordering": "row_key",
            "output_directory": "reports/baselines",
        }
    )
    validate_baseline_config(result)
    legacy = (training_config or {}).get("baselines")
    if legacy is not None and legacy != result["strategies"]:
        raise DataValidationError("Legacy training.baselines disagrees with baseline configuration")
    return result


def evaluate_baselines(train_y, validation_y, config, labels):
    """Class-frequency benchmark. No features, identifiers, or test input are accepted."""
    config = resolve_baseline_config(config)
    train_y, validation_y, labels = list(train_y), list(validation_y), list(labels)
    # Shared metric validation also rejects nulls, unknown labels and duplicate class names.
    evaluate_predictions(train_y, train_y, labels)
    evaluate_predictions(validation_y, validation_y, labels)
    if "down" not in labels or set(train_y) != set(labels) or set(validation_y) != set(labels):
        raise DataValidationError(
            "Both benchmark partitions must contain every class, including down"
        )
    train_x = np.zeros((len(train_y), 1))
    validation_x = np.zeros((len(validation_y), 1))
    counts = {label: train_y.count(label) for label in labels}
    majority = min(counts, key=lambda label: (-counts[label], label))
    runs, timings = [], []
    schedule = [("most_frequent", None)] + [("stratified", seed) for seed in config["repeat_seeds"]]
    for strategy, seed in schedule:
        estimator = DummyClassifier(strategy=strategy, random_state=seed)
        started = time.perf_counter()
        estimator.fit(train_x, train_y)
        fitted = time.perf_counter() - started
        started = time.perf_counter()
        predictions = estimator.predict(validation_x)
        predicted = time.perf_counter() - started
        if strategy == "most_frequent" and set(predictions) != {majority}:
            raise DataValidationError("Installed estimator majority tie behavior violates contract")
        metrics = evaluate_predictions(validation_y, predictions, labels)
        runs.append(
            {
                "strategy": strategy,
                "seed": seed,
                "metrics": metrics,
                "prediction_sha256": fingerprint(list(predictions)),
            }
        )
        timings.append(
            {
                "strategy": strategy,
                "seed": seed,
                "fit_seconds": fitted,
                "prediction_seconds": predicted,
            }
        )
    canonical = {
        run["strategy"]: run["metrics"]
        for run in runs
        if run["seed"] is None or run["seed"] == config["reference_seed"]
    }
    summary = {}
    for key in ("macro_f1", "down_recall"):
        values = [
            run["metrics"]["macro_f1"]
            if key == "macro_f1"
            else run["metrics"]["per_class"]["down"]["recall"]
            for run in runs
            if run["strategy"] == "stratified"
        ]
        summary[key] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=0)),
            "minimum": min(values),
            "maximum": max(values),
        }
    return {
        "canonical": canonical,
        "runs": runs,
        "timings": timings,
        "summary": {
            "reference_seed": config["reference_seed"],
            "repeat_seeds": config["repeat_seeds"],
            "std_ddof": 0,
            "stratified": summary,
            "interpretation": "Classifier randomness on fixed data; not a confidence interval",
        },
        "training_priors": {
            "counts": counts,
            "proportions": {label: count / len(train_y) for label, count in counts.items()},
            "majority_class": majority,
            "tie_rule": "lexicographic_first",
        },
    }


def build_baselines(random_seed: int = 42) -> dict[str, Any]:
    return {
        "most_frequent": DummyClassifier(strategy="most_frequent"),
        "stratified": DummyClassifier(strategy="stratified", random_state=random_seed),
    }
