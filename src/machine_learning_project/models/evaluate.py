from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from ..utils.exceptions import DataValidationError

METRIC_CONTRACT_VERSION = "1.0"
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


def evaluate_predictions(
    truth: Sequence[str], predictions: Sequence[str], labels: Sequence[str]
) -> dict[str, Any]:
    truth, predictions, labels = list(truth), list(predictions), list(labels)
    if (
        len(labels) < 2
        or not all(isinstance(label, str) and label for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise DataValidationError("Metric labels must contain at least two unique nonempty strings")
    if not truth or len(truth) != len(predictions):
        raise DataValidationError("Truth and predictions must have matching nonzero lengths")
    if any(not isinstance(value, str) or value not in labels for value in truth + predictions):
        raise DataValidationError("Truth/predictions contain null or unknown labels")
    precision, recall, f1, support = precision_recall_fscore_support(
        truth, predictions, labels=list(labels), zero_division=0
    )
    per_class = {
        label: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, label in enumerate(labels)
    }
    matrix = confusion_matrix(truth, predictions, labels=list(labels))
    return {
        "accuracy": float(accuracy_score(truth, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(truth, predictions)),
        "macro_f1": float(
            f1_score(truth, predictions, labels=labels, average="macro", zero_division=0)
        ),
        "weighted_f1": float(
            f1_score(truth, predictions, labels=labels, average="weighted", zero_division=0)
        ),
        "per_class": per_class,
        "down_false_negatives": int(
            sum(
                actual == "down" and predicted != "down"
                for actual, predicted in zip(truth, predictions)
            )
        ),
        "confusion_matrix": {
            "labels": list(labels),
            "values": matrix.astype(int).tolist(),
        },
    }


def subgroup_metrics(
    dataframe: pd.DataFrame,
    truth: Sequence[str],
    predictions: Sequence[str],
    columns: Sequence[str],
    *,
    minimum_rows: int = 25,
) -> dict[str, Any]:
    scored = dataframe.copy()
    scored["__truth"] = list(truth)
    scored["__prediction"] = list(predictions)
    output: dict[str, Any] = {}
    for column in columns:
        if column not in scored:
            continue
        groups: dict[str, Any] = {}
        for value, group in scored.groupby(column, dropna=False):
            if len(group) < minimum_rows:
                continue
            groups[str(value)] = {
                "row_count": len(group),
                "macro_f1": float(
                    f1_score(
                        group["__truth"], group["__prediction"], average="macro", zero_division=0
                    )
                ),
                "down_recall": float(
                    precision_recall_fscore_support(
                        group["__truth"], group["__prediction"], labels=["down"], zero_division=0
                    )[1][0]
                ),
            }
        output[column] = groups
    return output
