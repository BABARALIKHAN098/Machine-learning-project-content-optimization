import copy
import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import ConvergenceWarning

from machine_learning_project.models.tune import (
    make_training_folds,
    measured_fit_predict,
    rank_trials,
    summarize_trial,
)
from machine_learning_project.utils.exceptions import DataValidationError


def trial(name, score, recall, c=1.0):
    return {
        "trial_id": name,
        "family": "logistic_regression",
        "parameters": {"model__C": c},
        "status": "valid",
        "summary": {"macro_f1": {"mean": score}, "down_recall": {"mean": recall}},
    }


def test_eligibility_tolerance_complexity_and_fallback():
    rows = [trial("a", 0.7, 0.49), trial("b", 0.6, 0.5, 10), trial("c", 0.599, 0.5, 0.1)]
    chosen = rank_trials(rows, 0.5, 0.001)
    assert chosen["trial_id"] == "c" and not chosen["recall_fallback_used"]
    assert rank_trials(list(reversed(rows)), 0.5, 0.001) == chosen
    assert rank_trials(rows, 0.99, 0.001)["recall_fallback_used"]
    assert rank_trials(rows, 0.99, 0.001)["trial_id"] == "a"
    with pytest.raises(DataValidationError):
        rank_trials([], 0.5, 0.001)
    with pytest.raises(DataValidationError):
        rank_trials([trial("bad", float("nan"), 0.5)], 0.5, 0.001)


def test_forest_and_cross_family_ties():
    a = trial("a", 0.6, 0.6)
    forest = {
        "trial_id": "f",
        "family": "random_forest",
        "status": "valid",
        "parameters": {
            "model__max_depth": 20,
            "model__min_samples_leaf": 5,
            "model__max_features": "sqrt",
        },
        "summary": a["summary"],
    }
    other = copy.deepcopy(forest)
    other["trial_id"] = "g"
    other["parameters"]["model__max_depth"] = None
    assert rank_trials([other, forest], 0.5, 0.001)["trial_id"] == "f"
    assert rank_trials([forest, a], 0.5, 0.001, cross_family=True)["trial_id"] == "a"


def test_unweighted_aggregation_and_partial_trials():
    folds = [
        {
            "status": "valid",
            "metrics": {
                "macro_f1": score,
                "per_class": {"down": {"recall": recall, "support": count}},
            },
        }
        for score, recall, count in [(0.2, 0.3, 100), (0.4, 0.5, 10), (0.6, 0.7, 5)]
    ]
    result = summarize_trial({"trial_id": "x"}, folds)
    assert result["summary"]["macro_f1"]["mean"] == pytest.approx(0.4)
    assert result["summary"]["macro_f1"]["std"] == np.std([0.2, 0.4, 0.6])
    assert summarize_trial({}, folds[:2])["status"] == "invalid"
    folds[0]["status"] = "nonconverged"
    assert summarize_trial({}, folds)["status"] == "invalid"


@pytest.mark.parametrize("mode", ["nonconverged", "failed"])
def test_failure_diagnostics_do_not_publish_private_messages(mode):
    class Estimator:
        def fit(self, x, y):
            if mode == "failed":
                raise ValueError("private-client-secret")
            warnings.warn("private-row-secret", ConvergenceWarning)

        def predict(self, x):
            return ["down", "up"]

    class Names:
        def get_feature_names_out(self):
            return ["value"]

        def get_params(self):
            return {}

    estimator = Estimator()
    estimator.named_steps = {"preprocessor": Names(), "model": Names()}
    result, _ = measured_fit_predict(
        estimator, [[1], [2]], ["down", "up"], [[1], [2]], ["down", "up"], ["down", "up"], "scope"
    )
    assert result["status"] == mode
    assert "private" not in str(result)


def test_fold_coverage_and_missing_classes():
    frame = pd.DataFrame(
        [
            {"id": f"{g}-{label}", "group": g, "y": label}
            for g in range(9)
            for label in ["down", "up"]
        ]
    )
    data = {"target_column": "y", "allowed_target_values": ["down", "up"]}
    training = {"group_column": "group", "row_key": "id"}
    config = {"folds": 3, "random_seed": 42}
    folds, manifest = make_training_folds(frame, data, training, config, "scope")
    assert sorted(np.concatenate([right for _, right in folds])) == list(range(len(frame)))
    for left, right in folds:
        assert not set(frame.iloc[left]["group"]) & set(frame.iloc[right]["group"])
    assert manifest["fitting_population"] == "train"
    with pytest.raises(DataValidationError):
        make_training_folds(frame.iloc[:2], data, training, config, "scope")
    bad = frame.copy()
    bad["id"] = "same"
    with pytest.raises(DataValidationError):
        make_training_folds(bad, data, training, config, "scope")
