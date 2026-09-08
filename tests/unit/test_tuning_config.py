import copy

import pytest

from machine_learning_project.models.tune import expand_trials
from machine_learning_project.utils.config import load_yaml, validate_tuning_config
from machine_learning_project.utils.exceptions import DataValidationError


def configs():
    return (
        load_yaml("configs/tuning.yaml")["tuning"],
        load_yaml("configs/training.yaml")["training"],
    )


def test_fixed_schedule_and_budget():
    tuning, training = configs()
    validate_tuning_config(tuning, training)
    trials = expand_trials(tuning)
    assert len(trials) == 15 and len({t["trial_id"] for t in trials}) == 15
    assert sum(t["family"] == "logistic_regression" for t in trials) == 3
    assert len(trials) * tuning["folds"] + 2 == 47
    assert trials == expand_trials(copy.deepcopy(tuning))


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("folds", True),
        ("folds", 2),
        ("random_seed", 43),
        ("families", ["svm"]),
        ("selection_tolerance", float("nan")),
        ("selection_tolerance", -1),
        ("max_estimator_fits", 46),
        ("max_configurations", 16),
        ("search_n_jobs", -1),
        ("estimator_n_jobs", True),
        ("evaluation_partition", "test"),
        ("resume", True),
        ("output_root", ""),
        ("tuning_contract_version", "2.0"),
        ("unknown", 1),
        ("grids", []),
    ],
)
def test_invalid_protocol(key, value):
    tuning, training = configs()
    tuning[key] = value
    with pytest.raises(DataValidationError):
        validate_tuning_config(tuning, training)


@pytest.mark.parametrize("values", [[], [True], [1, 1], [float("inf")], [-1], "auto"])
def test_invalid_trial_grids(values):
    tuning, training = configs()
    tuning["grids"]["logistic_regression"]["model__C"] = values
    with pytest.raises(DataValidationError):
        validate_tuning_config(tuning, training)


def test_original_model_parameters_are_not_silently_ignored():
    tuning, training = configs()
    training["candidates"]["logistic_regression"]["typo"] = 1
    with pytest.raises(DataValidationError):
        validate_tuning_config(tuning, training)
