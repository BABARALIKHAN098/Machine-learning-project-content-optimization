import copy

import pytest

from machine_learning_project.utils.config import load_yaml, validate_evaluation_config
from machine_learning_project.utils.exceptions import DataValidationError


def test_defaults():
    validate_evaluation_config(
        load_yaml("configs/evaluation.yaml")["evaluation"],
        load_yaml("configs/training.yaml")["training"],
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("partition", "test"),
        ("partition", "all"),
        ("dimensions", "content_type"),
        ("dimensions", ["trend_pct"]),
        ("dimensions", ["age_tier", "age_tier"]),
        ("minimum_subgroup_rows", True),
        ("minimum_down_support", 0),
        ("comparison_tolerance", float("nan")),
        ("subgroup_alert_macro_f1_gap", -1),
        ("row_exports", True),
        ("prediction_n_jobs", True),
        ("output_root", ""),
        ("labels", ["down", "up"]),
        ("families", ["random_forest"]),
    ],
)
def test_invalid_contract(key, value):
    config = copy.deepcopy(load_yaml("configs/evaluation.yaml")["evaluation"])
    config[key] = value
    with pytest.raises(DataValidationError):
        validate_evaluation_config(config, load_yaml("configs/training.yaml")["training"])
