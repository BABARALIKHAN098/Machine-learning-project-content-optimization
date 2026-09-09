import pytest

from machine_learning_project.utils.config import (
    load_yaml,
    validate_inference_config,
    validate_packaging_config,
)
from machine_learning_project.utils.exceptions import DataValidationError


def test_defaults():
    validate_packaging_config(load_yaml("configs/packaging.yaml")["packaging"])
    validate_inference_config(load_yaml("configs/inference.yaml")["inference"])


@pytest.mark.parametrize(
    "key,value",
    [
        ("overwrite", True),
        ("families", []),
        ("families", ["recommended"]),
        ("families", ["random_forest", "random_forest"]),
        ("purpose", "production"),
        ("package_root", "https://remote"),
        ("resume", 0),
    ],
)
def test_bad_packaging(key, value):
    config = load_yaml("configs/packaging.yaml")["packaging"]
    config[key] = value
    with pytest.raises(DataValidationError):
        validate_packaging_config(config)


@pytest.mark.parametrize(
    "key,value",
    [
        ("chunk_rows", True),
        ("chunk_rows", 0),
        ("chunk_rows", 30001),
        ("maximum_batch_rows", 30001),
        ("maximum_batch_rows", 0),
        ("probability_tolerance", float("nan")),
        ("probability_tolerance", 0.1),
        ("numeric_string_coercion", True),
        ("extra_columns", "ignore"),
        ("thread_limit", True),
    ],
)
def test_bad_inference(key, value):
    config = load_yaml("configs/inference.yaml")["inference"]
    config[key] = value
    with pytest.raises(DataValidationError):
        validate_inference_config(config)
