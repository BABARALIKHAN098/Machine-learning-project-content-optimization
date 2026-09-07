import pytest

from machine_learning_project.features.registry import resolve_registry
from machine_learning_project.features.selection import configured_feature_columns
from machine_learning_project.utils.config import load_yaml
from machine_learning_project.utils.exceptions import DataValidationError


def test_real_registry_has_expected_order_and_lineage():
    data = load_yaml("configs/data.yaml")["data"]
    features = load_yaml("configs/features.yaml")["features"]
    registry = resolve_registry(data, features)
    assert len(registry) == 23
    assert [row["name"] for row in registry[:19]] == configured_feature_columns(data)
    assert registry[19]["dependencies"] == ["clicks_prev_30d", "impressions_prev_30d"]


@pytest.mark.parametrize("role", ["id_columns", "drop_columns", "sensitive_columns"])
def test_forbidden_sources_cannot_be_reenabled(role):
    data = {"numeric_columns": ["secret"], role: ["secret"]}
    with pytest.raises(DataValidationError, match="Forbidden"):
        configured_feature_columns(data)


@pytest.mark.parametrize(
    "change",
    [
        {"engineering_version": "unknown"},
        {"families": ["target_encoding"]},
        {"families": ["ratios", "ratios"]},
        {"excluded_inputs": ["clicks_prev_30d"]},
        {"log_sources": ["trend_pct"]},
    ],
)
def test_invalid_contract_fails(change):
    data = load_yaml("configs/data.yaml")["data"]
    features = load_yaml("configs/features.yaml")["features"]
    features.update(change)
    with pytest.raises(DataValidationError):
        resolve_registry(data, features)
