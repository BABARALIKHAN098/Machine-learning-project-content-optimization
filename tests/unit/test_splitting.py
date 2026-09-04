import pandas as pd
import pytest

from machine_learning_project.data.splitting import split_by_group, write_split_artifacts
from machine_learning_project.utils.config import validate_split_config
from machine_learning_project.utils.exceptions import DataValidationError


def make_frame(group_count=20):
    rows = []
    labels = ["down", "stable", "up", "new", "flat"]
    for group_index in range(group_count):
        for label in labels:
            rows.append(
                {
                    "row_id": f"{group_index}-{label}",
                    "client_id": f"private-{group_index}",
                    "target": label,
                }
            )
    return pd.DataFrame(rows), labels


def test_group_split_is_disjoint_complete_and_deterministic():
    dataframe, labels = make_frame()
    kwargs = {
        "target_column": "target",
        "group_column": "client_id",
        "row_key": "row_id",
        "allowed_labels": labels,
        "random_seed": 7,
        "source_sha256": "source-hash",
    }
    first = split_by_group(dataframe, **kwargs)
    second = split_by_group(dataframe, **kwargs)
    train_groups = set(first.train["client_id"])
    validation_groups = set(first.validation["client_id"])
    test_groups = set(first.test["client_id"])
    assert train_groups.isdisjoint(validation_groups)
    assert train_groups.isdisjoint(test_groups)
    assert validation_groups.isdisjoint(test_groups)
    assert len(first.train) + len(first.validation) + len(first.test) == len(dataframe)
    assert first.manifest == second.manifest
    assert first.assignments.equals(second.assignments)
    assert set(first.assignments["partition"]) == {"train", "validation", "test"}
    assert first.manifest["invariants"] == {
        "row_complete": True,
        "row_disjoint": True,
        "group_complete": True,
        "group_disjoint": True,
        "class_coverage": True,
    }


def test_assignments_are_independent_of_input_row_order():
    dataframe, labels = make_frame()
    kwargs = {
        "target_column": "target",
        "group_column": "client_id",
        "row_key": "row_id",
        "allowed_labels": labels,
        "random_seed": 7,
    }
    ordered = split_by_group(dataframe, **kwargs)
    shuffled = split_by_group(dataframe.sample(frac=1, random_state=99), **kwargs)
    assert ordered.assignments.equals(shuffled.assignments)
    assert ordered.manifest == shuffled.manifest


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda frame: frame.drop(columns="client_id"), "split column not found"),
        (lambda frame: frame.assign(client_id=None), "contains missing values"),
        (lambda frame: frame.assign(row_id="duplicate"), "duplicate values"),
    ],
)
def test_invalid_split_identity_is_rejected(mutation, message):
    dataframe, labels = make_frame()
    with pytest.raises(DataValidationError, match=message):
        split_by_group(
            mutation(dataframe),
            target_column="target",
            group_column="client_id",
            row_key="row_id",
            allowed_labels=labels,
        )


def test_class_must_occur_in_at_least_three_groups():
    dataframe, labels = make_frame()
    dataframe = dataframe[~((dataframe["target"] == "flat") & (dataframe["client_id"] != "private-0"))]
    with pytest.raises(DataValidationError, match="'flat' occurs in 1 groups"):
        split_by_group(
            dataframe,
            target_column="target",
            group_column="client_id",
            row_key="row_id",
            allowed_labels=labels,
        )


def test_manifest_and_assignments_do_not_publish_raw_identifiers(tmp_path):
    dataframe, labels = make_frame()
    splits = split_by_group(
        dataframe,
        target_column="target",
        group_column="client_id",
        row_key="row_id",
        allowed_labels=labels,
        source_sha256="source-hash",
    )
    manifest, assignments = write_split_artifacts(
        splits, tmp_path / "manifest.json", tmp_path / "assignments.csv"
    )
    published = manifest.read_text(encoding="utf-8") + assignments.read_text(encoding="utf-8")
    assert "private-" not in published
    assert "0-down" not in published
    assert splits.manifest["source"]["source_sha256"] == "source-hash"


def test_invalid_split_config_is_rejected():
    with pytest.raises(DataValidationError, match="sum below 1"):
        validate_split_config(
            {
                "test_size": 0.6,
                "validation_size": 0.5,
                "group_column": "client_id",
                "row_key": "row_id",
                "split_contract_version": "1.0",
                "split_algorithm_version": "v1",
                "search_attempts": 10,
                "row_ratio_tolerance": 0.1,
                "class_ratio_tolerance": 0.1,
                "split_objective_weights": {"size": 1, "class": 1},
            }
        )
