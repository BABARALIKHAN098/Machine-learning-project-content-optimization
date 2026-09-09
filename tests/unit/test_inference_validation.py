import numpy as np
import pandas as pd
import pytest

from machine_learning_project.inference.contracts import (
    input_schema,
    read_request_csv,
    validate_predictions,
    validate_request,
)
from machine_learning_project.utils.config import load_yaml
from machine_learning_project.utils.exceptions import DataValidationError


@pytest.fixture
def contract():
    return input_schema(
        {
            "numeric_columns": ["value"],
            "categorical_columns": ["kind"],
            "non_negative_columns": ["value"],
            "allowed_target_values": ["down", "stable", "up", "new", "flat"],
        }
    )


@pytest.fixture
def config():
    return load_yaml("configs/inference.yaml")["inference"]


def test_ids_column_order_missing_and_unknown(contract, config):
    frame = pd.DataFrame(
        {"kind": ["UNKNOWN", None], "value": [2, np.nan], "content_id": ["001", "NA"]}, index=[5, 5]
    )
    result = validate_request(frame, contract, config)
    assert list(result) == ["content_id", "value", "kind"]
    assert result.content_id.tolist() == ["001", "NA"]
    assert result.index.tolist() == [0, 1]
    assert result.kind[0] == "UNKNOWN"
    assert pd.isna(result.kind[1])


@pytest.mark.parametrize(
    "column,value",
    [
        ("value", True),
        ("value", "2"),
        ("value", np.inf),
        ("value", -1),
        ("value", []),
        ("kind", 2),
        ("kind", {}),
        ("content_id", None),
        ("content_id", 1),
        ("content_id", ""),
        ("content_id", " private-id "),
        ("content_id", []),
    ],
)
def test_invalid_cells_are_private(contract, config, column, value):
    frame = pd.DataFrame({"content_id": ["private-id"], "value": [1], "kind": ["private-category"]})
    frame[column] = pd.Series([value], dtype=object)
    with pytest.raises(DataValidationError) as error:
        validate_request(frame, contract, config)
    assert "private-id" not in str(error.value) and "private-category" not in str(error.value)


def test_batch_bounds_and_columns(contract, config):
    frame = pd.DataFrame({"content_id": ["a", "b"], "value": [1, 2], "kind": ["x", "y"]})
    bad = [
        frame.iloc[:0],
        frame.drop(columns="kind"),
        frame.assign(trend_pct=1),
        frame.assign(client_id="private"),
        frame.assign(content_id="duplicate"),
        pd.concat([frame, frame[["kind"]]], axis=1),
    ]
    for value in bad:
        with pytest.raises(DataValidationError):
            validate_request(value, contract, config)
    config.update(maximum_batch_rows=2, chunk_rows=1)
    assert len(validate_request(frame, contract, config)) == 2
    with pytest.raises(DataValidationError):
        validate_request(pd.concat([frame, frame]), contract, config)


def test_csv_literal_ids_and_duplicate_headers(contract, config, tmp_path):
    path = tmp_path / "input.csv"
    path.write_text("content_id,value,kind\n001,2,UNKNOWN\nNA,,NA\n", encoding="utf-8")
    frame = read_request_csv(path, contract, config)
    assert frame.content_id.tolist() == ["001", "NA"]
    assert pd.isna(frame.value[1]) and pd.isna(frame.kind[1])
    for text in (
        "content_id,value,kind,kind\na,2,x,x\n",
        "content_id,value,kind\na,2,x,extra\n",
        "content_id,value,kind\na,secret,x\n",
    ):
        path.write_text(text)
        with pytest.raises(DataValidationError) as error:
            read_request_csv(path, contract, config)
        assert "secret" not in str(error.value)


def test_probability_mapping_and_label_only(contract):
    classes = ["flat", "new", "up", "stable", "down"]
    records = validate_predictions(
        ["a"],
        ["up"],
        [[0.1, 0.1, 0.4, 0.1, 0.3]],
        classes,
        contract,
        {"model_version": "source-v1"},
        1e-9,
    )
    assert list(records[0]["probabilities"]) == contract["labels"]
    assert records[0]["probability_down"] == 0.3
    assert records[0]["predicted_trend"] == "up"
    assert (
        validate_predictions(["a"], ["up"], None, classes, contract, {"model_version": "v1"}, 1e-9)[
            0
        ]["probabilities"]
        is None
    )


@pytest.mark.parametrize(
    "probabilities",
    [
        [[1, 1, 0, 0, 0]],
        [[np.nan, 0, 0, 0, 1]],
        [[-0.1, 0, 0, 0, 1.1]],
        [[1, 0]],
        [1, 0, 0, 0, 0],
        [["1", "0", "0", "0", "0"]],
    ],
)
def test_bad_probability_outputs(contract, probabilities):
    with pytest.raises(DataValidationError):
        validate_predictions(
            ["a"],
            ["down"],
            probabilities,
            contract["labels"],
            contract,
            {"model_version": "v1"},
            1e-9,
        )


@pytest.mark.parametrize("predictions", [["unknown"], [], [["down"]], [None]])
def test_bad_label_outputs(contract, predictions):
    with pytest.raises(DataValidationError):
        validate_predictions(
            ["a"], predictions, None, contract["labels"], contract, {"model_version": "v1"}, 1e-9
        )
