"""Strict research inference v2; the legacy PredictionRecord remains unchanged."""

import csv
import numbers
from pathlib import Path

import numpy as np
import pandas as pd

from ..features.selection import configured_feature_columns
from ..utils.exceptions import DataValidationError
from .schemas import PredictionRecord

LABELS = {"down", "stable", "up", "new", "flat"}
OUTPUT_SCHEMA = {
    "inference_contract_version": "2.0",
    "record_fields": list(PredictionRecord.__dataclass_fields__),
    "purpose": "research",
    "production_ready": False,
    "order": "input_position",
    "probability_interpretation": "Uncalibrated model scores; no causal refresh benefit established",
    "probability_columns": "canonical_labels_mapped_from_estimator_classes",
}


def input_schema(data):
    columns = configured_feature_columns(data)
    labels = data["allowed_target_values"]
    if len(labels) != 5 or set(labels) != LABELS or "content_id" in columns:
        raise DataValidationError("Package requires the five-class content-trend contract")
    return {
        "inference_contract_version": "2.0",
        "id_column": "content_id",
        "features": columns,
        "numeric_columns": list(data["numeric_columns"]),
        "categorical_columns": list(data["categorical_columns"]),
        "non_negative_columns": [c for c in data.get("non_negative_columns", []) if c in columns],
        "labels": list(labels),
        "extra_columns": "reject",
        "missing_columns": "reject",
        "id_policy": "unique_nonempty_strings_no_surrounding_whitespace",
        "unknown_categories": "ignore",
        "missing_cells": "fitted_imputation",
    }


def validate_schema(schema):
    data = {
        "numeric_columns": schema.get("numeric_columns", []),
        "categorical_columns": schema.get("categorical_columns", []),
        "non_negative_columns": schema.get("non_negative_columns", []),
        "allowed_target_values": schema.get("labels", []),
    }
    roles = data["numeric_columns"] + data["categorical_columns"]
    if (
        not roles
        or any(not isinstance(c, str) or not c for c in roles)
        or not set(data["non_negative_columns"]) <= set(data["numeric_columns"])
        or schema != input_schema(data)
    ):
        raise DataValidationError("Invalid packaged input schema")


def _missing(value):
    return (
        value is None
        or value is pd.NA
        or isinstance(value, (float, np.floating))
        and np.isnan(value)
    )


def validate_request(frame, schema, config):
    if not isinstance(frame, pd.DataFrame) or frame.columns.duplicated().any():
        raise DataValidationError("request_columns: require unique dataframe columns")
    expected = [schema["id_column"], *schema["features"]]
    if set(frame.columns) != set(expected):
        raise DataValidationError("request_columns: missing or unapproved columns")
    if not 1 <= len(frame) <= config["maximum_batch_rows"]:
        raise DataValidationError("request_rows: outside configured batch bounds")
    ids = frame[schema["id_column"]].tolist()
    if any(not isinstance(v, str) or not v or v != v.strip() for v in ids) or len(set(ids)) != len(
        ids
    ):
        raise DataValidationError(
            "request_ids: require unique nonempty strings without surrounding whitespace"
        )
    # Work positionally, preserving values and allowing duplicate/non-consecutive caller indices.
    result = frame.loc[:, expected].reset_index(drop=True).copy()
    for column in schema["numeric_columns"]:
        values = []
        for value in result[column]:
            if _missing(value):
                values.append(np.nan)
            elif (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, numbers.Real)
                or not np.isfinite(value)
                or column in schema["non_negative_columns"]
                and value < 0
            ):
                raise DataValidationError(f"request_numeric: invalid cells in {column}")
            else:
                values.append(float(value))
        result[column] = values
    for column in schema["categorical_columns"]:
        values = []
        for value in result[column]:
            if _missing(value):
                values.append(np.nan)
            elif not isinstance(value, str):
                raise DataValidationError(f"request_categorical: invalid cells in {column}")
            else:
                values.append(value)
        result[column] = pd.Series(values, dtype=object)
    return result


def read_request_csv(path, schema, config):
    try:
        # csv.reader preserves duplicate headers and identifier strings (including literal NA).
        with Path(path).open(encoding=config["encoding"], newline="") as handle:
            reader = csv.reader(handle, delimiter=config["delimiter"], strict=True)
            header = next(reader, [])
            if len(set(header)) != len(header) or set(header) != {
                "content_id",
                *schema["features"],
            }:
                raise DataValidationError("request_csv: invalid or duplicate header")
            rows = []
            for row in reader:
                if len(row) != len(header):
                    raise DataValidationError("request_csv: malformed row width")
                rows.append(row)
                if len(rows) > config["maximum_batch_rows"]:
                    raise DataValidationError("request_rows: outside configured batch bounds")
        frame = pd.DataFrame(rows, columns=header)
        for column in schema["features"]:
            values = frame[column].mask(frame[column].isin(config["missing_tokens"]), np.nan)
            if column in schema["numeric_columns"]:
                values = pd.to_numeric(values, errors="raise")
            frame[column] = values
        return validate_request(frame, schema, config)
    except (csv.Error, UnicodeError, ValueError, TypeError, OSError):
        raise DataValidationError("request_csv: cannot parse the declared input contract") from None


def validate_predictions(ids, predicted, probabilities, classes, schema, metadata, tolerance):
    predicted = np.asarray(predicted)
    labels = schema["labels"]
    if predicted.shape != (len(ids),) or any(
        not isinstance(v, str) or v not in labels for v in predicted
    ):
        raise DataValidationError("model_output: invalid prediction shape or labels")
    if len(classes) != len(labels) or set(classes) != set(labels):
        raise DataValidationError("model_output: invalid estimator class mapping")
    if probabilities is not None:
        try:
            if np.asarray(probabilities).dtype.kind not in "fiu":
                raise DataValidationError("model_output: probability values must be numeric")
            probabilities = np.asarray(probabilities, dtype=float)
        except (TypeError, ValueError):
            raise DataValidationError("model_output: invalid probabilities") from None
        if (
            probabilities.shape != (len(ids), len(labels))
            or not np.isfinite(probabilities).all()
            or (probabilities < 0).any()
            or (probabilities > 1).any()
            or not np.allclose(probabilities.sum(axis=1), 1, atol=tolerance, rtol=0)
        ):
            raise DataValidationError("model_output: invalid probability shape, bounds or sum")
    records = []
    for i, content_id in enumerate(ids):
        scores = (
            None
            if probabilities is None
            else {label: float(probabilities[i, classes.index(label)]) for label in labels}
        )
        records.append(
            PredictionRecord(
                content_id,
                str(predicted[i]),
                scores["down"] if scores else None,
                scores,
                metadata["model_version"],
            ).to_dict()
        )
    return records


def synthetic_request(schema, rows):
    """Invented IDs/features only; suitable for smoke tests and resource observations."""
    frame = pd.DataFrame({"content_id": [f"synthetic-{i:06d}" for i in range(rows)]})
    for column in schema["numeric_columns"]:
        frame[column] = [np.nan if i % 17 == 0 else float(i % 101) for i in range(rows)]
    for column in schema["categorical_columns"]:
        frame[column] = pd.Series(
            [None if i % 13 == 0 else "__SYNTHETIC_UNKNOWN__" for i in range(rows)], dtype=object
        )
    return frame
