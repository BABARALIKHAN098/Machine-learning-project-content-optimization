"""Strict JSON adapter and generated public contract. No fitted logic lives here."""

import json
import math

import pandas as pd

from machine_learning_project.inference.contracts import validate_predictions, validate_request
from machine_learning_project.inference.schemas import PredictionRecord
from machine_learning_project.utils.exceptions import DataValidationError

from .errors import APIError


def decode_request(body):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError()
            result[key] = value
        return result

    def invalid(value):
        raise ValueError()

    try:
        text = body.decode("utf-8", errors="strict")
        depth, quoted, escaped = 0, False, False
        for char in text:
            if quoted:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    quoted = False
            elif char == '"':
                quoted = True
            elif char in "[{":
                depth += 1
                if depth > 8:
                    raise ValueError()
            elif char in "]}":
                depth -= 1
        return json.loads(text, object_pairs_hook=unique, parse_constant=invalid)
    except (ValueError, UnicodeError, RecursionError):
        raise APIError("invalid_json") from None


def request_frame(document, predictor, settings):
    schema = predictor.schema
    if (
        not isinstance(document, dict)
        or set(document) - {"purpose", "include_probabilities", "records"}
        or document.get("purpose") != "research"
        or type(document.get("include_probabilities", False)) is not bool
    ):
        raise APIError("invalid_request")
    rows = document.get("records")
    if not isinstance(rows, list) or not 1 <= len(rows) <= settings.maximum_batch_rows:
        raise APIError("invalid_request")
    expected = {"content_id", *schema["features"]}
    try:
        for row in rows:
            if not isinstance(row, dict) or set(row) != expected:
                raise ValueError()
            content_id = row["content_id"]
            if (
                not isinstance(content_id, str)
                or len(content_id.encode("utf-8")) > settings.maximum_content_id_bytes
            ):
                raise ValueError()
            for key in schema["numeric_columns"]:
                value = row[key]
                if value is not None and (
                    type(value) not in (int, float) or not math.isfinite(value)
                ):
                    raise ValueError()
            for key in schema["categorical_columns"]:
                value = row[key]
                if value is not None and (
                    not isinstance(value, str)
                    or len(value.encode("utf-8")) > settings.maximum_category_bytes
                ):
                    raise ValueError()
        frame = validate_request(
            pd.DataFrame(rows, dtype=object),
            schema,
            {"maximum_batch_rows": settings.maximum_batch_rows},
        )
    except (ValueError, TypeError, OverflowError, UnicodeError, DataValidationError):
        raise APIError("invalid_request") from None
    return frame, document.get("include_probabilities", False)


def validate_response(result, frame, probabilities, predictor):
    fixed = {
        "inference_contract_version": "2.0",
        "package_schema_version": "1.0",
        "package_id": predictor.manifest["package_id"],
        "package_manifest_sha256": predictor.manifest_sha256,
        "model_version": predictor.manifest["model_version"],
        "purpose": "research",
        "production_ready": False,
        "row_count": len(frame),
        "include_probabilities": probabilities,
        "probability_interpretation": "uncalibrated_noncausal"
        if probabilities
        else "not_requested",
    }
    if not isinstance(result, dict) or set(result) != set(fixed) | {"records"}:
        raise ValueError()
    if any(type(result[k]) is not type(v) or result[k] != v for k, v in fixed.items()):
        raise ValueError()
    records = result["records"]
    labels = predictor.schema["labels"]
    if not isinstance(records, list) or len(records) != len(frame):
        raise ValueError()
    for record, content_id in zip(records, frame.content_id, strict=True):
        if not isinstance(record, dict) or set(record) != set(
            PredictionRecord.__dataclass_fields__
        ):
            raise ValueError()
        if record["content_id"] != content_id or record["model_version"] != fixed["model_version"]:
            raise ValueError()
        scores = record["probabilities"]
        if probabilities:
            if not isinstance(scores, dict) or list(scores) != labels:
                raise ValueError()
            if any(type(v) not in (int, float) for v in scores.values()):
                raise ValueError()
            matrix = [[scores[k] for k in labels]]
        else:
            if scores is not None or record["probability_down"] is not None:
                raise ValueError()
            matrix = None
        canonical = validate_predictions(
            [content_id],
            [record["predicted_trend"]],
            matrix,
            labels,
            predictor.schema,
            {"model_version": fixed["model_version"]},
            predictor.config["probability_tolerance"],
        )[0]
        if record != canonical:
            raise ValueError()


def request_schema(schema, settings):
    properties = {
        "content_id": {
            "type": "string",
            "minLength": 1,
            "description": f"Unique, untrimmed string; at most {settings.maximum_content_id_bytes} UTF-8 bytes.",
        }
    }
    for key in schema["numeric_columns"]:
        properties[key] = {"type": ["number", "null"]}
        if key in schema["non_negative_columns"]:
            properties[key]["minimum"] = 0
    for key in schema["categorical_columns"]:
        properties[key] = {
            "type": ["string", "null"],
            "description": f"Unknown strings retained; at most {settings.maximum_category_bytes} UTF-8 bytes. Only JSON null is missing.",
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["purpose", "records"],
        "properties": {
            "purpose": {"type": "string", "const": "research"},
            "include_probabilities": {"type": "boolean", "default": False},
            "records": {
                "type": "array",
                "minItems": 1,
                "maxItems": settings.maximum_batch_rows,
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": list(properties),
                    "properties": properties,
                },
            },
        },
    }


def synthetic_document(schema, rows=5):
    from machine_learning_project.inference.contracts import synthetic_request

    frame = synthetic_request(schema, rows)
    return {
        "purpose": "research",
        "include_probabilities": False,
        "records": json.loads(frame.to_json(orient="records")),
    }
