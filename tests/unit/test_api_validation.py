import copy
from dataclasses import replace

import pandas as pd
import pytest

from app.errors import APIError
from app.schemas import decode_request, request_frame
from tests.api_helpers import fixture


@pytest.mark.parametrize(
    "body",
    [
        b'{"a":1,"a":2}',
        b'{"x":{"a":1,"a":2}}',
        b"NaN",
        b"Infinity",
        b"{} garbage",
        b"\xff",
        b"[" * 10 + b"]" * 10,
    ],
)
def test_strict_decoder(body):
    with pytest.raises(APIError, match="invalid_json"):
        decode_request(body)


@pytest.mark.parametrize(
    "field,value",
    [
        ("value", True),
        ("value", "1"),
        ("value", float("inf")),
        ("value", 10**400),
        ("value", -1),
        ("kind", []),
        ("kind", 4),
        ("content_id", None),
        ("content_id", " x "),
        ("content_id", ""),
        ("content_id", 1),
    ],
)
def test_raw_scalars(field, value):
    predictor, settings, document = fixture()
    document["records"][1][field] = value
    with pytest.raises(APIError, match="invalid_request"):
        request_frame(document, predictor, settings)
    assert not predictor._predictor.pipeline.calls


def test_per_row_keys_and_global_ids():
    predictor, settings, original = fixture()
    for edit in (
        lambda d: d["records"][1].pop("kind"),
        lambda d: d["records"][1].update(trend_pct=1),
        lambda d: d["records"][2].update(content_id=d["records"][0]["content_id"]),
        lambda d: d.update(include_probabilities="false"),
        lambda d: d.update(purpose="production"),
        lambda d: d.update(package_dir="private-path"),
    ):
        document = copy.deepcopy(original)
        edit(document)
        with pytest.raises(APIError):
            request_frame(document, predictor, settings)


def test_preservation_nulls_and_limits():
    predictor, settings, document = fixture()
    for row, content_id in zip(document["records"], ["001", "NA", "zero"], strict=True):
        row.update(content_id=content_id, kind="NA", value=None)
    frame, mode = request_frame(document, predictor, settings)
    assert frame.content_id.tolist() == ["001", "NA", "zero"]
    assert frame.kind.tolist() == ["NA"] * 3 and pd.isna(frame.value).all() and mode is False
    with pytest.raises(APIError):
        request_frame(document, predictor, replace(settings, maximum_batch_rows=2, chunk_rows=1))
    document["records"][0]["content_id"] = "é" * 128
    request_frame(document, predictor, settings)
    document["records"][0]["content_id"] += "é"
    with pytest.raises(APIError):
        request_frame(document, predictor, settings)
