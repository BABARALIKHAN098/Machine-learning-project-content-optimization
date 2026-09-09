from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from machine_learning_project.inference.contracts import input_schema
from machine_learning_project.inference.packaged_predictor import PackagedPredictor
from machine_learning_project.utils.config import load_yaml
from machine_learning_project.utils.exceptions import DataValidationError


class FakePipeline:
    classes_ = np.asarray(["flat", "new", "up", "stable", "down"])

    def __init__(self):
        self.calls = []

    def predict(self, x):
        self.calls.append(("predict", len(x)))
        return np.asarray(["down"] * len(x))

    def predict_proba(self, x):
        self.calls.append(("probability", len(x)))
        return np.tile([0.1, 0.1, 0.1, 0.1, 0.6], (len(x), 1))


def test_chunk_calls_order_ranking_and_invalid_requests():
    instance = PackagedPredictor()
    instance.manifest = {"package_id": "research", "model_version": "frozen"}
    instance.manifest_sha256 = "hash"
    instance.config = load_yaml("configs/inference.yaml")["inference"]
    instance.schema = input_schema(
        {
            "numeric_columns": ["value"],
            "categorical_columns": [],
            "allowed_target_values": ["down", "stable", "up", "new", "flat"],
        }
    )
    pipeline = FakePipeline()
    instance._predictor = SimpleNamespace(pipeline=pipeline, metadata={"model_version": "frozen"})
    frame = pd.DataFrame({"content_id": ["c", "a", "b"], "value": [1, 2, 3]}, index=[9, 9, 1])
    label = instance.predict(frame, chunk_rows=2)
    assert pipeline.calls == [("predict", 2), ("predict", 1)]
    assert [r["content_id"] for r in label["records"]] == ["c", "a", "b"]
    assert all(r["probabilities"] is None for r in label["records"])
    pipeline.calls.clear()
    full = instance.predict(frame, include_probabilities=True, chunk_rows=3)
    chunks = instance.predict(frame, include_probabilities=True, chunk_rows=2)
    assert full == chunks
    assert [r["content_id"] for r in instance.rank_review_queue(frame)["records"]] == [
        "c",
        "a",
        "b",
    ]
    before = pipeline.calls.copy()
    for invalid in (frame.assign(trend_pct=1), frame.assign(content_id="duplicate")):
        with pytest.raises(DataValidationError):
            instance.predict(invalid)
    assert before == pipeline.calls
    with pytest.raises(DataValidationError):
        instance.rank_review_queue(frame, include_probabilities=False)
    pipeline.predict_proba = lambda x: None
    with pytest.raises(DataValidationError, match="missing"):
        instance.predict(frame, include_probabilities=True)
