"""Synthetic transport fixtures sharing the real packaged-predictor behavior."""

from types import SimpleNamespace

import numpy as np

from app.config import Settings
from app.schemas import synthetic_document
from machine_learning_project.inference.contracts import input_schema
from machine_learning_project.inference.packaged_predictor import PackagedPredictor
from machine_learning_project.utils.config import load_yaml

TOKEN = "synthetic-test-token-never-a-real-secret-12345"


class Pipeline:
    classes_ = np.asarray(["flat", "new", "up", "stable", "down"])

    def __init__(self):
        self.calls = []

    def predict(self, frame):
        self.calls.append(("predict", len(frame)))
        return np.asarray(["down"] * len(frame))

    def predict_proba(self, frame):
        self.calls.append(("probability", len(frame)))
        return np.tile([0.1, 0.1, 0.1, 0.1, 0.6], (len(frame), 1))


def fixture():
    predictor = PackagedPredictor()
    predictor.manifest = {
        "package_id": "synthetic-research",
        "model_version": "frozen",
        "purpose": "research",
        "production_ready": False,
        "recommended_model": None,
        "development_reference": "random_forest",
        "family": "random_forest",
    }
    predictor.manifest_sha256 = "1" * 64
    predictor.config = load_yaml("configs/inference.yaml")["inference"]
    predictor.schema = input_schema(
        {
            "numeric_columns": ["value"],
            "categorical_columns": ["kind"],
            "non_negative_columns": ["value"],
            "allowed_target_values": ["down", "stable", "up", "new", "flat"],
        }
    )
    predictor._predictor = SimpleNamespace(
        pipeline=Pipeline(), metadata={"model_version": "frozen"}
    )
    settings = Settings(
        package_dir="synthetic", expected_manifest_sha256="1" * 64, token=TOKEN, chunk_rows=2
    )
    return predictor, settings, synthetic_document(predictor.schema, 3)


def headers():
    return {"Authorization": f"Bearer {TOKEN}", "Content-Type": "application/json"}
