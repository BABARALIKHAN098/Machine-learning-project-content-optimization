import joblib
import pandas as pd
import pytest

from machine_learning_project.inference.predictor import Predictor
from machine_learning_project.models.train import build_candidates
from machine_learning_project.utils.config import load_yaml
from machine_learning_project.utils.exceptions import DataValidationError


def test_legacy_bundle_keeps_original_pipeline_and_new_bundle_requires_contract(tmp_path):
    data = {"numeric_columns": ["value"], "id_columns": ["content_id"], "target_column": "target"}
    preprocessing = load_yaml("configs/preprocessing.yaml")["preprocessing"]
    pipeline = build_candidates(data, preprocessing, {})["logistic_regression"]
    pipeline.fit(pd.DataFrame({"value": [0.0, 1.0, 2.0, 3.0]}), ["down", "down", "up", "up"])
    metadata = {"model_version": "legacy"}
    bundle = {"pipeline": pipeline, "metadata": metadata, "data_config": data}
    path = tmp_path / "legacy.joblib"
    joblib.dump(bundle, path)
    predictor = Predictor.load(path)
    assert "engineering" not in predictor.pipeline.named_steps
    assert len(predictor.predict(pd.DataFrame({"content_id": ["one"], "value": [1.0]}))) == 1
    metadata["bundle_schema_version"] = "2.0"
    joblib.dump(bundle, path)
    with pytest.raises(DataValidationError, match="Incomplete"):
        Predictor.load(path)


def test_version_disagreement_rejected():
    data = {"numeric_columns": ["value"]}
    pre = load_yaml("configs/preprocessing.yaml")["preprocessing"]
    features = load_yaml("configs/features.yaml")["features"]
    pre["feature_contract_version"] = "1.0"
    with pytest.raises(DataValidationError, match="versions disagree"):
        build_candidates(data, pre, {}, features)
