import json

import joblib
import pandas as pd
import pytest

from machine_learning_project.features.registry import fingerprint
from machine_learning_project.inference.predictor import Predictor
from machine_learning_project.models.train import build_candidates
from machine_learning_project.utils.config import load_yaml
from machine_learning_project.utils.exceptions import DataValidationError
from pipelines.feature_pipeline import choose_variant, run_feature_study


def fixture():
    data = {
        "numeric_columns": ["impressions_prev_30d", "clicks_prev_30d", "sessions_prev_30d"],
        "categorical_columns": ["model_used"],
        "id_columns": ["content_id", "client_id"],
        "target_column": "trend_direction",
        "allowed_target_values": ["down", "up"],
        "drop_columns": [],
    }
    rows = []
    for group in range(8):
        for index in range(8):
            rows.append(
                {
                    "content_id": f"c{group}-{index}",
                    "client_id": f"g{group}",
                    "impressions_prev_30d": 10.0 + index,
                    "clicks_prev_30d": float(index),
                    "sessions_prev_30d": float(index * 2),
                    "model_used": "a",
                    "trend_direction": "down" if index % 2 == 0 else "up",
                }
            )
    frame = pd.DataFrame(rows)
    pre = load_yaml("configs/preprocessing.yaml")["preprocessing"]
    features = load_yaml("configs/features.yaml")["features"]
    features.update({"folds": 3, "log_sources": ["impressions_prev_30d"]})
    training = {
        "group_column": "client_id",
        "random_seed": 42,
        "candidates": {"random_forest": {"n_estimators": 3, "n_jobs": 1}},
    }
    return frame.iloc[:48].copy(), frame.iloc[48:].copy(), data, pre, training, features


def test_decision_guardrails_and_ties():
    rows = [
        {"variant": "A", "macro_f1": 0.6, "down_recall": 0.6, "width": 19},
        {"variant": "B", "macro_f1": 0.604, "down_recall": 0.7, "width": 23},
        {"variant": "C", "macro_f1": 0.9, "down_recall": 0.2, "width": 30},
    ]
    assert choose_variant(rows, 0.005, 0.5) == "A"
    assert choose_variant(rows, 0.005, 0.99) == "A"


def test_development_study_reproducible_and_group_isolated(tmp_path, monkeypatch):
    train, validation, data, pre, training, features = fixture()
    original_fit = __import__(
        "machine_learning_project.features.engineering", fromlist=["CutoffSafeFeatureEngineer"]
    ).CutoffSafeFeatureEngineer.fit
    fits = []

    def observed_fit(self, X, y=None):
        assert set(X.index) <= set(train.index)
        assert "trend_direction" not in X and "client_id" not in X
        fits.append(set(X.index))
        return original_fit(self, X, y)

    monkeypatch.setattr(
        "machine_learning_project.features.engineering.CutoffSafeFeatureEngineer.fit", observed_fit
    )
    one = run_feature_study(train, validation, data, pre, training, features, tmp_path / "one")
    two = run_feature_study(train, validation, data, pre, training, features, tmp_path / "two")
    assert fits
    assert one["decisions"] == two["decisions"]
    assert one["encoded_names"] == two["encoded_names"]
    assert one["test_accessed"] is False
    assert set(one["selected_configs"]) == {"logistic_regression", "random_forest"}
    text = (tmp_path / "one" / "feature_manifest.json").read_text()
    assert '"g0"' not in text and '"c0-0"' not in text
    json.loads(text)


def test_group_overlap_rejected_before_fitting(tmp_path):
    train, _, data, pre, training, features = fixture()
    with pytest.raises(DataValidationError, match="share groups"):
        run_feature_study(train, train, data, pre, training, features, tmp_path)


def test_prediction_round_trip_and_tamper_detection(tmp_path):
    train, validation, data, pre, training, features = fixture()
    raw_names = data["numeric_columns"] + data["categorical_columns"]
    pipeline = build_candidates(data, pre, training, features)["random_forest"]
    pipeline.fit(train[raw_names], train.trend_direction)
    bundle = {
        "pipeline": pipeline,
        "data_config": data,
        "feature_config": features,
        "metadata": {
            "model_version": "test",
            "bundle_schema_version": "2.0",
            "feature_config_sha256": fingerprint(features),
        },
    }
    path = tmp_path / "model.joblib"
    joblib.dump(bundle, path)
    loaded = Predictor.load(path)
    request = validation.drop(columns="trend_direction").copy()
    request["model_used"] = "unseen"
    pd.testing.assert_frame_equal(
        loaded.predict(request), Predictor(pipeline, bundle["metadata"], data).predict(request)
    )
    bundle["metadata"]["feature_config_sha256"] = "tampered"
    joblib.dump(bundle, path)
    with pytest.raises(DataValidationError, match="fingerprint"):
        Predictor.load(path)


def make_persisted_development(tmp_path):
    from machine_learning_project.data.ingestion import sha256_file
    from machine_learning_project.data.splitting import split_by_group, write_split_artifacts

    train, validation, data, pre, _, features = fixture()
    frame = pd.concat([train, validation], ignore_index=True)
    source = tmp_path / "source.csv"
    frame.to_csv(source, index=False)
    data.update({"csv_path": str(source), "unique_columns": ["content_id"]})
    training = load_yaml("configs/training.yaml")["training"]
    training.update(
        {
            "row_ratio_tolerance": 0.2,
            "class_ratio_tolerance": 0.2,
            "search_attempts": 8,
            "split_manifest_path": str(tmp_path / "split.json"),
            "split_assignments_path": str(tmp_path / "assignments.csv"),
            "artifact_path": str(tmp_path / "model.joblib"),
            "metadata_path": str(tmp_path / "model.json"),
            "metrics_path": str(tmp_path / "metrics.json"),
        }
    )
    training["candidates"]["random_forest"].update({"n_estimators": 3, "n_jobs": 1})
    splits = split_by_group(
        frame,
        target_column="trend_direction",
        group_column="client_id",
        row_key="content_id",
        allowed_labels=["down", "up"],
        source_sha256=sha256_file(source),
        row_ratio_tolerance=0.2,
        class_ratio_tolerance=0.2,
        search_attempts=8,
    )
    write_split_artifacts(
        splits, training["split_manifest_path"], training["split_assignments_path"]
    )
    return data, pre, training, features, splits


def test_persisted_split_integrity_and_training_never_receives_test(tmp_path, monkeypatch):
    from machine_learning_project.data.development import load_development
    from pipelines.training_pipeline import run_training

    data, pre, training, features, splits = make_persisted_development(tmp_path)
    train, validation, provenance = load_development(data, training)
    assert set(train.content_id) == set(splits.train.content_id)
    assert set(validation.content_id) == set(splits.validation.content_id)
    assert len(provenance["source_sha256"]) == 64
    test_ids = set(splits.test.content_id)
    from machine_learning_project.data.preparation import prepare_supervised_data

    def checked_preparation(frame, *args, **kwargs):
        assert not set(frame.content_id) & test_ids
        return prepare_supervised_data(frame, *args, **kwargs)

    monkeypatch.setattr("pipelines.training_pipeline.prepare_supervised_data", checked_preparation)
    result = run_training(data, pre, training, features)
    assert result.metrics["test_accessed"] is False
    assert "test" not in result.metrics
    assert Predictor.load(result.artifact_path).metadata["fitting_population"] == "train"
    assignments = pd.read_csv(training["split_assignments_path"])
    assignments.loc[0, "partition"] = "invalid"
    assignments.to_csv(training["split_assignments_path"], index=False)
    with pytest.raises(DataValidationError, match="partitions"):
        load_development(data, training)


def test_source_change_rejected(tmp_path):
    from pathlib import Path

    from machine_learning_project.data.development import load_development

    data, _, training, _, _ = make_persisted_development(tmp_path)
    path = Path(data["csv_path"])
    path.write_text(path.read_text().replace("10.0", "11.0"))
    with pytest.raises(DataValidationError, match="fingerprint mismatch"):
        load_development(data, training)


def test_frozen_handoff_and_contract_mismatch(tmp_path):
    from machine_learning_project.features.artifacts import load_frozen_configs

    train, validation, data, pre, training, features = fixture()
    provenance = {"source_sha256": "synthetic", "split_manifest_sha256": "split"}
    result = run_feature_study(
        train, validation, data, pre, training, features, tmp_path, provenance
    )
    configs, digest = load_frozen_configs(
        tmp_path / "feature_manifest.json", data, pre, training, provenance
    )
    assert configs == result["selected_configs"] and len(digest) == 64
    with pytest.raises(DataValidationError, match="provenance"):
        load_frozen_configs(tmp_path / "feature_manifest.json", data, pre, training, {})
    changed = {**pre, "scale_numeric": not pre["scale_numeric"]}
    with pytest.raises(DataValidationError, match="input contract"):
        load_frozen_configs(tmp_path / "feature_manifest.json", data, changed, training, provenance)
