import json

import pytest

from machine_learning_project.models.training_artifacts import (
    MODEL_FILES,
    REPORT_FILES,
    load_training_run,
    run_directories,
    safe_payload,
)
from machine_learning_project.utils.exceptions import DataValidationError


@pytest.mark.parametrize("run_id", ["../escape", "C:outside", "a/b", "", "CON", "a\\b"])
def test_unsafe_run_names_rejected(tmp_path, run_id):
    config = {
        "output_root": str(tmp_path / "reports"),
        "model_output_root": str(tmp_path / "models"),
    }
    with pytest.raises(DataValidationError):
        run_directories(config, run_id)


def test_overlapping_roots_rejected(tmp_path):
    with pytest.raises(DataValidationError):
        run_directories(
            {"output_root": str(tmp_path), "model_output_root": str(tmp_path / "nested")}, "x"
        )
    with pytest.raises(DataValidationError):
        safe_payload(tmp_path, "../outside")


def test_all_hashes_checked_before_deserialization(tmp_path, monkeypatch):
    import machine_learning_project.models.training_artifacts as module
    from machine_learning_project.data.ingestion import sha256_file

    reports, models = tmp_path / "reports", tmp_path / "models"
    reports.mkdir()
    models.mkdir()
    for root, names in ((reports, REPORT_FILES), (models, MODEL_FILES)):
        for name in names:
            (root / name).write_text("{}")
    manifest = {
        "artifact_schema_version": "1.0",
        "tuning_contract_version": "1.0",
        "status": "complete",
        "report_hashes": {name: sha256_file(reports / name) for name in REPORT_FILES},
        "model_hashes": {name: sha256_file(models / name) for name in MODEL_FILES},
    }
    (reports / "training_manifest.json").write_text(json.dumps(manifest))
    (models / "random_forest.joblib").write_text("tampered")

    def forbidden(*args, **kwargs):
        pytest.fail("Never deserialize before checking every hash")

    monkeypatch.setattr(module.Predictor, "load", forbidden)
    with pytest.raises(DataValidationError, match="fingerprint"):
        load_training_run(reports, models)
    manifest["model_hashes"].pop("random_forest.joblib")
    (reports / "training_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(DataValidationError, match="Incomplete"):
        load_training_run(reports, models)
