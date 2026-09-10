import copy
import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

from machine_learning_project.data.ingestion import sha256_file
from machine_learning_project.inference.contracts import synthetic_request
from machine_learning_project.inference.packaged_predictor import PackagedPredictor
from machine_learning_project.models.package_artifacts import (
    load_package_manifest,
    load_packaging_run,
    read_json,
    semantic_digest,
)
from machine_learning_project.utils.config import load_yaml
from machine_learning_project.utils.exceptions import DataValidationError
from pipelines.evaluation_pipeline import run_evaluation
from pipelines.inference_pipeline import run_packaged_batch
from pipelines.packaging_pipeline import build_runtime_wheel, preflight, run_packaging
from pipelines.tuning_pipeline import run_tuning
from scripts.verify_packaging import audit_calls


def test_copy_only_reproduction_integrity_batch_and_interruption(tuning_inputs, tmp_path):
    frozen = run_tuning(**tuning_inputs, run_id="frozen")
    evaluation = load_yaml("configs/evaluation.yaml")["evaluation"]
    evaluation.update(
        output_root=str(tmp_path / "evaluation"),
        dimensions=["numeric_missingness_bucket"],
        minimum_subgroup_rows=1,
        minimum_class_support=1,
        minimum_down_support=1,
    )
    run_evaluation(
        tuning_inputs["data"],
        tuning_inputs["training"],
        tuning_inputs["preprocessing"],
        evaluation,
        frozen["report_dir"],
        frozen["model_dir"],
        tuning_inputs["feature_manifest"],
        tuning_inputs["baseline_manifest"],
        tuning_inputs["baseline_config"],
        "frozen",
    )
    packaging = load_yaml("configs/packaging.yaml")["packaging"]
    packaging.update(
        package_root=str(tmp_path / "packages"), output_root=str(tmp_path / "packaging")
    )
    inputs = {
        "packaging_config": packaging,
        "inference_config": load_yaml("configs/inference.yaml")["inference"],
        "evaluation_dir": tmp_path / "evaluation/frozen",
        "training_report_dir": frozen["report_dir"],
        "model_dir": frozen["model_dir"],
        "purpose": "research",
    }
    state = preflight(**inputs, run_id="one")
    wheel = build_runtime_wheel(tmp_path / "wheel")
    with audit_calls() as audit:
        before = set(tmp_path.rglob("*"))
        run_packaging(**inputs, run_id="dry", dry_run=True)
        assert set(tmp_path.rglob("*")) == before
        one = run_packaging(**inputs, run_id="one", runtime_wheel=wheel)
        two = run_packaging(**inputs, run_id="two", runtime_wheel=wheel)
        assert audit["fit_calls"] == 0 and audit["counts"] == {}
    load_packaging_run(one["report_dir"])
    for family, item in one["packages"].items():
        assert item["semantic_sha256"] == two["packages"][family]["semantic_sha256"]
        assert (
            sha256_file(Path(item["path"]) / "model.joblib")
            == state["references"]["model_hashes"][f"{family}.joblib"]
        )
    with pytest.raises(DataValidationError, match="exists"):
        run_packaging(**inputs, run_id="one", runtime_wheel=wheel)
    with pytest.raises(DataValidationError, match="research"):
        run_packaging(**{**inputs, "purpose": "production"}, run_id="production")
    invalid = copy.deepcopy(inputs)
    invalid["packaging_config"]["package_root"] = str(Path(frozen["model_dir"]) / "nested")
    with pytest.raises(DataValidationError, match="overlaps"):
        run_packaging(**invalid, run_id="unsafe", dry_run=True)

    root = Path(one["packages"]["random_forest"]["path"])
    predictor = PackagedPredictor.load(root, purpose="research")
    frame = synthetic_request(predictor.schema, 7)
    frame.content_id = ["001", "NA", *[f"invented-{i}" for i in range(5)]]
    source, output = tmp_path / "request.csv", tmp_path / "response.json"
    frame.to_csv(source, index=False)
    run_packaged_batch(
        source, root, output, purpose="research", include_probabilities=True, chunk_rows=3
    )
    assert read_json(output) == predictor.predict(frame, include_probabilities=True, chunk_rows=7)
    command = [
        sys.executable,
        "scripts/predict_batch.py",
        "--package-dir",
        str(root),
        "--input-path",
        str(source),
        "--output-path",
        str(tmp_path / "cli.json"),
        "--purpose",
        "research",
        "--include-probabilities",
        "--chunk-rows",
        "3",
    ]
    cli = subprocess.run(command, capture_output=True, text=True, check=False)
    assert cli.returncode == 0, cli.stderr
    assert read_json(tmp_path / "cli.json") == read_json(output)
    assert "invented-" not in cli.stdout
    frame.loc[6, "content_id"] = "001"
    frame.to_csv(source, index=False)
    with (
        patch.object(
            PackagedPredictor, "predict", side_effect=AssertionError("scored invalid batch")
        ),
        pytest.raises(DataValidationError),
    ):
        run_packaged_batch(
            source, root, tmp_path / "invalid.json", purpose="research", chunk_rows=3
        )
    assert not (tmp_path / "invalid.json").exists()
    with pytest.raises(DataValidationError):
        run_packaged_batch(source, root, source, purpose="research")
    with pytest.raises(DataValidationError):
        run_packaged_batch(source, root, output, purpose="research")

    # Rehashed forged relationships still fail before deserialization.
    manifest_path = root / "package_manifest.json"
    original = manifest_path.read_bytes()
    with patch("joblib.load", side_effect=AssertionError("unverified deserialization")):
        for field in ("references", "comparison_flags"):
            manifest = json.loads(original)
            if field == "references":
                manifest[field]["feature_manifest_sha256"] = "0" * 64
            else:
                manifest[field]["project_macro_f1_target_met"] = not manifest[field][
                    "project_macro_f1_target_met"
                ]
            manifest["semantic_sha256"] = semantic_digest(manifest)
            manifest_path.write_text(json.dumps(manifest))
            with pytest.raises(DataValidationError):
                PackagedPredictor.load(root, purpose="research")
        manifest_path.write_bytes(original)
        with (
            patch(
                "machine_learning_project.models.package_artifacts.runtime_environment",
                return_value={},
            ),
            pytest.raises(DataValidationError, match="runtime"),
        ):
            PackagedPredictor.load(root, purpose="research")
        extra = root / "unexpected.txt"
        extra.write_text("extra")
        with pytest.raises(DataValidationError, match="Unexpected"):
            PackagedPredictor.load(root, purpose="research")
        extra.unlink()
        model = root / "model.joblib"
        model_bytes = model.read_bytes()
        model.write_bytes(b"tampered")
        with pytest.raises(DataValidationError, match="fingerprint"):
            PackagedPredictor.load(root, purpose="research")
        model.write_bytes(model_bytes)
        with pytest.raises(DataValidationError, match="pinned"):
            load_package_manifest(root, "0" * 64)
    original_copy = __import__("shutil").copyfile

    def interrupted(source, destination, *args, **kwargs):
        if Path(destination).name == "model.joblib" and "random_forest" in str(destination):
            raise RuntimeError("interrupted")
        return original_copy(source, destination, *args, **kwargs)

    with (
        patch("pipelines.packaging_pipeline.shutil.copyfile", side_effect=interrupted),
        pytest.raises(RuntimeError, match="interrupted"),
    ):
        run_packaging(**inputs, run_id="interrupted", runtime_wheel=wheel)
    assert not (tmp_path / "packaging/interrupted/packaging_manifest.json").exists()
    load_package_manifest(tmp_path / "packages/interrupted-logistic_regression")
    assert not (tmp_path / "packages/interrupted-random_forest/package_manifest.json").exists()
    for path, digest in state["protected"].items():
        assert sha256_file(path) == digest
