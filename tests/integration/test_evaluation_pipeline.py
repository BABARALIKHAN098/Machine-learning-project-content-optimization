import copy
from pathlib import Path
from unittest.mock import patch

import pytest
from sklearn.pipeline import Pipeline

from machine_learning_project.models.evaluation_artifacts import (
    load_evaluation,
    semantic_evaluation,
)
from machine_learning_project.utils.config import load_yaml
from machine_learning_project.utils.exceptions import DataValidationError
from pipelines.evaluation_pipeline import preflight, run_evaluation
from pipelines.tuning_pipeline import run_tuning
from scripts.verify_evaluation import audit_inference, verify_runs


def test_synthetic_replay_no_fit_drift_privacy_and_immutability(tuning_inputs, tmp_path):
    # Only small synthetic fixture training; production evaluation never builds or fits candidates.
    frozen = run_tuning(**tuning_inputs, run_id="frozen")
    evaluation = load_yaml("configs/evaluation.yaml")["evaluation"]
    evaluation.update(
        output_root=str(tmp_path / "evaluation"),
        dimensions=["numeric_missingness_bucket"],
        minimum_subgroup_rows=1,
        minimum_class_support=1,
        minimum_down_support=1,
    )
    inputs = {
        "data_config": tuning_inputs["data"],
        "training_config": tuning_inputs["training"],
        "preprocessing_config": tuning_inputs["preprocessing"],
        "evaluation_config": evaluation,
        "training_report_dir": frozen["report_dir"],
        "model_dir": frozen["model_dir"],
        "feature_manifest": tuning_inputs["feature_manifest"],
        "baseline_manifest": tuning_inputs["baseline_manifest"],
        "baseline_config": tuning_inputs["baseline_config"],
    }
    evidence, roots = verify_runs(inputs, "one", "two")
    assert evidence["counts"] == {"fit_calls": 0, "predict_calls": 4, "probability_calls": 0}
    assert semantic_evaluation(roots[0]) == semantic_evaluation(roots[1])
    with pytest.raises(DataValidationError, match="exists"):
        run_evaluation(**inputs, run_id="one")
    state = preflight(
        tuning_inputs["data"],
        tuning_inputs["training"],
        tuning_inputs["preprocessing"],
        evaluation,
        frozen["report_dir"],
        frozen["model_dir"],
        tuning_inputs["feature_manifest"],
        tuning_inputs["baseline_manifest"],
        tuning_inputs["baseline_config"],
        "drift",
    )
    original = Pipeline.predict

    def drift(self, x, **kwargs):
        predictions = original(self, x, **kwargs)
        predictions[:] = "up"
        return predictions

    with (
        audit_inference(state["prepared"].features),
        patch.object(Pipeline, "predict", drift),
        pytest.raises(DataValidationError, match="drift"),
    ):
        run_evaluation(**inputs, run_id="drift")
    assert not (tmp_path / "evaluation/drift").exists()
    changed = copy.deepcopy(inputs)
    changed["training_config"]["minimum_macro_f1"] = 0.1
    with pytest.raises(DataValidationError, match="mismatch"):
        run_evaluation(**changed, run_id="stale", dry_run=True)
    for path, digest in state["snapshots"].items():
        from machine_learning_project.data.ingestion import sha256_file

        assert sha256_file(path) == digest
    with (
        patch("pipelines.evaluation_pipeline._plots", side_effect=RuntimeError("interrupted")),
        pytest.raises(RuntimeError, match="interrupted"),
    ):
        run_evaluation(**inputs, run_id="interrupted")
    assert not (tmp_path / "evaluation/interrupted/evaluation_manifest.json").exists()
    for path, digest in state["snapshots"].items():
        assert sha256_file(path) == digest
    # Reject a complete-looking manifest with an escaped or missing payload inventory.
    import json

    original_manifest = (roots[1] / "evaluation_manifest.json").read_text()
    manifest_doc = json.loads(original_manifest)
    manifest_doc["payload_hashes"]["../escape.json"] = "0" * 64
    (roots[1] / "evaluation_manifest.json").write_text(json.dumps(manifest_doc))
    with pytest.raises(DataValidationError, match="payload list"):
        load_evaluation(roots[1])
    (roots[1] / "evaluation_manifest.json").write_text(original_manifest)
    (roots[1] / "figures/logistic_regression_rates.png").unlink()
    with pytest.raises(DataValidationError, match="tampered"):
        load_evaluation(roots[1])
    (roots[0] / "decision.json").write_text("{}")
    with pytest.raises(DataValidationError, match="tampered"):
        load_evaluation(roots[0])
    manifest = Path(frozen["report_dir"]) / "selection.json"
    manifest.write_text("{}")
    with pytest.raises(DataValidationError, match="fingerprint"):
        run_evaluation(**inputs, run_id="tampered", dry_run=True)
