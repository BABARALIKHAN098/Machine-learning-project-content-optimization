import copy
import json
from pathlib import Path

import pytest
from sklearn.pipeline import Pipeline

from machine_learning_project.models.train import build_trial
from machine_learning_project.models.training_artifacts import load_training_run
from machine_learning_project.utils.exceptions import DataValidationError
from pipelines.tuning_pipeline import preflight, run_tuning


def test_preflight_does_not_fit_or_write_and_preserves_frozen_inputs(tuning_inputs, monkeypatch):
    original = copy.deepcopy(tuning_inputs["training"])

    def forbidden(*args, **kwargs):
        pytest.fail("Preflight must not fit")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    result = run_tuning(**tuning_inputs, run_id="dry", dry_run=True)
    assert result["planned_fits"] == 47
    assert not Path(tuning_inputs["tuning"]["output_root"]).exists()
    assert original == tuning_inputs["training"]
    state = preflight(**tuning_inputs, run_id="dry")
    trial = state["trials"][0]
    pipeline = build_trial(
        tuning_inputs["data"],
        tuning_inputs["preprocessing"],
        original,
        state["frozen"][trial["family"]],
        trial,
    )
    assert pipeline.named_steps["model"].C == 0.1
    assert original == tuning_inputs["training"]


@pytest.mark.parametrize("kind", ["feature", "baseline", "original", "missing"])
def test_stale_dependencies_fail_before_fitting(tuning_inputs, kind, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Stale inputs must fail before fitting")

    monkeypatch.setattr(Pipeline, "fit", forbidden)
    if kind in ("feature", "baseline"):
        path = tuning_inputs[f"{kind}_manifest"]
        doc = json.loads(path.read_text())
        if kind == "feature":
            doc["input_contract_sha256"] = "stale"
        else:
            doc["evaluation_identity"]["validation_rows"] = 1
        path.write_text(json.dumps(doc))
    elif kind == "original":
        tuning_inputs["training"]["candidates"]["logistic_regression"]["C"] = 5
    else:
        tuning_inputs["feature_manifest"] = None
    with pytest.raises(DataValidationError):
        run_tuning(**tuning_inputs, run_id="bad")


def test_search_fold_isolation_reproduction_and_roundtrip(tuning_inputs, monkeypatch):
    state = preflight(**tuning_inputs, run_id="one")
    protected = {path: path.read_bytes() for path in state["snapshots"]}
    train_indices = set(state["train"].index)
    validation_indices = set(state["validation"].index)
    original_fit, original_predict = Pipeline.fit, Pipeline.predict
    fits, scores = [], []
    output = Path(tuning_inputs["tuning"]["output_root"])

    def fit(self, x, y=None, **kwargs):
        if "model" in self.named_steps:
            assert set(x.index) <= train_indices
            assert "content_id" not in x and "client_id" not in x and "target" not in x
            fits.append(set(x.index))
        return original_fit(self, x, y, **kwargs)

    def predict(self, x, **kwargs):
        if "model" in self.named_steps:
            scores.append(set(x.index))
            if set(x.index) == validation_indices:
                assert (output / "one/selection.json").is_file()
        return original_predict(self, x, **kwargs)

    monkeypatch.setattr(Pipeline, "fit", fit)
    monkeypatch.setattr(Pipeline, "predict", predict)
    one = run_tuning(**tuning_inputs, run_id="one")
    assert len(fits) == 47
    expected = [set(state["train"].iloc[left].index) for left, _ in state["folds"]]
    assert fits[:45] == expected * 15
    assert fits[-2:] == [train_indices, train_indices]
    assert len(scores) == 49  # 45 fold + 2 confirmation + 2 reload checks
    for index in range(45):
        assert not fits[index] & scores[index]
    assert one["selection"]["validation_used_for_selection"] is False
    monkeypatch.setattr(Pipeline, "fit", original_fit)
    monkeypatch.setattr(Pipeline, "predict", original_predict)
    two = run_tuning(**tuning_inputs, run_id="two")
    assert one["selection"] == two["selection"]
    assert one["validation"] == two["validation"]
    manifest, predictors = load_training_run(one["report_dir"], one["model_dir"])
    assert manifest["status"] == "complete" and len(predictors) == 2
    for path, before in protected.items():
        assert path.read_bytes() == before
    for root in (Path(one["report_dir"]), Path(one["model_dir"])):
        for path in root.iterdir():
            if path.suffix in (".json", ".md", ".csv"):
                assert (
                    "private-row-" not in path.read_text()
                    and "private-client-" not in path.read_text()
                )
    with pytest.raises(DataValidationError, match="exists"):
        run_tuning(**tuning_inputs, run_id="one")


def test_interruption_never_publishes_completion(tuning_inputs, monkeypatch):
    import pipelines.tuning_pipeline as module

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(module, "search_training", interrupted)
    with pytest.raises(KeyboardInterrupt):
        run_tuning(**tuning_inputs, run_id="interrupted")
    output = Path(tuning_inputs["tuning"]["output_root"]) / "interrupted"
    assert not (output / "training_manifest.json").exists()
    assert json.loads((output / "failure.json").read_text())["status"] == "incomplete"
