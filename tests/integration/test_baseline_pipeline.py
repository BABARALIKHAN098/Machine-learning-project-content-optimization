import json
from pathlib import Path

import pytest

from machine_learning_project.models.benchmark import load_benchmark
from machine_learning_project.utils.config import load_yaml
from machine_learning_project.utils.exceptions import DataValidationError
from pipelines.baseline_pipeline import run_baselines
from pipelines.training_pipeline import run_training


def test_standalone_reproducibility_privacy_and_training_reuse(
    baseline_inputs, tmp_path, monkeypatch
):
    data, training, config = baseline_inputs
    protected = [
        Path(data["csv_path"]),
        Path(training["split_manifest_path"]),
        Path(training["split_assignments_path"]),
    ]
    before = [p.read_bytes() for p in protected]
    import pipelines.training_pipeline as training_module

    original = training_module.build_candidates

    def forbidden(*args, **kwargs):
        pytest.fail("Standalone baseline invoked candidate building")

    monkeypatch.setattr(training_module, "build_candidates", forbidden)
    first = run_baselines(data, training, config, tmp_path / "one")
    second = run_baselines(data, training, config, tmp_path / "two")
    assert first == second
    assert not Path(training["artifact_path"]).exists()
    assert before == [p.read_bytes() for p in protected]
    manifest = tmp_path / "one/baseline_manifest.json"
    loaded, manifest_hash = load_benchmark(manifest, first["evaluation_identity"], config)
    assert loaded == first and len(manifest_hash) == 64
    for path in (tmp_path / "one").iterdir():
        assert "private-row-" not in path.read_text() and "private-client-" not in path.read_text()
    monkeypatch.setattr(training_module, "build_candidates", original)
    pre = load_yaml("configs/preprocessing.yaml")["preprocessing"]
    integrated = run_training(data, pre, training, baseline_config=config)
    assert integrated.metrics["baselines_validation"] == first["canonical"]
    reused = run_training(data, pre, training, baseline_config=config, baseline_manifest=manifest)
    assert reused.metrics["baseline_manifest_sha256"] == manifest_hash
    assert reused.metrics["selection_status"] == "selected_for_development"
    assert reused.metrics["test_accessed"] is False
    assert set(reused.metrics["baseline_comparisons"]) == {"logistic_regression", "random_forest"}


@pytest.mark.parametrize("kind", ["missing", "source", "assignment"])
def test_invalid_persisted_inputs_fail_without_publication(baseline_inputs, tmp_path, kind):
    data, training, config = baseline_inputs
    if kind == "missing":
        Path(training["split_manifest_path"]).unlink()
    elif kind == "source":
        with Path(data["csv_path"]).open("a") as handle:
            handle.write("\n")
    else:
        path = Path(training["split_assignments_path"])
        text = path.read_text().replace("validation", "train")
        path.write_text(text)
    with pytest.raises(DataValidationError):
        run_baselines(data, training, config, tmp_path / "output")
    assert not (tmp_path / "output/baseline_manifest.json").exists()


def test_interrupted_publication_is_rejected(baseline_inputs, tmp_path, monkeypatch):
    import pipelines.baseline_pipeline as module

    data, training, config = baseline_inputs
    output = tmp_path / "output"
    result = run_baselines(data, training, config, output)
    original = module._atomic_write

    def interrupted(path, text):
        if path.name == "baseline_summary.json":
            raise OSError("simulated interrupted write")
        original(path, text)

    monkeypatch.setattr(module, "_atomic_write", interrupted)
    with pytest.raises(OSError, match="interrupted"):
        run_baselines(data, training, config, output)
    with pytest.raises(DataValidationError, match="fingerprint"):
        load_benchmark(output / "baseline_manifest.json", result["evaluation_identity"], config)
    assert json.loads((output / "baseline_manifest.json").read_text())["artifact_hashes"]
