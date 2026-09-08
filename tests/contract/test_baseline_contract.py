import copy
import json

import pytest

from machine_learning_project.models.benchmark import compare_candidate, load_benchmark
from machine_learning_project.utils.exceptions import DataValidationError
from pipelines.baseline_pipeline import run_baselines


def test_comparison_boundaries_and_independent_flags(baseline_inputs, tmp_path):
    data, training, config = baseline_inputs
    result = run_baselines(data, training, config, tmp_path / "output")
    result["canonical"]["most_frequent"]["macro_f1"] = 0.2
    result["canonical"]["stratified"]["macro_f1"] = 0.3
    result["summary"]["stratified"]["macro_f1"]["mean"] = 0.3
    for score, beats, material in [
        (0.2, False, False),
        (0.3, False, False),
        (0.305, True, False),
        (0.31, True, True),
    ]:
        metrics = {"macro_f1": score, "per_class": {"down": {"recall": 0.5}}}
        flags = compare_candidate(metrics, result["evaluation_identity"], result, config, training)
        assert flags["beats_both_baselines"] is beats
        assert flags["material_improvement_met"] is material
        assert flags["down_recall_guardrail_met"] is True
        assert flags["project_macro_f1_target_met"] is False
    flags = compare_candidate(
        {"macro_f1": 0.45, "per_class": {"down": {"recall": 0.49}}},
        result["evaluation_identity"],
        result,
        config,
        training,
    )
    assert flags["project_macro_f1_target_met"] and not flags["down_recall_guardrail_met"]


def test_every_identity_field_and_legacy_evidence_rejected(baseline_inputs, tmp_path):
    data, training, config = baseline_inputs
    result = run_baselines(data, training, config, tmp_path / "output")
    metrics = result["canonical"]["most_frequent"]
    for key in result["evaluation_identity"]:
        identity = copy.deepcopy(result["evaluation_identity"])
        identity[key] = None
        with pytest.raises(DataValidationError):
            compare_candidate(metrics, identity, result, config, training)
    with pytest.raises(DataValidationError, match="legacy"):
        compare_candidate(metrics, {}, result, config, training)


def test_manifest_tampering_and_incomplete_payloads(baseline_inputs, tmp_path):
    data, training, config = baseline_inputs
    output = tmp_path / "output"
    result = run_baselines(data, training, config, output)
    path = output / "baseline_manifest.json"
    original = path.read_text()
    for mutation in ["version", "config", "path", "missing"]:
        manifest = json.loads(original)
        if mutation == "version":
            manifest["manifest_schema_version"] = "0.0"
        elif mutation == "config":
            manifest["config_sha256"] = "invalid"
        elif mutation == "path":
            manifest["artifact_hashes"]["../outside.json"] = "invalid"
        else:
            manifest["artifact_hashes"].pop("baseline_metrics.json")
        path.write_text(json.dumps(manifest))
        with pytest.raises(DataValidationError):
            load_benchmark(path, result["evaluation_identity"], config)
    path.write_text(original)
    (output / "baseline_metrics.json").write_text("{}")
    with pytest.raises(DataValidationError, match="fingerprint"):
        load_benchmark(path, result["evaluation_identity"], config)


def test_canonical_order_and_group_overlap(baseline_inputs):
    from machine_learning_project.data.development import load_development
    from machine_learning_project.models.benchmark import prepare_benchmark_partitions

    data, training, _ = baseline_inputs
    train, validation, provenance = load_development(data, training)
    one = prepare_benchmark_partitions(train, validation, data, training, provenance)
    two = prepare_benchmark_partitions(
        train.sample(frac=1, random_state=8),
        validation.sample(frac=1, random_state=9),
        data,
        training,
        provenance,
    )
    assert one[2] == two[2]
    assert one[0].equals(two[0]) and one[1].equals(two[1])
    with pytest.raises(DataValidationError, match="overlap"):
        prepare_benchmark_partitions(train, train, data, training, provenance)
