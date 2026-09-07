"""Benchmark identity, frozen evidence and candidate comparison contracts."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ..data.ingestion import sha256_file
from ..data.splitting import _hash_value
from ..features.registry import fingerprint
from ..utils.exceptions import DataValidationError
from .baseline import resolve_baseline_config
from .evaluate import METRIC_CONTRACT_VERSION

PAYLOAD_NAMES = {
    "baseline_metrics.json",
    "baseline_repeats.csv",
    "baseline_summary.json",
    "baseline_report.md",
}
IDENTITY_FIELDS = {
    "provenance",
    "metric_contract_version",
    "partition",
    "target",
    "labels",
    "train_rows",
    "validation_rows",
    "train_groups",
    "validation_groups",
    "ordered_train_sha256",
    "ordered_validation_sha256",
}


def prepare_benchmark_partitions(train, validation, data_config, training_config, provenance):
    row_key, group = training_config["row_key"], training_config["group_column"]
    frames = [
        train.sort_values(row_key, kind="stable").copy(),
        validation.sort_values(row_key, kind="stable").copy(),
    ]
    for frame in frames:
        if frame.empty or frame[row_key].isna().any() or frame[row_key].duplicated().any():
            raise DataValidationError("Benchmark requires nonempty partitions with unique row keys")
        if frame[group].isna().any():
            raise DataValidationError("Benchmark group contains null values")
    if set(frames[0][group]) & set(frames[1][group]):
        raise DataValidationError("Benchmark development groups overlap")
    if set(frames[0][row_key]) & set(frames[1][row_key]):
        raise DataValidationError("Benchmark development rows overlap")
    context = f"{training_config.get('split_alias_context', 'content-trend-split-v1')}:{provenance['source_sha256']}"
    identity = {
        "provenance": dict(provenance),
        "metric_contract_version": METRIC_CONTRACT_VERSION,
        "partition": "validation",
        "target": data_config["target_column"],
        "labels": list(data_config["allowed_target_values"]),
    }
    for name, frame in zip(("train", "validation"), frames):
        identity[f"{name}_rows"] = len(frame)
        identity[f"{name}_groups"] = int(frame[group].nunique())
        identity[f"ordered_{name}_sha256"] = fingerprint(
            [_hash_value(value, context) for value in frame[row_key]]
        )
    return *frames, identity


def require_matching_identity(candidate, reference):
    if (
        not isinstance(candidate, dict)
        or not isinstance(reference, dict)
        or set(candidate) != IDENTITY_FIELDS
        or set(reference) != IDENTITY_FIELDS
    ):
        raise DataValidationError("Missing or legacy benchmark evaluation identity")
    if candidate != reference or reference["metric_contract_version"] != METRIC_CONTRACT_VERSION:
        raise DataValidationError("Candidate/benchmark evaluation identity mismatch")
    if reference["partition"] != "validation":
        raise DataValidationError("Only development validation comparisons are supported")


def compare_candidate(metrics, candidate_identity, benchmark, baseline_config, training_config):
    """A development ranking is separate from the independent benchmark/recall/target flags."""
    config = resolve_baseline_config(baseline_config, training_config)
    require_matching_identity(candidate_identity, benchmark["evaluation_identity"])
    scores = benchmark["canonical"]
    candidate = metrics["macro_f1"]
    recall = metrics["per_class"]["down"]["recall"]
    references = {
        "most_frequent": scores["most_frequent"]["macro_f1"],
        "stratified_reference": scores["stratified"]["macro_f1"],
        "stratified_mean": benchmark["summary"]["stratified"]["macro_f1"]["mean"],
    }
    guardrail = training_config.get("down_recall_guardrail", 0.5)
    target = training_config.get("minimum_macro_f1", 0.45)
    for value in [candidate, recall, guardrail, target, *references.values()]:
        if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
            raise DataValidationError("Comparison scores/thresholds must be finite and in [0, 1]")
    delta = {name: candidate - score for name, score in references.items()}
    margin = config["minimum_macro_f1_improvement"]
    # Comparing sums rather than subtractions preserves equality at declared float boundaries.
    return {
        "macro_f1_deltas": delta,
        "beats_both_baselines": candidate > references["most_frequent"]
        and candidate > references["stratified_reference"],
        "material_improvement_met": all(
            candidate >= score + margin for score in references.values()
        ),
        "minimum_macro_f1_improvement": margin,
        "down_recall_guardrail_met": recall >= guardrail,
        "project_macro_f1_target_met": candidate >= target,
        "interpretation": "Development comparison flags; not model promotion",
    }


def load_benchmark(path, expected_identity, baseline_config):
    """Reject missing/tampered/stale evidence before reusing a published benchmark."""
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    config = resolve_baseline_config(baseline_config)
    if manifest.get("manifest_schema_version") != "1.0":
        raise DataValidationError("Unsupported baseline manifest schema")
    if manifest.get("baseline_contract_version") != config["baseline_contract_version"]:
        raise DataValidationError("Baseline contract version mismatch")
    require_matching_identity(expected_identity, manifest.get("evaluation_identity"))
    expected_config = {key: value for key, value in config.items() if key != "output_directory"}
    if manifest.get("config_sha256") != fingerprint(expected_config):
        raise DataValidationError("Baseline configuration mismatch")
    hashes = manifest.get("artifact_hashes", {})
    if set(hashes) != PAYLOAD_NAMES:
        raise DataValidationError("Incomplete baseline artifact manifest")
    for name, expected in hashes.items():
        if not (path.parent / name).is_file() or sha256_file(path.parent / name) != expected:
            raise DataValidationError(f"Baseline artifact missing or fingerprint mismatch: {name}")
    result = json.loads((path.parent / "baseline_metrics.json").read_text(encoding="utf-8"))
    require_matching_identity(expected_identity, result.get("evaluation_identity"))
    if result.get("baseline_contract_version") != config["baseline_contract_version"]:
        raise DataValidationError("Baseline metric contract mismatch")
    if result["summary"] != json.loads(
        (path.parent / "baseline_summary.json").read_text(encoding="utf-8")
    ):
        raise DataValidationError("Baseline summary disagrees with metrics")
    return result, sha256_file(path)
