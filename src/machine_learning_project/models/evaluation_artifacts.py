"""Manifest-last, integrity-checked private aggregate evaluation artifacts."""

import json
import re
from pathlib import Path

from ..data.ingestion import sha256_file
from ..utils.exceptions import DataValidationError
from .benchmark import require_matching_identity
from .training_artifacts import FAMILIES, safe_payload

PAYLOADS = {
    "resolved_config.json",
    "evaluation_metrics.json",
    "class_errors.csv",
    "confusion_matrices.json",
    "paired_comparison.json",
    "subgroup_metrics.csv",
    "subgroup_coverage.json",
    "client_sensitivity.json",
    "decision.json",
    "evaluation_report.md",
    "timings.json",
    "plot_data.json",
    "error_analysis.json",
    *{f"figures/{f}_{kind}.png" for f in FAMILIES for kind in ("counts", "rates")},
    "figures/supported_down_recall.png",
}


def evaluation_directory(config, run_id, protected):
    if not isinstance(run_id, str) or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id):
        raise DataValidationError("Run ID must be a safe local name")
    if run_id.upper() in {
        "CON",
        "PRN",
        "AUX",
        "NUL",
        *[f"COM{i}" for i in range(10)],
        *[f"LPT{i}" for i in range(10)],
    }:
        raise DataValidationError("Reserved run ID")
    root = Path(config["output_root"]).resolve()
    output = safe_payload(root, run_id)
    if output.exists():
        raise DataValidationError("Run destination already exists")
    for path in protected:
        path = Path(path).resolve()
        if (
            output == path
            or output in path.parents
            or path.parent == root
            or (path.suffix == ".joblib" and path.parent in root.parents)
        ):
            raise DataValidationError("Evaluation output overlaps a protected input directory")
    return output


def payload_path(root, name):
    root = Path(root).resolve()
    if name not in PAYLOADS:
        raise DataValidationError("Unknown or unsafe evaluation payload")
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise DataValidationError("Evaluation payload escapes run directory")
    return path


def load_evaluation(root):
    root = Path(root)
    manifest_path = safe_payload(root, "evaluation_manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if (
        manifest.get("status") != "complete"
        or any(
            manifest.get(k) != "1.0"
            for k in (
                "artifact_schema_version",
                "evaluation_contract_version",
                "metric_contract_version",
            )
        )
        or manifest.get("evaluation_identity", {}).get("partition") != "validation"
    ):
        raise DataValidationError("Incomplete or unsupported evaluation")
    hashes = manifest.get("payload_hashes", {})
    if set(hashes) != PAYLOADS:
        raise DataValidationError("Incomplete evaluation payload list")
    for name, expected in hashes.items():
        path = payload_path(root, name)
        if not path.is_file() or sha256_file(path) != expected:
            raise DataValidationError("Missing or tampered evaluation payload")
    for field in (
        "references",
        "code",
        "environment",
        "training_environment",
        "protected_input_hashes",
    ):
        if not isinstance(manifest.get(field), dict) or not manifest[field]:
            raise DataValidationError("Missing evaluation provenance")

    def read(name):
        return json.loads(payload_path(root, name).read_text(encoding="utf-8"))

    metrics, resolved, decision = [
        read(name) for name in ("evaluation_metrics.json", "resolved_config.json", "decision.json")
    ]
    require_matching_identity(manifest["evaluation_identity"], metrics.get("evaluation_identity"))
    if (
        manifest["references"] != resolved.get("references")
        or manifest["references"] != decision.get("frozen_references")
        or set(metrics.get("finalists", {})) != set(FAMILIES)
        or decision.get("execution_status") != "complete"
        or decision.get("production_ready") is not False
    ):
        raise DataValidationError("Evaluation semantic references disagree")
    return manifest


def semantic_evaluation(root):
    manifest = load_evaluation(root)
    output = {
        name: json.loads((Path(root) / name).read_text(encoding="utf-8"))
        for name in sorted(PAYLOADS)
        if name.endswith(".json") and name != "timings.json"
    }
    output["evaluation_identity"] = manifest["evaluation_identity"]
    for name in ("class_errors.csv", "subgroup_metrics.csv"):
        output[name] = (Path(root) / name).read_text(encoding="utf-8")
    return output
