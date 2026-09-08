"""Trusted local training artifacts; validate all hashes before loading model bytes."""

from __future__ import annotations

import json
import platform
import re
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
import threadpoolctl

from ..data.ingestion import sha256_file
from ..data.splitting import _atomic_write
from ..features.registry import fingerprint
from ..inference.predictor import Predictor
from ..utils.exceptions import DataValidationError

REPORT_FILES = {
    "resolved_config.json",
    "fold_manifest.json",
    "trial_metrics.json",
    "cv_results.csv",
    "selection.json",
    "validation_metrics.json",
    "timings.json",
    "training_report.md",
}
FAMILIES = ("logistic_regression", "random_forest")
MODEL_FILES = {f"{family}.{suffix}" for family in FAMILIES for suffix in ("joblib", "json")}


def write_json(path, value):
    _atomic_write(Path(path), json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def safe_payload(root, name):
    root = Path(root).resolve()
    if not isinstance(name, str) or Path(name).name != name or ":" in name or "\\" in name:
        raise DataValidationError("Unsafe artifact name")
    path = root / name
    if path.resolve().parent != root:
        raise DataValidationError("Artifact escapes its approved directory")
    return path


def run_directories(config, run_id):
    if (
        not isinstance(run_id, str)
        or not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,79}", run_id)
        or run_id.upper()
        in {
            "CON",
            "PRN",
            "AUX",
            "NUL",
            *[f"COM{i}" for i in range(10)],
            *[f"LPT{i}" for i in range(10)],
        }
    ):
        raise DataValidationError("Run ID must be a safe local name")
    roots = [Path(config[key]).resolve() for key in ("output_root", "model_output_root")]
    if roots[0] == roots[1] or roots[0] in roots[1].parents or roots[1] in roots[0].parents:
        raise DataValidationError("Report and model roots must be separate")
    paths = [safe_payload(root, run_id) for root in roots]
    if any(path.exists() for path in paths):
        raise DataValidationError("Run destination already exists; choose a new run ID")
    return paths


def code_environment():
    root = Path(__file__).resolve().parents[3]
    code = {
        "source_tree_sha256": fingerprint(
            {
                path.relative_to(root).as_posix(): sha256_file(path)
                for folder in ("src", "pipelines", "scripts")
                for path in sorted((root / folder).rglob("*.py"))
            }
        )
    }
    try:
        code["revision"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True
        ).stdout.strip()
        code["dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=root,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        code.update(revision="unavailable", dirty=None)
    return code, {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "sklearn": sklearn.__version__,
        "threadpoolctl": threadpoolctl.__version__,
        "thread_limit": 1,
        "estimator_n_jobs": 1,
        "search_n_jobs": 1,
    }


def load_training_run(report_dir, model_dir):
    report_dir, model_dir = Path(report_dir).resolve(), Path(model_dir).resolve()
    manifest = json.loads((report_dir / "training_manifest.json").read_text(encoding="utf-8"))
    if (
        manifest.get("artifact_schema_version") != "1.0"
        or manifest.get("tuning_contract_version") != "1.0"
        or manifest.get("status") != "complete"
    ):
        raise DataValidationError("Training run is incomplete or unsupported")
    for root, hashes, required in (
        (report_dir, manifest.get("report_hashes", {}), REPORT_FILES),
        (model_dir, manifest.get("model_hashes", {}), MODEL_FILES),
    ):
        if set(hashes) != required:
            raise DataValidationError("Incomplete training artifact manifest")
        for name, expected in hashes.items():
            path = safe_payload(root, name)
            if not path.is_file() or sha256_file(path) != expected:
                raise DataValidationError("Training artifact missing or fingerprint mismatch")
    # Hash every payload before any trusted local joblib deserialization.
    predictors = {family: Predictor.load(model_dir / f"{family}.joblib") for family in FAMILIES}
    for family, predictor in predictors.items():
        metadata = json.loads((model_dir / f"{family}.json").read_text(encoding="utf-8"))
        if metadata != predictor.metadata or not metadata.get("development_only"):
            raise DataValidationError("Model metadata disagrees or is not development-only")
        if metadata["evaluation_identity"] != manifest["evaluation_identity"]:
            raise DataValidationError("Model evaluation identity mismatch")
    return manifest, predictors
