"""Execute and verify two isolated SPEC-06 searches, preserving earlier evidence."""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machine_learning_project.data.ingestion import sha256_file
from machine_learning_project.models.training_artifacts import load_training_run, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", default="spec06")
    args = parser.parse_args()
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,59}", args.run_prefix):
        parser.error("Run prefix must be a safe local name")
    root = Path(__file__).resolve().parents[1]
    evidence_path = root / "reports/training" / f"{args.run_prefix}-verification.json"
    if evidence_path.exists():
        parser.error("Verification evidence already exists; use a new run prefix")
    protected = {root / "content_refresh_anonymized.csv", root / "data/raw/dataset.csv"}
    for directory in (
        "artifacts/metadata",
        "reports/features",
        "reports/baselines",
        "reports/metrics",
    ):
        protected.update(p for p in (root / directory).rglob("*") if p.is_file())
    protected.update(p for p in (root / "artifacts/models").rglob("*") if p.is_file())
    before = {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(protected)}
    evidence = {
        "status": "running",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "commands": [],
    }

    def run(arguments):
        print("Running: " + subprocess.list2cmdline(arguments), flush=True)
        completed = subprocess.run(arguments, cwd=root, capture_output=True, text=True, check=False)
        evidence["commands"].append(
            {
                "command": subprocess.list2cmdline(arguments),
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        write_json(evidence_path, evidence)
        if completed.returncode:
            print(completed.stderr, flush=True)
            raise RuntimeError("Verification command failed; see recorded stdout/stderr")
        print("Command passed.", flush=True)

    try:
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/test_tuning_config.py",
                "tests/unit/test_tuning.py",
                "tests/integration/test_tuning_pipeline.py",
                "tests/contract/test_training_contract.py",
                "tests/integration/test_training_pipeline.py",
                "tests/integration/test_baseline_pipeline.py",
            ]
        )
        run([sys.executable, "-m", "pytest"])
        run([sys.executable, "-m", "ruff", "check", "src", "pipelines", "scripts", "tests"])
        manifests, results = [], []
        for suffix in ("reference", "reproduction"):
            run_id = f"{args.run_prefix}-{suffix}"
            run(
                [
                    sys.executable,
                    "scripts/train_model.py",
                    "--tuning-config",
                    "configs/tuning.yaml",
                    "--feature-manifest",
                    "reports/features/feature_manifest.json",
                    "--baseline-manifest",
                    "reports/baselines/baseline_manifest.json",
                    "--run-id",
                    run_id,
                ]
            )
            report_dir, model_dir = (
                root / "reports/training" / run_id,
                root / "artifacts/models/training" / run_id,
            )
            manifest, predictors = load_training_run(report_dir, model_dir)
            manifests.append(manifest)
            results.append(
                {
                    name: json.loads((report_dir / name).read_text())
                    for name in (
                        "resolved_config.json",
                        "fold_manifest.json",
                        "trial_metrics.json",
                        "selection.json",
                        "validation_metrics.json",
                    )
                }
            )
            assert len(predictors) == 2
            assert (
                json.loads((report_dir / "timings.json").read_text())["actual_estimator_fits"] == 47
            )
        assert results[0] == results[1], "Semantic training outputs differ"
        for key in (
            "code",
            "environment",
            "evaluation_identity",
            "resolved_config_sha256",
            "selection_sha256",
            "fold_manifest_sha256",
        ):
            assert manifests[0][key] == manifests[1][key], f"Normalized provenance differs: {key}"
        after = {name: sha256_file(root / name) for name in before}
        assert before == after, "Protected prerequisite/model/report bytes changed"
        evidence.update(
            status="passed",
            protected_artifacts_unchanged=True,
            protected_artifact_sha256=before,
            reproducibility={
                "semantic_outputs_equal": True,
                "normalized_provenance_equal": True,
                "each_run_hashes_verified": True,
                "finalist_reload_checks_passed": True,
                "fits_per_run": 47,
                "total_real_fits": 94,
                "excluded": ["run IDs", "physical paths", "timings", "joblib bytes"],
            },
            execution_scope={
                "test_scoring": False,
                "train_plus_validation_refit": False,
                "public_artifacts": "Aggregates and scoped hashes; no row predictions",
                "fit_isolation": "Verified with synthetic Pipeline fit/predict spies",
            },
            finalist_summary=results[0]["validation_metrics.json"]["finalists"],
        )
        for suffix in ("reference", "reproduction"):
            report_dir = root / "reports/training" / f"{args.run_prefix}-{suffix}"
            write_json(
                report_dir / "verification.json",
                {
                    "status": "passed",
                    "evidence_file": evidence_path.name,
                    "manifest_sha256": sha256_file(report_dir / "training_manifest.json"),
                    "reproducibility": evidence["reproducibility"],
                    "protected_artifacts_unchanged": True,
                },
            )
    except BaseException as error:
        evidence.update(status="failed", error_type=type(error).__name__)
        raise
    finally:
        write_json(evidence_path, evidence)
    print(f"Verified evidence: {evidence_path}", flush=True)


if __name__ == "__main__":
    main()
