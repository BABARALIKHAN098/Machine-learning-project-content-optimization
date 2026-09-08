"""Rebuild and verify SPEC-05 evidence without training real-data candidates."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machine_learning_project.data.ingestion import sha256_file
from machine_learning_project.models.benchmark import load_benchmark
from machine_learning_project.utils.config import load_yaml


def main():
    root = Path(__file__).resolve().parents[1]
    output = root / "reports/baselines"
    output.mkdir(parents=True, exist_ok=True)
    data = load_yaml(root / "configs/data.yaml")["data"]
    config = load_yaml(root / "configs/baselines.yaml")["baselines"]
    protected = {root / data["csv_path"]}
    for folder in ("artifacts", "reports/features", "reports/metrics"):
        protected.update(p for p in (root / folder).rglob("*") if p.is_file())
    before = {str(p.relative_to(root)): sha256_file(p) for p in sorted(protected)}
    evidence = {
        "verified_at_utc": datetime.now(UTC).isoformat(),
        "commands": [],
        "status": "running",
    }

    def run(arguments):
        completed = subprocess.run(arguments, cwd=root, text=True, capture_output=True, check=False)
        evidence["commands"].append(
            {
                "command": subprocess.list2cmdline(arguments),
                "exit_code": completed.returncode,
                "stdout": completed.stdout,
                "stderr": completed.stderr,
            }
        )
        print(completed.stdout, end="", flush=True)
        if completed.returncode:
            raise RuntimeError(f"Verification command failed: {arguments}")

    try:
        run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/test_baselines.py",
                "tests/unit/test_evaluation.py",
                "tests/integration/test_baseline_pipeline.py",
                "tests/contract/test_baseline_contract.py",
                "tests/integration/test_training_pipeline.py",
            ]
        )
        run([sys.executable, "-m", "pytest"])
        run([sys.executable, "-m", "ruff", "check", "src", "pipelines", "scripts", "tests"])
        with tempfile.TemporaryDirectory(prefix="baseline-verification-") as scratch:
            for directory in (str(output), scratch):
                run([sys.executable, "scripts/run_baselines.py", "--output-dir", directory])
            manifest_path = output / "baseline_manifest.json"
            manifest = json.loads(manifest_path.read_text())
            identity = manifest["evaluation_identity"]
            first, manifest_hash = load_benchmark(manifest_path, identity, config)
            second, _ = load_benchmark(Path(scratch) / "baseline_manifest.json", identity, config)
            second_manifest = json.loads((Path(scratch) / "baseline_manifest.json").read_text())
            assert first == second, "Semantic metrics/priors/prediction fingerprints differ"
            for key in ("config_sha256", "code", "environment", "evaluation_identity"):
                assert manifest[key] == second_manifest[key], f"Provenance differs: {key}"
            for name in ("baseline_metrics.json", "baseline_summary.json", "baseline_report.md"):
                assert (output / name).read_bytes() == (Path(scratch) / name).read_bytes()
            evidence["reproducibility"] = {
                "semantic_artifacts_equal": True,
                "per_seed_prediction_fingerprints_equal": True,
                "each_run_payload_hashes_verified": True,
                "normalized_provenance_equal": True,
                "excluded": ["fit_seconds", "prediction_seconds", "timing-bearing payload hashes"],
                "benchmark_manifest_sha256": manifest_hash,
            }
        after = {name: sha256_file(root / name) for name in before}
        assert before == after, "Protected source/split/model/report bytes changed"
        evidence["protected_artifact_sha256"] = before
        evidence["protected_artifacts_unchanged"] = True
        evidence["execution_scope"] = {
            "real_candidate_training": False,
            "test_evaluation": False,
            "integrity_access": "Full-source validation upstream; runner accepts train/validation only",
            "privacy": "Aggregate outputs and scoped identity hashes; no row-level predictions",
            "tests": "Neutral-input fit/predict instrumentation and synthetic training reuse passed",
        }
        evidence["status"] = "passed"
    except Exception as error:
        evidence["status"] = "failed"
        evidence["error"] = str(error)
        raise
    finally:
        (output / "verification.json").write_text(
            json.dumps(evidence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )


if __name__ == "__main__":
    main()
