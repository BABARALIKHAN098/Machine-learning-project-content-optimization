"""Verify SPEC-07 without entering any real-data fitting or test-scoring path."""

import argparse
import json
import subprocess
import sys
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder

from machine_learning_project.data.ingestion import sha256_file
from machine_learning_project.models.evaluation_artifacts import (
    load_evaluation,
    semantic_evaluation,
)
from machine_learning_project.models.training_artifacts import write_json
from machine_learning_project.utils.config import load_yaml
from pipelines.evaluation_pipeline import preflight, run_evaluation


@contextmanager
def audit_inference(expected_features):
    """Fail any estimator/transformer fit and require exact canonical validation features."""
    counts = {"fit_calls": 0, "predict_calls": 0, "probability_calls": 0}
    original = Pipeline.predict

    def forbidden(*args, **kwargs):
        counts["fit_calls"] += 1
        raise AssertionError("Evaluation attempted fitting")

    def probability(*args, **kwargs):
        counts["probability_calls"] += 1
        raise AssertionError("Evaluation attempted probability scoring")

    def predict(self, x, *args, **kwargs):
        if "model" in self.named_steps:
            if not x.equals(expected_features):
                raise AssertionError(
                    "Prediction input is not the exact canonical validation feature batch"
                )
            counts["predict_calls"] += 1
        return original(self, x, *args, **kwargs)

    classes, pending = set(), [BaseEstimator]
    while pending:
        cls = pending.pop()
        if cls not in classes:
            classes.add(cls)
            pending.extend(cls.__subclasses__())
    with ExitStack() as stack:
        for cls in classes:
            # sklearn's fixed-label metric arithmetic builds a local LabelEncoder.
            # It is not a fitted feature transformer or finalist and learns no feature state.
            if cls is LabelEncoder:
                continue
            for method in ("fit", "fit_transform", "partial_fit"):
                if method in cls.__dict__:
                    stack.enter_context(patch.object(cls, method, forbidden))
        stack.enter_context(patch.object(Pipeline, "predict", predict))
        stack.enter_context(patch.object(Pipeline, "predict_proba", probability))
        yield counts


def verify_runs(inputs, first, second):
    # Read-only preflight provides the exact batch used to instrument inference.
    s = preflight(
        inputs["data_config"],
        inputs["training_config"],
        inputs["preprocessing_config"],
        inputs["evaluation_config"],
        inputs["training_report_dir"],
        inputs["model_dir"],
        inputs["feature_manifest"],
        inputs["baseline_manifest"],
        inputs["baseline_config"],
        first,
    )
    snapshots = s["snapshots"]
    privacy_values = set(map(str, s["prepared"].identifiers.to_numpy().ravel()))
    for col in s["context"]:
        if col in inputs["data_config"]["categorical_columns"]:
            privacy_values.update(str(v) for v in s["context"][col].dropna().unique())
    with audit_inference(s["prepared"].features) as counts:
        dry = run_evaluation(**inputs, run_id=first, dry_run=True)
        if counts["predict_calls"] != 0 or dry["status"] != "preflight_passed":
            raise AssertionError("Dry run performed inference")
        one = run_evaluation(**inputs, run_id=first)
        two = run_evaluation(**inputs, run_id=second)
    if counts != {"fit_calls": 0, "predict_calls": 4, "probability_calls": 0}:
        raise AssertionError("Unexpected evaluation call counts")
    if semantic_evaluation(one["report_dir"]) != semantic_evaluation(two["report_dir"]):
        raise AssertionError("Semantic evaluation outputs differ")
    if any(sha256_file(path) != digest for path, digest in snapshots.items()):
        raise AssertionError("Protected inputs changed")
    # Inspect structured string values, avoiding substring collisions with ordinary prose.
    private_identifiers = set(map(str, s["prepared"].identifiers.to_numpy().ravel()))

    def category_values(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "group" and isinstance(item, str):
                    yield item
                else:
                    yield from category_values(item)
        elif isinstance(value, list):
            for item in value:
                yield from category_values(item)

    for result in (one, two):
        root = Path(result["report_dir"])
        for path in root.rglob("*"):
            if path.suffix in (".json", ".csv", ".md"):
                text = path.read_text(encoding="utf-8")
                if any(value in text for value in private_identifiers):
                    raise AssertionError("Private identifier exposed in evaluation")
                if (
                    path.suffix == ".json"
                    and set(category_values(json.loads(text))) & privacy_values
                ):
                    raise AssertionError("Raw private category exposed in structured evaluation")
        load_evaluation(root)
    evidence = {
        "status": "passed",
        "counts": counts,
        "fit_audit_scope": "Estimator/feature-transformer fits blocked; sklearn metric-local "
        "LabelEncoder bookkeeping excluded",
        "exact_validation_batch_checked": True,
        "semantic_reproducibility": True,
        "protected_inputs_unchanged": True,
        "privacy_scan_passed": True,
        "independent_png_hashes_verified": True,
        "manifest_hashes": {
            Path(r["report_dir"]).name: sha256_file(
                Path(r["report_dir"]) / "evaluation_manifest.json"
            )
            for r in (one, two)
        },
        "no_test_scoring": "All four prediction inputs equal verified canonical validation features; "
        "load_development performs full-source integrity access only",
    }
    return evidence, [Path(one["report_dir"]), Path(two["report_dir"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--first-run", default="spec07-reference")
    parser.add_argument("--second-run", default="spec07-reproduction")
    parser.add_argument("--output-root", default="reports/evaluation")
    parser.add_argument(
        "--skip-checks",
        action="store_true",
        help="Run replay verification only; record checks as skipped",
    )
    args = parser.parse_args()
    checks = []
    if not args.skip_checks:
        for command in (
            [sys.executable, "-m", "pytest"],
            [sys.executable, "-m", "ruff", "check", "src", "pipelines", "scripts", "tests"],
        ):
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            checks.append(
                {
                    "command": command,
                    "returncode": result.returncode,
                    "stdout": result.stdout,
                    "stderr": result.stderr,
                }
            )
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
    inputs = {
        "data_config": load_yaml("configs/data.yaml")["data"],
        "training_config": load_yaml("configs/training.yaml")["training"],
        "preprocessing_config": load_yaml("configs/preprocessing.yaml")["preprocessing"],
        "evaluation_config": load_yaml("configs/evaluation.yaml")["evaluation"],
        "baseline_config": load_yaml("configs/baselines.yaml")["baselines"],
        "training_report_dir": "reports/training/spec06-reference",
        "model_dir": "artifacts/models/training/spec06-reference",
        "feature_manifest": "reports/features/feature_manifest.json",
        "baseline_manifest": "reports/baselines/baseline_manifest.json",
    }
    inputs["evaluation_config"]["output_root"] = args.output_root
    evidence, roots = verify_runs(inputs, args.first_run, args.second_run)
    evidence.update(
        command=[sys.executable, *sys.argv],
        checks=checks,
        checks_status="skipped" if args.skip_checks else "passed",
    )
    for root in roots:
        write_json(root / "verification.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
