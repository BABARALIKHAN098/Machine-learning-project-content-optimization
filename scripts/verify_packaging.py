"""Offline research package verification: no real model fitting or test scoring."""
import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack, contextmanager
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from threadpoolctl import threadpool_limits

from machine_learning_project.data.development import load_development
from machine_learning_project.data.ingestion import sha256_file
from machine_learning_project.data.splitting import _hash_value
from machine_learning_project.features.registry import fingerprint
from machine_learning_project.inference.contracts import synthetic_request, validate_request
from machine_learning_project.inference.packaged_predictor import PackagedPredictor
from machine_learning_project.models.benchmark import prepare_benchmark_partitions, require_matching_identity
from machine_learning_project.models.package_artifacts import load_packaging_run, read_json
from machine_learning_project.models.training_artifacts import FAMILIES, write_json
from machine_learning_project.utils.config import load_yaml
from pipelines.packaging_pipeline import build_runtime_wheel, preflight, run_packaging


@contextmanager
def audit_calls():
    state = {"phase": "packaging", "expected": None, "counts": {}, "fit_calls": 0}
    def forbidden(*args, **kwargs):
        state["fit_calls"] += 1
        raise AssertionError("Packaging/inference attempted a fit")
    def spy(method, name):
        def invoke(self, x, *args, **kwargs):
            if "model" in self.named_steps:
                expected = state["expected"]
                if expected is None or not x.equals(expected.loc[x.index]):
                    raise AssertionError("Inference batch is outside the declared validation/synthetic population")
                counts = state["counts"].setdefault(state["phase"], {"predict": 0, "predict_proba": 0, "rows": 0})
                counts[name] += 1
                counts["rows"] += len(x)
            return method(self, x, *args, **kwargs)
        return invoke
    pending, classes = [BaseEstimator], set()
    while pending:
        cls = pending.pop()
        if cls not in classes:
            classes.add(cls)
            pending.extend(cls.__subclasses__())
    with ExitStack() as stack:
        for cls in classes - {LabelEncoder}:
            for method in ("fit", "fit_transform", "partial_fit"):
                if method in cls.__dict__:
                    stack.enter_context(patch.object(cls, method, forbidden))
        stack.enter_context(patch.object(Pipeline, "predict", spy(Pipeline.predict, "predict")))
        stack.enter_context(patch.object(Pipeline, "predict_proba", spy(Pipeline.predict_proba, "predict_proba")))
        yield state


ISOLATED_SCRIPT = r'''
import json, pathlib, sys
settings = json.loads(pathlib.Path(sys.argv[1]).read_text())
# -I -S disables site/.pth/editable hooks; only the installed project and existing dependency files are added.
sys.path[:0] = [settings['installed'], settings['dependencies']]
forbidden = [pathlib.Path(p).resolve() for p in settings['forbidden']]
def audit(event, args):
    if event == 'open' and isinstance(args[0], (str, bytes)):
        path = pathlib.Path(args[0]).resolve()
        if any(path.is_relative_to(p) for p in forbidden):
            raise AssertionError('Isolated runtime attempted upstream/checkout access')
sys.addaudithook(audit)
from machine_learning_project.inference.packaged_predictor import PackagedPredictor
from machine_learning_project.inference.contracts import synthetic_request
from machine_learning_project.features.registry import fingerprint
from sklearn.base import BaseEstimator
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from unittest.mock import patch
from contextlib import ExitStack
predictor = PackagedPredictor.load(settings['package'], purpose='research')
pending, classes = [BaseEstimator], set()
while pending:
    cls = pending.pop()
    if cls not in classes:
        classes.add(cls)
        pending.extend(cls.__subclasses__())
counts = {'fit': 0, 'predict': 0, 'predict_proba': 0}
def forbidden_fit(*a, **k):
    counts['fit'] += 1
    raise AssertionError('isolated fit attempted')
def spy(original, name):
    def call(self, x, *a, **k):
        if 'model' in self.named_steps:
            counts[name] += 1
        return original(self, x, *a, **k)
    return call
with ExitStack() as stack:
    for cls in classes - {LabelEncoder}:
        for name in ('fit','fit_transform','partial_fit'):
            if name in cls.__dict__:
                stack.enter_context(patch.object(cls, name, forbidden_fit))
    stack.enter_context(patch.object(Pipeline, 'predict', spy(Pipeline.predict, 'predict')))
    stack.enter_context(patch.object(Pipeline, 'predict_proba', spy(Pipeline.predict_proba, 'predict_proba')))
    result = predictor.predict(synthetic_request(predictor.schema, 17), include_probabilities=True, chunk_rows=7)
origins = [m.__file__ for name,m in sys.modules.items() if name.startswith('machine_learning_project') and getattr(m,'__file__',None)]
assert origins and all(pathlib.Path(p).resolve().is_relative_to(pathlib.Path(settings['installed']).resolve()) for p in origins)
print(json.dumps({'prediction_sha256': fingerprint(result), 'counts': counts, 'installed_project_only': True}))
'''


def isolated_check(package, expected, workspace):
    """Install only the local project wheel; reuse local dependency files without editable hooks."""
    with tempfile.TemporaryDirectory(prefix="isolated-runtime-", dir=workspace) as directory:
        directory = Path(directory)
        clone = directory / "package"
        shutil.copytree(package.root, clone)
        installed = directory / "installed"
        command = [sys.executable, "-m", "pip", "install", "--no-index", "--no-deps", "--no-compile",
                   "--target", str(installed), str(clone / package.manifest["wheel"])]
        install = subprocess.run(command, capture_output=True, text=True, check=False)
        if install.returncode:
            raise AssertionError("Offline isolated project-wheel install failed")
        root = Path(__file__).resolve().parents[1]
        settings = {"installed": str(installed), "package": str(clone),
                    "dependencies": str(Path(np.__file__).resolve().parent.parent),
                    "forbidden": [str(root / p) for p in ("src", "pipelines", "data", "reports", "artifacts/models")]}
        write_json(directory / "settings.json", settings)
        (directory / "check.py").write_text(ISOLATED_SCRIPT, encoding="utf-8")
        invocation = [sys.executable, "-I", "-S", str(directory / "check.py"), str(directory / "settings.json")]
        result = subprocess.run(invocation, cwd=directory, capture_output=True, text=True, check=False)
        if result.returncode:
            raise AssertionError("Isolated runtime failed: " + result.stderr)
        evidence = json.loads(result.stdout)
        if evidence["prediction_sha256"] != fingerprint(expected) or evidence["counts"] != {"fit": 0, "predict": 3, "predict_proba": 3}:
            raise AssertionError("Isolated runtime prediction parity mismatch")
        return {"status": "passed", "install_command": command, "run_command": invocation,
                "dependency_policy": "Existing local dependency files reused; site/.pth/editable hooks disabled",
                "upstream_access_blocked": True, **evidence}


def verify_packages(inputs, training_config, prefix, workspace):
    workspace = Path(workspace).resolve()
    workspace.mkdir(parents=True, exist_ok=True)
    initial = preflight(**inputs, run_id=f"{prefix}-reference")
    wheel = build_runtime_wheel(workspace / f"{prefix}-wheel")
    evidence = {"purpose": "research", "production_ready": False, "parity": {}, "workload": {}, "isolated": {}}
    with audit_calls() as audit:
        dry = run_packaging(**inputs, run_id=f"{prefix}-dry", dry_run=True)
        one = run_packaging(**inputs, run_id=f"{prefix}-reference", runtime_wheel=wheel)
        two = run_packaging(**inputs, run_id=f"{prefix}-reproduction", runtime_wheel=wheel)
        if audit["counts"] or dry["status"] != "preflight_passed":
            raise AssertionError("Packaging/dry run performed inference")
        for f in inputs["packaging_config"]["families"]:
            if one["packages"][f]["semantic_sha256"] != two["packages"][f]["semantic_sha256"]:
                raise AssertionError("Two builds differ semantically")
        data = initial["predictors"][FAMILIES[0]].data_config
        train, validation, provenance = load_development(data, training_config)
        train, validation, identity = prepare_benchmark_partitions(train, validation, data, training_config, provenance)
        del train
        require_matching_identity(identity, initial["evaluated"]["evaluation_identity"])
        identifiers = set(map(str, validation[data["id_columns"]].to_numpy().ravel()))
        for family in inputs["packaging_config"]["families"]:
            load_start = time.perf_counter()
            package = PackagedPredictor.load(one["packages"][family]["path"], purpose="research")
            load_seconds = time.perf_counter() - load_start
            request = validate_request(validation[["content_id", *package.schema["features"]]], package.schema, package.config)
            audit.update(phase="validation", expected=request[package.schema["features"]])
            with threadpool_limits(limits=1):
                source = initial["predictors"][family].pipeline
                labels = source.predict(audit["expected"])
                probabilities = source.predict_proba(audit["expected"])
            digest = _hash_value(fingerprint(list(labels)), f"content-trend-tuning-v1:{provenance['source_sha256']}")
            if digest != initial["scores"][family]["prediction_sha256"]:
                raise AssertionError("Source prediction drift from frozen SPEC-06")
            for run in (one, two):
                loaded = PackagedPredictor.load(run["packages"][family]["path"], purpose="research")
                label_only = loaded.predict(request, chunk_rows=len(request))
                with_probabilities = loaded.predict(request, include_probabilities=True)
                if [r["predicted_trend"] for r in label_only["records"]] != list(labels) or any(r["probabilities"] is not None for r in label_only["records"]):
                    raise AssertionError("Label-only package parity/capability mismatch")
                if [r["predicted_trend"] for r in with_probabilities["records"]] != list(labels):
                    raise AssertionError("Chunked label mismatch")
                observed = np.asarray([[r["probabilities"][str(c)] for c in source.classes_] for r in with_probabilities["records"]])
                if not np.allclose(observed, probabilities, atol=1e-9, rtol=0):
                    raise AssertionError("Chunked source/package probability mismatch")
            evidence["parity"][family] = {"row_count": len(request), "partition": "validation",
                "prediction_sha256": digest, "exact_labels": True, "probabilities_match_atol": 1e-9,
                "model_bytes_unchanged": sha256_file(package.root / "model.joblib") == initial["references"]["model_hashes"][f"{family}.joblib"]}
            frame = synthetic_request(package.schema, 30000)
            validated_start = time.perf_counter()
            validated = validate_request(frame, package.schema, package.config)
            validation_seconds = time.perf_counter() - validated_start
            audit.update(phase="synthetic_30000", expected=validated[package.schema["features"]])
            modes = {}
            for proba in (False, True):
                start = time.perf_counter()
                response = package.predict(frame, include_probabilities=proba)
                predict_seconds = time.perf_counter() - start
                start = time.perf_counter()
                serialized = json.dumps(response, allow_nan=False)
                serialization_seconds = time.perf_counter() - start
                modes[str(proba)] = {"prediction_including_validation_seconds": predict_seconds,
                    "serialization_seconds": serialization_seconds, "serialized_bytes": len(serialized.encode()),
                    "rows": response["row_count"]}
            evidence["workload"][family] = {"source": "invented_synthetic_rows_only", "row_count": 30000,
                "cold_load_seconds": load_seconds, "validation_seconds": validation_seconds, "modes": modes,
                "chunk_rows": package.config["chunk_rows"], "sla_certified": False}
            tiny = synthetic_request(package.schema, 17)
            audit.update(phase="isolated_smoke_reference", expected=validate_request(tiny, package.schema, package.config)[package.schema["features"]])
            expected = package.predict(tiny, include_probabilities=True, chunk_rows=7)
            evidence["isolated"][family] = isolated_check(package, expected, workspace)
        evidence["call_counts"] = audit["counts"]
        evidence["fit_calls"] = audit["fit_calls"]
        if audit["fit_calls"] != 0:
            raise AssertionError("Unexpected fit calls")
    for path, digest in initial["protected"].items():
        if sha256_file(path) != digest:
            raise AssertionError("Protected input changed during verification")
    for run in (one, two):
        load_packaging_run(run["report_dir"])
        roots = [Path(run["report_dir"]), *[Path(v["path"]) for v in run["packages"].values()]]
        for root in roots:
            for path in root.rglob("*"):
                if path.suffix in (".json", ".md", ".txt"):
                    text = path.read_text(encoding="utf-8")
                    if any(value in text for value in identifiers):
                        raise AssertionError("Private source identifier exposed")
    import platform
    import psutil
    memory = psutil.Process().memory_info()
    evidence.update(status="passed", semantic_reproducibility=True, protected_inputs_unchanged=True,
        privacy_scan_passed=True, no_test_scoring=True, full_source_access="integrity_only",
        hardware={"platform": platform.platform(), "machine": platform.machine(), "processor": platform.processor(),
                  "logical_cpus": psutil.cpu_count(), "verification_process_rss_bytes": memory.rss,
                  "verification_process_peak_working_set_bytes": getattr(memory, "peak_wset", None)},
        manifest_hashes={Path(r["report_dir"]).name: sha256_file(Path(r["report_dir"]) / "packaging_manifest.json") for r in (one, two)})
    return evidence, [Path(one["report_dir"]), Path(two["report_dir"])]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-prefix", default="spec08-verification")
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()
    checks = []
    if not args.skip_checks:
        for command in ([sys.executable, "-m", "pytest"],
                        [sys.executable, "-m", "ruff", "check", "src", "pipelines", "scripts", "tests"]):
            result = subprocess.run(command, capture_output=True, text=True, check=False)
            checks.append({"command": command, "returncode": result.returncode, "stdout": result.stdout, "stderr": result.stderr})
            if result.returncode:
                raise RuntimeError(result.stdout + result.stderr)
    inputs = {"packaging_config": load_yaml("configs/packaging.yaml")["packaging"],
              "inference_config": load_yaml("configs/inference.yaml")["inference"],
              "evaluation_dir": "reports/evaluation/spec07-reference",
              "training_report_dir": "reports/training/spec06-reference",
              "model_dir": "artifacts/models/training/spec06-reference", "purpose": "research"}
    evidence, roots = verify_packages(inputs, load_yaml("configs/training.yaml")["training"], args.run_prefix, ".packaging-work")
    evidence.update(checks=checks, checks_status="skipped" if args.skip_checks else "passed",
                    command=[sys.executable, *sys.argv])
    for root in roots:
        write_json(root / "verification.json", evidence)
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
