"""Own real loopback server processes; verify only invented populations and publish aggregates."""

import argparse
import importlib.metadata
import io
import json
import logging
import multiprocessing
import platform
import secrets
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
import numpy as np
import psutil
import uvicorn

from app.config import Settings
from app.main import create_app
from app.schemas import request_frame, synthetic_document
from machine_learning_project.data.ingestion import sha256_file
from machine_learning_project.features.registry import fingerprint
from machine_learning_project.inference.packaged_predictor import PackagedPredictor
from machine_learning_project.models.package_artifacts import load_packaging_run, safe_name
from machine_learning_project.models.training_artifacts import write_json
from scripts.verify_packaging import audit_calls


def serve(settings, stop, results):
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    logger = logging.getLogger("research_api")
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)
    try:
        with audit_calls() as audit:

            def loader(*args, **kwargs):
                predictor = PackagedPredictor.load(*args, **kwargs)
                frame, _ = request_frame(
                    synthetic_document(predictor.schema, 30000), predictor, settings
                )
                audit.update(phase="http_synthetic", expected=frame[predictor.schema["features"]])
                return predictor

            server = uvicorn.Server(
                uvicorn.Config(
                    create_app(settings, loader),
                    host=settings.host,
                    port=settings.port,
                    access_log=False,
                    proxy_headers=False,
                    server_header=False,
                    ws="none",
                    log_level="critical",
                    log_config=None,
                )
            )

            def shutdown():
                stop.wait()
                server.should_exit = True

            threading.Thread(target=shutdown, daemon=True).start()
            server.run()
            memory = psutil.Process().memory_info()
            results.put(
                {
                    "status": "passed" if server.started else "failed",
                    "counts": audit["counts"],
                    "fit_calls": audit["fit_calls"],
                    "logs": stream.getvalue(),
                    "rss_bytes": memory.rss,
                    "peak_working_set_bytes": getattr(memory, "peak_wset", None),
                }
            )
    except Exception:  # noqa: BLE001 -- no raw subprocess exceptions in shareable reports
        results.put({"status": "failed", "code": "server_verification_failed"})
    finally:
        logger.removeHandler(handler)


def compare(response, expected, labels):
    assert set(response) == set(expected)
    assert {k: v for k, v in response.items() if k != "records"} == {
        k: v for k, v in expected.items() if k != "records"
    }
    for observed, original in zip(response["records"], expected["records"], strict=True):
        assert {
            k: v for k, v in observed.items() if k not in {"probabilities", "probability_down"}
        } == {k: v for k, v in original.items() if k not in {"probabilities", "probability_down"}}
        if original["probabilities"] is None:
            assert observed["probabilities"] is None and observed["probability_down"] is None
        else:
            assert list(observed["probabilities"]) == labels
            assert np.allclose(
                [observed["probabilities"][k] for k in labels],
                [original["probabilities"][k] for k in labels],
                atol=1e-9,
                rtol=0,
            )
            assert observed["probability_down"] == observed["probabilities"]["down"]


def verify_family(reference):
    token = secrets.token_urlsafe(36)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    settings = Settings(
        package_dir=reference["path"],
        expected_manifest_sha256=reference["manifest_sha256"],
        token=token,
        port=port,
    )
    predictor = PackagedPredictor.load(
        settings.package_dir,
        purpose="research",
        expected_manifest_sha256=settings.expected_manifest_sha256,
    )
    context = multiprocessing.get_context("spawn")
    stop, results = context.Event(), context.Queue()
    process = context.Process(target=serve, args=(settings, stop, results))
    started = time.perf_counter()
    process.start()
    observations = []
    try:
        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}", trust_env=False, timeout=120
        ) as client:
            deadline = time.monotonic() + 60
            while True:
                try:
                    if client.get("/ready").status_code == 200:
                        break
                except httpx.TransportError:
                    pass
                if not process.is_alive() or time.monotonic() > deadline:
                    raise AssertionError("Owned API server did not become ready")
                time.sleep(0.1)
            startup_seconds = time.perf_counter() - started
            assert client.get("/health").json() == {"status": "healthy"}
            assert client.get("/docs").status_code == 200
            assert client.get("/v1/model").status_code == 401
            headers = {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}
            assert client.get("/v1/model", headers=headers).json()["recommended_model"] is None
            assert client.get("/v1/schema", headers=headers).status_code == 200
            with audit_calls() as direct_audit:
                for count in (17, 30000):
                    document = synthetic_document(predictor.schema, count)
                    frame, _ = request_frame(document, predictor, settings)
                    direct_audit.update(
                        phase="direct_synthetic", expected=frame[predictor.schema["features"]]
                    )
                    for mode in (False, True):
                        document["include_probabilities"] = mode
                        body = json.dumps(document, allow_nan=False).encode()
                        assert len(body) <= settings.maximum_request_bytes
                        expected = predictor.predict(
                            frame, include_probabilities=mode, chunk_rows=count
                        )
                        start = time.perf_counter()
                        response = client.post("/v1/predictions", headers=headers, content=body)
                        elapsed = time.perf_counter() - start
                        assert response.status_code == 200, "HTTP prediction failed"
                        assert response.headers["cache-control"] == "no-store"
                        compare(response.json(), expected, predictor.schema["labels"])
                        observations.append(
                            {
                                "rows": count,
                                "include_probabilities": mode,
                                "request_bytes": len(body),
                                "response_bytes": len(response.content),
                                "http_seconds": elapsed,
                                "response_sha256": fingerprint(response.json()),
                                "exact_labels_ids_order": True,
                                "probability_atol": 1e-9,
                            }
                        )
            invalid = synthetic_document(predictor.schema, 17)
            invalid["records"][-1]["content_id"] = invalid["records"][0]["content_id"]
            assert client.post("/v1/predictions", headers=headers, json=invalid).status_code == 422
            assert client.get("/ready").status_code == 200
    finally:
        stop.set()
        try:
            child = results.get(timeout=60)
        finally:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
                process.join(timeout=10)
            results.close()
    assert process.exitcode == 0 and child["status"] == "passed"
    assert child["fit_calls"] == 0 and direct_audit["fit_calls"] == 0
    # 17 rows = 1 chunk, 30k = 6 chunks, each in two modes.
    assert child["counts"]["http_synthetic"] == {"predict": 14, "predict_proba": 7, "rows": 90051}
    logs = child.pop("logs")
    assert token not in logs and "synthetic-" not in logs and "__SYNTHETIC_UNKNOWN__" not in logs
    events = [json.loads(line) for line in logs.splitlines() if line.strip()]
    allowed = {
        "request_id",
        "route",
        "status",
        "duration_seconds",
        "row_count",
        "include_probabilities",
        "validation_seconds",
        "prediction_seconds",
        "serialization_seconds",
        "error_code",
    }
    assert all(set(event) <= allowed for event in events)
    return {
        "package_manifest_sha256": reference["manifest_sha256"],
        "startup_seconds": startup_seconds,
        "observations": observations,
        "server": child,
        "aggregate_events": events,
        "direct_call_counts": direct_audit["counts"],
        "privacy_scan_passed": True,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--skip-checks", action="store_true")
    args = parser.parse_args()
    output = Path("reports/application") / safe_name(args.run_id)
    if output.exists():
        raise ValueError("Application verification destination already exists")
    upstream = load_packaging_run("reports/packaging/spec08-final-reference")
    protected = {Path(p): h for p, h in upstream["protected_input_hashes"].items()}
    for root in (Path("artifacts/packages"), Path("reports/packaging")):
        protected.update({p: sha256_file(p) for p in root.rglob("*") if p.is_file()})
    checks = []
    if not args.skip_checks:
        for command in (
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/unit/test_api_config.py",
                "tests/unit/test_api_validation.py",
                "tests/contract/test_api_contract.py",
                "tests/integration/test_api_workflow.py",
            ],
            [sys.executable, "-m", "pytest"],
            [sys.executable, "-m", "ruff", "check", "src", "pipelines", "scripts", "tests", "app"],
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
    families = {
        family: verify_family(reference) for family, reference in upstream["packages"].items()
    }
    assert all(sha256_file(path) == digest for path, digest in protected.items())
    code = {
        str(p): sha256_file(p)
        for root in (Path("app"),)
        for p in root.rglob("*")
        if p.is_file() and "__pycache__" not in str(p)
    }
    code.update(
        {
            str(p): sha256_file(p)
            for p in [Path("configs/api.yaml"), Path(__file__), Path("pyproject.toml")]
        }
    )
    evidence = {
        "status": "passed",
        "purpose": "research",
        "production_ready": False,
        "recommended_model": None,
        "api_contract_version": "1.0",
        "inference_contract_version": "2.0",
        "package_schema_version": "1.0",
        "command": [sys.executable, *sys.argv],
        "checks": checks,
        "checks_status": "skipped" if args.skip_checks else "passed",
        "families": families,
        "protected_inputs_unchanged": True,
        "protected_input_hashes": {str(p): h for p, h in protected.items()},
        "code_hashes": code,
        "synthetic_only": True,
        "no_test_scoring": True,
        "fit_calls": 0,
        "platform": platform.platform(),
        "python": platform.python_version(),
        "logical_cpus": psutil.cpu_count(),
        "sla_certified": False,
        "api_dependencies": {
            k: importlib.metadata.version(k)
            for k in ("fastapi", "uvicorn", "pydantic", "starlette", "httpx", "anyio")
        },
    }
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "verification.json", evidence)
    (output / "verification_report.md").write_text(
        "# SPEC-09 local HTTP verification\n\nBoth frozen packages matched direct Python inference on invented requests. "
        "30,000-row label and probability workloads completed. Zero fitting or test scoring. "
        "Protected inputs unchanged. Research only; no production recommendation or SLA certification.\n\n"
        "See verification.json for actual checks, package pins, hardware, timings and aggregate counters.\n",
        encoding="utf-8",
    )
    write_json(
        output / "application_manifest.json",
        {
            "status": "complete",
            "run_id": args.run_id,
            "purpose": "research",
            "api_contract_version": "1.0",
            "payload_hashes": {
                name: sha256_file(output / name)
                for name in ("verification.json", "verification_report.md")
            },
        },
    )
    print(
        json.dumps(
            {
                "status": "passed",
                "report_dir": str(output),
                "families": list(families),
                "fit_calls": 0,
            }
        )
    )


if __name__ == "__main__":
    main()
