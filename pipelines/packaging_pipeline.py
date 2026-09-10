"""Copy-only research packaging of the verified SPEC-07 handoff."""

import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from machine_learning_project.data.ingestion import sha256_file
from machine_learning_project.features.registry import fingerprint
from machine_learning_project.inference.contracts import OUTPUT_SCHEMA, input_schema
from machine_learning_project.models.benchmark import require_matching_identity
from machine_learning_project.models.evaluation_artifacts import load_evaluation
from machine_learning_project.models.evaluation_decision import FLAGS
from machine_learning_project.models.package_artifacts import (
    FILES,
    load_package_manifest,
    local_path,
    read_json,
    runtime_environment,
    safe_name,
    safe_path,
    semantic_digest,
    validate_model,
    verify_wheel,
)
from machine_learning_project.models.training_artifacts import (
    MODEL_FILES,
    REPORT_FILES,
    code_environment,
    load_training_run,
    write_json,
)
from machine_learning_project.utils.config import (
    validate_inference_config,
    validate_packaging_config,
)
from machine_learning_project.utils.exceptions import DataValidationError


def preflight(
    packaging_config,
    inference_config,
    evaluation_dir,
    training_report_dir,
    model_dir,
    run_id,
    purpose,
):
    validate_packaging_config(packaging_config)
    validate_inference_config(inference_config)
    if purpose != "research":
        raise DataValidationError("Only explicit research packaging is supported")
    safe_name(run_id)
    evaluation, reports, models = map(local_path, (evaluation_dir, training_report_dir, model_dir))
    evaluated = load_evaluation(evaluation)
    decision = read_json(evaluation / "decision.json")
    training = read_json(reports / "training_manifest.json")
    references = decision["frozen_references"]
    require_matching_identity(evaluated["evaluation_identity"], training.get("evaluation_identity"))
    if (
        references["training_manifest_sha256"] != sha256_file(reports / "training_manifest.json")
        or training.get("model_hashes") != references["model_hashes"]
        or training.get("report_hashes") != references["report_hashes"]
        or references["selection_sha256"] != sha256_file(reports / "selection.json")
    ):
        raise DataValidationError("Evaluation/training handoff mismatch")
    protected = {local_path(p): digest for p, digest in evaluated["protected_input_hashes"].items()}
    protected[evaluation / "evaluation_manifest.json"] = sha256_file(
        evaluation / "evaluation_manifest.json"
    )
    for name, digest in evaluated["payload_hashes"].items():
        protected[safe_path(evaluation, name)] = digest
    for directory, hashes, required in (
        (reports, training["report_hashes"], REPORT_FILES),
        (models, training["model_hashes"], MODEL_FILES),
    ):
        if set(hashes) != required:
            raise DataValidationError("Incomplete frozen training inventory")
        protected.update({safe_path(directory, name): digest for name, digest in hashes.items()})
    for p, digest in protected.items():
        if not p.is_file() or sha256_file(p) != digest:
            raise DataValidationError("Protected upstream fingerprint mismatch")
    roots = [local_path(packaging_config[k]) for k in ("output_root", "package_root")]
    if roots[0].is_relative_to(roots[1]) or roots[1].is_relative_to(roots[0]):
        raise DataValidationError("Packaging output roots must be separate")
    output = safe_path(roots[0], run_id)
    destinations = {
        f: safe_path(roots[1], safe_name(f"{run_id}-{f}")) for f in packaging_config["families"]
    }
    for dest in [output, *destinations.values()]:
        if dest.exists():
            raise DataValidationError("Packaging destination already exists")
        if any(
            dest.is_relative_to(d) or d.is_relative_to(dest) for d in (evaluation, reports, models)
        ):
            raise DataValidationError("Packaging destination overlaps frozen artifacts")
        if any(dest == p or p.is_relative_to(dest) for p in protected):
            raise DataValidationError("Packaging destination overlaps protected input")
        repository = Path(__file__).resolve().parents[1]
        protected_directories = [
            repository / p
            for p in (
                "data",
                "reports/features",
                "reports/baselines",
                "reports/evaluation",
                "reports/training",
                "artifacts/models",
            )
        ]
        if any(dest.is_relative_to(p.resolve()) for p in protected_directories):
            raise DataValidationError("Packaging destination is inside a protected input directory")
    resolved = read_json(reports / "resolved_config.json")
    selection = read_json(reports / "selection.json")
    scores = read_json(evaluation / "evaluation_metrics.json")["finalists"]
    if (
        fingerprint(resolved) != training["resolved_config_sha256"]
        or fingerprint(selection) != training["selection_sha256"]
        or decision["development_reference"] != selection["preferred_for_development"]["family"]
        or set(decision["metric_eligible_models"])
        != {f for f, s in scores.items() if all(s["comparison"][k] for k in FLAGS)}
    ):
        raise DataValidationError("Frozen configuration/selection/eligibility mismatch")
    environment = runtime_environment()
    for old, current in (
        ("numpy", "numpy"),
        ("pandas", "pandas"),
        ("sklearn", "scikit-learn"),
        ("threadpoolctl", "threadpoolctl"),
    ):
        if training["environment"][old] != environment["dependencies"][current]:
            raise DataValidationError("Unsupported source training environment")
    if training["environment"]["python"] != environment["python"]:
        raise DataValidationError("Unsupported source Python version")
    _, predictors = load_training_run(reports, models)
    schemas = {}
    for family, predictor in predictors.items():
        metadata = predictor.metadata
        schemas[family] = input_schema(predictor.data_config)
        validate_model(predictor, schemas[family], metadata)
        if (
            metadata["selection_sha256"] != fingerprint(selection)
            or metadata["feature_manifest_sha256"] != references["feature_manifest_sha256"]
            or metadata["baseline_manifest_sha256"] != references["baseline_manifest_sha256"]
            or fingerprint(predictor.data_config) != resolved["original_config_hashes"]["data"]
            or any(
                metadata["effective_model_parameters"][key.removeprefix("model__")] != value
                for key, value in selection["finalists"][family]["parameters"].items()
            )
        ):
            raise DataValidationError("Source model semantic reference mismatch")
    return locals()


def build_runtime_wheel(destination):
    """Use the declared setuptools backend offline in an isolated build checkout."""
    destination = local_path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory(prefix="wheel-build-", dir=destination.parent) as staging:
        staging = Path(staging)
        shutil.copytree(
            root / "src",
            staging / "src",
            ignore=shutil.ignore_patterns("__pycache__", "*.egg-info"),
        )
        shutil.copyfile(root / "pyproject.toml", staging / "pyproject.toml")
        shutil.copyfile(root / "readme.md", staging / "README.md")
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pip",
                "wheel",
                "--no-deps",
                "--no-index",
                "--no-build-isolation",
                "--wheel-dir",
                str(destination),
                str(staging),
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode:
            raise DataValidationError(
                "Offline wheel build failed; install the declared local build dependencies"
            )
    wheels = list(destination.glob("*.whl"))
    if len(wheels) != 1:
        raise DataValidationError("Expected exactly one project runtime wheel")
    verify_wheel(wheels[0], runtime_environment())
    return wheels[0]


def run_packaging(
    packaging_config,
    inference_config,
    evaluation_dir,
    training_report_dir,
    model_dir,
    run_id,
    purpose,
    dry_run=False,
    runtime_wheel=None,
):
    state = preflight(
        packaging_config,
        inference_config,
        evaluation_dir,
        training_report_dir,
        model_dir,
        run_id,
        purpose,
    )
    if dry_run:
        return {
            "status": "preflight_passed",
            "purpose": purpose,
            "planned_fits": 0,
            "planned_predictions": 0,
            "packages": {f: str(p) for f, p in state["destinations"].items()},
        }
    output = state["output"]
    output.mkdir(parents=True, exist_ok=False)
    wheel = (
        local_path(runtime_wheel)
        if runtime_wheel
        else build_runtime_wheel(output / "build_runtime")
    )
    verify_wheel(wheel, state["environment"])
    wheel_hash = sha256_file(wheel)
    packages = {}
    code, _ = code_environment()
    for family, dest in state["destinations"].items():
        dest.mkdir(parents=True, exist_ok=False)
        (dest / "runtime").mkdir()
        try:
            shutil.copyfile(state["models"] / f"{family}.joblib", dest / "model.joblib")
            shutil.copyfile(state["models"] / f"{family}.json", dest / "model_metadata.json")
            shutil.copyfile(state["evaluation"] / "decision.json", dest / "decision.json")
            shutil.copyfile(wheel, dest / "runtime" / wheel.name)
            write_json(dest / "input_schema.json", state["schemas"][family])
            write_json(dest / "output_schema.json", OUTPUT_SCHEMA)
            write_json(dest / "inference_config.json", inference_config)
            write_json(
                dest / "environment.json",
                {
                    "runtime": state["environment"],
                    "training_recorded": state["training"]["environment"],
                    "compatibility_policy": "exact_python_os_arch_dependency_versions_and_project_source",
                    "training_unrecorded": ["scipy", "joblib", "project_distribution_version"],
                },
            )
            (dest / "runtime/requirements.txt").write_text(
                "\n".join(
                    f"{k}=={v}" for k, v in sorted(state["environment"]["dependencies"].items())
                )
                + "\n",
                encoding="utf-8",
            )
            result = state["scores"][family]
            card = [
                f"# {family}: research package",
                "",
                "Production ready: false.",
                f"Model version: {state['predictors'][family].metadata['model_version']}",
                f"Recommended model: {state['decision']['recommended_model']}",
                f"Development reference: {state['decision']['development_reference']}",
                f"Validation macro F1: {result['metrics']['macro_f1']}",
                "",
                *state["decision"]["limitations"],
                "",
                "Probabilities are optional, uncalibrated model scores, not causal refresh benefit.",
                "Human review required; do not automate content changes.",
                "Install runtime wheel with its exact pinned dependencies. Local trusted packages only.",
                "Hash verification is not authenticity when model and manifest are both attacker-controlled.",
                "SPEC-09 must select a concrete research package and enforce this input/output contract.",
            ]
            (dest / "MODEL_CARD.md").write_text("\n".join(card) + "\n", encoding="utf-8")
            manifest = {
                "status": "complete",
                "packaging_contract_version": "1.0",
                "package_schema_version": "1.0",
                "inference_contract_version": "2.0",
                "package_id": dest.name,
                "family": family,
                "model_version": state["predictors"][family].metadata["model_version"],
                "purpose": "research",
                "research_only": True,
                "production_ready": False,
                "recommended_model": state["decision"]["recommended_model"],
                "development_reference": state["decision"]["development_reference"],
                "metric_eligible": family in state["decision"]["metric_eligible_models"],
                "comparison_flags": result["comparison"],
                "wheel": f"runtime/{wheel.name}",
                "references": {
                    "evaluation_manifest_sha256": sha256_file(
                        state["evaluation"] / "evaluation_manifest.json"
                    ),
                    "decision_sha256": sha256_file(state["evaluation"] / "decision.json"),
                    **state["references"],
                },
                "payload_hashes": {
                    name: sha256_file(dest / name)
                    for name in sorted(FILES | {f"runtime/{wheel.name}"})
                },
            }
            manifest["semantic_sha256"] = semantic_digest(manifest)
            if (
                sha256_file(dest / "model.joblib")
                != state["training"]["model_hashes"][f"{family}.joblib"]
            ):
                raise DataValidationError("Copied model bytes changed")
            if runtime_environment() != state["environment"] or sha256_file(wheel) != wheel_hash:
                raise DataValidationError("Runtime changed during packaging")
            for path, digest in state["protected"].items():
                if sha256_file(path) != digest:
                    raise DataValidationError("Protected input changed during packaging")
            write_json(dest / "package_manifest.json", manifest)
            load_package_manifest(dest)
            packages[family] = {
                "path": str(dest),
                "manifest_sha256": sha256_file(dest / "package_manifest.json"),
                "semantic_sha256": manifest["semantic_sha256"],
            }
        except BaseException:
            (dest / "package_manifest.json").unlink(missing_ok=True)
            raise
    write_json(
        output / "resolved_config.json",
        {
            "packaging": packaging_config,
            "inference": inference_config,
            "purpose": purpose,
            "code": code,
        },
    )
    write_json(output / "package_inventory.json", packages)
    (output / "packaging_report.md").write_text(
        "# SPEC-08 research packaging\n\n"
        "Frozen finalist bytes copied without fitting or scoring. Production ready: false.\n"
        f"Recommended model: {state['decision']['recommended_model']}.\n\n"
        + "\n".join(state["decision"]["limitations"])
        + "\n",
        encoding="utf-8",
    )
    write_json(
        output / "packaging_manifest.json",
        {
            "status": "complete",
            "packaging_contract_version": "1.0",
            "run_id": run_id,
            "purpose": purpose,
            "packages": packages,
            "protected_input_hashes": {str(p): h for p, h in state["protected"].items()},
            "runtime_wheel_sha256": wheel_hash,
            "code": code,
            "report_hashes": {
                name: sha256_file(output / name)
                for name in (
                    "resolved_config.json",
                    "package_inventory.json",
                    "packaging_report.md",
                )
            },
        },
    )
    return {"status": "complete", "report_dir": str(output), "packages": packages}
