"""Trusted local research packages. Check every payload before deserialization."""

import importlib.metadata
import json
import os
import platform
import re
import tempfile
import zipfile
from email.parser import Parser
from pathlib import Path, PurePosixPath

from ..data.ingestion import sha256_file
from ..features.registry import fingerprint
from ..inference.contracts import OUTPUT_SCHEMA, input_schema, validate_schema
from ..utils.config import validate_inference_config
from ..utils.exceptions import DataValidationError
from .evaluation_decision import FLAGS

DEPENDENCIES = (
    "numpy",
    "pandas",
    "scipy",
    "scikit-learn",
    "joblib",
    "threadpoolctl",
    "PyYAML",
    "matplotlib",
    "python-dateutil",
    "six",
    "tzdata",
    "packaging",
    "contourpy",
    "cycler",
    "fonttools",
    "kiwisolver",
    "pillow",
    "pyparsing",
)
FILES = {
    "model.joblib",
    "model_metadata.json",
    "input_schema.json",
    "output_schema.json",
    "inference_config.json",
    "decision.json",
    "environment.json",
    "MODEL_CARD.md",
    "runtime/requirements.txt",
}


def read_json(path):
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise DataValidationError("Duplicate JSON key")
            result[key] = value
        return result

    try:
        return json.loads(
            Path(path).read_text(encoding="utf-8"),
            object_pairs_hook=unique,
            parse_constant=lambda v: (_ for _ in ()).throw(ValueError()),
        )
    except (OSError, ValueError, TypeError):
        raise DataValidationError("Cannot read a valid package JSON document") from None


def safe_name(value):
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,119}", value)
        or value.upper()
        in {
            "CON",
            "AUX",
            "PRN",
            "NUL",
            *[f"COM{i}" for i in range(10)],
            *[f"LPT{i}" for i in range(10)],
        }
    ):
        raise DataValidationError("Unsafe package/run name")
    return value


def local_path(value):
    if (
        not isinstance(value, (str, Path))
        or "://" in str(value)
        or str(value).startswith(("//", "\\\\"))
    ):
        raise DataValidationError("Only local paths are supported")
    return Path(value).resolve()


def safe_path(root, name):
    root = local_path(root)
    if (
        not isinstance(name, str)
        or not name
        or ":" in name
        or "\\" in name
        or PurePosixPath(name).is_absolute()
        or any(p in (".", "..", "") for p in name.split("/"))
    ):
        raise DataValidationError("Unsafe package payload name")
    path = root / name
    if not path.resolve().is_relative_to(root):
        raise DataValidationError("Package payload escapes its directory")
    return path


def source_digest():
    root = Path(__file__).resolve().parents[1]
    return fingerprint(
        {p.relative_to(root).as_posix(): sha256_file(p) for p in sorted(root.rglob("*.py"))}
    )


def runtime_environment():
    return {
        "python": platform.python_version(),
        "system": platform.system(),
        "machine": platform.machine(),
        "implementation": platform.python_implementation(),
        "dependencies": {k: importlib.metadata.version(k) for k in DEPENDENCIES},
        "project_version": importlib.metadata.version("machine-learning-project"),
        "project_source_sha256": source_digest(),
    }


def verify_wheel(path, environment):
    expected = f"machine_learning_project-{environment['project_version']}-py3-none-any.whl"
    if Path(path).name != expected:
        raise DataValidationError("Unexpected project wheel filename")
    try:
        with zipfile.ZipFile(path) as archive:
            entries = archive.namelist()
            if len(entries) != len(set(entries)) or any(
                ".." in PurePosixPath(n).parts for n in entries
            ):
                raise DataValidationError("Invalid project wheel inventory")
            import hashlib

            hashes = {
                n.removeprefix("machine_learning_project/"): hashlib.sha256(
                    archive.read(n)
                ).hexdigest()
                for n in entries
                if n.startswith("machine_learning_project/") and n.endswith(".py")
            }
            if fingerprint(hashes) != environment["project_source_sha256"]:
                raise DataValidationError("Project wheel source does not match supported runtime")
            metadata = Parser().parsestr(
                archive.read(
                    f"machine_learning_project-{environment['project_version']}.dist-info/METADATA"
                ).decode()
            )
            if metadata.get_all("Version") != [environment["project_version"]] or metadata.get_all(
                "Name"
            ) != ["machine-learning-project"]:
                raise DataValidationError("Project wheel distribution mismatch")
    except (OSError, zipfile.BadZipFile, KeyError, UnicodeError):
        raise DataValidationError("Invalid project wheel") from None


def semantic_digest(manifest):
    return fingerprint(
        {k: v for k, v in manifest.items() if k not in ("package_id", "semantic_sha256")}
    )


def validate_model(predictor, schema, metadata):
    pipeline = predictor.pipeline
    if (
        predictor.metadata != metadata
        or input_schema(predictor.data_config) != schema
        or metadata.get("bundle_schema_version") != "2.0"
        or metadata.get("development_only") is not True
        or metadata.get("fitting_population") != "train"
        or list(pipeline.named_steps) != ["engineering", "preprocessor", "model"]
        or list(map(str, pipeline.classes_)) != metadata.get("class_order")
        or set(pipeline.classes_) != set(schema["labels"])
        or list(pipeline.feature_names_in_) != schema["features"]
        or pipeline.named_steps["engineering"].data_config != predictor.data_config
        or pipeline.named_steps["model"].get_params() != metadata.get("effective_model_parameters")
    ):
        raise DataValidationError("Packaged model metadata/schema mismatch")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression

    expected = {"logistic_regression": LogisticRegression, "random_forest": RandomForestClassifier}
    if type(pipeline.named_steps["model"]) is not expected.get(metadata.get("selected_model")):
        raise DataValidationError("Unsupported model family")
    for key, value in pipeline.get_params(deep=True).items():
        if key.endswith("n_jobs") and value not in (None, 1):
            raise DataValidationError("Fitted model must retain serial resource settings")
        if key.endswith("handle_unknown") and value != "ignore":
            raise DataValidationError("Unsupported fitted category policy")
    if predictor.metadata["feature_config_sha256"] != fingerprint(
        pipeline.named_steps["engineering"].feature_config
    ):
        raise DataValidationError("Fitted feature contract mismatch")


def load_package_manifest(package_dir, expected_manifest_sha256=None):
    root = local_path(package_dir)
    manifest_path = safe_path(root, "package_manifest.json")
    if (
        expected_manifest_sha256 is not None
        and sha256_file(manifest_path) != expected_manifest_sha256
    ):
        raise DataValidationError("Package manifest does not match the pinned digest")
    manifest = read_json(manifest_path)
    fixed = {
        "status": "complete",
        "packaging_contract_version": "1.0",
        "package_schema_version": "1.0",
        "inference_contract_version": "2.0",
        "purpose": "research",
        "research_only": True,
        "production_ready": False,
    }
    if any(type(manifest.get(k)) is not type(v) or manifest[k] != v for k, v in fixed.items()):
        raise DataValidationError("Incomplete or unsupported research package")
    safe_name(manifest.get("package_id"))
    wheel = manifest.get("wheel")
    if not isinstance(wheel, str) or not re.fullmatch(
        r"runtime/machine_learning_project-[0-9][A-Za-z0-9.]*-py3-none-any\.whl", wheel
    ):
        raise DataValidationError("Invalid runtime wheel reference")
    required = FILES | {wheel}
    hashes = manifest.get("payload_hashes", {})
    if set(hashes) != required:
        raise DataValidationError("Incomplete package payload inventory")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != required | {"package_manifest.json"}:
        raise DataValidationError("Unexpected or missing package files")
    for name, expected in hashes.items():
        path = safe_path(root, name)
        if not path.is_file() or sha256_file(path) != expected:
            raise DataValidationError("Package payload fingerprint mismatch")
    if manifest.get("semantic_sha256") != semantic_digest(manifest):
        raise DataValidationError("Package semantic fingerprint mismatch")
    environment = read_json(root / "environment.json")
    if environment.get("runtime") != runtime_environment():
        raise DataValidationError("Unsupported package runtime environment")
    verify_wheel(root / wheel, environment["runtime"])
    requirements = (
        "\n".join(f"{k}=={v}" for k, v in sorted(environment["runtime"]["dependencies"].items()))
        + "\n"
    )
    if (root / "runtime/requirements.txt").read_text(encoding="utf-8") != requirements:
        raise DataValidationError("Runtime requirements disagree with supported environment")
    schema = read_json(root / "input_schema.json")
    validate_schema(schema)
    validate_inference_config(read_json(root / "inference_config.json"))
    if read_json(root / "output_schema.json") != OUTPUT_SCHEMA:
        raise DataValidationError("Unsupported output contract")
    decision, metadata = read_json(root / "decision.json"), read_json(root / "model_metadata.json")
    family = manifest.get("family")
    references = manifest.get("references", {})
    frozen = decision.get("frozen_references", {})
    flags = manifest.get("comparison_flags", {})
    if (
        family not in ("logistic_regression", "random_forest")
        or family != metadata.get("selected_model")
        or decision.get("execution_status") != "complete"
        or decision.get("production_ready") is not False
        or manifest.get("recommended_model") != decision.get("recommended_model")
        or manifest.get("development_reference") != decision.get("development_reference")
        or manifest.get("metric_eligible") != (family in decision.get("metric_eligible_models", []))
        or manifest.get("model_version") != metadata.get("model_version")
        or manifest.get("references", {}).get("decision_sha256") != hashes["decision.json"]
        or decision.get("frozen_references", {}).get("model_hashes", {}).get(f"{family}.joblib")
        != hashes["model.joblib"]
        or any(references.get(k) != v for k, v in frozen.items())
        or frozen.get("model_hashes", {}).get(f"{family}.json") != hashes["model_metadata.json"]
        or any(type(flags.get(k)) is not bool for k in FLAGS)
        or [k for k in FLAGS if not flags[k]] != decision.get("failed_requirements", {}).get(family)
        or metadata.get("feature_manifest_sha256") != references.get("feature_manifest_sha256")
        or metadata.get("baseline_manifest_sha256") != references.get("baseline_manifest_sha256")
    ):
        raise DataValidationError("Package decision/model references disagree")
    return manifest


def publish_private_json(path, value):
    """Atomic no-clobber publication via a same-filesystem hard link."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() or path.is_symlink():
        raise DataValidationError("Output destination already exists")
    descriptor, staging = tempfile.mkstemp(prefix=".prediction-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, sort_keys=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.link(staging, path)  # Atomic and fails if a concurrent writer claimed the final name.
    except FileExistsError:
        raise DataValidationError("Output destination already exists") from None
    finally:
        Path(staging).unlink(missing_ok=True)


def load_packaging_run(report_dir):
    root = local_path(report_dir)
    manifest = read_json(safe_path(root, "packaging_manifest.json"))
    if (
        manifest.get("status") != "complete"
        or manifest.get("packaging_contract_version") != "1.0"
        or manifest.get("purpose") != "research"
        or set(manifest.get("report_hashes", {}))
        != {"resolved_config.json", "package_inventory.json", "packaging_report.md"}
    ):
        raise DataValidationError("Incomplete packaging run")
    for name, digest in manifest["report_hashes"].items():
        if sha256_file(safe_path(root, name)) != digest:
            raise DataValidationError("Packaging report fingerprint mismatch")
    packages = manifest.get("packages", {})
    if not packages or packages != read_json(root / "package_inventory.json"):
        raise DataValidationError("Packaging inventory mismatch")
    for family, reference in packages.items():
        loaded = load_package_manifest(reference["path"], reference["manifest_sha256"])
        if family != loaded["family"] or reference["semantic_sha256"] != loaded["semantic_sha256"]:
            raise DataValidationError("Packaging package identity mismatch")
    return manifest
