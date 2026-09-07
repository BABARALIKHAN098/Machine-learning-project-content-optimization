"""Read the frozen study decision without running feature selection again."""

import json
from pathlib import Path

from ..data.ingestion import sha256_file
from ..utils.exceptions import DataValidationError
from .registry import fingerprint, resolve_registry


def input_contract_fingerprint(data, preprocessing, training):
    return fingerprint(
        {
            "data": data,
            "preprocessing": preprocessing,
            "candidates": training.get("candidates", {}),
            "random_seed": training.get("random_seed", 42),
        }
    )


def load_frozen_configs(path, data, preprocessing, training, provenance):
    path = Path(path)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != "1.0" or manifest.get("feature_contract_version") != "2.0":
        raise DataValidationError("Unsupported feature manifest")
    if manifest.get("provenance") != provenance:
        raise DataValidationError("Feature study source/split provenance mismatch")
    if manifest.get("input_contract_sha256") != input_contract_fingerprint(
        data, preprocessing, training
    ):
        raise DataValidationError("Feature study input contract mismatch")
    for name, expected in manifest["artifact_hashes"].items():
        if Path(name).name != name:
            raise DataValidationError("Invalid feature artifact name")
        if sha256_file(path.parent / name) != expected:
            raise DataValidationError(f"Feature artifact fingerprint mismatch: {name}")
    configs = manifest["selected_configs"]
    if set(configs) != {"logistic_regression", "random_forest"}:
        raise DataValidationError("Incomplete selected feature configurations")
    frozen = json.loads((path.parent / "selected_features.json").read_text(encoding="utf-8"))
    if configs != frozen:
        raise DataValidationError("Selected configurations disagree with frozen artifact")
    for config in configs.values():
        resolve_registry(data, config)
    return configs, sha256_file(path)
