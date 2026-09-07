"""Verify persisted partitions and expose only development rows to model code."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from ..utils.exceptions import DataValidationError
from .ingestion import load_csv, sha256_file
from .splitting import _hash_value
from .validation import require_valid_schema


def load_development(data_config, training_config):
    """Verify existing assignments, returning no test dataframe or test labels."""
    loaded = load_csv(data_config)
    require_valid_schema(loaded.dataframe, data_config)
    manifest_path = Path(training_config["split_manifest_path"])
    assignment_path = Path(training_config["split_assignments_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_schema_version") != "1.0":
        raise DataValidationError("Unsupported split manifest schema")
    if manifest["split_contract_version"] != training_config["split_contract_version"]:
        raise DataValidationError("Split contract version mismatch")
    if manifest["algorithm_version"] != training_config["split_algorithm_version"]:
        raise DataValidationError("Split algorithm version mismatch")
    if manifest["source"]["source_sha256"] != loaded.sha256:
        raise DataValidationError("Split/source fingerprint mismatch; regenerate split explicitly")
    parameters = manifest["parameters"]
    for key in (
        "row_key",
        "group_column",
        "random_seed",
        "test_size",
        "validation_size",
        "search_attempts",
        "row_ratio_tolerance",
        "class_ratio_tolerance",
    ):
        if parameters[key] != training_config[key]:
            raise DataValidationError(f"Split configuration mismatch: {key}")
    if parameters["target_column"] != data_config["target_column"]:
        raise DataValidationError("Split target mismatch")
    if manifest["source"]["data_schema_version"] != data_config.get(
        "schema_version", "unversioned"
    ):
        raise DataValidationError("Split data schema mismatch")
    if set(manifest["source"]["target_labels"]) != set(data_config["allowed_target_values"]):
        raise DataValidationError("Split target labels mismatch")
    assignments = pd.read_csv(assignment_path, dtype=str)
    if assignments.isna().any().any():
        raise DataValidationError("Missing split assignment values")
    if set(assignments["split_contract_version"]) != {manifest["split_contract_version"]}:
        raise DataValidationError("Assignment contract version mismatch")
    if assignments["row_key_hash"].duplicated().any():
        raise DataValidationError("Duplicate split assignment")
    context = (
        f"{training_config.get('split_alias_context', 'content-trend-split-v1')}:{loaded.sha256}"
    )
    frame = loaded.dataframe
    hashes = frame[parameters["row_key"]].map(lambda value: _hash_value(value, context))
    if hashes.duplicated().any() or frame[parameters["group_column"]].isna().any():
        raise DataValidationError("Duplicate row identity or null group")
    if set(hashes) != set(assignments["row_key_hash"]):
        raise DataValidationError("Split assignments do not cover source exactly")
    mapping = assignments.set_index("row_key_hash")
    partitions = hashes.map(mapping["partition"])
    if set(partitions) != {"train", "validation", "test"}:
        raise DataValidationError("Invalid split partitions")
    aliases = frame[parameters["group_column"]].map(lambda value: _hash_value(value, context)[:16])
    if list(aliases) != list(hashes.map(mapping["group_alias"])):
        raise DataValidationError("Group alias mismatch")
    if (
        pd.DataFrame({"group": aliases, "partition": partitions})
        .groupby("group")["partition"]
        .nunique()
        .max()
        != 1
    ):
        raise DataValidationError("Group overlap in split assignments")
    for name in ("train", "validation", "test"):
        selected_hashes = sorted(hashes[partitions.eq(name)])
        observed = hashlib.sha256("\n".join(selected_hashes).encode()).hexdigest()
        if observed != manifest["partitions"][name]["assignment_fingerprint"]:
            raise DataValidationError(f"Assignment fingerprint mismatch: {name}")
    provenance = {
        "source_sha256": loaded.sha256,
        "split_manifest_sha256": sha256_file(manifest_path),
        "split_assignments_sha256": sha256_file(assignment_path),
    }
    if sha256_file(loaded.source_path) != loaded.sha256:
        raise DataValidationError("Source changed while loading development partitions")
    return (
        frame.loc[partitions.eq("train")].copy(),
        frame.loc[partitions.eq("validation")].copy(),
        provenance,
    )
