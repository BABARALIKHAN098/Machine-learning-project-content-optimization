from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machine_learning_project.data.ingestion import load_csv, sha256_file
from machine_learning_project.data.splitting import split_by_group, write_split_artifacts
from machine_learning_project.data.validation import require_valid_schema
from machine_learning_project.utils.config import (
    load_yaml,
    validate_data_config,
    validate_split_config,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Create reproducible grouped split metadata.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    args = parser.parse_args()
    data_config = load_yaml(args.data_config)["data"]
    training_config = load_yaml(args.training_config)["training"]
    validate_data_config(data_config)
    validate_split_config(training_config)
    loaded = load_csv(data_config)
    require_valid_schema(loaded.dataframe, data_config)
    weights = training_config["split_objective_weights"]
    splits = split_by_group(
        loaded.dataframe,
        target_column=data_config["target_column"],
        group_column=training_config["group_column"],
        row_key=training_config["row_key"],
        allowed_labels=data_config["allowed_target_values"],
        test_size=training_config["test_size"],
        validation_size=training_config["validation_size"],
        random_seed=training_config["random_seed"],
        search_attempts=training_config["search_attempts"],
        row_ratio_tolerance=training_config["row_ratio_tolerance"],
        class_ratio_tolerance=training_config["class_ratio_tolerance"],
        size_weight=weights["size"],
        class_weight=weights["class"],
        source_sha256=loaded.sha256,
        data_schema_version=data_config.get("schema_version", "unversioned"),
        split_contract_version=training_config["split_contract_version"],
        algorithm_version=training_config["split_algorithm_version"],
        alias_context=training_config["split_alias_context"],
    )
    manifest, assignments = write_split_artifacts(
        splits,
        training_config["split_manifest_path"],
        training_config["split_assignments_path"],
    )
    if sha256_file(loaded.source_path) != loaded.sha256:
        raise RuntimeError("Raw CSV changed while splitting data.")
    for name in ("train", "validation", "test"):
        details = splits.manifest["partitions"][name]
        print(
            f"{name}: {details['row_count']} rows, {details['group_count']} groups, "
            f"ratio={details['actual_row_ratio']:.4f}"
        )
    print(f"Manifest: {manifest}")
    print(f"Assignments: {assignments}")


if __name__ == "__main__":
    main()
