from __future__ import annotations

import argparse

from machine_learning_project.data.ingestion import load_csv
from machine_learning_project.data.preparation import prepare_supervised_data
from machine_learning_project.data.splitting import split_by_group
from machine_learning_project.data.validation import require_valid_schema
from machine_learning_project.features.preprocessing import build_preprocessor_from_config
from machine_learning_project.utils.config import load_yaml, validate_split_config


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate, split, and verify train-only preprocessing."
    )
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    args = parser.parse_args()
    data_config = load_yaml(args.data_config)["data"]
    preprocessing_config = load_yaml(args.preprocessing_config)["preprocessing"]
    training_config = load_yaml(args.training_config)["training"]
    validate_split_config(training_config)
    loaded = load_csv(data_config)
    require_valid_schema(loaded.dataframe, data_config)
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
        size_weight=training_config["split_objective_weights"]["size"],
        class_weight=training_config["split_objective_weights"]["class"],
        source_sha256=loaded.sha256,
        data_schema_version=data_config.get("schema_version", "unversioned"),
        split_contract_version=training_config["split_contract_version"],
        algorithm_version=training_config["split_algorithm_version"],
        alias_context=training_config["split_alias_context"],
    )
    prepared = {
        name: prepare_supervised_data(
            getattr(splits, name),
            data_config,
            feature_contract_version=preprocessing_config["feature_contract_version"],
        )
        for name in ("train", "validation", "test")
    }
    transformer = build_preprocessor_from_config(
        data_config["numeric_columns"],
        data_config["categorical_columns"],
        preprocessing_config,
    )
    train_matrix = transformer.fit_transform(prepared["train"].features)
    matrices = {
        "train": train_matrix,
        "validation": transformer.transform(prepared["validation"].features),
        "test": transformer.transform(prepared["test"].features),
    }
    for name, matrix in matrices.items():
        frame = getattr(splits, name)
        print(
            f"{name}: {matrix.shape[0]} rows, {matrix.shape[1]} transformed features, "
            f"{frame[training_config['group_column']].nunique()} clients"
        )


if __name__ == "__main__":
    main()
