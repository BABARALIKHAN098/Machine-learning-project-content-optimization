import pandas as pd
import pytest

from machine_learning_project.data.ingestion import sha256_file
from machine_learning_project.data.splitting import split_by_group, write_split_artifacts
from machine_learning_project.utils.config import load_yaml


@pytest.fixture
def baseline_inputs(tmp_path):
    labels = ["down", "flat", "new", "stable", "up"]
    frame = pd.DataFrame(
        [
            {
                "content_id": f"private-row-{g}-{i}",
                "client_id": f"private-client-{g}",
                "target": label,
                "value": i + g / 10,
            }
            for g in range(20)
            for i, label in enumerate(labels)
        ]
    )
    source = tmp_path / "source.csv"
    frame.to_csv(source, index=False)
    data = {
        "csv_path": str(source),
        "target_column": "target",
        "allowed_target_values": labels,
        "required_columns": list(frame.columns),
        "id_columns": ["content_id", "client_id"],
        "unique_columns": ["content_id"],
        "numeric_columns": ["value"],
        "categorical_columns": [],
        "drop_columns": [],
    }
    training = load_yaml("configs/training.yaml")["training"]
    for key, name in [
        ("split_manifest_path", "split.json"),
        ("split_assignments_path", "assignments.csv"),
        ("metrics_path", "model_metrics.json"),
        ("metadata_path", "model_metadata.json"),
        ("artifact_path", "model.joblib"),
    ]:
        training[key] = str(tmp_path / name)
    training["candidates"]["random_forest"].update(n_estimators=3, n_jobs=1)
    splits = split_by_group(
        frame,
        target_column="target",
        group_column="client_id",
        row_key="content_id",
        allowed_labels=labels,
        source_sha256=sha256_file(source),
        random_seed=training["random_seed"],
        test_size=training["test_size"],
        validation_size=training["validation_size"],
        search_attempts=training["search_attempts"],
        row_ratio_tolerance=training["row_ratio_tolerance"],
        class_ratio_tolerance=training["class_ratio_tolerance"],
        split_contract_version=training["split_contract_version"],
        algorithm_version=training["split_algorithm_version"],
        alias_context=training["split_alias_context"],
    )
    write_split_artifacts(
        splits, training["split_manifest_path"], training["split_assignments_path"]
    )
    return data, training, load_yaml("configs/baselines.yaml")["baselines"]


@pytest.fixture
def tuning_inputs(baseline_inputs, tmp_path):
    import json

    from machine_learning_project.data.development import load_development
    from machine_learning_project.features.artifacts import input_contract_fingerprint
    from pipelines.baseline_pipeline import run_baselines

    data, training, baseline = baseline_inputs
    pre = load_yaml("configs/preprocessing.yaml")["preprocessing"]
    tuning = load_yaml("configs/tuning.yaml")["tuning"]
    tuning.update(output_root=str(tmp_path / "reports"), model_output_root=str(tmp_path / "models"))
    features = load_yaml("configs/features.yaml")["features"]
    features.update(families=[], log_sources=[])
    selected = {family: features for family in tuning["families"]}
    feature_dir = tmp_path / "features"
    feature_dir.mkdir()
    selected_path = feature_dir / "selected_features.json"
    selected_path.write_text(json.dumps(selected))
    _, _, provenance = load_development(data, training)
    manifest = {
        "schema_version": "1.0",
        "feature_contract_version": "2.0",
        "provenance": provenance,
        "input_contract_sha256": input_contract_fingerprint(data, pre, training),
        "selected_configs": selected,
        "artifact_hashes": {"selected_features.json": sha256_file(selected_path)},
    }
    feature_path = feature_dir / "feature_manifest.json"
    feature_path.write_text(json.dumps(manifest))
    run_baselines(data, training, baseline, tmp_path / "baselines")
    return {
        "data": data,
        "preprocessing": pre,
        "training": training,
        "tuning": tuning,
        "feature_manifest": feature_path,
        "baseline_manifest": tmp_path / "baselines/baseline_manifest.json",
        "baseline_config": baseline,
    }
