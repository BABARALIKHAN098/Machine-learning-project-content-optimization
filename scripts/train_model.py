from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machine_learning_project.utils.config import load_yaml
from pipelines.training_pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate content trend classifiers.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    args = parser.parse_args()
    result = run_training(
        load_yaml(args.data_config)["data"],
        load_yaml(args.preprocessing_config)["preprocessing"],
        load_yaml(args.training_config)["training"],
    )
    print(f"Selected model: {result.selected_model}")
    print(f"Artifact: {result.artifact_path}")
    print(f"Metrics: {result.metrics_path}")


if __name__ == "__main__":
    main()
