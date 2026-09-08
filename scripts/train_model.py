from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machine_learning_project.utils.config import load_yaml
from pipelines.training_pipeline import run_training


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classifiers using development data only.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--features-config", default="configs/features.yaml")
    parser.add_argument("--feature-manifest", help="Reuse estimator-specific frozen study choices")
    parser.add_argument("--baseline-config", default="configs/baselines.yaml")
    parser.add_argument("--baseline-manifest", help="Reuse a verified frozen validation benchmark")
    parser.add_argument("--tuning-config", help="Run the fixed SPEC-06 grouped training search")
    parser.add_argument("--run-id", default="spec06-reference")
    parser.add_argument("--output-root")
    parser.add_argument("--model-output-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.tuning_config:
        import json

        from pipelines.tuning_pipeline import run_tuning

        tuning = load_yaml(args.tuning_config)["tuning"]
        if args.output_root:
            tuning["output_root"] = args.output_root
        if args.model_output_root:
            tuning["model_output_root"] = args.model_output_root
        result = run_tuning(
            load_yaml(args.data_config)["data"],
            load_yaml(args.preprocessing_config)["preprocessing"],
            load_yaml(args.training_config)["training"],
            tuning,
            feature_manifest=args.feature_manifest,
            baseline_manifest=args.baseline_manifest,
            baseline_config=load_yaml(args.baseline_config)["baselines"],
            run_id=args.run_id,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2))
        return
    if args.dry_run or args.output_root or args.model_output_root:
        parser.error("Dry run/output roots require --tuning-config")
    result = run_training(
        load_yaml(args.data_config)["data"],
        load_yaml(args.preprocessing_config)["preprocessing"],
        load_yaml(args.training_config)["training"],
        load_yaml(args.features_config)["features"],
        feature_manifest=args.feature_manifest,
        baseline_config=load_yaml(args.baseline_config)["baselines"],
        baseline_manifest=args.baseline_manifest,
    )
    print(f"Selected model: {result.selected_model}")
    print(f"Artifact: {result.artifact_path}")
    print(f"Metrics: {result.metrics_path}")


if __name__ == "__main__":
    main()
