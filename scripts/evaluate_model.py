from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Replay frozen validation models or display a historical report."
    )
    parser.add_argument("--metrics", default="reports/metrics/model_metrics.json")
    parser.add_argument("--evaluation-config")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--preprocessing-config", default="configs/preprocessing.yaml")
    parser.add_argument("--baseline-config", default="configs/baselines.yaml")
    parser.add_argument("--training-report-dir", default="reports/training/spec06-reference")
    parser.add_argument("--model-dir", default="artifacts/models/training/spec06-reference")
    parser.add_argument("--feature-manifest", default="reports/features/feature_manifest.json")
    parser.add_argument("--baseline-manifest", default="reports/baselines/baseline_manifest.json")
    parser.add_argument("--run-id", default="spec07-reference")
    parser.add_argument("--output-root")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.evaluation_config:
        from machine_learning_project.utils.config import load_yaml
        from pipelines.evaluation_pipeline import run_evaluation

        config = load_yaml(args.evaluation_config)["evaluation"]
        if args.output_root:
            config["output_root"] = args.output_root
        result = run_evaluation(
            load_yaml(args.data_config)["data"],
            load_yaml(args.training_config)["training"],
            load_yaml(args.preprocessing_config)["preprocessing"],
            config,
            args.training_report_dir,
            args.model_dir,
            args.feature_manifest,
            args.baseline_manifest,
            load_yaml(args.baseline_config)["baselines"],
            args.run_id,
            dry_run=args.dry_run,
        )
        print(json.dumps(result, indent=2))
        return
    if args.dry_run or args.output_root:
        parser.error("Dry run/output root require --evaluation-config")
    payload = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    if "test" not in payload:
        validation = payload["candidates_validation"][payload["selected_model"]]
        print(f"Selected model: {payload['selected_model']}")
        print(f"Validation macro F1: {validation['macro_f1']:.4f}")
        print(
            "Development-only artifact. Historical test evaluation exists elsewhere; this display does not score data."
        )
        return
    test = payload["test"]
    print(f"Selected model: {payload['selected_model']}")
    print(f"Test macro F1: {test['macro_f1']:.4f}")
    print(f"Down recall: {test['per_class']['down']['recall']:.4f}")
    print(f"Threshold met: {payload['threshold_met']}")
    print("Historical test report; this display performs no new test evaluation.")


if __name__ == "__main__":
    main()
