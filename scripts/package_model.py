import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machine_learning_project.utils.config import load_yaml
from pipelines.packaging_pipeline import run_packaging


def main():
    parser = argparse.ArgumentParser(description="Package frozen finalists for research only")
    parser.add_argument("--packaging-config", default="configs/packaging.yaml")
    parser.add_argument("--inference-config", default="configs/inference.yaml")
    parser.add_argument("--evaluation-dir", default="reports/evaluation/spec07-reference")
    parser.add_argument("--training-report-dir", default="reports/training/spec06-reference")
    parser.add_argument("--model-dir", default="artifacts/models/training/spec06-reference")
    parser.add_argument("--purpose", required=True, choices=["research"])
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--runtime-wheel")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    result = run_packaging(
        load_yaml(args.packaging_config)["packaging"],
        load_yaml(args.inference_config)["inference"],
        args.evaluation_dir,
        args.training_report_dir,
        args.model_dir,
        args.run_id,
        args.purpose,
        dry_run=args.dry_run,
        runtime_wheel=args.runtime_wheel,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
