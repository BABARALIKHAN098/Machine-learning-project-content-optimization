from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machine_learning_project.utils.config import load_yaml
from pipelines.baseline_pipeline import run_baselines


def main():
    parser = argparse.ArgumentParser(description="Generate the development-only baseline benchmark")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--training-config", default="configs/training.yaml")
    parser.add_argument("--baseline-config", default="configs/baselines.yaml")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    config = load_yaml(args.baseline_config)["baselines"]
    result = run_baselines(
        load_yaml(args.data_config)["data"],
        load_yaml(args.training_config)["training"],
        config,
        args.output_dir,
    )
    for name, metrics in result["canonical"].items():
        print(f"{name}: validation macro F1 {metrics['macro_f1']:.6f}")
    print(f"Benchmark: {args.output_dir or config['output_directory']}")


if __name__ == "__main__":
    main()
