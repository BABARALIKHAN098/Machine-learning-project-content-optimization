from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machine_learning_project.utils.config import load_yaml
from pipelines.data_pipeline import run_eda


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible exploratory data analysis.")
    parser.add_argument("--data-config", default="configs/data.yaml")
    parser.add_argument("--eda-config", default="configs/eda.yaml")
    parser.add_argument("--output-dir")
    parser.add_argument("--figures-dir")
    args = parser.parse_args()
    data_config = load_yaml(args.data_config)["data"]
    eda_config = load_yaml(args.eda_config)["eda"]
    results, artifacts = run_eda(
        data_config,
        eda_config,
        output_directory=args.output_dir,
        figures_directory=args.figures_dir,
    )
    print(
        f"EDA completed for {results.summary['row_count']} rows and "
        f"{results.summary['column_count']} columns."
    )
    print(f"Created {len(artifacts)} artifacts.")


if __name__ == "__main__":
    main()
