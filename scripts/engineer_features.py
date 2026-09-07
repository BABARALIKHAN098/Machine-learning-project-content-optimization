from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machine_learning_project.data.development import load_development
from machine_learning_project.data.ingestion import sha256_file
from machine_learning_project.utils.config import load_yaml
from machine_learning_project.utils.exceptions import DataValidationError
from pipelines.feature_pipeline import run_feature_study


def main():
    parser = argparse.ArgumentParser(description="Run the fixed development-only feature study")
    for name in ("data", "preprocessing", "training", "features"):
        parser.add_argument(f"--{name}-config", default=f"configs/{name}.yaml")
    parser.add_argument("--output-dir", default="reports/features")
    args = parser.parse_args()
    configs = {
        name: load_yaml(getattr(args, f"{name}_config"))[name]
        for name in ("data", "preprocessing", "training", "features")
    }
    train, validation, provenance = load_development(configs["data"], configs["training"])
    run_feature_study(
        train,
        validation,
        configs["data"],
        configs["preprocessing"],
        configs["training"],
        configs["features"],
        args.output_dir,
        provenance,
        source_path=configs["data"]["csv_path"],
    )
    if sha256_file(Path(configs["data"]["csv_path"])) != provenance["source_sha256"]:
        raise DataValidationError("Source changed during feature study")
    print(f"Feature evidence: {args.output_dir}")


if __name__ == "__main__":
    main()
