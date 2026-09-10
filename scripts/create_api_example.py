"""Generate invented JSON rows from a verified package; no predictions or source data."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.schemas import synthetic_document
from machine_learning_project.inference.packaged_predictor import PackagedPredictor
from machine_learning_project.models.package_artifacts import local_path, publish_private_json
from machine_learning_project.utils.exceptions import DataValidationError


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--rows", type=int, default=5)
    parser.add_argument("--output-path", required=True)
    args = parser.parse_args()
    package, output = map(local_path, (args.package_dir, args.output_path))
    root = Path(__file__).resolve().parents[1]
    if (
        not 1 <= args.rows <= 30000
        or output.exists()
        or any(
            output.is_relative_to(p.resolve())
            for p in (
                package,
                root / "artifacts",
                root / "reports",
                root / "data/raw",
                root / "data/processed",
                root / "data/splits",
            )
        )
    ):
        raise DataValidationError("Invalid example destination or row count")
    predictor = PackagedPredictor.load(
        package, purpose="research", expected_manifest_sha256=args.expected_manifest_sha256
    )
    publish_private_json(output, synthetic_document(predictor.schema, args.rows))
    print(f"Invented JSON example created: {args.rows} rows.")


if __name__ == "__main__":
    main()
