from __future__ import annotations

from pathlib import Path

import pandas as pd

from machine_learning_project.inference.predictor import Predictor


def run_batch_inference(input_path: str | Path, artifact_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(input_path)
    return Predictor.load(artifact_path).predict(dataframe)


def run_packaged_batch(
    input_path, package_dir, output_path, *, purpose, include_probabilities=False, chunk_rows=None
):
    from machine_learning_project.inference.contracts import read_request_csv
    from machine_learning_project.inference.packaged_predictor import PackagedPredictor
    from machine_learning_project.models.package_artifacts import local_path, publish_private_json
    from machine_learning_project.utils.exceptions import DataValidationError

    source, package, output = map(local_path, (input_path, package_dir, output_path))
    if output == source or output.is_relative_to(package) or package.is_relative_to(output):
        raise DataValidationError("Inference output overlaps input/package")
    repository = Path(__file__).resolve().parents[1]
    protected = [
        repository / p
        for p in (
            "data/raw",
            "reports/evaluation",
            "reports/training",
            "reports/features",
            "reports/baselines",
            "reports/packaging",
            "artifacts",
        )
    ]
    if any(output.is_relative_to(p) for p in protected):
        raise DataValidationError("Inference output overlaps protected artifacts")
    if output.exists():
        raise DataValidationError("Inference output already exists")
    predictor = PackagedPredictor.load(package, purpose=purpose)
    frame = read_request_csv(source, predictor.schema, predictor.config)
    result = predictor.predict(
        frame, include_probabilities=include_probabilities, chunk_rows=chunk_rows
    )
    publish_private_json(output, result)
    return {
        "status": "complete",
        "row_count": result["row_count"],
        "output_path": str(output),
        "package_id": result["package_id"],
        "purpose": "research",
    }
