from __future__ import annotations

from pathlib import Path

import pandas as pd

from machine_learning_project.inference.predictor import Predictor


def run_batch_inference(input_path: str | Path, artifact_path: str | Path) -> pd.DataFrame:
    dataframe = pd.read_csv(input_path)
    return Predictor.load(artifact_path).predict(dataframe)
