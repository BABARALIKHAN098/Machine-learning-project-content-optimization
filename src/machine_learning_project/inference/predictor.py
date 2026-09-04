from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd

from ..features.selection import select_features
from .schemas import PredictionRecord
from .validation import validate_inference_frame


class Predictor:
    def __init__(self, pipeline: Any, metadata: dict[str, Any], data_config: dict[str, Any]):
        self.pipeline = pipeline
        self.metadata = metadata
        self.data_config = data_config

    @classmethod
    def load(cls, artifact_path: str | Path) -> Predictor:
        artifact = joblib.load(artifact_path)
        return cls(artifact["pipeline"], artifact["metadata"], artifact["data_config"])

    def predict(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        validate_inference_frame(dataframe, self.data_config)
        features = select_features(dataframe, self.data_config)
        predictions = self.pipeline.predict(features)
        probabilities = None
        classes: list[str] = []
        if hasattr(self.pipeline, "predict_proba"):
            probabilities = self.pipeline.predict_proba(features)
            classes = list(map(str, self.pipeline.classes_))
        down_index = classes.index("down") if "down" in classes else None
        records = []
        for index, predicted in enumerate(predictions):
            down_probability = (
                float(probabilities[index][down_index])
                if probabilities is not None and down_index is not None
                else None
            )
            records.append(
                PredictionRecord(
                    content_id=str(dataframe.iloc[index]["content_id"]),
                    predicted_trend=str(predicted),
                    probability_down=down_probability,
                    probabilities=(
                        {
                            label: float(probabilities[index][class_index])
                            for class_index, label in enumerate(classes)
                        }
                        if probabilities is not None
                        else None
                    ),
                    model_version=str(self.metadata["model_version"]),
                ).to_dict()
            )
        return pd.DataFrame(records)

    def rank_review_queue(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        predictions = self.predict(dataframe)
        return predictions.sort_values(
            "probability_down", ascending=False, na_position="last"
        ).reset_index(drop=True)
