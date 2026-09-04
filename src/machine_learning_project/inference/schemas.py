from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class PredictionRecord:
    content_id: str
    predicted_trend: str
    probability_down: float | None
    probabilities: dict[str, float] | None
    model_version: str
    warning: str = "Human review required; do not automate content changes."

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
