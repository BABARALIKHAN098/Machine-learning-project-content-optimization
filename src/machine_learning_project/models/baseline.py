from __future__ import annotations

from typing import Any

from sklearn.dummy import DummyClassifier


def build_baselines(random_seed: int = 42) -> dict[str, Any]:
    return {
        "most_frequent": DummyClassifier(strategy="most_frequent"),
        "stratified": DummyClassifier(strategy="stratified", random_state=random_seed),
    }
