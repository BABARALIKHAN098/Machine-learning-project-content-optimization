from __future__ import annotations

from typing import Any


def select_candidate(results: dict[str, dict[str, Any]], down_recall_guardrail: float) -> str:
    """Select by macro F1, preferring candidates that meet the recall guardrail."""
    eligible = {
        name: metrics
        for name, metrics in results.items()
        if metrics.get("per_class", {}).get("down", {}).get("recall", 0.0)
        >= down_recall_guardrail
    }
    pool = eligible or results
    if not pool:
        raise ValueError("No candidate evaluation results were supplied.")
    return max(pool, key=lambda name: (pool[name]["macro_f1"], pool[name]["weighted_f1"], name))
