"""Separate development preference, metric eligibility, and unresolved review."""

from ..utils.exceptions import DataValidationError
from .training_artifacts import FAMILIES

FLAGS = (
    "beats_both_baselines",
    "material_improvement_met",
    "down_recall_guardrail_met",
    "project_macro_f1_target_met",
)
LIMITATIONS = [
    "Validation informed earlier feature and SPEC-06 confirmation; replay is not independent evidence.",
    "Historical test evaluation exists; this analysis scores validation only. Final-test policy is unresolved.",
    "Historical metadata cutoff evidence remains unresolved.",
    "Client sensitivity is descriptive; subgroup support is not a formal privacy or fairness guarantee.",
    "Production runtime SLA and deployment readiness are not established.",
]


def decide_evaluation(metrics, comparisons, cv_reference, diagnostics, config):
    if (
        set(metrics) != set(FAMILIES)
        or set(comparisons) != set(FAMILIES)
        or cv_reference not in FAMILIES
    ):
        raise DataValidationError(
            "Decision requires both finalists and a valid frozen CV reference"
        )
    if any(type(comparisons[f].get(k)) is not bool for f in FAMILIES for k in FLAGS):
        raise DataValidationError("Missing decision comparison flags")
    eligible = [f for f in FAMILIES if all(comparisons[f][k] for k in FLAGS)]
    blockers = list(diagnostics.get("review_flags", []))
    recommended = None
    if eligible:
        best = max(metrics[f]["macro_f1"] for f in eligible)
        tied = [
            f for f in eligible if metrics[f]["macro_f1"] + config["comparison_tolerance"] >= best
        ]
        recommended = min(tied, key=lambda f: (f != cv_reference, f != "logistic_regression", f))
    return {
        "execution_status": "complete",
        "metric_eligible_models": eligible,
        "development_reference": cv_reference,
        "recommended_model": recommended,
        "failed_requirements": {f: [k for k in FLAGS if not comparisons[f][k]] for f in FAMILIES},
        "recommendation_status": "no_candidate_meets_metric_requirements"
        if not eligible
        else "eligible_pending_review"
        if blockers
        else "eligible_for_packaging_review",
        "review_flags": blockers,
        "production_ready": False,
        "limitations": LIMITATIONS,
        "selection_evidence": "reused_validation",
        "spec08_handoff": "Research artifacts and rejection evidence; no production pointer"
        if not eligible
        else "Eligible candidate requires separate packaging acceptance",
    }
