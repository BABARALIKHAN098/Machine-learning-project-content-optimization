import json

import numpy as np
import pandas as pd
import pytest

from machine_learning_project.models.error_analysis import (
    analyze_errors,
    analyze_subgroups,
    client_sensitivity,
    compare_paired_predictions,
    subgroup_memberships,
)
from machine_learning_project.models.evaluation_decision import FLAGS, decide_evaluation
from machine_learning_project.models.training_artifacts import FAMILIES
from machine_learning_project.utils.config import load_yaml
from machine_learning_project.utils.exceptions import DataValidationError

LABELS = ["down", "stable", "up", "new", "flat"]


def config():
    return load_yaml("configs/evaluation.yaml")["evaluation"]


def test_error_arithmetic_and_joint_marginals():
    truth = LABELS + ["down", "down"]
    a = ["down", "down", "new", "new", "flat", "stable", "up"]
    b = ["stable", "stable", "up", "flat", "down", "down", "up"]
    errors = analyze_errors(truth, a, LABELS)
    assert {k: errors["down"][k] for k in ("tp", "fp", "fn", "tn")} == {
        "tp": 1,
        "fp": 1,
        "fn": 2,
        "tn": 3,
    }
    assert errors["down"]["recall"]["value"] == 1 / 3
    assert errors["down"]["precision"]["value"] == 0.5
    assert errors["down"]["false_positive_rate"]["value"] == 0.25
    assert sum(r["false_negatives"] for r in errors["classes"]) == 4
    paired = compare_paired_predictions(truth, dict(zip(FAMILIES, [a, b])), LABELS)
    assert sum(paired["correctness"].values()) == 7
    assert sum(paired["true_down"].values()) == 3
    assert paired["agreement"] + paired["disagreement"] == 7
    for family, key, pred in zip(FAMILIES, ["logistic_prediction", "forest_prediction"], [a, b]):
        marginal = [
            [
                sum(r["count"] for r in paired["joint_table"] if r["truth"] == t and r[key] == p)
                for p in LABELS
            ]
            for t in LABELS
        ]
        assert marginal == analyze_errors(truth, pred, LABELS)["confusion"]["counts"]


@pytest.mark.parametrize(
    "truth,pred,undefined",
    [
        (["up"], ["up"], ["recall", "precision", "false_negative_rate"]),
        (["down"], ["down"], ["false_positive_rate"]),
        (["down"], ["up"], ["precision"]),
    ],
)
def test_undefined_rates(truth, pred, undefined):
    result = analyze_errors(truth, pred, LABELS)
    for name in undefined:
        assert result["down"][name]["value"] is None
        assert result["down"][name]["denominator"] == 0
    for label in set(LABELS) - set(truth):
        assert result["confusion"]["row_normalized"][LABELS.index(label)] == [None] * 5


def test_unpaired_rejected():
    with pytest.raises(DataValidationError):
        compare_paired_predictions(LABELS, dict(zip(FAMILIES, [LABELS, LABELS[:-1]])), LABELS)


def test_membership_edges_collision_privacy_and_coverage():
    c = config()
    c.update(
        dimensions=["content_type", "previous_impressions_bucket", "numeric_missingness_bucket"],
        minimum_subgroup_rows=2,
        maximum_categories_per_dimension=2,
    )
    context = pd.DataFrame(
        {
            "content_type": ["OTHER", "OTHER", None, "private-category", "MISSING", "MISSING", "z"],
            "impressions_prev_30d": [None, 0, 1, 100, 101, 1000, 1001],
        }
    )
    groups = subgroup_memberships(context, c, ["impressions_prev_30d"])
    assert groups["previous_impressions_bucket"] == [
        ("missing", ""),
        ("bin", "zero"),
        ("bin", "(0,100]"),
        ("bin", "(0,100]"),
        ("bin", "(100,1000]"),
        ("bin", "(100,1000]"),
        ("bin", ">1000"),
    ]
    assert groups["content_type"][0][0] == "category"
    assert groups["content_type"][2][0] == "missing"
    assert groups["content_type"][3][0] == "pooled"
    truth = LABELS + ["down", "up"]
    result = analyze_subgroups(
        context,
        truth,
        {f: truth for f in FAMILIES},
        LABELS,
        c,
        numeric_columns=["impressions_prev_30d"],
        down_recall_guardrail=0.5,
    )
    assert "private-category" not in json.dumps(result)
    for coverage in result["coverage"].values():
        assert coverage["published_rows"] + coverage["suppressed_rows"] == 7
    assert not result["alerts"]
    assert all(r["down_recall"] is None for r in result["records"])


@pytest.mark.parametrize("n,supported", [(19, False), (20, True), (21, True)])
def test_class_support_boundary(n, supported):
    c = config()
    c.update(dimensions=["content_type"], minimum_subgroup_rows=1)
    truth = LABELS * n
    context = pd.DataFrame({"content_type": ["a"] * len(truth)})
    result = analyze_subgroups(
        context,
        truth,
        {f: truth for f in FAMILIES},
        LABELS,
        c,
        numeric_columns=[],
        down_recall_guardrail=0.5,
    )
    assert all(r["macro_supported"] == supported for r in result["records"])


@pytest.mark.parametrize("n,supported", [(29, False), (30, True), (31, True)])
def test_recall_support_boundary(n, supported):
    c = config()
    c.update(dimensions=["content_type"], minimum_subgroup_rows=1)
    truth = ["down"] * n
    result = analyze_subgroups(
        pd.DataFrame({"content_type": ["a"] * n}),
        truth,
        {f: ["up"] * n for f in FAMILIES},
        LABELS,
        c,
        numeric_columns=[],
        down_recall_guardrail=0.5,
    )
    assert all(r["down_recall_alert"] == supported for r in result["records"])
    assert all(not r["macro_alert"] for r in result["records"])


def test_client_sensitivity_no_identifiers_and_reversal():
    c = config()
    c.update(minimum_class_support=1, minimum_down_support=1)
    truth = LABELS * 3
    a, b = truth.copy(), truth.copy()
    a[:5] = np.roll(LABELS, 1).tolist()
    b[5:9] = np.roll(LABELS, 1).tolist()[:4]
    b[10:] = np.roll(LABELS, 1).tolist()
    result = client_sensitivity(
        ["private-client-a"] * 5 + ["b"] * 5 + ["c"] * 5,
        truth,
        dict(zip(FAMILIES, [a, b])),
        LABELS,
        c,
        hash_context="fixture",
    )
    assert result["scenario_count"] == 3
    assert result["remaining_rows"]["minimum"] == 10
    assert result["ordering_reverses"]
    assert "private-client" not in json.dumps(result)


@pytest.mark.parametrize("eligible", [[], [FAMILIES[0]], [FAMILIES[1]], list(FAMILIES)])
def test_decision_eligibility_and_tie(eligible):
    comparisons = {f: {flag: f in eligible for flag in FLAGS} for f in FAMILIES}
    metrics = {f: {"macro_f1": 0.5} for f in FAMILIES}
    result = decide_evaluation(
        metrics, comparisons, FAMILIES[1], {"review_flags": [{"reason": "cutoff"}]}, config()
    )
    assert result["recommended_model"] == (
        FAMILIES[1] if FAMILIES[1] in eligible else FAMILIES[0] if eligible else None
    )
    assert result["metric_eligible_models"] == eligible
    assert result["production_ready"] is False
    assert result["development_reference"] == FAMILIES[1]
    reverse = decide_evaluation(
        dict(reversed(list(metrics.items()))),
        comparisons,
        FAMILIES[1],
        {"review_flags": [{"reason": "cutoff"}]},
        config(),
    )
    assert result == reverse


def test_failed_target_cannot_be_overridden_by_baselines_or_cv():
    comparisons = {f: {flag: True for flag in FLAGS} for f in FAMILIES}
    for f in FAMILIES:
        comparisons[f]["project_macro_f1_target_met"] = False
    result = decide_evaluation(
        {f: {"macro_f1": 0.43} for f in FAMILIES},
        comparisons,
        FAMILIES[1],
        {"review_flags": []},
        config(),
    )
    assert result["recommended_model"] is None
    assert result["recommendation_status"] == "no_candidate_meets_metric_requirements"
    assert all(v == ["project_macro_f1_target_met"] for v in result["failed_requirements"].values())


@pytest.mark.parametrize("n,published", [(99, False), (100, True), (101, True)])
def test_publication_support_boundary(n, published):
    c = config()
    c["dimensions"] = ["content_type"]
    truth = ["down"] * n
    result = analyze_subgroups(
        pd.DataFrame({"content_type": ["private"] * n}),
        truth,
        {f: truth for f in FAMILIES},
        LABELS,
        c,
        numeric_columns=[],
        down_recall_guardrail=0.5,
    )
    assert bool(result["records"]) == published
    coverage = result["coverage"]["content_type"]
    assert coverage["published_rows"] == (n if published else 0)
    assert coverage["suppressed_rows"] == (0 if published else n)


@pytest.mark.parametrize("value", [-1, float("inf"), float("-inf")])
def test_invalid_impressions(value):
    c = config()
    c["dimensions"] = ["previous_impressions_bucket"]
    with pytest.raises(DataValidationError):
        subgroup_memberships(pd.DataFrame({"impressions_prev_30d": [value]}), c, [])
