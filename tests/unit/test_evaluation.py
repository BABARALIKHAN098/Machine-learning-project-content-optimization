import pytest

from machine_learning_project.models.evaluate import evaluate_predictions
from machine_learning_project.utils.exceptions import DataValidationError

LABELS = ["down", "flat", "new", "stable", "up"]


def test_analytic_majority_metrics():
    truth = LABELS + ["down"] * 5
    metrics = evaluate_predictions(truth, ["down"] * 10, LABELS)
    q = 0.6
    assert metrics["accuracy"] == q
    assert metrics["balanced_accuracy"] == 1 / 5
    assert metrics["macro_f1"] == pytest.approx(2 * q / (5 * (1 + q)))
    assert metrics["weighted_f1"] == pytest.approx(q * 2 * q / (1 + q))
    assert metrics["per_class"]["down"]["recall"] == 1
    assert metrics["down_false_negatives"] == 0
    assert sum(map(sum, metrics["confusion_matrix"]["values"])) == len(truth)
    assert sum(v["support"] for v in metrics["per_class"].values()) == len(truth)
    assert metrics["per_class"]["up"]["f1"] == 0


def test_missing_truth_class_preserves_fixed_label_macro():
    metrics = evaluate_predictions(["down"] * 3, ["down"] * 3, LABELS)
    assert metrics["macro_f1"] == 0.2
    assert metrics["balanced_accuracy"] == 1
    assert metrics["per_class"]["up"]["support"] == 0
    assert len(metrics["confusion_matrix"]["values"]) == 5


@pytest.mark.parametrize(
    ("truth", "predictions", "labels"),
    [
        ([], [], LABELS),
        (["down"], [], LABELS),
        (["down"], ["other"], LABELS),
        ([None], ["down"], LABELS),
        (["down"], ["down"], ["down", "down"]),
        (["down"], ["down"], []),
        (["down"], ["down"], ["down", None]),
    ],
)
def test_invalid_metric_inputs(truth, predictions, labels):
    with pytest.raises(DataValidationError):
        evaluate_predictions(truth, predictions, labels)
