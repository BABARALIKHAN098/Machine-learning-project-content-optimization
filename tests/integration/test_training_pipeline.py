from machine_learning_project.models.tune import select_candidate


def test_candidate_selection_obeys_down_recall_guardrail():
    results = {
        "high_f1_low_recall": {
            "macro_f1": 0.8,
            "weighted_f1": 0.8,
            "per_class": {"down": {"recall": 0.2}},
        },
        "eligible": {
            "macro_f1": 0.6,
            "weighted_f1": 0.7,
            "per_class": {"down": {"recall": 0.7}},
        },
    }
    assert select_candidate(results, down_recall_guardrail=0.5) == "eligible"
