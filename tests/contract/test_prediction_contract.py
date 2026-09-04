from machine_learning_project.inference.schemas import PredictionRecord


def test_prediction_record_contract():
    record = PredictionRecord(
        "content-1", "down", 0.8, {"down": 0.8, "up": 0.2}, "1.0.0"
    ).to_dict()
    assert set(record) == {
        "content_id",
        "predicted_trend",
        "probability_down",
        "probabilities",
        "model_version",
        "warning",
    }
    assert "Human review" in record["warning"]
    assert abs(sum(record["probabilities"].values()) - 1.0) < 1e-9
