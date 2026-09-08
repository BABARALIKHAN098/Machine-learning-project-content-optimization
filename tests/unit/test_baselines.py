import numpy as np
import pytest

from machine_learning_project.models.baseline import evaluate_baselines, resolve_baseline_config
from machine_learning_project.utils.exceptions import DataValidationError

LABELS = ["down", "flat", "new", "stable", "up"]


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("strategies", ["uniform"]),
        ("strategies", ["stratified", "stratified"]),
        ("repeat_seeds", []),
        ("repeat_seeds", [42, 42]),
        ("repeat_seeds", [True]),
        ("repeat_seeds", [42, -1]),
        ("repeat_seeds", [42, 2**32]),
        ("repeat_seeds", [42, 43]),
        ("reference_seed", True),
        ("reference_seed", 43),
        ("metric_contract_version", "0.0"),
        ("baseline_contract_version", "2.0"),
        ("minimum_macro_f1_improvement", float("nan")),
        ("minimum_macro_f1_improvement", True),
        ("minimum_macro_f1_improvement", 0),
        ("evaluation_partition", "test"),
        ("ordering", "input"),
        ("output_directory", ""),
    ],
)
def test_invalid_protocol(key, value):
    config = resolve_baseline_config()
    config[key] = value
    with pytest.raises(DataValidationError):
        resolve_baseline_config(config)


def test_mapping_and_legacy_conflicts():
    with pytest.raises(DataValidationError):
        resolve_baseline_config([])
    with pytest.raises(DataValidationError):
        resolve_baseline_config(training_config={"baselines": ["uniform"]})


def test_training_priors_ties_repeats_and_rng_independence():
    config = resolve_baseline_config()
    train = LABELS * 3 + ["up", "down"]
    first = evaluate_baselines(train, LABELS * 7, config, LABELS)
    np.random.seed(987)
    second = evaluate_baselines(train, LABELS * 7, config, LABELS)
    first_timing, second_timing = first.pop("timings"), second.pop("timings")
    assert first == second
    assert len(first["runs"]) == 6
    assert first["training_priors"]["majority_class"] == "down"
    assert first["training_priors"]["counts"]["down"] == 4
    assert first["training_priors"]["proportions"]["up"] == 4 / 17
    changed = evaluate_baselines(train, LABELS + ["up"] * 20, config, LABELS)
    assert changed["training_priors"] == first["training_priors"]
    for key, extract in [
        ("macro_f1", lambda m: m["macro_f1"]),
        ("down_recall", lambda m: m["per_class"]["down"]["recall"]),
    ]:
        values = [extract(run["metrics"]) for run in first["runs"][1:]]
        summary = first["summary"]["stratified"][key]
        assert summary == {
            "mean": np.mean(values),
            "std": np.std(values),
            "minimum": min(values),
            "maximum": max(values),
        }
    assert first["canonical"]["stratified"] == first["runs"][1]["metrics"]
    for timing in first_timing + second_timing:
        assert timing["fit_seconds"] >= 0 and timing["prediction_seconds"] >= 0


@pytest.mark.parametrize("bad", [[], [None], ["unknown"], ["down"]])
def test_invalid_partitions(bad):
    for train, validation in [(bad, LABELS), (LABELS, bad)]:
        with pytest.raises(DataValidationError):
            evaluate_baselines(train, validation, resolve_baseline_config(), LABELS)


def test_fit_predict_once_with_neutral_features_and_separate_timing(monkeypatch):
    import machine_learning_project.models.baseline as module

    original = module.DummyClassifier
    calls = []

    class Observed(original):
        def fit(self, X, y, sample_weight=None):
            assert X.shape == (len(y), 1) and not X.any()
            assert sample_weight is None
            calls.append("fit")
            return super().fit(X, y)

        def predict(self, X):
            assert X.shape == (10, 1) and not X.any()
            calls.append("predict")
            return super().predict(X)

    ticks = iter(range(24))
    monkeypatch.setattr(module, "DummyClassifier", Observed)
    monkeypatch.setattr(module.time, "perf_counter", lambda: next(ticks))
    result = evaluate_baselines(LABELS, LABELS * 2, resolve_baseline_config(), LABELS)
    assert calls == ["fit", "predict"] * 6
    assert all(t["fit_seconds"] == t["prediction_seconds"] == 1 for t in result["timings"])
