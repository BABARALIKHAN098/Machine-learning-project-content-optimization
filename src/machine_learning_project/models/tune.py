from __future__ import annotations

from typing import Any


def select_candidate(results: dict[str, dict[str, Any]], down_recall_guardrail: float) -> str:
    """Select by macro F1, preferring candidates that meet the recall guardrail."""
    eligible = {
        name: metrics
        for name, metrics in results.items()
        if metrics.get("per_class", {}).get("down", {}).get("recall", 0.0) >= down_recall_guardrail
    }
    pool = eligible or results
    if not pool:
        raise ValueError("No candidate evaluation results were supplied.")
    return max(pool, key=lambda name: (pool[name]["macro_f1"], pool[name]["weighted_f1"], name))


def expand_trials(config):
    from itertools import product

    from ..features.registry import fingerprint

    trials = []
    for family in config["families"]:
        grid = config["grids"][family]
        keys = sorted(grid)
        for values in product(*(grid[key] for key in keys)):
            trial = {"family": family, "parameters": dict(zip(keys, values))}
            trial["trial_id"] = family + "-" + fingerprint(trial)[:16]
            trials.append(trial)
    return trials


def make_training_folds(frame, data, training, config, context):
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold

    from ..data.splitting import _hash_value
    from ..features.registry import fingerprint
    from ..utils.exceptions import DataValidationError

    group, row, target = training["group_column"], training["row_key"], data["target_column"]
    labels = data["allowed_target_values"]
    if (
        frame.empty
        or frame[group].isna().any()
        or frame[row].isna().any()
        or frame[row].duplicated().any()
        or frame[group].nunique() < config["folds"]
    ):
        raise DataValidationError("Invalid training identities or insufficient clients for folds")
    if set(frame[target]) != set(labels):
        raise DataValidationError("Training folds require all configured classes")
    splitter = StratifiedGroupKFold(
        n_splits=config["folds"], shuffle=True, random_state=config["random_seed"]
    )
    folds = list(splitter.split(np.zeros((len(frame), 1)), frame[target], frame[group]))
    coverage = np.zeros(len(frame), dtype=int)
    records = []
    for number, (left, right) in enumerate(folds):
        if set(frame.iloc[left][group]) & set(frame.iloc[right][group]):
            raise DataValidationError("Training fold clients overlap")
        if set(left) & set(right) or len(left) + len(right) != len(frame):
            raise DataValidationError("Training fold rows overlap or are incomplete")
        coverage[right] += 1
        record = {"fold": number}
        for name, positions in (("fit", left), ("score", right)):
            part = frame.iloc[positions]
            if set(part[target]) != set(labels):
                raise DataValidationError(
                    "Training fold lacks class coverage; do not search new seeds"
                )
            record[name] = {
                "rows": len(part),
                "groups": int(part[group].nunique()),
                "class_counts": {label: int(part[target].eq(label).sum()) for label in labels},
                "ordered_sha256": fingerprint([_hash_value(key, context) for key in part[row]]),
            }
        records.append(record)
    if not (coverage == 1).all():
        raise DataValidationError("Every training row must be scored once")
    return folds, {
        "algorithm": "StratifiedGroupKFold",
        "seed": config["random_seed"],
        "folds": records,
        "fitting_population": "train",
        "shuffle": True,
    }


def measured_fit_predict(pipeline, fit_x, fit_y, score_x, score_y, labels, context):
    """No diagnostics include raw exception strings, feature values or row identifiers."""
    import time
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from threadpoolctl import threadpool_limits

    from ..data.splitting import _hash_value
    from ..features.registry import fingerprint
    from .evaluate import evaluate_predictions

    timing = {}
    with warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always")
        try:
            with threadpool_limits(limits=1):
                started = time.perf_counter()
                pipeline.fit(fit_x, fit_y)
                timing["fit_seconds"] = time.perf_counter() - started
                started = time.perf_counter()
                predictions = pipeline.predict(score_x)
                timing["prediction_seconds"] = time.perf_counter() - started
            metrics = evaluate_predictions(score_y, predictions, labels)
            status = (
                "nonconverged"
                if any(issubclass(item.category, ConvergenceWarning) for item in observed)
                else "valid"
            )
            result = {
                "status": status,
                "metrics": metrics,
                "prediction_sha256": _hash_value(fingerprint(list(predictions)), context),
                "transformed_width": len(
                    pipeline.named_steps["preprocessor"].get_feature_names_out()
                ),
                "effective_parameters": pipeline.named_steps["model"].get_params(),
            }
        except Exception as error:  # noqa: BLE001 - retain failed trial status without private error text
            result = {"status": "failed", "error_type": type(error).__name__}
    result["warnings"] = [item.category.__name__ for item in observed]
    for category in sorted(set(result["warnings"])):
        if category != "ConvergenceWarning":
            warnings.warn(f"Trial emitted {category}; recorded in trial diagnostics", stacklevel=2)
    return result, timing


def summarize_trial(trial, folds):
    import numpy as np

    result = {**trial, "folds": folds, "status": "invalid"}
    if len(folds) != 3 or any(fold["status"] != "valid" for fold in folds):
        return result
    result["status"] = "valid"
    result["summary"] = {}
    for name in ("macro_f1", "down_recall"):
        values = [
            fold["metrics"]["macro_f1"]
            if name == "macro_f1"
            else fold["metrics"]["per_class"]["down"]["recall"]
            for fold in folds
        ]
        result["summary"][name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "minimum": min(values),
            "maximum": max(values),
        }
    return result


def rank_trials(trials, guardrail, tolerance, *, cross_family=False):
    import math

    from ..utils.exceptions import DataValidationError

    valid = [trial for trial in trials if trial["status"] == "valid"]
    if not valid:
        raise DataValidationError("No valid trial available for finalist selection")
    for trial in valid:
        for key in ("macro_f1", "down_recall"):
            value = trial["summary"][key]["mean"]
            if type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1:
                raise DataValidationError("Invalid CV selection score")
    eligible = [trial for trial in valid if trial["summary"]["down_recall"]["mean"] >= guardrail]
    pool = eligible or valid
    best = max(trial["summary"]["macro_f1"]["mean"] for trial in pool)
    tied = [trial for trial in pool if trial["summary"]["macro_f1"]["mean"] >= best - tolerance]

    def complexity(trial):
        params = trial["parameters"]
        if cross_family:
            return (trial["family"] != "logistic_regression", trial["trial_id"])
        if trial["family"] == "logistic_regression":
            return (params["model__C"], trial["trial_id"])
        depth = params["model__max_depth"]
        return (
            depth is None,
            depth or 0,
            -params["model__min_samples_leaf"],
            params["model__max_features"] != "sqrt",
            trial["trial_id"],
        )

    chosen = min(tied, key=complexity)
    return {
        "trial_id": chosen["trial_id"],
        "family": chosen["family"],
        "parameters": chosen["parameters"],
        "cv_summary": chosen["summary"],
        "recall_fallback_used": not bool(eligible),
        "tied_trial_ids": sorted(t["trial_id"] for t in tied),
        "reason": "CV macro F1 within tolerance, declared complexity, stable ID",
    }


def search_training(
    train, data, preprocessing, training, frozen, config, folds, context, progress=None
):
    from ..data.preparation import prepare_supervised_data
    from .train import build_trial

    prepared = prepare_supervised_data(train, data, feature_contract_version="2.0")
    trials, timings = [], []
    for trial in expand_trials(config):
        records = []
        for number, (left, right) in enumerate(folds):
            print(f"{trial['trial_id']} fold {number + 1}/{len(folds)}", flush=True)
            pipeline = build_trial(data, preprocessing, training, frozen[trial["family"]], trial)
            record, timing = measured_fit_predict(
                pipeline,
                prepared.features.iloc[left],
                prepared.target.iloc[left],
                prepared.features.iloc[right],
                prepared.target.iloc[right],
                data["allowed_target_values"],
                context,
            )
            records.append({"fold": number, **record})
            timings.append({"trial_id": trial["trial_id"], "fold": number, **timing})
        trials.append(summarize_trial(trial, records))
        if progress:
            progress(trials, timings)
    return trials, timings
