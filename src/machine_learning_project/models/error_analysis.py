"""Pure aggregate diagnostics on aligned frozen predictions; no fitting or row exports."""

from collections import Counter

import numpy as np
import pandas as pd

from ..data.splitting import _hash_value
from ..features.registry import fingerprint
from ..utils.exceptions import DataValidationError
from .evaluate import evaluate_predictions
from .training_artifacts import FAMILIES


def rate(numerator, denominator):
    return {
        "numerator": int(numerator),
        "denominator": int(denominator),
        "value": numerator / denominator if denominator else None,
        "status": "supported" if denominator else "undefined_zero_denominator",
    }


def analyze_errors(truth, predictions, labels):
    metrics = evaluate_predictions(truth, predictions, labels)
    truth, predictions = np.asarray(truth), np.asarray(predictions)
    matrix = metrics["confusion_matrix"]["values"]
    rows, normalized, off_diagonal = [], [], []
    for i, label in enumerate(labels):
        support, predicted = sum(matrix[i]), sum(row[i] for row in matrix)
        rows.append(
            {
                "class": label,
                **metrics["per_class"][label],
                "predicted_support": predicted,
                "false_negatives": support - matrix[i][i],
                "false_positives": predicted - matrix[i][i],
            }
        )
        normalized.append([v / support if support else None for v in matrix[i]])
        for j, other in enumerate(labels):
            if i != j:
                off_diagonal.append({"truth": label, "prediction": other, "count": matrix[i][j]})
    tp = int(((truth == "down") & (predictions == "down")).sum())
    fp = int(((truth != "down") & (predictions == "down")).sum())
    fn = int(((truth == "down") & (predictions != "down")).sum())
    tn = len(truth) - tp - fp - fn
    return {
        "metrics": metrics,
        "classes": rows,
        "confusion": {
            "labels": list(labels),
            "counts": matrix,
            "row_normalized": normalized,
            "undefined_rows": [r["class"] for r in rows if not r["support"]],
        },
        "off_diagonal": sorted(
            off_diagonal,
            key=lambda r: (-r["count"], labels.index(r["truth"]), labels.index(r["prediction"])),
        ),
        "down": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "tn": tn,
            "recall": rate(tp, tp + fn),
            "precision": rate(tp, tp + fp),
            "false_negative_rate": rate(fn, tp + fn),
            "false_positive_rate": rate(fp, fp + tn),
            "workload": rate(tp + fp, len(truth)),
            "missed_down": rate(fn, len(truth)),
            "false_alert_sources": {
                label: int(((truth == label) & (predictions == "down")).sum())
                for label in labels
                if label != "down"
            },
        },
    }


def _paired(truth, predictions_by_family, labels):
    if set(predictions_by_family) != set(FAMILIES):
        raise DataValidationError("Both aligned finalists are required")
    metrics = {f: evaluate_predictions(truth, predictions_by_family[f], labels) for f in FAMILIES}
    return np.asarray(truth), {f: np.asarray(predictions_by_family[f]) for f in FAMILIES}, metrics


def compare_paired_predictions(truth, predictions_by_family, labels):
    truth, predictions, metrics = _paired(truth, predictions_by_family, labels)
    a, b = [predictions[f] for f in FAMILIES]
    counts = Counter(zip(truth, a, b))
    for family, position in zip(FAMILIES, (1, 2)):
        marginal = [
            [
                sum(n for cell, n in counts.items() if cell[0] == t and cell[position] == p)
                for p in labels
            ]
            for t in labels
        ]
        if marginal != metrics[family]["confusion_matrix"]["values"]:
            raise DataValidationError("Paired joint-table marginal mismatch")
    joint = [
        {"truth": t, "logistic_prediction": x, "forest_prediction": y, "count": counts[t, x, y]}
        for t in labels
        for x in labels
        for y in labels
    ]

    def partition(x, y):
        return {
            "both": int((x & y).sum()),
            "only_logistic": int((x & ~y).sum()),
            "only_forest": int((~x & y).sum()),
            "neither": int((~x & ~y).sum()),
        }

    down = truth == "down"
    return {
        "row_count": len(truth),
        "both_confusion_marginals_verified": True,
        "correctness": partition(a == truth, b == truth),
        "true_down": partition(a[down] == "down", b[down] == "down"),
        "agreement": int((a == b).sum()),
        "disagreement": int((a != b).sum()),
        "joint_table": joint,
        "delta_convention": "forest_minus_logistic",
        "macro_f1_delta": metrics[FAMILIES[1]]["macro_f1"] - metrics[FAMILIES[0]]["macro_f1"],
        "down_recall_delta": (
            metrics[FAMILIES[1]]["per_class"]["down"]["recall"]
            - metrics[FAMILIES[0]]["per_class"]["down"]["recall"]
        ),
    }


def subgroup_memberships(context, config, numeric_columns):
    """Typed internal keys; aliases prevent identifier-like category values leaking out."""
    result = {}
    for dimension in config["dimensions"]:
        if dimension == "previous_impressions_bucket":
            if "impressions_prev_30d" not in context:
                raise DataValidationError("Missing approved impressions context")
            values = pd.to_numeric(context["impressions_prev_30d"], errors="raise")
            if ((values.dropna() < 0) | ~np.isfinite(values.dropna())).any():
                raise DataValidationError("Invalid impressions context")
            keys = [
                ("missing", "")
                if pd.isna(v)
                else (
                    "bin",
                    "zero"
                    if v == 0
                    else "(0,100]"
                    if v <= 100
                    else "(100,1000]"
                    if v <= 1000
                    else ">1000",
                )
                for v in values
            ]
        elif dimension == "numeric_missingness_bucket":
            if not numeric_columns or not set(numeric_columns) <= set(context.columns):
                raise DataValidationError("Missing approved numeric context")
            values = context[numeric_columns].isna().sum(axis=1)
            keys = [("bin", "0" if v == 0 else "1-2" if v <= 2 else ">=3") for v in values]
        else:
            if dimension not in context:
                raise DataValidationError("Missing approved categorical context")
            keys = [
                ("missing", "") if pd.isna(v) else ("value", str(v)) for v in context[dimension]
            ]
        counts = Counter(keys)
        eligible = sorted(
            (k for k in counts if k[0] == "value" and counts[k] >= config["minimum_subgroup_rows"]),
            key=lambda k: (-counts[k], k),
        )[: config["maximum_categories_per_dimension"]]
        aliases = {k: ("category", f"category_{i + 1:02d}") for i, k in enumerate(eligible)}
        result[dimension] = [
            aliases.get(k, ("pooled", "OTHER")) if k[0] == "value" else k for k in keys
        ]
    return result


def analyze_subgroups(
    context, truth, predictions_by_family, labels, config, *, numeric_columns, down_recall_guardrail
):
    truth, predictions, whole = _paired(truth, predictions_by_family, labels)
    if len(context) != len(truth):
        raise DataValidationError("Subgroup context must be positionally aligned")
    memberships = subgroup_memberships(context, config, numeric_columns)
    records, coverage, alerts = [], {}, []
    for dimension, keys in memberships.items():
        counts = Counter(keys)
        published = {k for k, n in counts.items() if n >= config["minimum_subgroup_rows"]}
        coverage[dimension] = {
            "total_rows": len(truth),
            "published_rows": sum(counts[k] for k in published),
            "suppressed_rows": sum(n for k, n in counts.items() if k not in published),
            "published_groups": len(published),
            "suppressed_groups": len(counts) - len(published),
            "pooled_rows": sum(n for k, n in counts.items() if k[0] == "pooled"),
            "missing_rows": sum(n for k, n in counts.items() if k[0] == "missing"),
            "membership_sha256": fingerprint(keys),
            "category_policy": "frequency_then_value; anonymous category aliases; typed sentinels",
        }
        for key in sorted(published):
            mask = np.asarray([k == key for k in keys])
            for family in FAMILIES:
                analysis = analyze_errors(truth[mask], predictions[family][mask], labels)
                m, down = analysis["metrics"], analysis["down"]
                supports = {label: m["per_class"][label]["support"] for label in labels}
                macro_supported = min(supports.values()) >= config["minimum_class_support"]
                recall_supported = supports["down"] >= config["minimum_down_support"]
                precision_supported = (
                    down["tp"] + down["fp"] >= config["minimum_predicted_down_support"]
                )
                macro_alert = macro_supported and whole[family]["macro_f1"] >= (
                    m["macro_f1"] + config["subgroup_alert_macro_f1_gap"]
                )
                recall_alert = recall_supported and down["recall"]["value"] < down_recall_guardrail
                row = {
                    "dimension": dimension,
                    "group_kind": key[0],
                    "group": key[1],
                    "family": family,
                    "row_count": int(mask.sum()),
                    "status": "published",
                    "macro_f1": m["macro_f1"],
                    "balanced_accuracy": m["balanced_accuracy"],
                    "all_classes_present": min(supports.values()) > 0,
                    "macro_supported": macro_supported,
                    "down_recall_supported": recall_supported,
                    "down_precision_supported": precision_supported,
                    "down_recall": down["recall"]["value"] if recall_supported else None,
                    "down_precision": down["precision"]["value"] if precision_supported else None,
                    "down_false_negatives": down["fn"],
                    "predicted_down": down["tp"] + down["fp"],
                    "macro_alert": macro_alert,
                    "down_recall_alert": recall_alert,
                }
                for label in labels:
                    row[f"support_{label}"] = supports[label]
                    row[f"interpretable_{label}"] = (
                        supports[label] >= config["minimum_class_support"]
                    )
                    for metric in ("precision", "recall", "f1"):
                        row[f"{metric}_{label}"] = m["per_class"][label][metric]
                records.append(row)
                if macro_alert or recall_alert:
                    alerts.append(
                        {
                            k: row[k]
                            for k in (
                                "dimension",
                                "group",
                                "family",
                                "row_count",
                                "macro_alert",
                                "down_recall_alert",
                            )
                        }
                    )
    return {"records": records, "coverage": coverage, "alerts": alerts}


def client_sensitivity(groups, truth, predictions_by_family, labels, config, *, hash_context):
    truth, predictions, whole = _paired(truth, predictions_by_family, labels)
    groups = np.asarray(groups)
    if len(groups) != len(truth) or pd.isna(groups).any():
        raise DataValidationError("Client context must be complete and aligned")
    clients = sorted(set(groups))
    scores = {f: [] for f in FAMILIES}
    recalls = {f: [] for f in FAMILIES}
    rows, gaps, recall_gaps = [], [], []
    complete = 0
    for client in clients:
        mask = groups != client
        if not mask.any():
            continue
        rows.append(int(mask.sum()))
        m = {f: evaluate_predictions(truth[mask], predictions[f][mask], labels) for f in FAMILIES}
        supports = m[FAMILIES[0]]["per_class"]
        supported = min(v["support"] for v in supports.values()) >= config["minimum_class_support"]
        complete += int(supported)
        for f in FAMILIES:
            if supported:
                scores[f].append(m[f]["macro_f1"])
            if supports["down"]["support"] >= config["minimum_down_support"]:
                recalls[f].append(m[f]["per_class"]["down"]["recall"])
        if supported:
            gaps.append(m[FAMILIES[1]]["macro_f1"] - m[FAMILIES[0]]["macro_f1"])
        if supports["down"]["support"] >= config["minimum_down_support"]:
            recall_gaps.append(
                m[FAMILIES[1]]["per_class"]["down"]["recall"]
                - m[FAMILIES[0]]["per_class"]["down"]["recall"]
            )

    def span(values):
        return {
            "minimum": min(values) if values else None,
            "maximum": max(values) if values else None,
            "supported_scenarios": len(values),
        }

    whole_gap = whole[FAMILIES[1]]["macro_f1"] - whole[FAMILIES[0]]["macro_f1"]
    return {
        "method": "descriptive_leave_one_client_out_no_refit",
        "client_count": len(clients),
        "scenario_count": len(rows),
        "remaining_rows": span(rows),
        "all_class_supported_scenarios": complete,
        "macro_f1": {f: span(scores[f]) for f in FAMILIES},
        "down_recall": {f: span(recalls[f]) for f in FAMILIES},
        "paired_macro_f1_gap": span(gaps),
        "paired_down_recall_gap": span(recall_gaps),
        "ordering_reverses": any(g * whole_gap < 0 for g in gaps),
        "assignment_sha256": _hash_value(fingerprint(list(groups)), hash_context),
    }
