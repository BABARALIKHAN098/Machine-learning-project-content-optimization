from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from ..utils.exceptions import DataValidationError


@dataclass(frozen=True)
class DatasetSplits:
    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    assignments: pd.DataFrame
    manifest: dict[str, Any]


def _hash_value(value: object, context: str) -> str:
    payload = f"{context}\0{value}".encode()
    return hashlib.sha256(payload).hexdigest()


def _distribution(frame: pd.DataFrame, target: str, labels: list[str]) -> dict[str, int]:
    counts = frame[target].astype(str).value_counts()
    return {label: int(counts.get(label, 0)) for label in labels}


def _distribution_proportions(
    frame: pd.DataFrame, target: str, labels: list[str]
) -> dict[str, float]:
    counts = frame[target].astype(str).value_counts(normalize=True)
    return {label: round(float(counts.get(label, 0.0)), 8) for label in labels}


def _distribution_error(frame: pd.DataFrame, target: str, reference: pd.Series) -> float:
    observed = frame[target].astype(str).value_counts(normalize=True)
    return float(sum(abs(observed.get(label, 0.0) - share) for label, share in reference.items()))


def _assignment_fingerprint(frame: pd.DataFrame, row_key: str, context: str) -> str:
    hashed = sorted(_hash_value(value, context) for value in frame[row_key].astype(str))
    return hashlib.sha256("\n".join(hashed).encode("utf-8")).hexdigest()


def _best_group_split(
    dataframe: pd.DataFrame,
    *,
    target: str,
    group: str,
    row_key: str,
    test_size: float,
    seed: int,
    attempts: int,
    size_weight: float,
    class_weight: float,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    labels = set(dataframe[target].astype(str))
    reference = dataframe[target].astype(str).value_counts(normalize=True)
    best: tuple[float, str, pd.DataFrame, pd.DataFrame] | None = None
    for offset in range(attempts):
        splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed + offset)
        left_index, right_index = next(
            splitter.split(dataframe, dataframe[target], groups=dataframe[group])
        )
        left = dataframe.iloc[left_index].copy()
        right = dataframe.iloc[right_index].copy()
        if set(left[target].astype(str)) != labels or set(right[target].astype(str)) != labels:
            continue
        size_error = abs(len(right) / len(dataframe) - test_size)
        class_error = _distribution_error(left, target, reference) + _distribution_error(
            right, target, reference
        )
        score = size_weight * size_error + class_weight * class_error
        tie_breaker = "|".join(sorted(right[row_key].astype(str)))
        if best is None or (score, tie_breaker) < (best[0], best[1]):
            best = (score, tie_breaker, left, right)
    if best is None:
        raise DataValidationError(
            f"Unable to create a group-disjoint split containing every target class "
            f"after {attempts} attempts."
        )
    return best[2], best[3], best[0]


def _validate_split_inputs(
    dataframe: pd.DataFrame,
    *,
    target: str,
    group: str,
    row_key: str,
    labels: list[str],
    test_size: float,
    validation_size: float,
    attempts: int,
) -> None:
    errors: list[str] = []
    if not 0 < test_size < 1 or not 0 < validation_size < 1 or test_size + validation_size >= 1:
        errors.append("test_size and validation_size must be positive and sum to less than 1")
    if attempts <= 0:
        errors.append("search_attempts must be positive")
    for column in (target, group, row_key):
        if column not in dataframe:
            errors.append(f"split column not found: {column}")
    if errors:
        raise DataValidationError("; ".join(errors))
    for column in (target, group, row_key):
        if dataframe[column].isna().any():
            errors.append(f"split column contains missing values: {column}")
    duplicate_count = int(dataframe[row_key].duplicated().sum())
    if duplicate_count:
        errors.append(f"row key {row_key} contains {duplicate_count} duplicate values")
    group_count = int(dataframe[group].nunique())
    if group_count < 3:
        errors.append(f"at least three groups are required, found {group_count}")
    observed = set(dataframe[target].astype(str))
    unknown = sorted(observed - set(labels))
    missing = sorted(set(labels) - observed)
    if unknown:
        errors.append(f"unknown target labels: {unknown}")
    if missing:
        errors.append(f"configured target labels absent from data: {missing}")
    for label in labels:
        distinct_groups = dataframe.loc[dataframe[target].astype(str) == label, group].nunique()
        if distinct_groups < 3:
            errors.append(
                f"target class {label!r} occurs in {distinct_groups} groups; at least 3 required"
            )
    if errors:
        raise DataValidationError("; ".join(errors))


def split_by_group(
    dataframe: pd.DataFrame,
    *,
    target_column: str,
    group_column: str,
    row_key: str | None = None,
    allowed_labels: list[str] | None = None,
    test_size: float = 0.2,
    validation_size: float = 0.2,
    random_seed: int = 42,
    search_attempts: int = 128,
    row_ratio_tolerance: float = 0.05,
    class_ratio_tolerance: float = 0.05,
    size_weight: float = 1.0,
    class_weight: float = 1.0,
    source_sha256: str = "unknown",
    data_schema_version: str = "unversioned",
    split_contract_version: str = "1.0",
    algorithm_version: str = "group-shuffle-search-v2",
    alias_context: str = "content-trend-split-v1",
) -> DatasetSplits:
    working = dataframe.copy(deep=True)
    internal_row_key = row_key or "__split_row_key__"
    if row_key is None:
        working[internal_row_key] = dataframe.index.astype(str)
    if target_column not in working:
        raise DataValidationError(f"split column not found: {target_column}")
    labels = allowed_labels or sorted(working[target_column].dropna().astype(str).unique())
    _validate_split_inputs(
        working,
        target=target_column,
        group=group_column,
        row_key=internal_row_key,
        labels=labels,
        test_size=test_size,
        validation_size=validation_size,
        attempts=search_attempts,
    )
    if min(row_ratio_tolerance, class_ratio_tolerance, size_weight, class_weight) < 0:
        raise DataValidationError("split tolerances and objective weights must be non-negative")

    working = working.sort_values(internal_row_key, kind="stable").reset_index(drop=True)
    development, test, test_score = _best_group_split(
        working,
        target=target_column,
        group=group_column,
        row_key=internal_row_key,
        test_size=test_size,
        seed=random_seed,
        attempts=search_attempts,
        size_weight=size_weight,
        class_weight=class_weight,
    )
    relative_validation_size = validation_size / (1.0 - test_size)
    train, validation, validation_score = _best_group_split(
        development,
        target=target_column,
        group=group_column,
        row_key=internal_row_key,
        test_size=relative_validation_size,
        seed=random_seed + 10_000,
        attempts=search_attempts,
        size_weight=size_weight,
        class_weight=class_weight,
    )
    frames = {"train": train, "validation": validation, "test": test}
    group_sets = {name: set(frame[group_column].astype(str)) for name, frame in frames.items()}
    row_sets = {name: set(frame[internal_row_key].astype(str)) for name, frame in frames.items()}
    names = list(frames)
    for index, left in enumerate(names):
        for right in names[index + 1 :]:
            if group_sets[left] & group_sets[right]:
                raise AssertionError(f"Group leakage detected between {left} and {right}.")
            if row_sets[left] & row_sets[right]:
                raise AssertionError(f"Row leakage detected between {left} and {right}.")
    if set().union(*group_sets.values()) != set(working[group_column].astype(str)):
        raise AssertionError("Dataset split omitted one or more groups.")
    if set().union(*row_sets.values()) != set(working[internal_row_key].astype(str)):
        raise AssertionError("Dataset split omitted one or more rows.")

    reference = working[target_column].astype(str).value_counts(normalize=True)
    intended = {"train": 1.0 - test_size - validation_size, "validation": validation_size, "test": test_size}
    context = f"{alias_context}:{source_sha256}"
    partition_details: dict[str, Any] = {}
    assignment_rows: list[dict[str, str]] = []
    tolerance_violations: list[str] = []
    for name, frame in frames.items():
        actual_ratio = len(frame) / len(working)
        proportions = _distribution_proportions(frame, target_column, labels)
        class_deviations = {
            label: round(abs(proportions[label] - float(reference.get(label, 0.0))), 8)
            for label in labels
        }
        ratio_deviation = abs(actual_ratio - intended[name])
        if ratio_deviation > row_ratio_tolerance:
            tolerance_violations.append(
                f"{name} row-ratio deviation {ratio_deviation:.4f} exceeds {row_ratio_tolerance:.4f}"
            )
        if max(class_deviations.values(), default=0.0) > class_ratio_tolerance:
            tolerance_violations.append(
                f"{name} class deviation exceeds {class_ratio_tolerance:.4f}"
            )
        aliases = sorted({_hash_value(value, context)[:16] for value in frame[group_column]})
        partition_details[name] = {
            "row_count": len(frame),
            "actual_row_ratio": round(actual_ratio, 8),
            "intended_row_ratio": intended[name],
            "row_ratio_deviation": round(ratio_deviation, 8),
            "group_count": int(frame[group_column].nunique()),
            "group_aliases": aliases,
            "class_distribution": _distribution(frame, target_column, labels),
            "class_proportions": proportions,
            "class_ratio_deviations": class_deviations,
            "assignment_fingerprint": _assignment_fingerprint(frame, internal_row_key, context),
        }
        for row in frame[[internal_row_key, group_column]].itertuples(index=False, name=None):
            assignment_rows.append(
                {
                    "row_key_hash": _hash_value(row[0], context),
                    "group_alias": _hash_value(row[1], context)[:16],
                    "partition": name,
                    "split_contract_version": split_contract_version,
                }
            )
    if tolerance_violations:
        raise DataValidationError("Split quality tolerances not met: " + "; ".join(tolerance_violations))

    assignments = pd.DataFrame(assignment_rows).sort_values("row_key_hash").reset_index(drop=True)
    manifest = {
        "manifest_schema_version": "1.0",
        "split_contract_version": split_contract_version,
        "algorithm_version": algorithm_version,
        "source": {
            "source_sha256": source_sha256,
            "data_schema_version": data_schema_version,
            "row_count": len(working),
            "group_count": int(working[group_column].nunique()),
            "target_labels": labels,
        },
        "parameters": {
            "random_seed": random_seed,
            "target_column": target_column,
            "group_column": group_column,
            "row_key": row_key or "dataframe_index",
            "test_size": test_size,
            "validation_size": validation_size,
            "search_attempts": search_attempts,
            "row_ratio_tolerance": row_ratio_tolerance,
            "class_ratio_tolerance": class_ratio_tolerance,
            "objective_weights": {"size": size_weight, "class": class_weight},
        },
        "search_scores": {"test": round(test_score, 8), "validation": round(validation_score, 8)},
        "partitions": partition_details,
        "invariants": {
            "row_complete": True,
            "row_disjoint": True,
            "group_complete": True,
            "group_disjoint": True,
            "class_coverage": True,
        },
        "limitations": [
            "Client-grouped snapshot validation does not measure future-period performance."
        ],
    }
    if row_key is None:
        frames = {name: frame.drop(columns=[internal_row_key]) for name, frame in frames.items()}
    return DatasetSplits(
        train=frames["train"],
        validation=frames["validation"],
        test=frames["test"],
        assignments=assignments,
        manifest=manifest,
    )


def _atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def write_split_artifacts(
    splits: DatasetSplits, manifest_path: str | Path, assignments_path: str | Path
) -> tuple[Path, Path]:
    manifest = Path(manifest_path)
    assignments = Path(assignments_path)
    _atomic_write(manifest, json.dumps(splits.manifest, indent=2, sort_keys=True) + "\n")
    _atomic_write(assignments, splits.assignments.to_csv(index=False, lineterminator="\n"))
    return manifest, assignments
