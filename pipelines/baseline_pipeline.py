"""Standalone class-frequency benchmark: no candidate or feature-pipeline imports."""

from __future__ import annotations

import json
import platform
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn

from machine_learning_project.data.development import load_development
from machine_learning_project.data.ingestion import sha256_file
from machine_learning_project.data.splitting import _atomic_write
from machine_learning_project.features.registry import fingerprint
from machine_learning_project.models.baseline import evaluate_baselines, resolve_baseline_config
from machine_learning_project.models.benchmark import prepare_benchmark_partitions
from machine_learning_project.utils.config import validate_data_config, validate_split_config
from machine_learning_project.utils.exceptions import DataValidationError


def run_baselines(data_config, training_config, baseline_config=None, output_dir=None):
    config = resolve_baseline_config(baseline_config, training_config)
    validate_data_config(data_config)
    validate_split_config(training_config)
    train, validation, provenance = load_development(data_config, training_config)
    train, validation, identity = prepare_benchmark_partitions(
        train, validation, data_config, training_config, provenance
    )
    target = data_config["target_column"]
    evaluated = evaluate_baselines(train[target], validation[target], config, identity["labels"])
    timings = evaluated.pop("timings")
    result = {
        **evaluated,
        "baseline_contract_version": config["baseline_contract_version"],
        "evaluation_identity": identity,
        "test_accessed": False,
    }
    repeats = []
    for run, timing in zip(result["runs"], timings):
        metrics = run["metrics"]
        repeats.append(
            {
                **timing,
                **{
                    key: metrics[key]
                    for key in (
                        "macro_f1",
                        "weighted_f1",
                        "accuracy",
                        "balanced_accuracy",
                        "down_false_negatives",
                    )
                },
                "down_recall": metrics["per_class"]["down"]["recall"],
            }
        )
    report = [
        "# Baseline benchmark",
        "",
        "Development validation only. No feature or candidate model fitting.",
        (
            f"Training: {identity['train_rows']} rows / {identity['train_groups']} clients. "
            f"Validation: {identity['validation_rows']} rows / {identity['validation_groups']} clients."
        ),
        "",
        "| Baseline | Macro F1 | Weighted F1 | Accuracy | Balanced accuracy | Down recall |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, metrics in result["canonical"].items():
        report.append(
            f"| {name} | {metrics['macro_f1']:.6f} | {metrics['weighted_f1']:.6f} | "
            f"{metrics['accuracy']:.6f} | {metrics['balanced_accuracy']:.6f} | "
            f"{metrics['per_class']['down']['recall']:.6f} |"
        )
    summary = result["summary"]["stratified"]["macro_f1"]
    report.extend(
        [
            "",
            f"Reference seed: {config['reference_seed']}; repeat seeds: {config['repeat_seeds']}.",
            (
                f"Stratified macro F1 mean {summary['mean']:.6f}, population SD {summary['std']:.6f}, "
                f"range {summary['minimum']:.6f} to {summary['maximum']:.6f}."
            ),
            "Seed dispersion describes classifier randomness, not a generalization confidence interval.",
            "",
            (
                f"Training majority: {result['training_priors']['majority_class']} "
                "(ties: lexicographically first). Class priors use training labels only."
            ),
            "A majority-down model has down recall 1 despite failing to discriminate other classes.",
            (
                "Per-class metrics and ordered confusion matrices are in baseline_metrics.json. "
                "Unpredicted classes retain zero precision/recall/F1; main partitions contain every class."
            ),
            "",
            (
                f"Adopted material improvement: +{config['minimum_macro_f1_improvement']:.3f} "
                "absolute macro F1 over majority, canonical stratified and stratified mean. "
                "Recall and project-target checks are separate from this comparison."
            ),
            "",
            (
                "Fit/prediction timings are diagnostic and exclude metrics computation; "
                "dummy timing does not establish the full model's 30,000-row runtime."
            ),
            (
                "The validation set was previously used for feature confirmation and the test holdout "
                "was previously evaluated. These results are a development benchmark, not an independent "
                "future-period estimate. Historical metadata cutoff evidence remains unverified."
            ),
            "",
        ]
    )
    output = Path(output_dir or config["output_directory"])
    payloads = {
        "baseline_metrics.json": json.dumps(result, indent=2, sort_keys=True, allow_nan=False)
        + "\n",
        "baseline_repeats.csv": pd.DataFrame(repeats).to_csv(index=False, lineterminator="\n"),
        "baseline_summary.json": json.dumps(result["summary"], indent=2, sort_keys=True) + "\n",
        "baseline_report.md": "\n".join(report),
    }
    root = Path(__file__).resolve().parents[1]
    code = {
        "source_tree_sha256": fingerprint(
            {
                str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
                for folder in ("src", "pipelines", "scripts")
                for path in sorted((root / folder).rglob("*.py"))
            }
        )
    }
    try:
        code["revision"] = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True, cwd=root
        ).stdout.strip()
        code["dirty"] = bool(
            subprocess.run(
                ["git", "status", "--porcelain"],
                capture_output=True,
                text=True,
                check=True,
                cwd=root,
            ).stdout.strip()
        )
    except (OSError, subprocess.CalledProcessError):
        code.update({"revision": "unavailable", "dirty": None})
    manifest = {
        "manifest_schema_version": "1.0",
        "baseline_contract_version": config["baseline_contract_version"],
        "evaluation_identity": identity,
        "config": config,
        "config_sha256": fingerprint(
            {key: value for key, value in config.items() if key != "output_directory"}
        ),
        "code": code,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "sklearn": sklearn.__version__,
        },
    }
    for path, expected in (
        (Path(data_config["csv_path"]), provenance["source_sha256"]),
        (Path(training_config["split_manifest_path"]), provenance["split_manifest_sha256"]),
        (Path(training_config["split_assignments_path"]), provenance["split_assignments_sha256"]),
    ):
        if sha256_file(path) != expected:
            raise DataValidationError(
                "Source or split changed during benchmark; outputs not published"
            )
    for name, text in payloads.items():
        _atomic_write(output / name, text)
    manifest["artifact_hashes"] = {name: sha256_file(output / name) for name in payloads}
    _atomic_write(
        output / "baseline_manifest.json", json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return result
