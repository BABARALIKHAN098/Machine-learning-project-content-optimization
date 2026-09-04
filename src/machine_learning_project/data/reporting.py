from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from .eda import EDAResults


def _atomic_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(content)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def _write_csv(path: Path, dataframe: pd.DataFrame) -> None:
    _atomic_text(path, dataframe.to_csv(index=False, lineterminator="\n"))


def render_eda_markdown(results: EDAResults, figure_paths: list[Path]) -> str:
    target = results.summary["target"]
    anomalies = results.anomalies.query("violation_count > 0")
    high_correlations = results.associations.query(
        "association_type == 'numeric_pair' and high_association"
    )
    missing = results.missingness_summary.query("kind == 'column' and count > 0").sort_values(
        "percentage", ascending=False
    )
    leakage = results.leakage_register.query("model_eligible == False")
    lines = [
        "# Exploratory Data Analysis",
        "",
        "## Run manifest",
        "",
        f"- Rows: {results.summary['row_count']:,}",
        f"- Columns: {results.summary['column_count']}",
        f"- Source SHA-256: `{results.summary['source_sha256']}`",
        f"- Artifact schema: `{results.summary['artifact_schema_version']}`",
        "",
        "## Target distribution",
        "",
        "| Class | Rows | Proportion |",
        "| --- | ---: | ---: |",
    ]
    for label, count in target["counts"].items():
        lines.append(f"| {label} | {count:,} | {target['proportions'][label]:.2%} |")
    lines.extend(
        [
            "",
            (
                f"Imbalance ratio: {target['imbalance_ratio']:.3f}. "
                f"Entropy: {target['entropy_bits']:.3f} bits."
            ),
            "",
            "## Material data-quality findings",
            "",
        ]
    )
    if missing.empty:
        lines.append("- No missing values were detected.")
    else:
        for row in missing.head(10).itertuples():
            lines.append(f"- `{row.name}`: {row.count:,} missing ({row.percentage:.2f}%).")
    if anomalies.empty:
        lines.append("- No configured anomaly rule reported violations.")
    else:
        for row in anomalies.itertuples():
            lines.append(
                f"- `{row.rule_id}` ({row.severity}): {row.violation_count:,} violations "
                f"({row.violation_percentage:.2f}%)."
            )
    lines.extend(["", "## Relationships and redundancy", ""])
    if high_correlations.empty:
        lines.append("No numeric pair exceeded the configured association threshold.")
    else:
        for row in high_correlations.head(15).itertuples():
            lines.append(
                f"- `{row.left}` / `{row.right}`: Pearson {row.pearson:.3f}, "
                f"Spearman {row.spearman:.3f}."
            )
    lines.extend(
        [
            "",
            "## Leakage review",
            "",
            (
                f"{len(leakage)} columns are prohibited or unavailable as model inputs. See "
                "`leakage_register.csv` for the field-level rationale."
            ),
            "",
            (
                "Target-derived and outcome-window associations describe label construction or "
                "contemporaneous relationships; they are not evidence of deployable predictive signal."
            ),
            "",
            "## Figures",
            "",
        ]
    )
    lines.extend(f"- `{path.as_posix()}`" for path in figure_paths)
    lines.extend(
        [
            "",
            "## Limitations and decisions",
            "",
            "- This analysis is descriptive and does not establish causation.",
            "- The source is a snapshot; prospective validity requires timestamped feature snapshots.",
            "- Client identifiers are masked in cohort artifacts and excluded from category summaries.",
            "- Rate domains and target/tier formulas require stakeholder confirmation.",
            "- No row was removed, corrected, imputed, or otherwise mutated by EDA.",
            "",
        ]
    )
    return "\n".join(lines)


def write_eda_artifacts(
    results: EDAResults, output_directory: str | Path, figure_paths: list[Path]
) -> list[Path]:
    output = Path(output_directory)
    paths = {
        "summary": output / "eda_summary.json",
        "numeric": output / "numeric_summary.csv",
        "categorical": output / "categorical_summary.csv",
        "crosstabs": output / "target_crosstabs.csv",
        "missingness": output / "missingness_summary.csv",
        "associations": output / "associations.csv",
        "anomalies": output / "anomalies.csv",
        "leakage": output / "leakage_register.csv",
        "report": output / "eda_report.md",
    }
    _atomic_text(paths["summary"], json.dumps(results.summary, indent=2, sort_keys=True) + "\n")
    _write_csv(paths["numeric"], results.numeric_summary)
    _write_csv(paths["categorical"], results.categorical_summary)
    _write_csv(paths["crosstabs"], results.target_crosstabs)
    _write_csv(paths["missingness"], results.missingness_summary)
    _write_csv(paths["associations"], results.associations)
    _write_csv(paths["anomalies"], results.anomalies)
    _write_csv(paths["leakage"], results.leakage_register)
    _atomic_text(paths["report"], render_eda_markdown(results, figure_paths))
    return list(paths.values())
