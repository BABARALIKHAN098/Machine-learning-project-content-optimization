from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def _save(path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, metadata={"Software": "machine-learning-project"})
    plt.close()


def render_eda_figures(
    dataframe: pd.DataFrame,
    data_config: dict[str, Any],
    eda_config: dict[str, Any],
    output_directory: str | Path,
) -> list[Path]:
    output = Path(output_directory)
    dpi = int(eda_config["plot_dpi"])
    paths: list[Path] = []

    target = data_config["target_column"]
    counts = dataframe[target].astype("string").value_counts().sort_index()
    counts.plot(kind="bar", color="#2563eb", title="Target distribution")
    plt.ylabel("Rows")
    target_path = output / "target_distribution.png"
    _save(target_path, dpi)
    paths.append(target_path)

    missing = (dataframe.isna().mean() * 100).sort_values(ascending=False)
    missing = missing[missing > 0]
    if not missing.empty:
        missing.plot(kind="bar", color="#f59e0b", title="Missing values by column")
        plt.ylabel("Missing (%)")
        missing_path = output / "missingness.png"
        _save(missing_path, dpi)
        paths.append(missing_path)

    for column in eda_config.get("priority_numeric_columns", []):
        if column not in dataframe:
            continue
        values = pd.to_numeric(dataframe[column], errors="coerce").dropna()
        if values.empty:
            continue
        values.plot(kind="hist", bins=30, color="#10b981", title=f"Distribution: {column}")
        plt.xlabel(column)
        path = output / f"distribution_{column}.png"
        _save(path, dpi)
        paths.append(path)

    numeric = dataframe.select_dtypes(include="number")
    if len(numeric.columns) > 1:
        matrix = numeric.corr(method="spearman")
        figure, axis = plt.subplots(figsize=(12, 10))
        image = axis.imshow(matrix, vmin=-1, vmax=1, cmap="coolwarm")
        axis.set_xticks(range(len(matrix.columns)), matrix.columns, rotation=90, fontsize=6)
        axis.set_yticks(range(len(matrix.columns)), matrix.columns, fontsize=6)
        axis.set_title("Spearman correlation matrix")
        figure.colorbar(image, ax=axis, shrink=0.7)
        correlation_path = output / "spearman_correlations.png"
        _save(correlation_path, dpi)
        paths.append(correlation_path)
    return paths
