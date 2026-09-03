from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DataLoadResult:
    dataframe: pd.DataFrame
    source_path: Path
    sha256: str


def sha256_file(path: Path, chunk_size: int = 1_048_576) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_csv(config: dict[str, Any]) -> DataLoadResult:
    path = Path(config["csv_path"])
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    if path.stat().st_size == 0:
        raise ValueError(f"CSV file is empty: {path}")

    dataframe = pd.read_csv(
        path,
        sep=config.get("delimiter", ","),
        encoding=config.get("encoding", "utf-8"),
        na_values=config.get("missing_value_tokens", []),
    )
    if dataframe.empty:
        raise ValueError("CSV contains a header but no data rows.")
    if len(dataframe.columns) < 2:
        raise ValueError("CSV must contain at least two columns.")

    normalized = [str(column).strip() for column in dataframe.columns]
    if any(not column for column in normalized):
        raise ValueError("CSV contains a blank column name.")
    if len(set(normalized)) != len(normalized):
        raise ValueError("CSV contains duplicate column names after normalization.")
    dataframe.columns = normalized

    return DataLoadResult(dataframe, path, sha256_file(path))
