#!/usr/bin/env python3
"""Create a specification-driven machine-learning project scaffold.

Usage:
    python create_ml_project_structure.py
    python create_ml_project_structure.py --name delivery-risk-ml
    python create_ml_project_structure.py --name my-project --destination C:/projects

The script never overwrites existing files unless --overwrite is supplied.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from textwrap import dedent


DEFAULT_PROJECT_NAME = "machine-learning-project"


def normalize_project_name(value: str) -> str:
    """Return a safe kebab-case project directory name."""
    normalized = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip()).strip("-").lower()
    if not normalized:
        raise argparse.ArgumentTypeError("Project name must contain letters or numbers.")
    return normalized


def package_name(project_name: str) -> str:
    """Convert a project directory name into a valid Python package name."""
    result = re.sub(r"[^a-zA-Z0-9]+", "_", project_name).strip("_").lower()
    if result[0].isdigit():
        result = f"ml_{result}"
    return result


def spec_placeholder(spec_id: str, title: str, outcome: str) -> str:
    return dedent(
        f"""\
        # {spec_id} — {title}

        **Status:** Draft  
        **Owner:** Babar Ali Khan  
        **Outcome:** {outcome}

        ## Purpose

        Define what this project phase must achieve before implementation begins.

        ## In Scope

        - TBD

        ## Out of Scope

        - TBD

        ## Requirements

        | ID | Requirement |
        | --- | --- |
        | {spec_id}-REQ-001 | TBD |

        ## Tests

        | Test ID | Given | When | Then |
        | --- | --- | --- | --- |
        | {spec_id}-TEST-001 | TBD | TBD | TBD |

        ## Acceptance Criteria

        - [ ] Requirements are approved.
        - [ ] Tests pass.
        - [ ] Decisions and limitations are documented.

        ## Open Decisions

        - TBD

        ## Definition of Done

        This phase is complete when every acceptance criterion is verified.
        """
    )


def build_files(project_name: str) -> dict[str, str]:
    """Return every starter file and its content."""
    package = package_name(project_name)

    files = {
        "README.md": dedent(
            f"""\
            # {project_name}

            A specification-driven machine-learning project owned by Babar Ali Khan.

            ## Development flow

            1. Approve the relevant file in `specs/`.
            2. Write or update tests from its acceptance criteria.
            3. Implement the smallest approved scope.
            4. Run the test suite and verify the acceptance criteria.
            5. Record artifacts, results, assumptions, and limitations.

            ## Setup

            ```bash
            python -m venv .venv
            # Windows: .venv\\Scripts\\activate
            # macOS/Linux: source .venv/bin/activate
            pip install -e ".[dev]"
            ```

            ## Run the first data validation

            1. Put the original CSV at `data/raw/dataset.csv`.
            2. Update `configs/data.yaml` with the target and column roles.
            3. Run:

            ```bash
            python scripts/validate_data.py --config configs/data.yaml
            pytest
            ```

            Never modify the source file in `data/raw/`. Learned preprocessing
            objects must be fitted on training data only.
            """
        ),
        "pyproject.toml": dedent(
            f"""\
            [build-system]
            requires = ["setuptools>=69", "wheel"]
            build-backend = "setuptools.build_meta"

            [project]
            name = "{project_name}"
            version = "0.1.0"
            description = "A specification-driven machine-learning project"
            readme = "README.md"
            requires-python = ">=3.11"
            dependencies = [
              "joblib>=1.4",
              "pandas>=2.2",
              "pyyaml>=6.0",
              "scikit-learn>=1.5",
            ]

            [project.optional-dependencies]
            dev = [
              "pytest>=8.0",
              "pytest-cov>=5.0",
              "ruff>=0.6",
            ]
            api = [
              "fastapi>=0.115",
              "uvicorn[standard]>=0.30",
            ]

            [tool.setuptools.packages.find]
            where = ["src"]

            [tool.pytest.ini_options]
            testpaths = ["tests"]
            addopts = "-q"

            [tool.ruff]
            line-length = 100
            target-version = "py311"
            """
        ),
        "requirements.txt": dedent(
            """\
            pandas>=2.2
            pyyaml>=6.0
            scikit-learn>=1.5
            joblib>=1.4
            pytest>=8.0
            """
        ),
        ".gitignore": dedent(
            """\
            .venv/
            __pycache__/
            *.py[cod]
            .pytest_cache/
            .ruff_cache/
            .coverage
            htmlcov/
            .env
            data/raw/*
            data/interim/*
            data/processed/*
            artifacts/models/*
            artifacts/preprocessors/*
            !**/.gitkeep
            """
        ),
        ".env.example": "APP_ENV=development\nLOG_LEVEL=INFO\n",
        "Makefile": dedent(
            """\
            install:
            \tpython -m pip install -e ".[dev]"

            test:
            \tpytest

            lint:
            \truff check .

            validate-data:
            \tpython scripts/validate_data.py --config configs/data.yaml
            """
        ),
        "Dockerfile": dedent(
            """\
            FROM python:3.12-slim

            WORKDIR /app
            COPY . .
            RUN pip install --no-cache-dir -e ".[api]"

            CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
            """
        ),
        "specs/SPEC-00-problem-definition.md": dedent(
            """\
            # SPEC-00 — Problem Definition

            **Status:** Draft  
            **Owner:** Babar Ali Khan

            ## Problem

            - Business or user problem: TBD
            - Intended user: TBD
            - Decision supported by the prediction: TBD
            - Prediction target: TBD
            - Task type: classification or regression — TBD
            - What one row represents: TBD

            ## Success Criteria

            - Primary model metric: TBD
            - Minimum acceptable value: TBD
            - Baseline: TBD
            - Operational constraint: TBD

            ## Risks and Constraints

            - Data leakage risk: TBD
            - Privacy or sensitive attributes: TBD
            - Cost of false positive: TBD
            - Cost of false negative: TBD

            ## Acceptance Criteria

            - [ ] Target and task type are approved.
            - [ ] Prediction timing is defined.
            - [ ] Success is measurable.
            - [ ] Stakeholders and limitations are documented.
            """
        ),
        "specs/SPEC-01-data-ingestion-preprocessing.md": dedent(
            """\
            # SPEC-01 — CSV Ingestion, Review, and Preprocessing

            **Status:** Draft  
            **Owner:** Babar Ali Khan

            ## Purpose

            Load the approved CSV, validate its schema, report data-quality risks,
            assign every column a role, and construct an unfitted preprocessing pipeline.

            ## Requirements

            | ID | Requirement |
            | --- | --- |
            | ING-001 | Load the CSV using configured path, encoding, delimiter, and missing tokens. |
            | ING-002 | Fail clearly when the source is missing, empty, unreadable, or malformed. |
            | ING-003 | Calculate a SHA-256 source fingerprint. |
            | REV-001 | Report shape, types, missingness, duplicates, constants, and cardinality. |
            | REV-002 | Validate required, target, ID, dropped, and sensitive columns. |
            | PRE-001 | Separate the target before feature preprocessing. |
            | PRE-002 | Exclude IDs, target, dropped fields, and confirmed leakage columns. |
            | PRE-003 | Handle numeric and categorical missing values through configured strategies. |
            | PRE-004 | Allow unseen inference categories without crashing. |
            | PRE-005 | Fit every learned transformation on training data only. |

            ## Tests

            | Test ID | Given | When | Then |
            | --- | --- | --- | --- |
            | T-ING-001 | A valid CSV | Loading runs | Shape and fingerprint are returned. |
            | T-ING-002 | A missing or empty file | Loading runs | A useful validation error is raised. |
            | T-REV-001 | Invalid schema | Validation runs | Exact issues are reported. |
            | T-PRE-001 | Numeric and categorical features | Training transformation runs | No unexpected null values remain. |
            | T-PRE-002 | An unseen category | Inference transformation runs | Transformation succeeds. |

            ## Acceptance Criteria

            - [ ] The original raw CSV remains unchanged.
            - [ ] Dataset fingerprint and profile are reproducible.
            - [ ] Every column has an approved role.
            - [ ] No row or column is silently removed.
            - [ ] Target leakage and preprocessing leakage are prevented.
            - [ ] All mandatory tests pass.
            """
        ),
        "configs/data.yaml": dedent(
            """\
            data:
              csv_path: data/raw/dataset.csv
              delimiter: ","
              encoding: utf-8
              target_column: TBD
              task_type: classification
              required_columns: []
              id_columns: []
              drop_columns: []
              sensitive_columns: []
              numeric_columns: []
              categorical_columns: []
              missing_value_tokens: ["", "NA", "N/A", "null", "None", "?"]
              maximum_missing_ratio: 0.60
              high_cardinality_threshold: 100
              random_seed: 42
            """
        ),
        "configs/preprocessing.yaml": dedent(
            """\
            preprocessing:
              numeric_imputation: median
              categorical_imputation: __MISSING__
              categorical_encoding: one_hot
              scale_numeric: true
              handle_unknown_categories: ignore
            """
        ),
        "configs/training.yaml": dedent(
            """\
            training:
              random_seed: 42
              test_size: 0.20
              validation_size: 0.20
              primary_metric: TBD
              baseline_model: TBD
            """
        ),
        f"src/{package}/__init__.py": '__version__ = "0.1.0"\n',
        f"src/{package}/utils/config.py": dedent(
            """\
            from __future__ import annotations

            from pathlib import Path
            from typing import Any

            import yaml


            def load_yaml(path: str | Path) -> dict[str, Any]:
                config_path = Path(path)
                if not config_path.is_file():
                    raise FileNotFoundError(f"Configuration file not found: {config_path}")
                with config_path.open("r", encoding="utf-8") as handle:
                    content = yaml.safe_load(handle) or {}
                if not isinstance(content, dict):
                    raise ValueError("Configuration root must be a mapping.")
                return content
            """
        ),
        f"src/{package}/utils/logging.py": dedent(
            """\
            import logging


            def configure_logging(level: str = "INFO") -> None:
                logging.basicConfig(
                    level=getattr(logging, level.upper(), logging.INFO),
                    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
                )
            """
        ),
        f"src/{package}/utils/exceptions.py": dedent(
            """\
            class DataValidationError(ValueError):
                '''Raised when input data violates an approved data contract.'''
            """
        ),
        f"src/{package}/data/ingestion.py": dedent(
            """\
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
            """
        ),
        f"src/{package}/data/validation.py": dedent(
            """\
            from __future__ import annotations

            from typing import Any

            import pandas as pd

            from ..utils.exceptions import DataValidationError


            def validate_schema(dataframe: pd.DataFrame, config: dict[str, Any]) -> list[str]:
                errors: list[str] = []
                target = config.get("target_column")
                required = set(config.get("required_columns", []))

                if not target or target == "TBD":
                    errors.append("Set data.target_column in configs/data.yaml.")
                elif target not in dataframe.columns:
                    errors.append(f"Target column not found: {target}")

                missing_required = sorted(required.difference(dataframe.columns))
                if missing_required:
                    errors.append(f"Required columns not found: {missing_required}")

                if target in dataframe.columns:
                    usable_values = dataframe[target].dropna()
                    if usable_values.empty:
                        errors.append("Target column contains no usable values.")
                    elif usable_values.nunique() < 2:
                        errors.append("Target column must contain at least two distinct values.")
                return errors


            def require_valid_schema(dataframe: pd.DataFrame, config: dict[str, Any]) -> None:
                errors = validate_schema(dataframe, config)
                if errors:
                    raise DataValidationError("; ".join(errors))
            """
        ),
        f"src/{package}/data/profiling.py": dedent(
            """\
            from __future__ import annotations

            from typing import Any

            import pandas as pd


            def build_profile(dataframe: pd.DataFrame, fingerprint: str) -> dict[str, Any]:
                row_count = len(dataframe)
                columns: dict[str, Any] = {}
                for name in dataframe.columns:
                    series = dataframe[name]
                    null_count = int(series.isna().sum())
                    columns[name] = {
                        "dtype": str(series.dtype),
                        "null_count": null_count,
                        "null_percentage": round((null_count / row_count) * 100, 4),
                        "unique_count": int(series.nunique(dropna=True)),
                        "is_constant": bool(series.nunique(dropna=False) <= 1),
                    }
                return {
                    "row_count": row_count,
                    "column_count": len(dataframe.columns),
                    "duplicate_row_count": int(dataframe.duplicated().sum()),
                    "memory_bytes": int(dataframe.memory_usage(deep=True).sum()),
                    "source_sha256": fingerprint,
                    "columns": columns,
                }
            """
        ),
        f"src/{package}/data/splitting.py": dedent(
            """\
            '''Data-splitting implementation belongs to SPEC-03.'''
            """
        ),
        f"src/{package}/features/preprocessing.py": dedent(
            """\
            from __future__ import annotations

            from typing import Sequence

            from sklearn.compose import ColumnTransformer
            from sklearn.impute import SimpleImputer
            from sklearn.pipeline import Pipeline
            from sklearn.preprocessing import OneHotEncoder, StandardScaler


            def build_preprocessor(
                numeric_columns: Sequence[str],
                categorical_columns: Sequence[str],
                *,
                scale_numeric: bool = True,
            ) -> ColumnTransformer:
                numeric_steps: list[tuple[str, object]] = [
                    ("imputer", SimpleImputer(strategy="median"))
                ]
                if scale_numeric:
                    numeric_steps.append(("scaler", StandardScaler()))

                numeric_pipeline = Pipeline(numeric_steps)
                categorical_pipeline = Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(strategy="constant", fill_value="__MISSING__"),
                        ),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=True),
                        ),
                    ]
                )
                return ColumnTransformer(
                    transformers=[
                        ("numeric", numeric_pipeline, list(numeric_columns)),
                        ("categorical", categorical_pipeline, list(categorical_columns)),
                    ],
                    remainder="drop",
                    verbose_feature_names_out=True,
                )
            """
        ),
        f"src/{package}/features/engineering.py": '"""Feature engineering belongs to SPEC-04."""\n',
        f"src/{package}/features/selection.py": '"""Feature selection belongs to SPEC-04."""\n',
        f"src/{package}/models/baseline.py": '"""Baseline modeling belongs to SPEC-05."""\n',
        f"src/{package}/models/train.py": '"""Model training belongs to SPEC-06."""\n',
        f"src/{package}/models/tune.py": '"""Model tuning belongs to SPEC-06."""\n',
        f"src/{package}/models/evaluate.py": '"""Evaluation belongs to SPEC-07."""\n',
        f"src/{package}/inference/schemas.py": '"""Prediction schemas belong to SPEC-08."""\n',
        f"src/{package}/inference/predictor.py": '"""Prediction logic belongs to SPEC-08."""\n',
        f"src/{package}/inference/validation.py": '"""Inference validation belongs to SPEC-08."""\n',
        "pipelines/data_pipeline.py": dedent(
            f"""\
            from {package}.data.ingestion import load_csv
            from {package}.data.profiling import build_profile
            from {package}.data.validation import require_valid_schema


            def run_data_review(config: dict) -> dict:
                result = load_csv(config)
                require_valid_schema(result.dataframe, config)
                return build_profile(result.dataframe, result.sha256)
            """
        ),
        "pipelines/training_pipeline.py": '"""Training pipeline belongs to SPEC-05 and SPEC-06."""\n',
        "pipelines/inference_pipeline.py": '"""Inference pipeline belongs to SPEC-08."""\n',
        "scripts/validate_data.py": dedent(
            f"""\
            from __future__ import annotations

            import argparse
            import json
            from pathlib import Path

            from {package}.data.ingestion import load_csv
            from {package}.data.profiling import build_profile
            from {package}.data.validation import require_valid_schema
            from {package}.utils.config import load_yaml


            def main() -> None:
                parser = argparse.ArgumentParser(description="Validate and profile a CSV dataset.")
                parser.add_argument("--config", default="configs/data.yaml")
                parser.add_argument("--output", default="reports/data_profile.json")
                args = parser.parse_args()

                config = load_yaml(args.config).get("data", {{}})
                result = load_csv(config)
                require_valid_schema(result.dataframe, config)
                profile = build_profile(result.dataframe, result.sha256)

                output_path = Path(args.output)
                output_path.parent.mkdir(parents=True, exist_ok=True)
                output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
                print(f"Validated {{profile['row_count']}} rows and {{profile['column_count']}} columns.")
                print(f"Profile saved to {{output_path}}")


            if __name__ == "__main__":
                main()
            """
        ),
        "scripts/preprocess_data.py": '"""Implement after SPEC-03 defines the approved data split."""\n',
        "scripts/train_model.py": '"""Implement after SPEC-05 and SPEC-06 are approved."""\n',
        "scripts/evaluate_model.py": '"""Implement after SPEC-07 is approved."""\n',
        "tests/unit/test_ingestion.py": dedent(
            f"""\
            import pandas as pd
            import pytest

            from {package}.data.ingestion import load_csv


            def test_load_valid_csv(tmp_path):
                path = tmp_path / "valid.csv"
                pd.DataFrame({{"feature": [1, 2], "target": [0, 1]}}).to_csv(path, index=False)
                result = load_csv({{"csv_path": str(path)}})
                assert result.dataframe.shape == (2, 2)
                assert len(result.sha256) == 64


            def test_missing_csv_raises(tmp_path):
                with pytest.raises(FileNotFoundError):
                    load_csv({{"csv_path": str(tmp_path / "missing.csv")}})
            """
        ),
        "tests/unit/test_validation.py": dedent(
            f"""\
            import pandas as pd

            from {package}.data.validation import validate_schema


            def test_missing_target_is_reported():
                dataframe = pd.DataFrame({{"feature": [1, 2]}})
                errors = validate_schema(dataframe, {{"target_column": "target"}})
                assert any("Target column not found" in error for error in errors)


            def test_valid_target_has_no_errors():
                dataframe = pd.DataFrame({{"feature": [1, 2], "target": [0, 1]}})
                assert validate_schema(dataframe, {{"target_column": "target"}}) == []
            """
        ),
        "tests/unit/test_preprocessing.py": dedent(
            f"""\
            import numpy as np
            import pandas as pd

            from {package}.features.preprocessing import build_preprocessor


            def test_preprocessor_handles_missing_and_unseen_values():
                train = pd.DataFrame(
                    {{"age": [20.0, np.nan, 40.0], "city": ["A", "B", None]}}
                )
                inference = pd.DataFrame({{"age": [30.0], "city": ["UNSEEN"]}})
                transformer = build_preprocessor(["age"], ["city"])
                transformer.fit(train)
                output = transformer.transform(inference)
                assert output.shape[0] == 1
            """
        ),
        "tests/integration/test_data_pipeline.py": dedent(
            """\
            def test_data_pipeline_placeholder():
                # Replace with an end-to-end test after configuring the real dataset contract.
                assert True
            """
        ),
        "tests/integration/test_training_pipeline.py": dedent(
            """\
            def test_training_pipeline_placeholder():
                # Implement when SPEC-05 and SPEC-06 are approved.
                assert True
            """
        ),
        "tests/contract/test_prediction_contract.py": dedent(
            """\
            def test_prediction_contract_placeholder():
                # Implement when SPEC-08 defines the request and response schemas.
                assert True
            """
        ),
        "app/__init__.py": "",
        "app/main.py": dedent(
            """\
            try:
                from fastapi import FastAPI
            except ImportError as exc:
                raise RuntimeError('Install API dependencies with: pip install -e ".[api]"') from exc

            app = FastAPI(title="ML Prediction API", version="0.1.0")


            @app.get("/health")
            def health() -> dict[str, str]:
                return {"status": "healthy"}
            """
        ),
        "app/routes.py": '"""Prediction routes belong to SPEC-09."""\n',
        "app/schemas.py": '"""API schemas belong to SPEC-09."""\n',
        ".github/workflows/tests.yml": dedent(
            """\
            name: tests

            on:
              push:
              pull_request:

            jobs:
              test:
                runs-on: ubuntu-latest
                steps:
                  - uses: actions/checkout@v4
                  - uses: actions/setup-python@v5
                    with:
                      python-version: "3.12"
                  - run: pip install -e ".[dev]"
                  - run: ruff check .
                  - run: pytest
            """
        ),
        ".github/workflows/deployment.yml": dedent(
            """\
            name: deployment-placeholder

            on:
              workflow_dispatch:

            jobs:
              explain:
                runs-on: ubuntu-latest
                steps:
                  - run: echo "Define the deployment target under SPEC-10 before enabling deployment."
            """
        ),
    }

    specs = [
        ("SPEC-02", "Exploratory Data Analysis", "Document distributions, relationships, anomalies, and data risks."),
        ("SPEC-03", "Data Splitting and Leakage Control", "Create reproducible train, validation, and test sets."),
        ("SPEC-04", "Feature Engineering and Selection", "Produce justified, reproducible model inputs."),
        ("SPEC-05", "Baseline Modeling", "Establish a simple performance benchmark."),
        ("SPEC-06", "Model Training and Tuning", "Train and tune approved candidate models reproducibly."),
        ("SPEC-07", "Evaluation and Error Analysis", "Select a model using metrics, errors, and subgroup evidence."),
        ("SPEC-08", "Model Packaging and Inference", "Create a versioned prediction pipeline and contract."),
        ("SPEC-09", "API or Application", "Expose a tested user-facing prediction workflow."),
        ("SPEC-10", "CI/CD and Deployment", "Create repeatable build, release, and rollback processes."),
        ("SPEC-11", "Monitoring and Maintenance", "Monitor input data, predictions, performance, and drift."),
    ]
    for spec_id, title, outcome in specs:
        slug = title.lower().replace("/", "-").replace(" ", "-")
        files[f"specs/{spec_id}-{slug}.md"] = spec_placeholder(spec_id, title, outcome)

    empty_directories = [
        "data/raw",
        "data/interim",
        "data/processed",
        "data/external",
        "notebooks",
        "artifacts/preprocessors",
        "artifacts/models",
        "artifacts/metadata",
        "reports/figures",
        "tests/fixtures",
    ]
    for directory in empty_directories:
        files[f"{directory}/.gitkeep"] = ""

    init_directories = [
        f"src/{package}/data",
        f"src/{package}/features",
        f"src/{package}/models",
        f"src/{package}/inference",
        f"src/{package}/utils",
        "pipelines",
    ]
    for directory in init_directories:
        files.setdefault(f"{directory}/__init__.py", "")

    return files


def create_project(root: Path, files: dict[str, str], overwrite: bool) -> tuple[int, int]:
    """Create the scaffold and return counts of created and skipped files."""
    created = 0
    skipped = 0
    root.mkdir(parents=True, exist_ok=True)

    for relative_path, content in sorted(files.items()):
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        if destination.exists() and not overwrite:
            skipped += 1
            continue
        destination.write_text(content, encoding="utf-8")
        created += 1
    return created, skipped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a specification-driven machine-learning project."
    )
    parser.add_argument(
        "--name",
        type=normalize_project_name,
        default=DEFAULT_PROJECT_NAME,
        help=f"Project name (default: {DEFAULT_PROJECT_NAME}).",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path.cwd(),
        help="Parent directory in which the project folder will be created.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite scaffold files that already exist. Unrelated files are preserved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.destination.expanduser().resolve() / args.name
    files = build_files(args.name)
    created, skipped = create_project(root, files, args.overwrite)

    print(f"Project created at: {root}")
    print(f"Files written: {created}")
    print(f"Existing files skipped: {skipped}")
    print("Next: update specs/SPEC-00-problem-definition.md and configs/data.yaml")


if __name__ == "__main__":
    main()
