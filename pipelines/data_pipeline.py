from machine_learning_project.data.eda import EDAResults, run_analyses
from machine_learning_project.data.ingestion import load_csv, sha256_file
from machine_learning_project.data.profiling import build_profile
from machine_learning_project.data.reporting import write_eda_artifacts
from machine_learning_project.data.validation import require_valid_schema
from machine_learning_project.data.visualization import render_eda_figures
from machine_learning_project.utils.config import (
    validate_data_config,
    validate_eda_config,
)


def run_data_review(config: dict, *, feature_contract_version: str = "1.0") -> dict:
    validate_data_config(config)
    result = load_csv(config)
    require_valid_schema(result.dataframe, config)
    profile = build_profile(
        result.dataframe,
        result.sha256,
        config,
        feature_contract_version=feature_contract_version,
    )
    if sha256_file(result.source_path) != result.sha256:
        raise RuntimeError("Raw CSV changed while the data review was running.")
    return profile


def run_eda(
    data_config: dict,
    eda_config: dict,
    *,
    output_directory: str | None = None,
    figures_directory: str | None = None,
) -> tuple[EDAResults, list]:
    validate_data_config(data_config)
    validate_eda_config(eda_config)
    loaded = load_csv(data_config)
    require_valid_schema(loaded.dataframe, data_config)
    original = loaded.dataframe.copy(deep=True)
    results = run_analyses(loaded.dataframe, loaded.sha256, data_config, eda_config)
    figure_paths = render_eda_figures(
        loaded.dataframe,
        data_config,
        eda_config,
        figures_directory or eda_config["figures_directory"],
    )
    artifacts = write_eda_artifacts(
        results, output_directory or eda_config["output_directory"], figure_paths
    )
    if not loaded.dataframe.equals(original):
        raise RuntimeError("EDA mutated the source dataframe.")
    if sha256_file(loaded.source_path) != loaded.sha256:
        raise RuntimeError("Raw CSV changed while EDA was running.")
    return results, artifacts + figure_paths
