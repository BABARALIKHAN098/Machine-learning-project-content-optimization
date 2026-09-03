from machine_learning_project.data.ingestion import load_csv
from machine_learning_project.data.profiling import build_profile
from machine_learning_project.data.validation import require_valid_schema


def run_data_review(config: dict) -> dict:
    result = load_csv(config)
    require_valid_schema(result.dataframe, config)
    return build_profile(result.dataframe, result.sha256)
