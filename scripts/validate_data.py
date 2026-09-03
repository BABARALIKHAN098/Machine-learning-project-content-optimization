from __future__ import annotations

import argparse
import json
from pathlib import Path

from machine_learning_project.data.ingestion import load_csv
from machine_learning_project.data.profiling import build_profile
from machine_learning_project.data.validation import require_valid_schema
from machine_learning_project.utils.config import load_yaml


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and profile a CSV dataset.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output", default="reports/data_profile.json")
    args = parser.parse_args()

    config = load_yaml(args.config).get("data", {})
    result = load_csv(config)
    require_valid_schema(result.dataframe, config)
    profile = build_profile(result.dataframe, result.sha256)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
    print(f"Validated {profile['row_count']} rows and {profile['column_count']} columns.")
    print(f"Profile saved to {output_path}")


if __name__ == "__main__":
    main()
