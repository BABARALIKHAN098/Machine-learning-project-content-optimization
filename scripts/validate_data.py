from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machine_learning_project.data.profiling import render_profile_markdown
from machine_learning_project.utils.config import load_yaml
from pipelines.data_pipeline import run_data_review


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate and profile a CSV dataset.")
    parser.add_argument("--config", default="configs/data.yaml")
    parser.add_argument("--output", default="reports/data_profile.json")
    parser.add_argument("--markdown-output", default="reports/data_audit.md")
    args = parser.parse_args()

    config = load_yaml(args.config).get("data", {})
    profile = run_data_review(config)

    output_path = Path(args.output)
    _atomic_write(output_path, json.dumps(profile, indent=2, sort_keys=True) + "\n")
    markdown_path = Path(args.markdown_output)
    _atomic_write(markdown_path, render_profile_markdown(profile))
    print(f"Validated {profile['row_count']} rows and {profile['column_count']} columns.")
    print(f"Profile saved to {output_path}")
    print(f"Audit saved to {markdown_path}")


if __name__ == "__main__":
    main()
