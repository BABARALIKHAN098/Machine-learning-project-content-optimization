import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from machine_learning_project.utils.exceptions import DataValidationError
from pipelines.inference_pipeline import run_packaged_batch


def main():
    parser = argparse.ArgumentParser(
        description="Private local research inference with a verified package"
    )
    parser.add_argument("--package-dir", required=True)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--purpose", required=True, choices=["research"])
    parser.add_argument("--include-probabilities", action="store_true")
    parser.add_argument("--chunk-rows", type=int)
    args = parser.parse_args()
    try:
        result = run_packaged_batch(
            args.input_path,
            args.package_dir,
            args.output_path,
            purpose=args.purpose,
            include_probabilities=args.include_probabilities,
            chunk_rows=args.chunk_rows,
        )
    except DataValidationError as error:
        print(json.dumps({"status": "rejected", "reason": str(error)}))
        raise SystemExit(2) from None
    print(json.dumps(result))


if __name__ == "__main__":
    main()
