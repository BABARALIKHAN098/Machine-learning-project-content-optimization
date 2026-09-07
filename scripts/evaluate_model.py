from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Display the sealed-test evaluation report.")
    parser.add_argument("--metrics", default="reports/metrics/model_metrics.json")
    args = parser.parse_args()
    payload = json.loads(Path(args.metrics).read_text(encoding="utf-8"))
    if "test" not in payload:
        validation = payload["candidates_validation"][payload["selected_model"]]
        print(f"Selected model: {payload['selected_model']}")
        print(f"Validation macro F1: {validation['macro_f1']:.4f}")
        print("Development-only artifact; final test evaluation has not been run.")
        return
    test = payload["test"]
    print(f"Selected model: {payload['selected_model']}")
    print(f"Test macro F1: {test['macro_f1']:.4f}")
    print(f"Down recall: {test['per_class']['down']['recall']:.4f}")
    print(f"Threshold met: {payload['threshold_met']}")


if __name__ == "__main__":
    main()
