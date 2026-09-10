"""Submit an authorized local JSON request; print aggregate response information only."""

import argparse
import json
import os
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    token = os.environ.get("CONTENT_TREND_API_TOKEN", "")
    if not token or not 1 <= args.port <= 65535:
        parser.error("A valid port and environment token are required")
    with httpx.Client(trust_env=False, timeout=120) as client:
        response = client.post(
            f"http://127.0.0.1:{args.port}/v1/predictions",
            content=Path(args.input_path).read_bytes(),
            headers={"Authorization": f"Bearer {token}", "Content-Type": "application/json"},
        )
    if response.status_code != 200:
        print(json.dumps({"status": "rejected", "http_status": response.status_code}))
        return 2
    if response.headers.get("content-type", "").split(";")[0] != "application/json":
        raise RuntimeError("Unexpected response content type")
    result = response.json()
    print(
        json.dumps(
            {
                k: result[k]
                for k in (
                    "row_count",
                    "package_id",
                    "purpose",
                    "production_ready",
                    "include_probabilities",
                )
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
