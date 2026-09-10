"""Supported single-worker loopback launcher. Never downloads dependencies."""

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.config import ConfigurationError, load_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/api.yaml")
    parser.add_argument("--purpose", required=True, choices=["research"])
    parser.add_argument("--package-dir")
    parser.add_argument("--expected-manifest-sha256")
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    args = vars(parser.parse_args())
    path = args.pop("config")
    try:
        settings = load_settings(path, args)
    except ConfigurationError:
        print(
            "API configuration rejected; supply a concrete package, trusted pin and environment token.",
            file=sys.stderr,
        )
        return 2
    import uvicorn

    from app.main import create_app

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    uvicorn.run(
        create_app(settings),
        host=settings.host,
        port=settings.port,
        workers=1,
        reload=False,
        proxy_headers=False,
        access_log=False,
        server_header=False,
        ws="none",
        log_level="warning",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
