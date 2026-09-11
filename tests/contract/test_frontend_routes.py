from pathlib import Path

from fastapi.testclient import TestClient

from app.main import create_app
from tests.api_helpers import fixture


def test_frontend_assets_headers_and_existing_api():
    predictor, settings, _ = fixture()
    with TestClient(
        create_app(settings, lambda *a, **k: predictor), base_url="http://127.0.0.1:8000"
    ) as client:
        response = client.get("/")
        assert response.status_code == 200
        assert 'id="connect-form"' in response.text
        assert 'type="module"' in response.text
        assert response.headers["cache-control"] == "no-store"
        assert "script-src 'self'" in response.headers["content-security-policy"]
        root = Path("app/static/frontend")
        for file in root.rglob("*"):
            if file.is_file() and file.suffix in {".css", ".js", ".svg"}:
                asset = client.get(f"/assets/{file.relative_to(root).as_posix()}")
                assert asset.status_code == 200
                assert asset.content == file.read_bytes()
                assert asset.headers["x-content-type-options"] == "nosniff"
                assert asset.headers["cache-control"] == "no-store"
                if file.suffix == ".js":
                    assert "javascript" in asset.headers["content-type"]
        assert client.get("/docs").status_code == 200
        assert client.get("/openapi.json").status_code == 200
        assert client.get("/v1/schema").status_code == 401
        assert client.get("/assets/missing.js").status_code == 404
        assert client.get("/assets/%2e%2e/%2e%2e/config.py").status_code == 404
        assert client.get("/", headers={"Host": "foreign.example"}).status_code == 400
        assert not predictor._predictor.pipeline.calls


def test_unknown_asset_paths_are_not_logged(caplog):
    import logging

    caplog.set_level(logging.INFO, logger="research_api")
    predictor, settings, _ = fixture()
    with TestClient(
        create_app(settings, lambda *a, **k: predictor), base_url="http://127.0.0.1:8000"
    ) as client:
        assert client.get("/assets/private-synthetic-identifier.js").status_code == 404
    assert "private-synthetic-identifier" not in caplog.text
    assert '"route": "unmatched"' in caplog.text
