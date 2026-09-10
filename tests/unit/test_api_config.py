from dataclasses import replace

import pytest

from app.config import ConfigurationError, load_settings
from tests.api_helpers import TOKEN, fixture


@pytest.mark.parametrize(
    "key,value",
    [
        ("purpose", "production"),
        ("host", "0.0.0.0"),
        ("expected_manifest_sha256", "invalid"),
        ("token", "short"),
        ("token", " " * 40),
        ("maximum_batch_rows", True),
        ("chunk_rows", 0),
        ("chunk_rows", 30001),
        ("maximum_active_predictions", 2),
        ("maximum_request_bytes", float("inf")),
        ("package_dir", "https://remote"),
        ("package_dir", "\\\\server\\share"),
        ("port", 65536),
        ("api_contract_version", "2.0"),
    ],
)
def test_invalid_settings(key, value):
    _, settings, _ = fixture()
    with pytest.raises(ConfigurationError):
        replace(settings, **{key: value})
    assert TOKEN not in repr(settings)


def test_closed_yaml_and_precedence(tmp_path, monkeypatch):
    from pathlib import Path

    monkeypatch.setenv("CONTENT_TREND_API_TOKEN", TOKEN)
    overrides = {
        "package_dir": "explicit-package",
        "expected_manifest_sha256": "1" * 64,
        "port": 8123,
    }
    settings = load_settings(overrides=overrides)
    assert settings.port == 8123 and settings.package_dir == "explicit-package"
    for content in (
        "api: {}\napi: {}",
        "api: {}\nunknown: 1",
        "api: {}",
        "api: [1]",
        Path("configs/api.yaml").read_text() + "  purpose: research\n",
    ):
        path = tmp_path / "bad.yaml"
        path.write_text(content)
        with pytest.raises(ConfigurationError):
            load_settings(path, overrides)
    monkeypatch.delenv("CONTENT_TREND_API_TOKEN")
    with pytest.raises(ConfigurationError):
        load_settings(overrides=overrides)
