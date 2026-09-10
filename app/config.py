"""Closed transport settings. Inference runtime configuration remains package-owned."""

import os
import re
from dataclasses import dataclass, field, fields
from pathlib import Path

import yaml


class ConfigurationError(Exception):
    """Intentionally contains no configuration values or credentials."""


@dataclass(frozen=True)
class Settings:
    package_dir: str
    expected_manifest_sha256: str
    token: str = field(repr=False)
    api_contract_version: str = "1.0"
    purpose: str = "research"
    host: str = "127.0.0.1"
    port: int = 8000
    maximum_request_bytes: int = 33554432
    maximum_response_bytes: int = 67108864
    maximum_batch_rows: int = 30000
    chunk_rows: int = 5000
    maximum_content_id_bytes: int = 256
    maximum_category_bytes: int = 1024
    maximum_active_predictions: int = 1
    body_timeout_seconds: int = 30

    def __post_init__(self):
        invalid = (
            self.api_contract_version != "1.0"
            or self.purpose != "research"
            or self.host != "127.0.0.1"
            or not isinstance(self.token, str)
            or len(self.token) < 32
            or not self.token.isascii()
            or any(c.isspace() or not c.isprintable() for c in self.token)
            or not isinstance(self.expected_manifest_sha256, str)
            or not re.fullmatch(r"[0-9a-f]{64}", self.expected_manifest_sha256)
            or not isinstance(self.package_dir, str)
            or not self.package_dir.strip()
            or "://" in self.package_dir
            or self.package_dir.startswith(("//", "\\\\"))
        )
        bounds = {
            "port": 65535,
            "maximum_request_bytes": 33554432,
            "maximum_response_bytes": 67108864,
            "maximum_batch_rows": 30000,
            "chunk_rows": 30000,
            "maximum_content_id_bytes": 256,
            "maximum_category_bytes": 1024,
            "maximum_active_predictions": 1,
            "body_timeout_seconds": 30,
        }
        if (
            invalid
            or any(
                type(getattr(self, k)) is not int or not 1 <= getattr(self, k) <= upper
                for k, upper in bounds.items()
            )
            or self.chunk_rows > self.maximum_batch_rows
        ):
            raise ConfigurationError("Invalid local research API configuration")


def load_settings(path="configs/api.yaml", overrides=None):
    class UniqueLoader(yaml.SafeLoader):
        pass

    def mapping(loader, node):
        result = {}
        for key_node, value_node in node.value:
            key = loader.construct_object(key_node)
            if not isinstance(key, str) or key in result:
                raise ConfigurationError("Invalid API configuration mapping")
            result[key] = loader.construct_object(value_node)
        return result

    UniqueLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, mapping)
    try:
        document = yaml.load(Path(path).read_text(encoding="utf-8"), Loader=UniqueLoader)
        if not isinstance(document, dict) or set(document) != {"api"}:
            raise ConfigurationError("Invalid API configuration root")
        config = document["api"]
        allowed = {f.name for f in fields(Settings)} - {"token"}
        if not isinstance(config, dict) or set(config) != allowed:
            raise ConfigurationError("Invalid API configuration fields")
        overrides = overrides or {}
        if set(overrides) - {"package_dir", "expected_manifest_sha256", "purpose", "host", "port"}:
            raise ConfigurationError("Invalid API overrides")
        config.update({k: v for k, v in overrides.items() if v is not None})
        return Settings(**config, token=os.environ.get("CONTENT_TREND_API_TOKEN", ""))
    except (OSError, yaml.YAMLError, ValueError, TypeError, RecursionError):
        raise ConfigurationError("Cannot read valid API configuration") from None
