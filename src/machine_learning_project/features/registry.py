"""Versioned, bounded feature definitions: arbitrary formulas are intentionally unsupported."""

from __future__ import annotations

import hashlib
import json

from ..utils.exceptions import DataValidationError
from .selection import configured_feature_columns

RATIOS = {
    "previous_ctr": ("clicks_prev_30d", "impressions_prev_30d"),
    "previous_sessions_per_click": ("sessions_prev_30d", "clicks_prev_30d"),
}
LOG_SOURCES = {
    "search_volume",
    "cpc",
    "word_count",
    "char_count",
    "impressions_prev_30d",
    "clicks_prev_30d",
    "sessions_prev_30d",
}


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False).encode()).hexdigest()


def resolve_registry(data_config, config):
    raw = configured_feature_columns(data_config)
    for key, supported in (
        ("feature_contract_version", "2.0"),
        ("registry_schema_version", "1.0"),
        ("engineering_version", "1.0"),
    ):
        if config.get(key) != supported:
            raise DataValidationError(f"Unsupported {key}: {config.get(key)!r}")
    for key in ("families", "excluded_inputs", "log_sources"):
        values = config.get(key, [])
        if (
            not isinstance(values, list)
            or not all(isinstance(x, str) for x in values)
            or len(values) != len(set(values))
        ):
            raise DataValidationError(f"{key} must contain unique names")
    if set(config.get("families", [])) - {"ratios", "logs"}:
        raise DataValidationError("Unknown feature family")
    excluded = set(config.get("excluded_inputs", []))
    if excluded - set(raw):
        raise DataValidationError("Excluded inputs must be approved raw inputs")
    numeric = [x for x in data_config.get("numeric_columns", []) if x not in excluded]
    categorical = [x for x in data_config.get("categorical_columns", []) if x not in excluded]
    entries = [
        {
            "name": x,
            "role": "numeric" if x in numeric else "categorical",
            "dependencies": [x],
            "operation": "identity",
            "family": "raw",
            "availability": "Configured cutoff input; historical availability unverified",
        }
        for x in numeric + categorical
    ]
    if "ratios" in config.get("families", []):
        for name, dependencies in RATIOS.items():
            for suffix, operation in (("", "ratio"), ("_unavailable", "unavailable")):
                entries.append(
                    {
                        "name": name + suffix,
                        "role": "numeric",
                        "dependencies": list(dependencies),
                        "operation": operation,
                        "family": "ratios",
                        "availability": "Previous observation period",
                    }
                )
    if set(config.get("log_sources", [])) - LOG_SOURCES:
        raise DataValidationError("Unsupported log source")
    if "logs" in config.get("families", []):
        for source in config.get("log_sources", []):
            entries.append(
                {
                    "name": f"log1p_{source}",
                    "role": "numeric",
                    "dependencies": [source],
                    "operation": "log1p",
                    "family": "logs",
                    "availability": "Inherits source availability",
                }
            )
    names = [entry["name"] for entry in entries]
    if not names or len(names) != len(set(names)):
        raise DataValidationError("Empty feature set or output name collision")
    for entry in entries:
        dependencies = set(entry["dependencies"])
        if dependencies - set(numeric + categorical):
            raise DataValidationError(f"Forbidden or unavailable dependency for {entry['name']}")
        if entry["operation"] != "identity" and dependencies - set(numeric):
            raise DataValidationError("Derived dependencies must be numeric")
        if entry["operation"] != "identity" and entry["name"] in raw:
            raise DataValidationError("Derived output collides with raw input")
        operation = entry["operation"]
        dependencies = entry["dependencies"]
        entry["formula"] = {
            "identity": dependencies[0],
            "log1p": f"log1p({dependencies[0]})",
            "ratio": " / ".join(dependencies),
            "unavailable": "1 if source missing or denominator zero; else 0",
        }[operation]
        entry["unit"] = {
            "identity": "source units",
            "log1p": "log-transformed source units",
            "ratio": "numerator units per denominator unit",
            "unavailable": "binary",
        }[operation]
        entry["missing_policy"] = (
            "Undefined ratio remains missing with unavailable indicator; train median; empty fallback zero"
            if operation == "ratio"
            else "Train median; entirely missing numeric fallback zero; categorical missing token"
        )
        entry["invalid_policy"] = "Reject negative/non-finite numeric sources and ratio overflow"
        entry["justification"] = {
            "raw": "Existing approved reference input",
            "ratios": "Historical efficiency and unavailable-history signal",
            "logs": "Test skew compression without deleting source counts",
        }[entry["family"]]
    return entries
