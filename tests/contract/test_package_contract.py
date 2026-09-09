import pytest

from machine_learning_project.models.package_artifacts import (
    load_package_manifest,
    publish_private_json,
    read_json,
    safe_name,
    safe_path,
)
from machine_learning_project.utils.exceptions import DataValidationError


@pytest.mark.parametrize("name", ["../escape", "CON", "AUX", "x/y", "", "a:b", "LPT0"])
def test_bad_names(name):
    with pytest.raises(DataValidationError):
        safe_name(name)


@pytest.mark.parametrize("name", ["../model.joblib", "x/../../x", "x\\y", "/x", "a:b", "x//y"])
def test_bad_payloads(tmp_path, name):
    with pytest.raises(DataValidationError):
        safe_path(tmp_path, name)


def test_duplicate_json_and_incomplete_package(tmp_path):
    path = tmp_path / "package_manifest.json"
    path.write_text('{"status":"complete","status":"incomplete"}')
    with pytest.raises(DataValidationError):
        read_json(path)
    path.write_text('{"status":"incomplete"}')
    with pytest.raises(DataValidationError):
        load_package_manifest(tmp_path)


def test_atomic_output_no_clobber_or_partial(tmp_path):
    path = tmp_path / "output.json"
    publish_private_json(path, {"result": 1})
    with pytest.raises(DataValidationError):
        publish_private_json(path, {"result": 2})
    assert read_json(path) == {"result": 1}
    with pytest.raises(ValueError):
        publish_private_json(tmp_path / "bad.json", {"result": float("nan")})
    assert not (tmp_path / "bad.json").exists()
    assert not list(tmp_path.glob(".prediction-*"))
