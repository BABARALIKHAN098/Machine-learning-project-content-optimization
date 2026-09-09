import json

import pytest

from machine_learning_project.models.evaluation_artifacts import (
    evaluation_directory,
    load_evaluation,
    payload_path,
)
from machine_learning_project.utils.exceptions import DataValidationError


@pytest.mark.parametrize("name", ["../escape", "a/b", "CON", "LPT1", "a:b", ""])
def test_unsafe_run_names(tmp_path, name):
    with pytest.raises(DataValidationError):
        evaluation_directory({"output_root": str(tmp_path)}, name, [])


def test_output_protection_and_incomplete_manifest(tmp_path):
    protected = tmp_path / "models/model.joblib"
    with pytest.raises(DataValidationError):
        evaluation_directory({"output_root": str(protected.parent)}, "run", [protected])
    with pytest.raises(DataValidationError):
        payload_path(tmp_path, "../decision.json")
    (tmp_path / "evaluation_manifest.json").write_text(json.dumps({"status": "incomplete"}))
    with pytest.raises(DataValidationError):
        load_evaluation(tmp_path)


def test_legacy_cli_wording(tmp_path, capsys, monkeypatch):
    from scripts.evaluate_model import main

    path = tmp_path / "old.json"
    path.write_text(
        json.dumps(
            {
                "selected_model": "logistic_regression",
                "candidates_validation": {"logistic_regression": {"macro_f1": 0.4}},
            }
        )
    )
    monkeypatch.setattr("sys.argv", ["evaluate_model.py", "--metrics", str(path)])
    main()
    output = capsys.readouterr().out
    assert "Historical test evaluation exists" in output
    assert "has not been run" not in output
