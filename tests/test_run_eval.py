import json
import sys
from unittest.mock import Mock

import pytest

from app.domain.models import Listing
from app.services.extraction import ExtractionError
from evals import run_eval


@pytest.fixture
def evaluation(monkeypatch, tmp_path):
    listing = Listing(title="Tour", city="Rome", duration_minutes=60, meeting_point="Gate")
    (tmp_path / "listing.txt").write_text("A tour in Rome", encoding="utf-8")
    cases = {
        "tour": {
            "source_file": "listing.txt",
            "expected_fields": listing.model_dump(),
            "expected_issue_codes": [],
        }
    }
    monkeypatch.setattr(run_eval, "EXAMPLES_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(run_eval, "load_cases", lambda: cases)
    monkeypatch.setattr(sys, "argv", ["run_eval"])
    client = Mock()
    client.models.generate_content.return_value.parsed = listing
    monkeypatch.setattr(run_eval, "create_gemini_client", lambda: client)
    return tmp_path, listing, client


def test_live_evaluation_and_replay(evaluation, monkeypatch):
    directory, _, client = evaluation
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    run_eval.main()
    result = json.loads((directory / "baseline.json").read_text())
    assert result["passed_cases"] == 1
    assert client.models.generate_content.call_args.kwargs["model"] == "test-model"
    client.close.assert_called_once()

    monkeypatch.setattr(sys, "argv", ["run_eval", "--replay"])
    monkeypatch.setattr(
        run_eval, "create_gemini_client", Mock(side_effect=AssertionError("No live calls"))
    )
    run_eval.main()
    replay = json.loads((directory / "replay.json").read_text())
    assert replay["passed_cases"] == 1


def test_live_evaluation_records_extraction_failure(evaluation):
    directory, _, client = evaluation
    client.models.generate_content.side_effect = ExtractionError("Unavailable")
    run_eval.main()
    result = json.loads((directory / "baseline.json").read_text())
    assert result["evaluated_cases"] == 0
    assert result["cases"][0]["passed"] is False
    assert "error" in result["cases"][0]
    client.close.assert_called_once()


def test_live_evaluation_closes_client_on_unexpected_failure(evaluation, monkeypatch):
    _, _, client = evaluation
    monkeypatch.setattr(run_eval, "run_case", Mock(side_effect=RuntimeError("Unexpected")))
    with pytest.raises(RuntimeError, match="Unexpected"):
        run_eval.main()
    client.close.assert_called_once()


@pytest.mark.parametrize("relative", [False, True])
def test_live_evaluation_uses_custom_output_without_overwriting_baseline(
    evaluation, monkeypatch, relative
):
    directory, _, _ = evaluation
    baseline_path = directory / "baseline.json"
    baseline_path.write_text("existing baseline", encoding="utf-8")
    monkeypatch.chdir(directory)
    output_path = directory / "new" / "nested" / "candidate.json"
    argument = output_path.relative_to(directory) if relative else output_path
    monkeypatch.setattr(sys, "argv", ["run_eval", "--output", str(argument)])

    run_eval.main()

    assert json.loads(output_path.read_text(encoding="utf-8"))["passed_cases"] == 1
    assert baseline_path.read_text(encoding="utf-8") == "existing baseline"


def test_replay_keeps_existing_paths_when_output_is_specified(evaluation, monkeypatch):
    directory, _, _ = evaluation
    run_eval.main()
    baseline_path = directory / "baseline.json"
    baseline = baseline_path.read_bytes()
    output_path = directory / "unused" / "candidate.json"
    monkeypatch.setattr(
        sys, "argv", ["run_eval", "--replay", "--output", str(output_path)]
    )
    monkeypatch.setattr(
        run_eval, "create_gemini_client", Mock(side_effect=AssertionError("No live calls"))
    )

    run_eval.main()

    assert json.loads((directory / "replay.json").read_text())["passed_cases"] == 1
    assert baseline_path.read_bytes() == baseline
    assert not output_path.parent.exists()


@pytest.mark.parametrize("default_writable", [True, False])
def test_custom_output_does_not_create_or_require_default_results_dir(
    evaluation, monkeypatch, default_writable
):
    directory, _, _ = evaluation
    default_results = directory / "default-results"
    output_path = directory / "custom" / "candidate.json"
    monkeypatch.setattr(run_eval, "RESULTS_DIR", default_results)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--output", str(output_path)])
    original_mkdir = run_eval.Path.mkdir

    def guarded_mkdir(path, *args, **kwargs):
        if path == default_results and not default_writable:
            raise PermissionError("Default results directory is read-only")
        return original_mkdir(path, *args, **kwargs)

    monkeypatch.setattr(run_eval.Path, "mkdir", guarded_mkdir)

    run_eval.main()

    assert json.loads(output_path.read_text())["passed_cases"] == 1
    assert not default_results.exists()


def test_directory_output_is_rejected_before_client_creation(evaluation, monkeypatch):
    directory, _, client = evaluation
    create_client = Mock(return_value=client)
    monkeypatch.setattr(run_eval, "create_gemini_client", create_client)
    monkeypatch.setattr(sys, "argv", ["run_eval", "--output", str(directory)])

    with pytest.raises(SystemExit) as exc_info:
        run_eval.main()

    assert exc_info.value.code == 2
    create_client.assert_not_called()
    client.models.generate_content.assert_not_called()
