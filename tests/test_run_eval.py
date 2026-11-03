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
