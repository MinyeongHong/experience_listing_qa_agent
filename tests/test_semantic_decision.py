import pytest
from pydantic import ValidationError

from app.domain.models import SemanticDecision


@pytest.mark.parametrize("reason", ["", " ", "\t\r\n", "\u2003\u3000"])
def test_semantic_decision_rejects_blank_reason(reason):
    with pytest.raises(ValidationError) as exc_info:
        SemanticDecision(verdict="uncertain", reason=reason)

    assert exc_info.value.errors()[0]["loc"] == ("reason",)


@pytest.mark.parametrize("verdict", ["possible_conflict", "no_conflict", "uncertain"])
@pytest.mark.parametrize("reason", ["x", "Different services", "  Unclear scope\n"])
def test_semantic_decision_preserves_valid_reason(verdict, reason):
    decision = SemanticDecision(verdict=verdict, reason=reason)

    assert decision.verdict == verdict
    assert decision.reason == reason


@pytest.mark.parametrize("reason", ["\x00", "Scope\x00 unclear", "Unclear\x00"])
def test_semantic_decision_rejects_nul_in_reason(reason):
    with pytest.raises(ValidationError) as exc_info:
        SemanticDecision(verdict="uncertain", reason=reason)
    assert exc_info.value.errors()[0]["loc"] == ("reason",)
