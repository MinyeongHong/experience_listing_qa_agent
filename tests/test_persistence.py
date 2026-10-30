from unittest.mock import MagicMock

import pytest
from sqlalchemy.exc import SQLAlchemyError

from app.domain.models import AnalysisResult, Listing
from app.services.persistence import save_analysis


def test_save_analysis_rolls_back_when_commit_fails():
    session = MagicMock()
    session.commit.side_effect = SQLAlchemyError("Commit failed")

    listing = Listing(
        title="Barcelona Walking Tour",
        city="Barcelona",
        duration_minutes=120,
        price=35,
        meeting_point="Plaça de Catalunya",
        inclusions=["Local guide"],
        exclusions=[],
    )

    result = AnalysisResult(
        extracted_fields=listing,
        issues=[],
        confidence=1.0,
    )

    with pytest.raises(SQLAlchemyError, match="Commit failed"):
        save_analysis(
            session=session,
            raw_text="Barcelona walking tour",
            result=result,
        )

    session.add.assert_called_once()
    session.commit.assert_called_once()
    session.rollback.assert_called_once()
    session.refresh.assert_not_called()
