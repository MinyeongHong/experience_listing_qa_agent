import pytest
from pydantic import ValidationError

from app.domain.models import Listing


def test_listing_allows_missing_city_and_duration():
    listing = Listing(
        title="Historic Walking Tour",
        city=None,
        price=45.0,
        meeting_point="La Rambla",
        inclusions=["Lunch"],
        exclusions=["Drinks"],
    )

    assert listing.title == "Historic Walking Tour"
    assert listing.city is None
    assert listing.duration_minutes is None


def test_listing_rejects_zero_duration():
    with pytest.raises(ValidationError):
        Listing(
            title="Historic Walking Tour",
            city="Rome",
            duration_minutes=0,
            price=45.0,
            meeting_point="La Rambla",
            inclusions=["Lunch"],
            exclusions=["Drinks"],
        )


def test_listing_rejects_blank_title():
    with pytest.raises(ValidationError):
        Listing(
            title="   ",
            city="Rome",
            duration_minutes=90,
            price=45.0,
            meeting_point="La Rambla",
            inclusions=["Lunch"],
            exclusions=["Drinks"],
        )


@pytest.mark.parametrize("mentions", [[0], [-1], [1, 0], [1, -1]])
def test_listing_rejects_nonpositive_duration_mentions(mentions):
    with pytest.raises(ValidationError) as exc_info:
        Listing(title="Tour", duration_mentions=mentions)

    assert exc_info.value.errors()[0]["loc"][0] == "duration_mentions"


@pytest.mark.parametrize("mentions", [[], [1], [120, 120], [120, 180]])
def test_listing_preserves_valid_duration_mentions_and_optional_fields(mentions):
    listing = Listing(title="Tour", duration_mentions=mentions)

    assert listing.duration_mentions == mentions
    assert listing.duration_minutes is None
    assert listing.city is None
    assert listing.price is None
    assert listing.meeting_point is None


def test_listing_defaults_duration_mentions_to_independent_empty_lists():
    first = Listing(title="First tour")
    second = Listing(title="Second tour")

    assert first.duration_mentions == []
    first.duration_mentions.append(1)
    assert second.duration_mentions == []
