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
