from datetime import date

from calendar_utils import is_weekend


def test_saturday_and_sunday():
    assert is_weekend(date(2026, 10, 31))   # Saturday
    assert is_weekend(date(2026, 11, 1))    # Sunday


def test_weekdays():
    assert not is_weekend(date(2026, 10, 30))   # Friday
    assert not is_weekend(date(2026, 11, 2))    # Monday
