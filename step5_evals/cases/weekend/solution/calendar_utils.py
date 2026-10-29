from datetime import date


def is_weekend(d: date) -> bool:
    return d.weekday() >= 5
