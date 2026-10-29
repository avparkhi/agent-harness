import pytest

from client import fetch_with_retry


class Flaky:
    def __init__(self, errors, result="ok"):
        self.errors, self.result, self.calls = list(errors), result, 0

    def __call__(self):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


def test_succeeds_after_a_retry():
    fetch = Flaky([ConnectionError()])
    assert fetch_with_retry(fetch, delay=0) == "ok"
    assert fetch.calls == 2


def test_raises_after_last_attempt():
    fetch = Flaky([ConnectionError("1"), ConnectionError("2"), ConnectionError("3")])
    with pytest.raises(ConnectionError, match="3"):
        fetch_with_retry(fetch, attempts=3, delay=0)
    assert fetch.calls == 3


def test_other_errors_are_not_retried():
    fetch = Flaky([ValueError("bad input")])
    with pytest.raises(ValueError):
        fetch_with_retry(fetch, delay=0)
    assert fetch.calls == 1
