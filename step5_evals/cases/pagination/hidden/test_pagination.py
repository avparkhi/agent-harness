from pager import paginate

ITEMS = list(range(25))


def test_first_page():
    assert paginate(ITEMS, 1, 10) == list(range(10))


def test_last_partial_page():
    assert paginate(ITEMS, 3, 10) == [20, 21, 22, 23, 24]


def test_past_the_end_is_empty():
    assert paginate(ITEMS, 4, 10) == []
