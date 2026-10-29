from pager import paginate


def test_returns_a_list():
    assert isinstance(paginate(list(range(5)), 1, 2), list)
