from tags import add_tag


def test_calls_do_not_share_state():
    assert add_tag("a") == ["a"]
    assert add_tag("b") == ["b"]


def test_appends_to_a_given_list():
    existing = ["x"]
    assert add_tag("y", existing) is existing
    assert existing == ["x", "y"]
