from dedupe import dedupe


def test_keeps_first_occurrence_order():
    assert dedupe(["c@x", "a@x", "c@x", "b@x", "a@x"]) == ["c@x", "a@x", "b@x"]


def test_order_is_deterministic():
    # Small ints hash to themselves, so set() returns them sorted: [1, 2, 3]. A string-only
    # test can pass by luck, because string hashing is randomized per process.
    assert dedupe([3, 1, 3, 2, 1]) == [3, 1, 2]


def test_empty():
    assert dedupe([]) == []
