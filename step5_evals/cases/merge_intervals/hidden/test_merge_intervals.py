from intervals import merge


def test_unsorted_overlap():
    assert merge([(5, 7), (1, 6)]) == [[1, 7]]


def test_disjoint_stay_apart():
    assert merge([(3, 4), (1, 2)]) == [[1, 2], [3, 4]]


def test_touching_and_empty():
    assert merge([(1, 3), (3, 5)]) == [[1, 5]]
    assert merge([]) == []
