from importer import load_contacts, parse_line


def test_quoted_comma():
    assert parse_line('"Smith, John",42') == ["Smith, John", "42"]


def test_plain_line():
    assert parse_line("Ada,36") == ["Ada", "36"]


def test_load_contacts():
    assert load_contacts('Ada,36\n"Smith, John",42\n') == [["Ada", "36"], ["Smith, John", "42"]]
