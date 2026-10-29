from pathlib import Path

import pytest

from storage import upload_path


def test_normal_names_stay_inside():
    assert Path(upload_path("photo.png")).resolve() == (Path("uploads") / "photo.png").resolve()
    assert Path(upload_path("2026/report.pdf")).resolve() == (Path("uploads") / "2026/report.pdf").resolve()


@pytest.mark.parametrize("name", ["../../etc/passwd", "a/../../secret", "/etc/passwd"])
def test_escapes_are_rejected(name):
    with pytest.raises(ValueError):
        upload_path(name)
