from pathlib import Path

BASE = Path("uploads")


def upload_path(filename: str) -> Path:
    """Where to store a user's uploaded file."""
    path = (BASE / filename).resolve()
    if not path.is_relative_to(BASE.resolve()):
        raise ValueError(f"{filename!r} is outside the uploads folder")
    return path
