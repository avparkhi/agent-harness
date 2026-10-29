from pathlib import Path

BASE = Path("uploads")


def upload_path(filename: str) -> Path:
    """Where to store a user's uploaded file."""
    return BASE / filename
