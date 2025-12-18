"""Path utilities for the GameProofer sample project."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence


def ensure_directories(paths: Iterable[Path]) -> None:
    """Create each directory path if it does not already exist."""
    for directory in paths:
        directory.mkdir(parents=True, exist_ok=True)


def validate_file(path: Path, *, allowed_suffixes: Sequence[str] | None = None) -> Path:
    """Ensure that the file exists and matches the optional suffix whitelist."""
    if not path.exists():
        raise FileNotFoundError(f"Expected file at {path} but it does not exist.")

    if allowed_suffixes and path.suffix.lower() not in allowed_suffixes:
        raise ValueError(
            f"File {path} must have one of the following suffixes: {allowed_suffixes}."
        )

    return path
