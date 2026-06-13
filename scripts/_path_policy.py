"""Shared path checks for local maintenance scripts."""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def checked_output_path(
    path: Path,
    *,
    approved_roots: Iterable[Path],
    allow_outside_repo: bool,
    force: bool,
) -> Path:
    output = path if path.is_absolute() else ROOT / path
    output = output.resolve(strict=False)
    approved = tuple(
        (root if root.is_absolute() else ROOT / root).resolve(strict=False)
        for root in approved_roots
    )
    if not allow_outside_repo and not any(
        output == root or output.is_relative_to(root)
        for root in approved
    ):
        formatted_roots = ", ".join(_display_path(root) for root in approved)
        raise ValueError(
            f"output path must be under one of: {formatted_roots}; "
            "pass --allow-outside-repo for an explicit scratch path"
        )
    if output.exists() and not force:
        raise FileExistsError(f"{output} already exists; pass --force to overwrite it")
    return output
