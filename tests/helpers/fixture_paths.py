from __future__ import annotations

from pathlib import Path


CHECKED_IN_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


def checked_in_fixture_path(*parts: str) -> Path:
    return CHECKED_IN_FIXTURE_ROOT.joinpath(*parts)
