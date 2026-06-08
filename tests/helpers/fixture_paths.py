from __future__ import annotations

import json
from pathlib import Path
from typing import Any


CHECKED_IN_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


def checked_in_fixture_path(*parts: str) -> Path:
    return CHECKED_IN_FIXTURE_ROOT.joinpath(*parts)


def read_fixture_json_object(path: Path, *, context: str = "fixture") -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{context} {path} is not readable JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{context} {path} must contain a JSON object")
    return payload
