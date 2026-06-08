from __future__ import annotations

import json
from pathlib import Path
import re
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
HISTORY_PATH = REPO_ROOT / "notes" / "004_perf_history.jsonl"
REPORT_DIR = REPO_ROOT / "notes" / "perf_reports"

_PERF_REPORT_LINE = re.compile(
    r"^\s*(?:\d+:)?\s*"
    r"(?P<inclusive>[0-9.]+)%\s+"
    r"(?P<self>[0-9.]+)%\s+"
    r"(?P<comm>\S+)\s+"
    r"(?P<dso>\S+)\s+\[\.\]\s+"
    r"(?P<symbol>.+?)\s*$"
)


def append_history_record(record: dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


def latest_history_record(kind: str) -> dict[str, Any] | None:
    if not HISTORY_PATH.is_file():
        return None
    lines = HISTORY_PATH.read_text(encoding="utf-8").splitlines()
    for line_number in range(len(lines), 0, -1):
        line = lines[line_number - 1]
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Malformed timing history record at {HISTORY_PATH}:{line_number}"
            ) from exc
        if not isinstance(record, dict):
            raise ValueError(
                f"Timing history record at {HISTORY_PATH}:{line_number} "
                "is not an object"
            )
        if record.get("kind") == kind:
            return record
    return None


def parse_perf_report_top_symbols(report_text: str, *, limit: int = 20) -> list[dict[str, Any]]:
    hotspots: list[dict[str, Any]] = []
    for line in report_text.splitlines():
        match = _PERF_REPORT_LINE.match(line)
        if match is None:
            continue
        hotspots.append(
            {
                "inclusive_pct": float(match.group("inclusive")),
                "self_pct": float(match.group("self")),
                "comm": match.group("comm"),
                "dso": match.group("dso"),
                "symbol": match.group("symbol"),
            }
        )
        if len(hotspots) >= limit:
            break
    return hotspots


def sanitize_label(label: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "-", label).strip("-")
    return safe or "perf"
