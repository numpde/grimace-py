from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import re
import socket
import subprocess
import sys
from typing import Any

from rdkit import rdBase


REPO_ROOT = Path(__file__).resolve().parents[2]
HISTORY_PATH = REPO_ROOT / "notes" / "perf_history.jsonl"
REPORT_DIR = REPO_ROOT / "notes" / "perf_reports"

_PERF_REPORT_LINE = re.compile(
    r"^\s*(?:\d+:)?\s*"
    r"(?P<inclusive>[0-9.]+)%\s+"
    r"(?P<self>[0-9.]+)%\s+"
    r"(?P<comm>\S+)\s+"
    r"(?P<dso>\S+)\s+\[\.\]\s+"
    r"(?P<symbol>.+?)\s*$"
)


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def current_run_metadata(*, change_label: str | None = None) -> dict[str, Any]:
    commit = git_output("rev-parse", "--short=12", "HEAD")
    change = change_label or git_output("log", "-1", "--format=%s", "HEAD")
    dirty = bool(git_output("status", "--short"))
    return {
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": commit,
        "git_change": change,
        "git_dirty": dirty,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "rdkit": rdBase.rdkitVersion,
    }


def append_history_record(record: dict[str, Any]) -> None:
    HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True) + "\n")


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
