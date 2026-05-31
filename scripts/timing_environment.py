from __future__ import annotations

import os
from datetime import datetime, timezone
from pathlib import Path
import platform
import socket
import subprocess
import sys
from typing import Any

from rdkit import rdBase


REPO_ROOT = Path(__file__).resolve().parents[1]


def git_output(*args: str) -> str:
    return subprocess.check_output(
        ["git", *args],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _cpu_model_name() -> str | None:
    cpuinfo = Path("/proc/cpuinfo")
    if not cpuinfo.is_file():
        return None

    for line in cpuinfo.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition(":")
        if separator and key.strip() in {"model name", "Hardware", "Processor"}:
            value = value.strip()
            if value:
                return value
    return None


def _visible_cpu_count() -> int | None:
    try:
        return len(os.sched_getaffinity(0))
    except AttributeError:
        return os.cpu_count()


def _cgroup_memory_limit_bytes() -> int | None:
    candidates = (
        Path("/sys/fs/cgroup/memory.max"),
        Path("/sys/fs/cgroup/memory/memory.limit_in_bytes"),
    )
    for path in candidates:
        if not path.is_file():
            continue
        value = path.read_text(encoding="utf-8").strip()
        if not value or value == "max":
            continue
        try:
            limit = int(value)
        except ValueError:
            continue
        if limit > 0:
            return limit
    return None


def current_machine_metadata() -> dict[str, Any]:
    return {
        "cpu_model": _cpu_model_name(),
        "visible_cpus": _visible_cpu_count(),
        "cgroup_memory_limit_bytes": _cgroup_memory_limit_bytes(),
    }


def current_run_metadata(*, change_label: str | None = None) -> dict[str, Any]:
    commit = os.environ.get("GRIMACE_PERF_GIT_COMMIT") or git_output(
        "rev-parse",
        "--short=12",
        "HEAD",
    )
    change = (
        change_label
        or os.environ.get("GRIMACE_PERF_GIT_CHANGE")
        or git_output("log", "-1", "--format=%s", "HEAD")
    )
    dirty_env = os.environ.get("GRIMACE_PERF_GIT_DIRTY")
    dirty = (
        dirty_env.strip().lower() in {"1", "true", "yes"}
        if dirty_env is not None and dirty_env != ""
        else bool(git_output("status", "--short"))
    )
    return {
        "recorded_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "git_commit": commit,
        "git_change": change,
        "git_dirty": dirty,
        "hostname": socket.gethostname(),
        "platform": platform.platform(),
        "python": sys.version.split()[0],
        "rdkit": rdBase.rdkitVersion,
        "machine": current_machine_metadata(),
    }
