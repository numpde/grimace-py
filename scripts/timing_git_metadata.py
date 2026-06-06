from __future__ import annotations

import argparse
from dataclasses import dataclass
import fnmatch
from pathlib import Path, PurePosixPath
import shlex
import subprocess


REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class StatusRecord:
    paths: tuple[str, ...]


def _git_output(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=REPO_ROOT, text=True).strip()


def _normalize_path(path: str) -> str:
    return PurePosixPath(path).as_posix().removeprefix("./")


def _path_is_ignored(path: str, ignore_patterns: tuple[str, ...]) -> bool:
    normalized = _normalize_path(path)
    for pattern in ignore_patterns:
        normalized_pattern = _normalize_path(pattern)
        if any(marker in normalized_pattern for marker in "*?["):
            if fnmatch.fnmatchcase(normalized, normalized_pattern):
                return True
            continue
        directory = normalized_pattern.rstrip("/")
        if normalized == directory or normalized.startswith(f"{directory}/"):
            return True
    return False


def parse_status_records(raw: bytes) -> tuple[StatusRecord, ...]:
    records: list[StatusRecord] = []
    fields = raw.decode("utf-8", errors="surrogateescape").split("\0")
    index = 0
    while index < len(fields):
        field = fields[index]
        index += 1
        if not field:
            continue
        if len(field) < 4 or field[2] != " ":
            raise ValueError(f"Unexpected git status record: {field!r}")

        status = field[:2]
        paths = [field[3:]]
        if "R" in status or "C" in status:
            if index >= len(fields) or not fields[index]:
                raise ValueError(f"Missing source path for git status record: {field!r}")
            paths.append(fields[index])
            index += 1
        records.append(StatusRecord(paths=tuple(paths)))
    return tuple(records)


def has_unignored_status(
    records: tuple[StatusRecord, ...],
    *,
    ignore_patterns: tuple[str, ...],
) -> bool:
    return any(
        not all(_path_is_ignored(path, ignore_patterns) for path in record.paths)
        for record in records
    )


def git_dirty_ignoring(*, ignore_patterns: tuple[str, ...]) -> bool:
    raw = subprocess.check_output(
        ["git", "status", "--porcelain=v1", "-z", "--untracked-files=all"],
        cwd=REPO_ROOT,
    )
    return has_unignored_status(
        parse_status_records(raw),
        ignore_patterns=ignore_patterns,
    )


def _shell_assignment(name: str, value: str) -> str:
    return f"{name}={shlex.quote(value)}"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Emit shell exports for timing metadata, ignoring generated timing "
            "artifacts when computing git_dirty."
        ),
    )
    parser.add_argument(
        "--ignore",
        action="append",
        default=[],
        help=(
            "Repository-relative generated artifact path or glob to ignore when "
            "computing GRIMACE_PERF_GIT_DIRTY. Directories ignore their contents."
        ),
    )
    args = parser.parse_args(argv)

    commit = _git_output("rev-parse", "--short=12", "HEAD")
    change = _git_output("log", "-1", "--format=%s", "HEAD")
    dirty = git_dirty_ignoring(ignore_patterns=tuple(args.ignore))

    print(_shell_assignment("GRIMACE_PERF_GIT_COMMIT", commit))
    print(_shell_assignment("GRIMACE_PERF_GIT_CHANGE", change))
    print(_shell_assignment("GRIMACE_PERF_GIT_DIRTY", "1" if dirty else "0"))
    print("export GRIMACE_PERF_GIT_COMMIT GRIMACE_PERF_GIT_CHANGE GRIMACE_PERF_GIT_DIRTY")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
