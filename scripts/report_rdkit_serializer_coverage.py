from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from tests.helpers.rdkit_serializer_coverage import (
    COVERAGE_STATUS_COVERED,
    DEFAULT_RDKIT_SERIALIZER_VERSION,
    UNTRIAGED_COVERAGE_STATUSES,
    default_serializer_coverage_path,
)

DEFAULT_COVERAGE = default_serializer_coverage_path(DEFAULT_RDKIT_SERIALIZER_VERSION)


def _load_entries(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    return payload["entries"]


def _counter(values) -> Counter[str]:
    return Counter(str(value) for value in values)


def _print_counter(title: str, counter: Counter[str]) -> None:
    print(title)
    for key, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
        print(f"  {count:4d}  {key}")
    print()


def _entry_label(entry: dict[str, Any]) -> str:
    parent = entry["parent"]
    name = entry["name"]
    title = f"{parent} / {name}" if parent else name
    return (
        f"{entry['id']} | {entry['upstream_file']}:{entry['start_line']}-"
        f"{entry['end_line']} | {title}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Summarize the RDKit serializer coverage triage ledger.",
    )
    parser.add_argument("--coverage", type=Path, default=DEFAULT_COVERAGE)
    parser.add_argument(
        "--status",
        action="append",
        help="print entries with this status; may be passed more than once",
    )
    parser.add_argument("--limit", type=int, default=40)
    parser.add_argument(
        "--fail-untriaged",
        action="store_true",
        help="exit nonzero if any entries remain unreviewed or need fixtures",
    )
    args = parser.parse_args(argv)

    coverage_path = args.coverage.resolve()
    entries = _load_entries(coverage_path)
    print(f"{coverage_path.relative_to(REPO_ROOT)}")
    print(f"entries: {len(entries)}")
    print()

    _print_counter("by status", _counter(entry["status"] for entry in entries))
    _print_counter("by claim", _counter(entry["claim"] for entry in entries))
    _print_counter("by upstream file", _counter(entry["upstream_file"] for entry in entries))
    _print_counter("by kind", _counter(entry["kind"] for entry in entries))
    _print_counter(
        "by matched term",
        _counter(term for entry in entries for term in entry["matched_terms"]),
    )

    linked = sum(len(entry["grimace_links"]) for entry in entries)
    covered = [entry for entry in entries if entry["status"] == COVERAGE_STATUS_COVERED]
    print(f"covered entries: {len(covered)}")
    print(f"total grimace links: {linked}")
    print()

    if args.status:
        wanted = set(args.status)
        selected = [entry for entry in entries if entry["status"] in wanted]
        print(f"entries with status in {sorted(wanted)}: {len(selected)}")
        for entry in selected[:args.limit]:
            print(f"  {_entry_label(entry)}")
        if len(selected) > args.limit:
            print(f"  ... {len(selected) - args.limit} more")

    if args.fail_untriaged:
        untriaged = [
            entry
            for entry in entries
            if entry["status"] in UNTRIAGED_COVERAGE_STATUSES
        ]
        if untriaged:
            print(
                f"untriaged entries remain: {len(untriaged)} "
                "(status is unreviewed or needs-fixture)"
            )
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
