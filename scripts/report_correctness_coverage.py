"""Report pinned RDKit correctness evidence from fixture JSON.

The report is intentionally derived from pinned RDKit fixture families and the
RDKit serializer coverage ledger. It does not import RDKit or Grimace, so it
can run inside the offline checks lane.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys
from typing import Any, cast


REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.helpers.pinned_rdkit_fixtures import (  # noqa: E402
    PINNED_RDKIT_KNOWN_QUIRKS,
    PINNED_RDKIT_KNOWN_STEREO_GAPS,
    PINNED_RDKIT_PARITY_FIXTURE_FAMILIES,
    PINNED_RDKIT_ROOTED_RANDOM,
    PinnedFixtureCase,
    load_pinned_rdkit_fixture_cases,
    pinned_rdkit_fixture_versions,
)
from tests.helpers.rdkit_serializer_coverage import (  # noqa: E402
    COVERAGE_STATUS_COVERED,
    COVERAGE_STATUS_KNOWN_GAP,
    COVERAGE_STATUS_OUT_OF_SCOPE,
    load_serializer_coverage,
)

_PINNED_RDKIT_REPORT_FIXTURE_FAMILIES = (
    *PINNED_RDKIT_PARITY_FIXTURE_FAMILIES,
    PINNED_RDKIT_KNOWN_QUIRKS,
    PINNED_RDKIT_KNOWN_STEREO_GAPS,
)

_SOURCE_CLASS_UPSTREAM_RDKIT = "upstream-rdkit"
_SOURCE_CLASS_LOCAL_PROBE = "local-probe"
_SOURCE_CLASS_DATASET_DERIVED = "dataset-derived"
_SOURCE_CLASS_RANDOM_WRITER_OBSERVATION = "random-writer-observation"
_SOURCE_CLASS_KNOWN_RDKIT_GAP = "known-rdkit-gap"
_SOURCE_CLASS_RDKIT_QUIRK = "rdkit-quirk"
_SOURCE_CLASS_ORDER: tuple[str, ...] = (
    _SOURCE_CLASS_UPSTREAM_RDKIT,
    _SOURCE_CLASS_LOCAL_PROBE,
    _SOURCE_CLASS_DATASET_DERIVED,
    _SOURCE_CLASS_RANDOM_WRITER_OBSERVATION,
    _SOURCE_CLASS_KNOWN_RDKIT_GAP,
    _SOURCE_CLASS_RDKIT_QUIRK,
)
_STATUSES_REQUIRING_EXECUTABLE_LINKS = (
    COVERAGE_STATUS_COVERED,
    COVERAGE_STATUS_KNOWN_GAP,
)


def _source_class(family: str, case: PinnedFixtureCase) -> str:
    source = case.source
    if family == PINNED_RDKIT_KNOWN_STEREO_GAPS:
        return _SOURCE_CLASS_KNOWN_RDKIT_GAP
    if family == PINNED_RDKIT_KNOWN_QUIRKS:
        return _SOURCE_CLASS_RDKIT_QUIRK
    if source.startswith("Dataset-derived "):
        return _SOURCE_CLASS_DATASET_DERIVED
    if (
        family == PINNED_RDKIT_ROOTED_RANDOM
        or "rdkit_sample_draw_budget" in case.raw
        or "rdkit_random_vector_seed" in case.raw
    ):
        return _SOURCE_CLASS_RANDOM_WRITER_OBSERVATION
    if source.startswith("RDKit "):
        return _SOURCE_CLASS_UPSTREAM_RDKIT
    if source.startswith("Local ") or source.startswith("RDKit-grounded"):
        return _SOURCE_CLASS_LOCAL_PROBE
    raise ValueError(
        f"{case.fixture_path}: {case.case_id} has unclassified source: {source!r}"
    )


def _fixture_case_ids(
    *,
    repo_root: Path,
    fixture_path: str,
    case_ids_by_fixture: dict[str, frozenset[str]],
) -> frozenset[str]:
    cached = case_ids_by_fixture.get(fixture_path)
    if cached is not None:
        return cached
    if Path(fixture_path).is_absolute() or ".." in Path(fixture_path).parts:
        raise ValueError(
            "serializer ledger fixture path is not repo-relative: "
            f"{fixture_path}"
        )
    resolved = repo_root / fixture_path
    fixture_root = repo_root / "tests" / "fixtures"
    if not resolved.is_relative_to(fixture_root):
        raise ValueError(
            "serializer ledger fixture path is outside tests/fixtures: "
            f"{fixture_path}"
        )
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or not cases:
        raise ValueError(f"serializer ledger fixture has no cases: {fixture_path}")
    case_ids = frozenset(
        case["id"]
        for case in cases
        if isinstance(case, dict) and isinstance(case.get("id"), str)
    )
    if len(case_ids) != len(cases):
        raise ValueError(
            f"serializer ledger fixture has invalid case ids: {fixture_path}"
        )
    case_ids_by_fixture[fixture_path] = case_ids
    return case_ids


def _validate_serializer_links(
    *,
    repo_root: Path,
    coverage_path: Path,
    entry: dict[str, Any],
    case_ids_by_fixture: dict[str, frozenset[str]],
) -> None:
    entry_id = entry.get("id")
    status = cast(str, entry["status"])
    links = entry.get("grimace_links")
    if not isinstance(links, list):
        raise ValueError(
            f"{coverage_path}: entry {entry_id!r} must define grimace_links"
        )
    if status in _STATUSES_REQUIRING_EXECUTABLE_LINKS and not links:
        raise ValueError(
            f"{coverage_path}: entry {entry_id!r} with status {status!r} "
            "must link executable fixture cases"
        )
    if status == COVERAGE_STATUS_OUT_OF_SCOPE and links:
        raise ValueError(
            f"{coverage_path}: entry {entry_id!r} with status {status!r} "
            "must not link executable fixture cases"
        )

    for link in links:
        if not isinstance(link, dict):
            raise ValueError(f"{coverage_path}: entry {entry_id!r} has invalid link")
        fixture_path = link.get("fixture")
        cases = link.get("cases")
        if not isinstance(fixture_path, str) or not fixture_path:
            raise ValueError(
                f"{coverage_path}: entry {entry_id!r} has link without fixture"
            )
        if not isinstance(cases, list) or not cases or not all(
            isinstance(case_id, str) and case_id for case_id in cases
        ):
            raise ValueError(
                f"{coverage_path}: entry {entry_id!r} has link without cases"
            )

        available_case_ids = _fixture_case_ids(
            repo_root=repo_root,
            fixture_path=fixture_path,
            case_ids_by_fixture=case_ids_by_fixture,
        )
        missing = sorted(set(cases) - available_case_ids)
        if missing:
            raise ValueError(
                f"{coverage_path}: entry {entry_id!r} references missing "
                f"cases in {fixture_path}: {missing}"
            )


def build_summary(repo_root: Path = REPO_ROOT) -> dict[str, Any]:
    fixture_root = repo_root / "tests" / "fixtures"
    fixture_cases: dict[str, Counter[str]] = {
        family: Counter() for family in _PINNED_RDKIT_REPORT_FIXTURE_FAMILIES
    }
    source_classes: Counter[str] = Counter()

    for family in _PINNED_RDKIT_REPORT_FIXTURE_FAMILIES:
        family_root = fixture_root / family
        if not family_root.is_dir():
            raise FileNotFoundError(
                f"missing pinned RDKit fixture family: {family_root}"
            )
        versions = pinned_rdkit_fixture_versions(family_root)
        if not versions:
            raise FileNotFoundError(
                f"pinned RDKit fixture family has no JSON cases: {family_root}"
            )
        for version in versions:
            cases = load_pinned_rdkit_fixture_cases(
                fixture_root=family_root,
                rdkit_version=version,
                fixture_label=family,
            )
            if not cases:
                raise FileNotFoundError(
                    f"pinned RDKit fixture has no cases: {family_root / version}"
                )
            for case in cases:
                fixture_cases[family][version] += 1
                source_classes[_source_class(family, case)] += 1

    coverage_root = fixture_root / "rdkit_upstream_serializer_coverage"
    if not coverage_root.is_dir():
        raise FileNotFoundError(
            f"missing RDKit serializer coverage ledger: {coverage_root}"
        )
    coverage_paths = sorted(coverage_root.glob("*.json"))
    if not coverage_paths:
        raise FileNotFoundError(
            f"RDKit serializer coverage ledger has no JSON files: {coverage_root}"
        )
    serializer_ledger_statuses: dict[str, Counter[str]] = defaultdict(Counter)
    case_ids_by_fixture: dict[str, frozenset[str]] = {}
    for path in coverage_paths:
        payload = load_serializer_coverage(path)
        version = payload.get("rdkit_version")
        if not isinstance(version, str) or not version:
            raise ValueError(f"{path} must define nonempty string rdkit_version")
        entries = cast(list[dict[str, Any]], payload["entries"])
        for entry in entries:
            _validate_serializer_links(
                repo_root=repo_root,
                coverage_path=path,
                entry=entry,
                case_ids_by_fixture=case_ids_by_fixture,
            )
            status = cast(str, entry["status"])
            serializer_ledger_statuses[version][status] += 1

    return {
        "fixture_cases": {
            family: dict(sorted(counts.items()))
            for family, counts in fixture_cases.items()
        },
        "source_classes": {
            source_class: source_classes[source_class]
            for source_class in _SOURCE_CLASS_ORDER
            if source_classes[source_class]
        },
        "serializer_ledger_statuses": {
            version: dict(sorted(counts.items()))
            for version, counts in sorted(serializer_ledger_statuses.items())
        },
    }


def _format_counts(counts: dict[str, int]) -> list[str]:
    return [f"- `{key}`: {value}" for key, value in counts.items()]


def format_text(summary: dict[str, Any]) -> str:
    lines = ["Pinned RDKit correctness evidence summary", ""]
    lines.append("Pinned fixture cases by family and RDKit version:")
    for family, versions in summary["fixture_cases"].items():
        for version, count in versions.items():
            lines.append(f"- `{family}` / `{version}`: {count}")
    lines.append("")
    lines.append("Fixture cases by source class:")
    lines.extend(_format_counts(summary["source_classes"]))
    lines.append("")
    lines.append("RDKit serializer ledger statuses:")
    for version, statuses in summary["serializer_ledger_statuses"].items():
        lines.append(f"- `{version}`:")
        for status, count in statuses.items():
            lines.append(f"  - `{status}`: {count}")
    return "\n".join(lines)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Report pinned RDKit correctness evidence from fixture JSON.",
    )
    parser.add_argument(
        "--format",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = build_summary()
    if args.format == "json":
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        print(format_text(summary))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
