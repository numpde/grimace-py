from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PinnedFixtureCase:
    case_id: str
    source: str
    raw: dict[str, object]
    fixture_path: Path


def normalized_unique_sorted_strings(
    values: list[object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    expected = tuple(sorted(set(normalized)))
    if normalized != expected:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} has non-canonical {field_name}; "
            f"expected sorted unique strings {expected!r}, got {normalized!r}"
        )
    return normalized


def optional_positive_int(
    raw_case: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> int | None:
    raw_value = raw_case.get(field_name)
    if raw_value is None:
        return None
    value = int(raw_value)
    if value <= 0:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} has nonpositive "
            f"{field_name}={value}"
        )
    return value


def load_pinned_rdkit_fixture_cases(
    *,
    fixture_root: Path,
    rdkit_version: str,
    fixture_label: str,
) -> tuple[PinnedFixtureCase, ...]:
    fixture_path = fixture_root / f"{rdkit_version}.json"
    if not fixture_path.is_file():
        raise FileNotFoundError(
            f"no pinned {fixture_label} fixture for RDKit {rdkit_version}"
        )

    data = json.loads(fixture_path.read_text())
    if data.get("rdkit_version") != rdkit_version:
        raise ValueError(
            f"fixture {fixture_path} declares rdkit_version={data.get('rdkit_version')!r}, "
            f"expected {rdkit_version!r}"
        )

    cases = []
    seen_ids: set[str] = set()
    for raw_case in data["cases"]:
        case_id = str(raw_case["id"])
        if not case_id:
            raise ValueError(f"fixture {fixture_path} contains an empty case id")
        if case_id in seen_ids:
            raise ValueError(
                f"fixture {fixture_path} contains duplicate case id {case_id!r}"
            )
        seen_ids.add(case_id)

        source = str(raw_case["source"])
        if not source:
            raise ValueError(
                f"fixture {fixture_path} case {case_id!r} contains an empty source"
            )

        cases.append(
            PinnedFixtureCase(
                case_id=case_id,
                source=source,
                raw=raw_case,
                fixture_path=fixture_path,
            )
        )

    return tuple(cases)
