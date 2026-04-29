from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.helpers.pinned_rdkit_fixtures import (
    load_pinned_rdkit_fixture_cases,
    required_string,
    required_string_tuple,
)


@dataclass(frozen=True, slots=True)
class PinnedRootedRandomCase:
    case_id: str
    source: str
    smiles: str
    rooted_outputs: tuple[str, ...]


_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "rdkit_rooted_random"
)


def load_pinned_rooted_random_cases(
    rdkit_version: str,
    *,
    fixture_root: Path = _FIXTURE_ROOT,
) -> tuple[PinnedRootedRandomCase, ...]:
    cases = []
    for fixture_case in load_pinned_rdkit_fixture_cases(
        fixture_root=fixture_root,
        rdkit_version=rdkit_version,
        fixture_label="rooted-random",
    ):
        raw_case = fixture_case.raw
        cases.append(
            PinnedRootedRandomCase(
                case_id=fixture_case.case_id,
                source=fixture_case.source,
                smiles=required_string(
                    raw_case,
                    field_name="smiles",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                rooted_outputs=required_string_tuple(
                    list(raw_case["rooted_outputs"]),
                    field_name="rooted_outputs",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
            )
        )

    return tuple(cases)
