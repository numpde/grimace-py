from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.helpers.pinned_rdkit_fixtures import (
    load_pinned_rdkit_fixture_cases,
    normalized_unique_sorted_strings,
)


@dataclass(frozen=True, slots=True)
class PinnedExactSmallSupportCase:
    case_id: str
    source: str
    smiles: str
    rooted_at_atom: int
    isomeric_smiles: bool
    expected: tuple[str, ...]
    expected_inventory: tuple[str, ...]
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False


_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "rdkit_exact_small_support"
)


def load_pinned_exact_small_support_cases(
    rdkit_version: str,
) -> tuple[PinnedExactSmallSupportCase, ...]:
    cases = []
    for fixture_case in load_pinned_rdkit_fixture_cases(
        fixture_root=_FIXTURE_ROOT,
        rdkit_version=rdkit_version,
        fixture_label="exact small-support",
    ):
        raw_case = fixture_case.raw
        cases.append(
            PinnedExactSmallSupportCase(
                case_id=fixture_case.case_id,
                source=fixture_case.source,
                smiles=str(raw_case["smiles"]),
                rooted_at_atom=int(raw_case["rooted_at_atom"]),
                isomeric_smiles=bool(raw_case["isomeric_smiles"]),
                expected=normalized_unique_sorted_strings(
                    list(raw_case["expected"]),
                    field_name="expected",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                expected_inventory=normalized_unique_sorted_strings(
                    list(raw_case["expected_inventory"]),
                    field_name="expected_inventory",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                kekule_smiles=bool(raw_case.get("kekule_smiles", False)),
                all_bonds_explicit=bool(raw_case.get("all_bonds_explicit", False)),
                all_hs_explicit=bool(raw_case.get("all_hs_explicit", False)),
                ignore_atom_map_numbers=bool(
                    raw_case.get("ignore_atom_map_numbers", False)
                ),
            )
        )

    return tuple(cases)
