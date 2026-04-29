from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.helpers.pinned_rdkit_fixtures import (
    load_pinned_rdkit_fixture_cases,
    normalized_unique_sorted_strings,
    optional_bool,
    required_bool,
    required_int,
    required_string,
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
    *,
    fixture_root: Path = _FIXTURE_ROOT,
) -> tuple[PinnedExactSmallSupportCase, ...]:
    cases = []
    for fixture_case in load_pinned_rdkit_fixture_cases(
        fixture_root=fixture_root,
        rdkit_version=rdkit_version,
        fixture_label="exact small-support",
    ):
        raw_case = fixture_case.raw
        cases.append(
            PinnedExactSmallSupportCase(
                case_id=fixture_case.case_id,
                source=fixture_case.source,
                smiles=required_string(
                    raw_case,
                    field_name="smiles",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                rooted_at_atom=required_int(
                    raw_case,
                    field_name="rooted_at_atom",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                isomeric_smiles=required_bool(
                    raw_case,
                    field_name="isomeric_smiles",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
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
                kekule_smiles=optional_bool(
                    raw_case,
                    field_name="kekule_smiles",
                    default=False,
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                all_bonds_explicit=optional_bool(
                    raw_case,
                    field_name="all_bonds_explicit",
                    default=False,
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                all_hs_explicit=optional_bool(
                    raw_case,
                    field_name="all_hs_explicit",
                    default=False,
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                ignore_atom_map_numbers=optional_bool(
                    raw_case,
                    field_name="ignore_atom_map_numbers",
                    default=False,
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
            )
        )

    return tuple(cases)
