from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.helpers.pinned_rdkit_fixtures import (
    PINNED_RDKIT_WRITER_MEMBERSHIP,
    load_pinned_rdkit_fixture_cases,
    optional_bool,
    optional_int,
    pinned_rdkit_fixture_root,
    required_bool,
    required_string,
)


@dataclass(frozen=True, slots=True)
class PinnedWriterMembershipCase:
    case_id: str
    source: str
    smiles: str
    expected: str
    rooted_at_atom: int | None
    isomeric_smiles: bool
    rdkit_canonical: bool
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False


_FIXTURE_ROOT = pinned_rdkit_fixture_root(PINNED_RDKIT_WRITER_MEMBERSHIP)


def load_pinned_writer_membership_cases(
    rdkit_version: str,
    *,
    fixture_root: Path = _FIXTURE_ROOT,
) -> tuple[PinnedWriterMembershipCase, ...]:
    cases = []
    for fixture_case in load_pinned_rdkit_fixture_cases(
        fixture_root=fixture_root,
        rdkit_version=rdkit_version,
        fixture_label="writer-membership",
    ):
        raw_case = fixture_case.raw
        cases.append(
            PinnedWriterMembershipCase(
                case_id=fixture_case.case_id,
                source=fixture_case.source,
                smiles=required_string(
                    raw_case,
                    field_name="smiles",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                expected=required_string(
                    raw_case,
                    field_name="expected",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                rooted_at_atom=optional_int(
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
                rdkit_canonical=required_bool(
                    raw_case,
                    field_name="rdkit_canonical",
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
