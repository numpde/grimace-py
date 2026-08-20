from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.helpers.pinned_rdkit_fixtures import (
    PINNED_RDKIT_SERIALIZER_REGRESSIONS,
    load_pinned_rdkit_fixture_cases,
    normalized_unique_sorted_strings,
    optional_bool,
    optional_positive_int,
    optional_string,
    pinned_rdkit_fixture_root,
    required_bool,
    required_int,
)


@dataclass(frozen=True, slots=True)
class PinnedSerializerRegressionCase:
    case_id: str
    source: str
    smiles: str | None
    molblock: str | None
    rooted_at_atom: int
    isomeric_smiles: bool
    expected: tuple[str, ...]
    expected_inventory: tuple[str, ...]
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False
    rdkit_sample_draw_budget: int | None = None


_FIXTURE_ROOT = pinned_rdkit_fixture_root(PINNED_RDKIT_SERIALIZER_REGRESSIONS)


def load_pinned_serializer_regression_cases(
    rdkit_version: str,
    *,
    fixture_root: Path = _FIXTURE_ROOT,
) -> tuple[PinnedSerializerRegressionCase, ...]:
    cases = []
    for fixture_case in load_pinned_rdkit_fixture_cases(
        fixture_root=fixture_root,
        rdkit_version=rdkit_version,
        fixture_label="serializer-regression",
    ):
        raw_case = fixture_case.raw
        smiles = optional_string(
            raw_case,
            field_name="smiles",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        molblock = optional_string(
            raw_case,
            field_name="molblock",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        if (smiles is None) == (molblock is None):
            raise ValueError(
                f"fixture {fixture_case.fixture_path} case {fixture_case.case_id!r} "
                "must define exactly one of 'smiles' or 'molblock'"
            )
        cases.append(
            PinnedSerializerRegressionCase(
                case_id=fixture_case.case_id,
                source=fixture_case.source,
                smiles=smiles,
                molblock=molblock,
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
                rdkit_sample_draw_budget=optional_positive_int(
                    raw_case,
                    field_name="rdkit_sample_draw_budget",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
            )
        )

    return tuple(cases)
