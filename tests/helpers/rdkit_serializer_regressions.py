from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.helpers.pinned_rdkit_fixtures import (
    load_pinned_rdkit_fixture_cases,
    normalized_unique_sorted_strings,
    optional_positive_int,
)


@dataclass(frozen=True, slots=True)
class PinnedSerializerRegressionCase:
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
    rdkit_sample_draw_budget: int | None = None


_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "rdkit_serializer_regressions"
)


def load_pinned_serializer_regression_cases(
    rdkit_version: str,
) -> tuple[PinnedSerializerRegressionCase, ...]:
    cases = []
    for fixture_case in load_pinned_rdkit_fixture_cases(
        fixture_root=_FIXTURE_ROOT,
        rdkit_version=rdkit_version,
        fixture_label="serializer-regression",
    ):
        raw_case = fixture_case.raw
        cases.append(
            PinnedSerializerRegressionCase(
                case_id=fixture_case.case_id,
                source=fixture_case.source,
                smiles=str(raw_case["smiles"]),
                rooted_at_atom=int(raw_case.get("rooted_at_atom", -1)),
                isomeric_smiles=bool(raw_case.get("isomeric_smiles", True)),
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
                rdkit_sample_draw_budget=optional_positive_int(
                    raw_case,
                    field_name="rdkit_sample_draw_budget",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
            )
        )

    return tuple(cases)
