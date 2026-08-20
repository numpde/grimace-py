from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.helpers.pinned_rdkit_fixtures import (
    PINNED_RDKIT_KNOWN_QUIRKS,
    load_pinned_rdkit_fixture_cases,
    pinned_rdkit_fixture_root,
    required_string,
    required_string_tuple,
)


@dataclass(frozen=True, slots=True)
class PinnedRdkitKnownQuirkCase:
    case_id: str
    source: str
    category: str
    smiles: str
    canonical_smiles: str
    source_stereo_double_bonds: tuple[str, ...]
    roundtrip_stereo_double_bonds: tuple[str, ...]


_FIXTURE_ROOT = pinned_rdkit_fixture_root(PINNED_RDKIT_KNOWN_QUIRKS)


def load_pinned_rdkit_known_quirk_cases(
    rdkit_version: str,
    *,
    fixture_root: Path = _FIXTURE_ROOT,
) -> tuple[PinnedRdkitKnownQuirkCase, ...]:
    cases = []
    for fixture_case in load_pinned_rdkit_fixture_cases(
        fixture_root=fixture_root,
        rdkit_version=rdkit_version,
        fixture_label="known-quirk",
    ):
        raw_case = fixture_case.raw
        cases.append(
            PinnedRdkitKnownQuirkCase(
                case_id=fixture_case.case_id,
                source=fixture_case.source,
                category=required_string(
                    raw_case,
                    field_name="category",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                smiles=required_string(
                    raw_case,
                    field_name="smiles",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                canonical_smiles=required_string(
                    raw_case,
                    field_name="canonical_smiles",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                source_stereo_double_bonds=required_string_tuple(
                    list(raw_case["source_stereo_double_bonds"]),
                    field_name="source_stereo_double_bonds",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                roundtrip_stereo_double_bonds=required_string_tuple(
                    list(raw_case["roundtrip_stereo_double_bonds"]),
                    field_name="roundtrip_stereo_double_bonds",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
            )
        )
    return tuple(cases)
