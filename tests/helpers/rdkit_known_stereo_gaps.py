from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.helpers.pinned_rdkit_fixtures import (
    PINNED_RDKIT_KNOWN_STEREO_GAPS,
    load_pinned_rdkit_fixture_cases,
    optional_bool,
    optional_int,
    optional_string,
    pinned_rdkit_fixture_root,
    required_bool,
    required_string,
)


@dataclass(frozen=True, slots=True)
class PinnedKnownStereoGapCase:
    case_id: str
    source: str
    smiles: str | None
    molblock: str | None
    writer_membership_case_id: str | None
    expected: str
    rooted_at_atom: int | None
    isomeric_smiles: bool
    rdkit_canonical: bool
    rdkit_random_vector_seed: int | None
    rdkit_random_vector_index: int | None
    check_grimace_decoder_path: bool
    check_grimace_support: bool


_FIXTURE_ROOT = pinned_rdkit_fixture_root(PINNED_RDKIT_KNOWN_STEREO_GAPS)


def load_pinned_known_stereo_gap_cases(
    rdkit_version: str,
    *,
    fixture_root: Path = _FIXTURE_ROOT,
) -> tuple[PinnedKnownStereoGapCase, ...]:
    cases = []
    for fixture_case in load_pinned_rdkit_fixture_cases(
        fixture_root=fixture_root,
        rdkit_version=rdkit_version,
        fixture_label="known-stereo-gap",
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
        writer_membership_case_id = optional_string(
            raw_case,
            field_name="writer_membership_case_id",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        molecule_source_count = sum(
            value is not None
            for value in (smiles, molblock, writer_membership_case_id)
        )
        if molecule_source_count != 1:
            raise ValueError(
                f"fixture {fixture_case.fixture_path} case "
                f"{fixture_case.case_id!r} must define exactly one of "
                "'smiles', 'molblock', or 'writer_membership_case_id'"
            )

        random_vector_seed = optional_int(
            raw_case,
            field_name="rdkit_random_vector_seed",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        random_vector_index = optional_int(
            raw_case,
            field_name="rdkit_random_vector_index",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        if (random_vector_seed is None) != (random_vector_index is None):
            raise ValueError(
                f"fixture {fixture_case.fixture_path} case "
                f"{fixture_case.case_id!r} must define both "
                "'rdkit_random_vector_seed' and 'rdkit_random_vector_index', "
                "or neither"
            )

        cases.append(
            PinnedKnownStereoGapCase(
                case_id=fixture_case.case_id,
                source=fixture_case.source,
                smiles=smiles,
                molblock=molblock,
                writer_membership_case_id=writer_membership_case_id,
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
                rdkit_random_vector_seed=random_vector_seed,
                rdkit_random_vector_index=random_vector_index,
                check_grimace_decoder_path=optional_bool(
                    raw_case,
                    field_name="check_grimace_decoder_path",
                    default=False,
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                check_grimace_support=optional_bool(
                    raw_case,
                    field_name="check_grimace_support",
                    default=True,
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
            )
        )

    return tuple(cases)
