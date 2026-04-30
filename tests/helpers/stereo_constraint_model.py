from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.helpers.pinned_rdkit_fixtures import (
    PINNED_STEREO_CONSTRAINT_MODEL,
    load_pinned_rdkit_fixture_cases,
    pinned_rdkit_fixture_root,
    required_int_tuple,
    required_positive_int,
    required_string,
)


@dataclass(frozen=True, slots=True)
class PinnedStereoConstraintModelCase:
    case_id: str
    source: str
    smiles: str
    expected_side_count: int
    expected_component_domain_assignment_counts: tuple[int, ...]
    expected_semantic_assignment_count: int
    expected_rdkit_local_writer_assignment_count: int
    expected_rdkit_traversal_writer_assignment_count: int

    @property
    def expected_component_count(self) -> int:
        return len(self.expected_component_domain_assignment_counts)


_FIXTURE_ROOT = pinned_rdkit_fixture_root(PINNED_STEREO_CONSTRAINT_MODEL)


def load_pinned_stereo_constraint_model_cases(
    rdkit_version: str,
    *,
    fixture_root: Path = _FIXTURE_ROOT,
) -> tuple[PinnedStereoConstraintModelCase, ...]:
    cases = []
    for fixture_case in load_pinned_rdkit_fixture_cases(
        fixture_root=fixture_root,
        rdkit_version=rdkit_version,
        fixture_label="stereo-constraint-model",
    ):
        raw_case = fixture_case.raw
        component_domain_assignment_counts = required_int_tuple(
            list(raw_case["expected_component_domain_assignment_counts"]),
            field_name="expected_component_domain_assignment_counts",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        if any(count <= 0 for count in component_domain_assignment_counts):
            raise ValueError(
                f"fixture {fixture_case.fixture_path} case {fixture_case.case_id!r} "
                "must define positive expected_component_domain_assignment_counts"
            )
        cases.append(
            PinnedStereoConstraintModelCase(
                case_id=fixture_case.case_id,
                source=fixture_case.source,
                smiles=required_string(
                    raw_case,
                    field_name="smiles",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                expected_side_count=required_positive_int(
                    raw_case,
                    field_name="expected_side_count",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                expected_component_domain_assignment_counts=component_domain_assignment_counts,
                expected_semantic_assignment_count=required_positive_int(
                    raw_case,
                    field_name="expected_semantic_assignment_count",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                expected_rdkit_local_writer_assignment_count=required_positive_int(
                    raw_case,
                    field_name="expected_rdkit_local_writer_assignment_count",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                expected_rdkit_traversal_writer_assignment_count=required_positive_int(
                    raw_case,
                    field_name="expected_rdkit_traversal_writer_assignment_count",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
            )
        )

    return tuple(cases)
