from __future__ import annotations

import math
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
    expected_component_side_domain_sizes: tuple[tuple[int, ...], ...]
    expected_semantic_assignment_count: int
    expected_rdkit_local_writer_assignment_count: int
    expected_rdkit_traversal_writer_assignment_count: int
    expected_grimace_runtime_support_count: int

    @property
    def expected_component_count(self) -> int:
        return len(self.expected_component_side_domain_sizes)

    @property
    def expected_side_count(self) -> int:
        return sum(
            len(side_domain_sizes)
            for side_domain_sizes in self.expected_component_side_domain_sizes
        )

    @property
    def expected_component_domain_assignment_counts(self) -> tuple[int, ...]:
        return tuple(
            math.prod(side_domain_sizes)
            for side_domain_sizes in self.expected_component_side_domain_sizes
        )


_FIXTURE_ROOT = pinned_rdkit_fixture_root(PINNED_STEREO_CONSTRAINT_MODEL)


def _required_component_side_domain_sizes(
    raw_case: dict[str, object],
    *,
    fixture_path: Path,
    case_id: str,
) -> tuple[tuple[int, ...], ...]:
    raw_values = raw_case.get("expected_component_side_domain_sizes")
    if not isinstance(raw_values, list) or not raw_values:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define nonempty "
            "expected_component_side_domain_sizes"
        )

    parsed_components = []
    for raw_component in raw_values:
        if not isinstance(raw_component, list):
            raise ValueError(
                f"fixture {fixture_path} case {case_id!r} must define "
                "expected_component_side_domain_sizes as lists of integers"
            )
        parsed_components.append(
            required_int_tuple(
                raw_component,
                field_name="expected_component_side_domain_sizes",
                fixture_path=fixture_path,
                case_id=case_id,
            )
        )
    parsed = tuple(parsed_components)
    if any(size <= 0 for component in parsed for size in component):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define positive "
            "expected_component_side_domain_sizes"
        )
    return parsed


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
        component_side_domain_sizes = _required_component_side_domain_sizes(
            raw_case,
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
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
                expected_component_side_domain_sizes=component_side_domain_sizes,
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
                expected_grimace_runtime_support_count=required_positive_int(
                    raw_case,
                    field_name="expected_grimace_runtime_support_count",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
            )
        )

    return tuple(cases)
