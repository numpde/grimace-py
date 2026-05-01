from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

from tests.helpers.pinned_rdkit_fixtures import (
    PINNED_STEREO_CONSTRAINT_MODEL,
    load_pinned_rdkit_fixture_cases,
    pinned_rdkit_fixture_root,
    optional_nonnegative_int,
    optional_positive_int,
    required_int_tuple,
    required_positive_int,
    required_string,
)


@dataclass(frozen=True, slots=True)
class PinnedStereoMarkerSequenceTransition:
    direction_erased_skeleton: str
    grimace_ordered_markers: tuple[str, ...]
    rdkit_ordered_markers: tuple[str, ...]


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
    expected_rdkit_sampled_support_count: int | None
    expected_rdkit_sampled_exact_support_overlap_count: int | None
    expected_rdkit_sampled_exact_local_invalid_overlap_count: int | None
    expected_rdkit_sampled_outside_current_exact_support_count: int | None
    expected_rdkit_sampled_outside_current_exact_identity_equivalent_count: int | None
    expected_rdkit_sampled_outside_current_exact_parse_failure_count: int | None
    expected_rdkit_sampled_outside_current_exact_ring_digit_direction_erased_overlap_count: (
        int | None
    )
    expected_rdkit_sampled_outside_current_exact_direction_erased_overlap_count: int | None
    expected_grimace_runtime_outputs_with_ring_digit_direction_count: int | None
    expected_rdkit_sampled_outputs_with_ring_digit_direction_count: int | None
    expected_rdkit_sampled_outside_current_exact_with_ring_digit_direction_count: int | None
    expected_direction_erased_skeletons_with_same_marker_sequence_count: int | None
    expected_direction_erased_skeletons_with_different_marker_sequence_count: int | None
    expected_ring_closure_marker_transform_support_count: int | None
    expected_ring_closure_marker_transform_exact_overlap_count: int | None
    expected_ring_closure_marker_transform_outside_rdkit_count: int | None
    expected_ring_closure_marker_transform_residual_provenance_classes: (
        tuple[tuple[str, int], ...]
    )
    expected_marker_sequence_transitions: tuple[PinnedStereoMarkerSequenceTransition, ...]

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


def _required_marker_sequence(
    raw_transition: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> tuple[str, ...]:
    raw_markers = raw_transition.get(field_name)
    if (
        not isinstance(raw_markers, list)
        or not raw_markers
        or not all(marker in ("/", "\\") for marker in raw_markers)
    ):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define nonempty "
            f"{field_name} as '/' or '\\\\' marker strings"
        )
    return tuple(str(marker) for marker in raw_markers)


def _optional_marker_sequence_transitions(
    raw_case: dict[str, object],
    *,
    fixture_path: Path,
    case_id: str,
) -> tuple[PinnedStereoMarkerSequenceTransition, ...]:
    raw_transitions = raw_case.get("expected_marker_sequence_transitions")
    if raw_transitions is None:
        return ()
    if not isinstance(raw_transitions, list):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define "
            "expected_marker_sequence_transitions as a list"
        )

    transitions = []
    for raw_transition in raw_transitions:
        if not isinstance(raw_transition, dict):
            raise ValueError(
                f"fixture {fixture_path} case {case_id!r} must define marker "
                "sequence transitions as objects"
            )
        transitions.append(
            PinnedStereoMarkerSequenceTransition(
                direction_erased_skeleton=required_string(
                    raw_transition,
                    field_name="direction_erased_skeleton",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                grimace_ordered_markers=_required_marker_sequence(
                    raw_transition,
                    field_name="grimace_ordered_markers",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                rdkit_ordered_markers=_required_marker_sequence(
                    raw_transition,
                    field_name="rdkit_ordered_markers",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
            )
        )

    parsed = tuple(sorted(transitions, key=lambda transition: transition.direction_erased_skeleton))
    if parsed != tuple(transitions) or len(
        {transition.direction_erased_skeleton for transition in parsed}
    ) != len(parsed):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define marker "
            "sequence transitions sorted by unique direction_erased_skeleton"
        )
    return parsed


def _optional_string_int_counts(
    raw_case: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> tuple[tuple[str, int], ...]:
    raw_counts = raw_case.get(field_name)
    if raw_counts is None:
        return ()
    if not isinstance(raw_counts, dict) or not raw_counts:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define {field_name} "
            "as a nonempty object"
        )
    parsed = []
    for key, value in raw_counts.items():
        if not isinstance(key, str) or not key or type(value) is not int or value < 0:
            raise ValueError(
                f"fixture {fixture_path} case {case_id!r} must define {field_name} "
                "as string keys with nonnegative integer values"
            )
        parsed.append((key, value))
    return tuple(sorted(parsed))


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
        expected_rdkit_sampled_support_count = optional_positive_int(
            raw_case,
            field_name="expected_rdkit_sampled_support_count",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        expected_rdkit_sampled_exact_support_overlap_count = optional_nonnegative_int(
            raw_case,
            field_name="expected_rdkit_sampled_exact_support_overlap_count",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        expected_rdkit_sampled_exact_local_invalid_overlap_count = optional_nonnegative_int(
            raw_case,
            field_name="expected_rdkit_sampled_exact_local_invalid_overlap_count",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        expected_rdkit_sampled_outside_current_exact_support_count = optional_nonnegative_int(
            raw_case,
            field_name="expected_rdkit_sampled_outside_current_exact_support_count",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        expected_rdkit_sampled_outside_current_exact_identity_equivalent_count = (
            optional_nonnegative_int(
                raw_case,
                field_name=(
                    "expected_rdkit_sampled_outside_current_exact_identity_equivalent_count"
                ),
                fixture_path=fixture_case.fixture_path,
                case_id=fixture_case.case_id,
            )
        )
        expected_rdkit_sampled_outside_current_exact_parse_failure_count = optional_nonnegative_int(
            raw_case,
            field_name="expected_rdkit_sampled_outside_current_exact_parse_failure_count",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        expected_rdkit_sampled_outside_current_exact_ring_digit_direction_erased_overlap_count = (
            optional_nonnegative_int(
                raw_case,
                field_name=(
                    "expected_rdkit_sampled_outside_current_exact_ring_digit_direction_erased_overlap_count"
                ),
                fixture_path=fixture_case.fixture_path,
                case_id=fixture_case.case_id,
            )
        )
        expected_rdkit_sampled_outside_current_exact_direction_erased_overlap_count = (
            optional_nonnegative_int(
                raw_case,
                field_name=(
                    "expected_rdkit_sampled_outside_current_exact_direction_erased_overlap_count"
                ),
                fixture_path=fixture_case.fixture_path,
                case_id=fixture_case.case_id,
            )
        )
        expected_grimace_runtime_outputs_with_ring_digit_direction_count = (
            optional_nonnegative_int(
                raw_case,
                field_name="expected_grimace_runtime_outputs_with_ring_digit_direction_count",
                fixture_path=fixture_case.fixture_path,
                case_id=fixture_case.case_id,
            )
        )
        expected_rdkit_sampled_outputs_with_ring_digit_direction_count = optional_nonnegative_int(
            raw_case,
            field_name="expected_rdkit_sampled_outputs_with_ring_digit_direction_count",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        expected_rdkit_sampled_outside_current_exact_with_ring_digit_direction_count = (
            optional_nonnegative_int(
                raw_case,
                field_name=(
                    "expected_rdkit_sampled_outside_current_exact_with_ring_digit_direction_count"
                ),
                fixture_path=fixture_case.fixture_path,
                case_id=fixture_case.case_id,
            )
        )
        expected_direction_erased_skeletons_with_same_marker_sequence_count = (
            optional_nonnegative_int(
                raw_case,
                field_name=(
                    "expected_direction_erased_skeletons_with_same_marker_sequence_count"
                ),
                fixture_path=fixture_case.fixture_path,
                case_id=fixture_case.case_id,
            )
        )
        expected_direction_erased_skeletons_with_different_marker_sequence_count = (
            optional_nonnegative_int(
                raw_case,
                field_name=(
                    "expected_direction_erased_skeletons_with_different_marker_sequence_count"
                ),
                fixture_path=fixture_case.fixture_path,
                case_id=fixture_case.case_id,
            )
        )
        expected_ring_closure_marker_transform_support_count = optional_nonnegative_int(
            raw_case,
            field_name="expected_ring_closure_marker_transform_support_count",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        expected_ring_closure_marker_transform_exact_overlap_count = optional_nonnegative_int(
            raw_case,
            field_name="expected_ring_closure_marker_transform_exact_overlap_count",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        expected_ring_closure_marker_transform_outside_rdkit_count = optional_nonnegative_int(
            raw_case,
            field_name="expected_ring_closure_marker_transform_outside_rdkit_count",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        expected_ring_closure_marker_transform_residual_provenance_classes = (
            _optional_string_int_counts(
                raw_case,
                field_name=(
                    "expected_ring_closure_marker_transform_residual_provenance_classes"
                ),
                fixture_path=fixture_case.fixture_path,
                case_id=fixture_case.case_id,
            )
        )
        rdkit_sampled_expectations = (
            expected_rdkit_sampled_support_count,
            expected_rdkit_sampled_exact_support_overlap_count,
            expected_rdkit_sampled_exact_local_invalid_overlap_count,
            expected_rdkit_sampled_outside_current_exact_support_count,
            expected_rdkit_sampled_outside_current_exact_identity_equivalent_count,
            expected_rdkit_sampled_outside_current_exact_parse_failure_count,
            expected_rdkit_sampled_outside_current_exact_ring_digit_direction_erased_overlap_count,
            expected_rdkit_sampled_outside_current_exact_direction_erased_overlap_count,
            expected_grimace_runtime_outputs_with_ring_digit_direction_count,
            expected_rdkit_sampled_outputs_with_ring_digit_direction_count,
            expected_rdkit_sampled_outside_current_exact_with_ring_digit_direction_count,
            expected_direction_erased_skeletons_with_same_marker_sequence_count,
            expected_direction_erased_skeletons_with_different_marker_sequence_count,
            expected_ring_closure_marker_transform_support_count,
            expected_ring_closure_marker_transform_exact_overlap_count,
            expected_ring_closure_marker_transform_outside_rdkit_count,
        )
        if any(value is not None for value in rdkit_sampled_expectations) and not all(
            value is not None for value in rdkit_sampled_expectations
        ):
            raise ValueError(
                f"fixture {fixture_case.fixture_path} case {fixture_case.case_id!r} "
                "must define all RDKit sampled diagnostic counts or none of them"
            )
        expected_marker_sequence_transitions = _optional_marker_sequence_transitions(
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
                expected_rdkit_sampled_support_count=expected_rdkit_sampled_support_count,
                expected_rdkit_sampled_exact_support_overlap_count=(
                    expected_rdkit_sampled_exact_support_overlap_count
                ),
                expected_rdkit_sampled_exact_local_invalid_overlap_count=(
                    expected_rdkit_sampled_exact_local_invalid_overlap_count
                ),
                expected_rdkit_sampled_outside_current_exact_support_count=(
                    expected_rdkit_sampled_outside_current_exact_support_count
                ),
                expected_rdkit_sampled_outside_current_exact_identity_equivalent_count=(
                    expected_rdkit_sampled_outside_current_exact_identity_equivalent_count
                ),
                expected_rdkit_sampled_outside_current_exact_parse_failure_count=(
                    expected_rdkit_sampled_outside_current_exact_parse_failure_count
                ),
                expected_rdkit_sampled_outside_current_exact_ring_digit_direction_erased_overlap_count=(
                    expected_rdkit_sampled_outside_current_exact_ring_digit_direction_erased_overlap_count
                ),
                expected_rdkit_sampled_outside_current_exact_direction_erased_overlap_count=(
                    expected_rdkit_sampled_outside_current_exact_direction_erased_overlap_count
                ),
                expected_grimace_runtime_outputs_with_ring_digit_direction_count=(
                    expected_grimace_runtime_outputs_with_ring_digit_direction_count
                ),
                expected_rdkit_sampled_outputs_with_ring_digit_direction_count=(
                    expected_rdkit_sampled_outputs_with_ring_digit_direction_count
                ),
                expected_rdkit_sampled_outside_current_exact_with_ring_digit_direction_count=(
                    expected_rdkit_sampled_outside_current_exact_with_ring_digit_direction_count
                ),
                expected_direction_erased_skeletons_with_same_marker_sequence_count=(
                    expected_direction_erased_skeletons_with_same_marker_sequence_count
                ),
                expected_direction_erased_skeletons_with_different_marker_sequence_count=(
                    expected_direction_erased_skeletons_with_different_marker_sequence_count
                ),
                expected_ring_closure_marker_transform_support_count=(
                    expected_ring_closure_marker_transform_support_count
                ),
                expected_ring_closure_marker_transform_exact_overlap_count=(
                    expected_ring_closure_marker_transform_exact_overlap_count
                ),
                expected_ring_closure_marker_transform_outside_rdkit_count=(
                    expected_ring_closure_marker_transform_outside_rdkit_count
                ),
                expected_ring_closure_marker_transform_residual_provenance_classes=(
                    expected_ring_closure_marker_transform_residual_provenance_classes
                ),
                expected_marker_sequence_transitions=expected_marker_sequence_transitions,
            )
        )

    return tuple(cases)
