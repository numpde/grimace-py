from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.helpers.pinned_rdkit_fixtures import (
    load_pinned_rdkit_fixture_cases,
    normalized_unique_sorted_strings,
    pinned_rdkit_fixture_root,
    required_int,
    required_positive_int,
    required_string,
)


VISIBLE_MARKER_BASIS_FIXTURE = "visible_marker_basis"
VISIBLE_MARKER_BASIS_CLASSES = frozenset(
    (
        "non_selected_visible_edge",
        "selected_carrier",
        "shared_visible_edge",
    )
)
VISIBLE_MARKER_POLICY_VARIANTS = frozenset(
    (
        "current_visible_basis",
        "frontier_shared_nonselected_visible_basis",
        "legacy_topology_gated_visible_basis",
        "raw_selected_carrier",
        "remaining_shared_visible_basis",
        "legacy_remaining_visible_basis",
    )
)


@dataclass(frozen=True, slots=True)
class PinnedVisibleMarkerBasisCase:
    case_id: str
    source: str
    smiles: str
    root_idx: int
    diagnostic_limit: int
    diagnostic_max_states: int
    expected_grimace_runtime_support_count: int
    expected_diagnostic_row_count: int
    expected_current_support_accepts_candidate_count: int
    expected_raw_selected_carrier_explained_component_count: int
    expected_visible_edge_explained_component_count: int
    expected_legacy_topology_guard_component_count: int
    expected_frontier_replacement_mismatch_count: int
    expected_policy_variant_accept_counts: tuple[tuple[str, int], ...]
    expected_basis_candidate_count: int
    expected_max_row_survivor_count: int
    expected_basis_class_counts: tuple[tuple[str, int], ...]
    required_basis_classes: tuple[str, ...]
    forbidden_basis_classes: tuple[str, ...]


_FIXTURE_ROOT = pinned_rdkit_fixture_root(VISIBLE_MARKER_BASIS_FIXTURE)


def _required_nonnegative_int(
    raw_case: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> int:
    value = required_int(
        raw_case,
        field_name=field_name,
        fixture_path=fixture_path,
        case_id=case_id,
    )
    if value < 0:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define "
            f"nonnegative integer {field_name}; got {value!r}"
        )
    return value


def _basis_class_tuple(
    raw_case: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> tuple[str, ...]:
    raw_values = raw_case.get(field_name)
    if not isinstance(raw_values, list):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define {field_name} "
            "as a list"
        )
    values = normalized_unique_sorted_strings(
        raw_values,
        field_name=field_name,
        fixture_path=fixture_path,
        case_id=case_id,
    )
    unknown = set(values) - VISIBLE_MARKER_BASIS_CLASSES
    if unknown:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} has unknown {field_name} "
            f"values: {sorted(unknown)!r}"
        )
    return values


def _positive_counts(
    raw_case: dict[str, object],
    *,
    field_name: str,
    valid_keys: frozenset[str],
    fixture_path: Path,
    case_id: str,
) -> tuple[tuple[str, int], ...]:
    raw_counts = raw_case.get(field_name)
    if not isinstance(raw_counts, dict) or not raw_counts:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define {field_name} "
            "as a nonempty object"
        )
    counts = []
    for key, value in raw_counts.items():
        if key not in valid_keys:
            raise ValueError(
                f"fixture {fixture_path} case {case_id!r} has unknown "
                f"{field_name} key {key!r}"
            )
        if type(value) is not int or value <= 0:
            raise ValueError(
                f"fixture {fixture_path} case {case_id!r} must define positive "
                f"integer counts in {field_name}"
            )
        counts.append((str(key), value))
    return tuple(sorted(counts))


def _basis_class_counts(
    raw_case: dict[str, object],
    *,
    fixture_path: Path,
    case_id: str,
) -> tuple[tuple[str, int], ...]:
    return _positive_counts(
        raw_case,
        field_name="expected_basis_class_counts",
        valid_keys=VISIBLE_MARKER_BASIS_CLASSES,
        fixture_path=fixture_path,
        case_id=case_id,
    )


def load_pinned_visible_marker_basis_cases(
    rdkit_version: str,
    *,
    fixture_root: Path = _FIXTURE_ROOT,
) -> tuple[PinnedVisibleMarkerBasisCase, ...]:
    cases = []
    for fixture_case in load_pinned_rdkit_fixture_cases(
        fixture_root=fixture_root,
        rdkit_version=rdkit_version,
        fixture_label="visible-marker-basis",
    ):
        raw_case = fixture_case.raw
        case_id = fixture_case.case_id
        fixture_path = fixture_case.fixture_path
        cases.append(
            PinnedVisibleMarkerBasisCase(
                case_id=case_id,
                source=fixture_case.source,
                smiles=required_string(
                    raw_case,
                    field_name="smiles",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                root_idx=required_int(
                    raw_case,
                    field_name="root_idx",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                diagnostic_limit=required_positive_int(
                    raw_case,
                    field_name="diagnostic_limit",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                diagnostic_max_states=required_positive_int(
                    raw_case,
                    field_name="diagnostic_max_states",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                expected_grimace_runtime_support_count=required_positive_int(
                    raw_case,
                    field_name="expected_grimace_runtime_support_count",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                expected_diagnostic_row_count=required_positive_int(
                    raw_case,
                    field_name="expected_diagnostic_row_count",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                expected_current_support_accepts_candidate_count=_required_nonnegative_int(
                    raw_case,
                    field_name="expected_current_support_accepts_candidate_count",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                expected_raw_selected_carrier_explained_component_count=_required_nonnegative_int(
                    raw_case,
                    field_name="expected_raw_selected_carrier_explained_component_count",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                expected_visible_edge_explained_component_count=_required_nonnegative_int(
                    raw_case,
                    field_name="expected_visible_edge_explained_component_count",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                expected_legacy_topology_guard_component_count=_required_nonnegative_int(
                    raw_case,
                    field_name="expected_legacy_topology_guard_component_count",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                expected_frontier_replacement_mismatch_count=_required_nonnegative_int(
                    raw_case,
                    field_name="expected_frontier_replacement_mismatch_count",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                expected_policy_variant_accept_counts=_positive_counts(
                    raw_case,
                    field_name="expected_policy_variant_accept_counts",
                    valid_keys=VISIBLE_MARKER_POLICY_VARIANTS,
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                expected_basis_candidate_count=required_positive_int(
                    raw_case,
                    field_name="expected_basis_candidate_count",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                expected_max_row_survivor_count=required_positive_int(
                    raw_case,
                    field_name="expected_max_row_survivor_count",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                expected_basis_class_counts=_basis_class_counts(
                    raw_case,
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                required_basis_classes=_basis_class_tuple(
                    raw_case,
                    field_name="required_basis_classes",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                forbidden_basis_classes=_basis_class_tuple(
                    raw_case,
                    field_name="forbidden_basis_classes",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
            )
        )
    return tuple(cases)
