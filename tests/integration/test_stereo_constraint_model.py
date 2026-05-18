from __future__ import annotations

import math
import unittest
from collections import Counter
from dataclasses import dataclass

from rdkit import Chem
from rdkit import rdBase

from grimace import _core, _runtime
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import public_enum_support, supported_public_kwargs
from tests.helpers.stereo_constraint_model import (
    load_pinned_stereo_constraint_model_cases,
)


SUPPORTED_STEREO_FLAGS = _runtime.MolToSmilesFlags(
    isomeric_smiles=True,
    kekule_smiles=False,
    rooted_at_atom=-1,
    canonical=False,
    all_bonds_explicit=False,
    all_hs_explicit=False,
    do_random=True,
    ignore_atom_map_numbers=False,
)
RDKIT_SAMPLE_DRAW_COUNT = 128
RDKIT_SAMPLE_SEED = 1
TOKEN_FLIP_INFERENCE_BRANCHES = frozenset(
    (
        "isolated_all_single_candidate",
        "isolated_selected_begin_side",
        "coupled_one_candidate_begin_side",
        "coupled_two_candidate_begin_side",
    )
)
TOKEN_OBSERVATION_KIND_BY_SUPPORTED_BRANCH = {
    "isolated_all_single_candidate": "all_single_candidate",
    "isolated_selected_begin_side": "selected_begin_side",
    "coupled_one_candidate_begin_side": "selected_begin_side",
    "coupled_two_candidate_begin_side": "two_candidate_begin_side",
}
TOKEN_FLIP_ADJUSTMENT_REASON_COUNT_KEYS = frozenset(
    (
        "value_true",
        "root_begin_side_orientation",
        "adjacent_two_candidate_first_emitted",
    )
)


@dataclass(frozen=True, slots=True)
class _DirectionMarkerSlot:
    slot: int
    marker: str


@dataclass(frozen=True, slots=True)
class _RingLabelSpan:
    label: str
    start_slot: int
    end_slot: int


def _effective_layer_assignment_count(
    *,
    layer: dict[str, object],
    semantic_assignment_count: int,
) -> int:
    assignment_count = layer["assignment_count"]
    if assignment_count is None:
        return semantic_assignment_count
    if type(assignment_count) is not int:
        raise AssertionError(f"unexpected layer assignment count: {assignment_count!r}")
    return assignment_count


def _marker_row_diagnostics(prepared: object) -> dict[str, int]:
    summary = _core._stereo_constraint_model_summary(prepared)
    rows = _core._stereo_constraint_output_facts(prepared)
    semantic_survivor_counts = [
        component["row_count_after_marker_events"]
        for row in rows
        for component in row["marker_placement_state"]["semantic"]
    ]
    semantic_obligation_survivor_counts = [
        component["row_count_after_marker_events"]
        for row in rows
        for component in row["marker_obligation_state"]["semantic"]
    ]
    return {
        "output_row_count": len(rows),
        "max_model_marker_row_count": max(
            component["marker_placement_row_count"]
            for component in summary["components"]
        ),
        "total_model_marker_row_count": sum(
            component["marker_placement_row_count"]
            for component in summary["components"]
        ),
        "max_semantic_marker_survivor_row_count": max(semantic_survivor_counts),
        "max_semantic_marker_obligation_survivor_row_count": max(
            semantic_obligation_survivor_counts
        ),
    }


def _canonical_isomeric_smiles(smiles: str) -> str | None:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)


def _erase_direction_markers(smiles: str) -> str:
    return smiles.replace("/", "").replace("\\", "")


def _erase_ring_digit_adjacent_direction_markers(smiles: str) -> str:
    out = []
    for idx, char in enumerate(smiles):
        if char in ("/", "\\"):
            previous_char = smiles[idx - 1] if idx > 0 else ""
            next_char = smiles[idx + 1] if idx + 1 < len(smiles) else ""
            if (
                previous_char.isdigit()
                or previous_char == "%"
                or next_char.isdigit()
                or next_char == "%"
            ):
                continue
        out.append(char)
    return "".join(out)


def _directional_spelling_summary(smiles: str) -> dict[str, object]:
    ring_digit_adjacent_count = 0
    total_count = 0
    ordered_markers = []
    direction_erased_skeleton = []
    marker_slots = []
    for idx, char in enumerate(smiles):
        if char not in ("/", "\\"):
            direction_erased_skeleton.append(char)
            continue
        total_count += 1
        ordered_markers.append(char)
        slot = len(direction_erased_skeleton)

        previous_char = smiles[idx - 1] if idx > 0 else ""
        next_char = smiles[idx + 1] if idx + 1 < len(smiles) else ""
        if (
            previous_char.isdigit()
            or previous_char == "%"
            or next_char.isdigit()
            or next_char == "%"
        ):
            ring_digit_adjacent_count += 1
        marker_slots.append(
            {
                "slot": slot,
                "marker": char,
                "after_ring_label": previous_char.isdigit() or previous_char == "%",
                "before_ring_label": next_char.isdigit() or next_char == "%",
                "before_bracket_atom": next_char == "[",
                "after_branch_open": previous_char == "(",
            }
        )

    return {
        "total": total_count,
        "ring_digit_adjacent": ring_digit_adjacent_count,
        "non_ring": total_count - ring_digit_adjacent_count,
        "direction_erased_skeleton": "".join(direction_erased_skeleton),
        "ordered_markers": ordered_markers,
        "marker_slots": marker_slots,
    }


def _direction_marker_slots(smiles: str) -> tuple[str, tuple[_DirectionMarkerSlot, ...]]:
    skeleton = []
    markers = []
    for char in smiles:
        if char in ("/", "\\"):
            markers.append(_DirectionMarkerSlot(slot=len(skeleton), marker=char))
        else:
            skeleton.append(char)
    return "".join(skeleton), tuple(markers)


def _direction_marker_slots_from_summary(
    summary: dict[str, object],
) -> tuple[str, tuple[_DirectionMarkerSlot, ...]]:
    return (
        str(summary["direction_erased_skeleton"]),
        tuple(
            _DirectionMarkerSlot(slot=int(marker["slot"]), marker=str(marker["marker"]))
            for marker in summary["marker_slots"]
        ),
    )


def _ring_label_spans(skeleton: str) -> tuple[_RingLabelSpan, ...]:
    spans = []
    idx = 0
    while idx < len(skeleton):
        char = skeleton[idx]
        if char.isdigit():
            spans.append(_RingLabelSpan(label=char, start_slot=idx, end_slot=idx + 1))
            idx += 1
            continue
        if (
            char == "%"
            and idx + 2 < len(skeleton)
            and skeleton[idx + 1].isdigit()
            and skeleton[idx + 2].isdigit()
        ):
            spans.append(
                _RingLabelSpan(
                    label=skeleton[idx : idx + 3],
                    start_slot=idx,
                    end_slot=idx + 3,
                )
            )
            idx += 3
            continue
        idx += 1
    return tuple(spans)


def _rdkit_ring_closure_marker_slots(
    skeleton: str,
    markers: tuple[_DirectionMarkerSlot, ...],
) -> tuple[_DirectionMarkerSlot, ...]:
    """Model RDKit's sampled spelling move for ring-carrier direction markers.

    This is still a test-side candidate, not runtime behavior: for the pinned
    witness, RDKit moves a direction marker from just after the ring-opening
    label to just before the matching ring-closure label.
    """

    markers_by_slot = {marker.slot: marker for marker in markers}
    moved_slots = set()
    rewritten = []
    spans_by_label = {}
    for span in _ring_label_spans(skeleton):
        spans_by_label.setdefault(span.label, []).append(span)

    for spans in spans_by_label.values():
        for left, right in zip(spans, spans[1:], strict=False):
            marker = markers_by_slot.get(left.end_slot)
            closure_is_bracket_atom = (
                right.start_slot > 0 and skeleton[right.start_slot - 1] == "]"
            )
            if (
                marker is None
                or right.start_slot in markers_by_slot
                or not closure_is_bracket_atom
            ):
                continue
            moved_slots.add(marker.slot)
            rewritten.append(
                _DirectionMarkerSlot(slot=right.start_slot, marker=marker.marker)
            )

    rewritten.extend(marker for marker in markers if marker.slot not in moved_slots)
    return tuple(sorted(rewritten, key=lambda marker: marker.slot))


def _ordered_markers_from_slots(
    markers: tuple[_DirectionMarkerSlot, ...],
) -> tuple[str, ...]:
    return tuple(marker.marker for marker in markers)


def _marker_slot_pairs(
    markers: tuple[_DirectionMarkerSlot, ...],
) -> tuple[tuple[int, str], ...]:
    return tuple((marker.slot, marker.marker) for marker in markers)


def _marker_slots_by_slot(
    marker_slots: object,
) -> dict[int, dict[str, object]]:
    return {int(marker["slot"]): marker for marker in marker_slots}


def _target_slot_context(skeleton: str, slot: int) -> tuple[str, ...]:
    context = []
    if slot < len(skeleton) and skeleton[slot] == "[":
        context.append("before_bracket_atom")
    if slot < len(skeleton) and (skeleton[slot].isdigit() or skeleton[slot] == "%"):
        context.append("before_ring_label")
    if slot > 0 and skeleton[slot - 1] == "(":
        context.append("after_branch_open")
    if slot > 0 and (skeleton[slot - 1].isdigit() or skeleton[slot - 1] == "%"):
        context.append("after_ring_label")
    return tuple(context)


def _slot_local_role(skeleton: str, slot: int) -> str:
    context = _target_slot_context(skeleton, slot)
    if "after_ring_label" in context:
        return "after_ring_label"
    if "before_ring_label" in context:
        return "before_ring_label"
    if "after_branch_open" in context:
        return "branch_edge"
    if "before_bracket_atom" in context:
        return "before_bracket_atom"
    return "tree_or_chain_edge"


def _residual_marker_provenance_class(
    *,
    source_role: str,
    target_context: tuple[str, ...],
) -> str:
    if "before_ring_label" in target_context:
        target_role = "before_ring_label"
    elif "before_bracket_atom" in target_context:
        target_role = "before_bracket_atom"
    else:
        target_role = "other"
    return f"{source_role}_to_{target_role}"


def _smiles_from_direction_marker_slots(
    skeleton: str,
    markers: tuple[_DirectionMarkerSlot, ...],
) -> str:
    markers_by_slot: dict[int, list[str]] = {}
    for marker in markers:
        markers_by_slot.setdefault(marker.slot, []).append(marker.marker)

    out = []
    for slot in range(len(skeleton) + 1):
        out.extend(markers_by_slot.get(slot, ()))
        if slot < len(skeleton):
            out.append(skeleton[slot])
    return "".join(out)


def _apply_residual_slot_transition(
    *,
    skeleton: str,
    markers: tuple[_DirectionMarkerSlot, ...],
    residual_slot_transition_map: dict[
        tuple[str, tuple[tuple[int, str], ...]], tuple[tuple[int, str], ...]
    ],
) -> tuple[_DirectionMarkerSlot, ...]:
    marker_pairs = residual_slot_transition_map.get(
        (skeleton, _marker_slot_pairs(markers))
    )
    if marker_pairs is None:
        return markers
    return tuple(_DirectionMarkerSlot(slot=slot, marker=marker) for slot, marker in marker_pairs)


def _rdkit_sampled_outputs(mol: Chem.Mol) -> frozenset[str]:
    return frozenset(
        Chem.MolToRandomSmilesVect(
            mol,
            RDKIT_SAMPLE_DRAW_COUNT,
            randomSeed=RDKIT_SAMPLE_SEED,
            isomericSmiles=True,
            kekuleSmiles=False,
            allBondsExplicit=False,
            allHsExplicit=False,
        )
    )


def _rdkit_respelling_family(smileses: frozenset[str]) -> frozenset[str]:
    rewritten = set()
    for smiles in smileses:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise AssertionError(f"candidate output does not parse with RDKit: {smiles!r}")
        rewritten.update(_rdkit_sampled_outputs(mol))
    return frozenset(rewritten)


class StereoConstraintModelFixtureTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.cases = load_pinned_stereo_constraint_model_cases(rdBase.rdkitVersion)
        except FileNotFoundError:
            raise unittest.SkipTest(
                f"no pinned stereo-constraint-model corpus for RDKit "
                f"{rdBase.rdkitVersion}"
            )

    def test_native_model_shape_matches_pinned_witnesses(self) -> None:
        for case in self.cases:
            mol = parse_smiles(case.smiles)
            prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
            summary = _core._stereo_constraint_model_summary(prepared)

            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertEqual(
                    case.expected_component_count,
                    summary["component_count"],
                )
                self.assertEqual(case.expected_side_count, summary["side_count"])
                self.assertEqual(
                    case.expected_component_side_domain_sizes,
                    tuple(
                        tuple(component["side_domain_sizes"])
                        for component in summary["components"]
                    ),
                )
                self.assertEqual(
                    case.expected_component_domain_assignment_counts,
                    tuple(
                        component["domain_assignment_count"]
                        for component in summary["components"]
                    ),
                )
                self.assertEqual(
                    case.expected_semantic_assignment_count,
                    math.prod(case.expected_component_domain_assignment_counts),
                )
                layers_by_name = {
                    layer["layer"]: layer
                    for component in summary["components"]
                    for layer in component["layers"]
                }
                self.assertEqual(
                    case.expected_rdkit_local_writer_assignment_count,
                    _effective_layer_assignment_count(
                        layer=layers_by_name["rdkit_local_writer"],
                        semantic_assignment_count=case.expected_semantic_assignment_count,
                    )
                )
                modeled_traversal_count = _effective_layer_assignment_count(
                    layer=layers_by_name["rdkit_traversal_writer"],
                    semantic_assignment_count=case.expected_semantic_assignment_count,
                )
                if (
                    case.expected_rdkit_traversal_writer_assignment_count
                    == case.expected_rdkit_local_writer_assignment_count
                ):
                    self.assertEqual(
                        case.expected_rdkit_traversal_writer_assignment_count,
                        modeled_traversal_count,
                    )
                else:
                    self.assertEqual(
                        case.expected_rdkit_local_writer_assignment_count,
                        modeled_traversal_count,
                    )
                for component in summary["components"]:
                    self.assertEqual(
                        component["marker_placement_row_count"],
                        len(component["marker_placement_rows"]),
                    )
                    self.assertEqual(
                        component["token_phase_assignment_count"]
                        * math.prod(component["marker_placement_domain_sizes"]),
                        component["marker_placement_row_count"],
                    )
                    for row in component["marker_placement_rows"]:
                        self.assertLess(
                            row["token_phase_assignment_id"],
                            component["token_phase_assignment_count"],
                        )
                        self.assertEqual(
                            len(component["side_ids"]),
                            len(row["carrier_neighbors"]),
                        )
                        self.assertEqual(
                            len(component["side_ids"]),
                            len(row["marker_neighbor_sets"]),
                        )

    def test_pinned_layer_counts_are_ordered_by_contract_strength(self) -> None:
        for case in self.cases:
            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertLessEqual(
                    case.expected_rdkit_local_writer_assignment_count,
                    case.expected_semantic_assignment_count,
                )
                self.assertLessEqual(
                    case.expected_rdkit_traversal_writer_assignment_count,
                    case.expected_rdkit_local_writer_assignment_count,
                )

    def test_pinned_marker_row_diagnostics_match_fixture_metadata(self) -> None:
        saw_expected_diagnostics = False

        for case in self.cases:
            expected = case.expected_marker_row_diagnostics
            if expected is None:
                continue
            saw_expected_diagnostics = True
            mol = parse_smiles(case.smiles)
            prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
            actual = _marker_row_diagnostics(prepared)

            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertEqual(expected.output_row_count, actual["output_row_count"])
                self.assertEqual(
                    expected.max_model_marker_row_count,
                    actual["max_model_marker_row_count"],
                )
                self.assertEqual(
                    expected.total_model_marker_row_count,
                    actual["total_model_marker_row_count"],
                )
                self.assertEqual(
                    expected.max_semantic_marker_survivor_row_count,
                    actual["max_semantic_marker_survivor_row_count"],
                )
                self.assertEqual(
                    expected.max_semantic_marker_obligation_survivor_row_count,
                    actual["max_semantic_marker_obligation_survivor_row_count"],
                )

        self.assertTrue(saw_expected_diagnostics)

    def test_shared_carrier_resolution_is_assignment_state_explained(self) -> None:
        saw_shared_group = False

        for case in self.cases:
            mol = parse_smiles(case.smiles)
            prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
            summary = _core._stereo_constraint_model_summary(prepared)
            rows = _core._stereo_constraint_output_facts(prepared)
            saw_shared_group |= bool(summary["shared_carrier_groups"])

            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertEqual(
                    case.expected_shared_carrier_group_count,
                    len(summary["shared_carrier_groups"]),
                )
                changed_resolution_row_count = 0
                joined_boundary_mismatch_count = 0
                for row in rows:
                    self.assertEqual(
                        len(summary["shared_carrier_groups"]),
                        len(row["shared_carrier_resolution"]),
                    )
                    if not row["shadow_debug"]["joined_support_boundary_matches_runtime"]:
                        joined_boundary_mismatch_count += 1
                    for group in row["shared_carrier_resolution"]:
                        changed = group["left_changed"] or group["right_changed"]
                        if changed:
                            changed_resolution_row_count += 1
                        self.assertTrue(
                            group["left_change_explained_by_assignment_state"]
                        )
                        self.assertTrue(
                            group["right_change_explained_by_assignment_state"]
                        )
                self.assertEqual(
                    case.expected_shared_carrier_resolution_changed_row_count,
                    changed_resolution_row_count,
                )
                self.assertEqual(
                    case.expected_joined_support_boundary_mismatch_count,
                    joined_boundary_mismatch_count,
                )

        self.assertTrue(saw_shared_group)

    def test_terminal_carrier_resolution_uses_support_boundary_shadow(self) -> None:
        for case in self.cases:
            mol = parse_smiles(case.smiles)
            prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
            rows = _core._stereo_constraint_output_facts(prepared)

            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertTrue(rows)
                for row in rows:
                    self.assertIn(
                        "resolved_selected_neighbors_from_assignment_state",
                        row["shadow_debug"],
                    )
                    self.assertIn(
                        "joined_support_boundary_selected_neighbors",
                        row["shadow_debug"],
                    )
                    self.assertIn(
                        "legacy_field_resolved_selected_neighbors",
                        row["shadow_debug"],
                    )
                    self.assertTrue(
                        row["shadow_debug"][
                            "assignment_state_resolution_matches_support_boundary"
                        ]
                    )
                    self.assertTrue(
                        row["shadow_debug"]["joined_support_boundary_matches_runtime"]
                    )
                    self.assertTrue(
                        row["shadow_debug"][
                            "legacy_field_resolution_matches_support_boundary"
                        ]
                    )

    def test_current_runtime_support_count_matches_pinned_witnesses(self) -> None:
        for case in self.cases:
            mol = parse_smiles(case.smiles)

            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertEqual(
                    case.expected_grimace_runtime_support_count,
                    len(
                        public_enum_support(
                            mol,
                            **supported_public_kwargs(
                                isomericSmiles=True,
                                rootedAtAtom=-1,
                            ),
                        )
                    ),
                )

    def test_output_fact_diagnostic_maps_minimal_witness_to_runtime_support(self) -> None:
        case = next(
            case
            for case in self.cases
            if case.case_id == "minimal_nonstereo_double_hazard"
        )
        mol = parse_smiles(case.smiles)
        prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)

        rows = _core._stereo_constraint_output_facts(prepared)
        diagnostic_outputs = frozenset(row["smiles"] for row in rows)
        runtime_outputs = public_enum_support(
            mol,
            **supported_public_kwargs(
                isomericSmiles=True,
                rootedAtAtom=-1,
            ),
        )

        self.assertEqual(runtime_outputs, diagnostic_outputs)
        self.assertEqual(
            case.expected_grimace_runtime_support_count,
            len(diagnostic_outputs),
        )
        self.assertGreater(len(rows), len(diagnostic_outputs))
        self.assertTrue(
            all(row["resolved_layer_completions"]["semantic"] for row in rows)
        )
        self.assertTrue(
            all(row["traversal_layer_completions"]["semantic"] for row in rows)
        )
        self.assertTrue(
            all(
                all(
                    component["remaining_count"] > 0
                    for component in row["traversal_assignment_state"]["semantic"]
                )
                for row in rows
            )
        )
        self.assertTrue(
            all(
                row["traversal_layer_completions"]["rdkit_traversal_writer"]
                == all(
                    component["remaining_count"] > 0
                    for component in row["traversal_assignment_state"][
                        "rdkit_traversal_writer"
                    ]
                )
                for row in rows
            )
        )
        self.assertEqual(
            {False, True},
            {
                row["resolved_layer_completions"]["rdkit_local_writer"]
                for row in rows
            },
        )
        self.assertTrue(
            all(
                row["directional_spelling"]
                == _directional_spelling_summary(row["smiles"])
                for row in rows
            )
        )
        self.assertEqual(
            {0, 1},
            {
                component["component_idx"]
                for row in rows
                for component in row["component_token_phase"]
            },
        )
        self.assertEqual(
            {0},
            {
                component["model_component_idx"]
                for row in rows
                for component in row["component_token_phase"]
            },
        )
        self.assertTrue(
            all(
                component["model_component_is_consistent"]
                for row in rows
                for component in row["component_token_phase"]
            )
        )
        self.assertTrue(
            all(
                component["carrier_assignment_singleton"]
                for row in rows
                for component in row["component_token_phase"]
            )
        )
        self.assertEqual(
            {2},
            {
                component["model_token_phase_component_count"]
                for row in rows
                for component in row["component_token_phase"]
            },
        )
        self.assertEqual(
            {
                ("stored", "stored"),
                ("flipped", "flipped"),
            },
            {
                tuple(
                    component["inferred_token_flip"]
                    for component in row["component_token_phase"]
                )
                for row in rows
            },
        )
        self.assertTrue(
            all(
                len(row["resolved_constraint_state"]["semantic"]) == 1
                and row["resolved_constraint_state"]["semantic"][0][
                    "runtime_component_ids"
                ]
                == [0, 1]
                and row["resolved_constraint_state"]["semantic"][0]["is_empty"] is False
                and row["resolved_constraint_state"]["semantic"][0][
                    "carrier_assignment_count"
                ]
                == 1
                and row["resolved_constraint_state"]["semantic"][0][
                    "token_phase_assignment_count"
                ]
                == 1
                for row in rows
            )
        )

    def test_shared_carrier_witness_diagnostics_match_runtime_support(self) -> None:
        cases = tuple(
            case
            for case in self.cases
            if case.expected_shared_carrier_group_count > 0
        )
        self.assertTrue(cases)

        for case in cases:
            mol = parse_smiles(case.smiles)
            prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)

            rows = _core._stereo_constraint_output_facts(prepared)
            diagnostic_outputs = frozenset(row["smiles"] for row in rows)
            runtime_outputs = public_enum_support(
                mol,
                **supported_public_kwargs(
                    isomericSmiles=True,
                    rootedAtAtom=-1,
                ),
            )

            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertEqual(runtime_outputs, diagnostic_outputs)
                self.assertEqual(
                    case.expected_grimace_runtime_support_count,
                    len(runtime_outputs),
                )
                self.assertTrue(
                    all(
                        row["shadow_debug"]["joined_support_boundary_matches_runtime"]
                        for row in rows
                    )
                )

    def test_component_token_phase_is_separate_from_carrier_assignment(self) -> None:
        saw_stored_token_flip = False
        saw_flipped_token_flip = False
        saw_rdkit_token_flip_adjustment = False
        saw_phase_dimension_needed = False
        saw_supported_token_observation = False
        seen_token_flip_inference_branches = set()

        for case in self.cases:
            mol = parse_smiles(case.smiles)
            prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
            rows = _core._stereo_constraint_output_facts(prepared)
            case_token_flip_inference_branches = Counter(
                component["token_flip_inference_inputs"]["inference_branch"]
                for row in rows
                for component in row["component_token_phase"]
            )

            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertTrue(rows)
                if case.expected_token_flip_inference_branch_counts:
                    self.assertEqual(
                        dict(case.expected_token_flip_inference_branch_counts),
                        dict(sorted(case_token_flip_inference_branches.items())),
                    )
                for row in rows:
                    row_token_flip_inference_branches = {
                        component["token_flip_inference_inputs"]["inference_branch"]
                        for component in row["component_token_phase"]
                    }
                    observation_state_key = (
                        "resolved_constraint_state_from_supported_token_observations"
                    )
                    mixed_observation_state_key = (
                        "resolved_constraint_state_from_known_token_flips_and_inferred_token_observations"
                    )
                    supported_observation_branches = set(
                        TOKEN_OBSERVATION_KIND_BY_SUPPORTED_BRANCH
                    )
                    if row_token_flip_inference_branches <= supported_observation_branches:
                        self.assertEqual(
                            row["resolved_constraint_state"],
                            row[observation_state_key],
                        )
                        self.assertEqual(
                            row["resolved_constraint_state"],
                            row[mixed_observation_state_key],
                        )
                    runtime_token_constraint_facts = row[
                        "runtime_token_constraint_facts"
                    ]
                    component_phase_boundary_facts = row[
                        "component_phase_boundary_facts"
                    ]
                    self.assertEqual(
                        component_phase_boundary_facts["phase_fact_count"],
                        len(component_phase_boundary_facts["phase_facts"]),
                    )
                    self.assertEqual(
                        component_phase_boundary_facts["begin_atom_fact_count"],
                        len(component_phase_boundary_facts["begin_atom_facts"]),
                    )
                    self.assertTrue(
                        all(
                            fact["source"] == "state"
                            for fact in component_phase_boundary_facts[
                                "phase_facts"
                            ]
                            + component_phase_boundary_facts["begin_atom_facts"]
                        )
                    )
                    runtime_known_token_flip_facts = runtime_token_constraint_facts[
                        "known_token_flip_facts"
                    ]
                    runtime_inferred_token_observation_facts = (
                        runtime_token_constraint_facts[
                            "inferred_token_observation_facts"
                        ]
                    )
                    self.assertEqual(
                        row["resolved_constraint_state"],
                        row[mixed_observation_state_key],
                    )
                    forced_token_flips = {
                        forced["runtime_component_idx"]: forced["token_flip"]
                        for component in row["resolved_constraint_state"]["semantic"]
                        for forced in component["forced_token_flips"]
                    }
                    self.assertTrue(
                        all(
                            not component["is_empty"]
                            and component["carrier_assignment_count"] > 0
                            and component["token_phase_assignment_count"] > 0
                            for component in row["resolved_constraint_state"]["semantic"]
                        )
                    )
                    self.assertTrue(row["component_token_phase"])
                    known_component_ids = {
                        component["component_idx"]
                        for component in row["component_token_phase"]
                        if component["token_constraint_kind"] == "known_token_flip"
                    }
                    inferred_component_ids = {
                        component["component_idx"]
                        for component in row["component_token_phase"]
                        if component["token_constraint_kind"]
                        == "inferred_token_observation"
                    }
                    self.assertEqual(
                        known_component_ids,
                        {
                            fact["runtime_component_idx"]
                            for fact in runtime_known_token_flip_facts
                        },
                    )
                    self.assertEqual(
                        inferred_component_ids,
                        {
                            fact["runtime_component_idx"]
                            for fact in runtime_inferred_token_observation_facts
                        },
                    )
                    self.assertEqual(
                        len(known_component_ids),
                        runtime_token_constraint_facts["known_token_flip_count"],
                    )
                    self.assertEqual(
                        len(inferred_component_ids),
                        runtime_token_constraint_facts[
                            "inferred_token_observation_count"
                        ],
                    )
                    for component in row["component_token_phase"]:
                        self.assertTrue(component["model_component_is_consistent"])
                        self.assertIsNotNone(component["model_component_idx"])
                        self.assertGreater(component["remaining_assignment_count"], 0)
                        self.assertTrue(component["inferred_matches_state"])
                        self.assertIsNotNone(component["inferred_token_flip"])
                        self.assertIn(
                            component["token_constraint_kind"],
                            (
                                "known_token_flip",
                                "inferred_token_observation",
                                "no_token_constraint",
                            ),
                        )
                        if component["state_token_flip"] != "unknown":
                            self.assertEqual(
                                "known_token_flip",
                                component["token_constraint_kind"],
                            )
                            self.assertEqual(
                                [],
                                [
                                    fact
                                    for fact in runtime_inferred_token_observation_facts
                                    if fact["runtime_component_idx"]
                                    == component["component_idx"]
                                ],
                            )
                        elif component["inferred_token_flip"] is not None:
                            self.assertEqual(
                                "inferred_token_observation",
                                component["token_constraint_kind"],
                            )
                            self.assertEqual(
                                [],
                                [
                                    fact
                                    for fact in runtime_known_token_flip_facts
                                    if fact["runtime_component_idx"]
                                    == component["component_idx"]
                                ],
                            )
                        token_phase_component_count = component[
                            "model_token_phase_component_count"
                        ]
                        self.assertGreater(token_phase_component_count, 0)
                        self.assertEqual(
                            component["remaining_assignment_count"]
                            * (2**token_phase_component_count),
                            component["token_phase_assignment_count_before_token"],
                        )
                        self.assertEqual(
                            component["token_phase_assignment_count_before_token"],
                            2 * component["token_observation_assignment_count_after"],
                        )
                        self.assertEqual(
                            component["inferred_token_flip"],
                            component["token_observation_forced_flip"],
                        )
                        self.assertEqual(
                            component["inferred_token_flip"],
                            forced_token_flips[component["component_idx"]],
                        )
                        self.assertTrue(
                            component["token_phase_dimension_explains_inferred_flip"]
                        )
                        self.assertTrue(
                            component["shadow_debug"][
                                "token_flip_matches_observation_backed_state"
                            ]
                        )
                        self.assertEqual(
                            component["state_token_flip"],
                            component["inferred_token_flip"],
                        )
                        token_flip_inputs = component["token_flip_inference_inputs"]
                        seen_token_flip_inference_branches.add(
                            token_flip_inputs["inference_branch"]
                        )
                        required_input_facts = frozenset(
                            token_flip_inputs["required_input_facts"]
                        )
                        self.assertEqual(
                            component["component_idx"],
                            token_flip_inputs["component_idx"],
                        )
                        input_observation_facts = {
                            fact["fact"]: fact
                            for fact in token_flip_inputs["input_observation_facts"]
                        }
                        self.assertEqual(
                            {
                                "component_phase",
                                "component_begin_atom",
                                "begin_side",
                                "selected_begin_token",
                                "first_emitted_candidate",
                                "rdkit_token_flip_adjustment",
                            },
                            set(input_observation_facts),
                        )
                        self.assertEqual(
                            token_flip_inputs["input_phase"],
                            input_observation_facts["component_phase"][
                                "input_phase"
                            ],
                        )
                        self.assertEqual(
                            token_flip_inputs["effective_phase"],
                            input_observation_facts["component_phase"][
                                "effective_phase"
                            ],
                        )
                        self.assertEqual(
                            token_flip_inputs["phase_source"],
                            input_observation_facts["component_phase"]["source"],
                        )
                        self.assertEqual(
                            token_flip_inputs["input_begin_atom_idx"],
                            input_observation_facts["component_begin_atom"][
                                "input_atom_idx"
                            ],
                        )
                        self.assertEqual(
                            token_flip_inputs["effective_begin_atom_idx"],
                            input_observation_facts["component_begin_atom"][
                                "effective_atom_idx"
                            ],
                        )
                        self.assertEqual(
                            token_flip_inputs["begin_atom_source"],
                            input_observation_facts["component_begin_atom"][
                                "source"
                            ],
                        )
                        self.assertEqual(
                            token_flip_inputs["begin_side_idx"],
                            input_observation_facts["begin_side"]["side_idx"],
                        )
                        self.assertEqual(
                            token_flip_inputs["begin_side_candidate_count"],
                            input_observation_facts["begin_side"][
                                "candidate_count"
                            ],
                        )
                        self.assertEqual(
                            token_flip_inputs["selected_begin_neighbor_idx"],
                            input_observation_facts["selected_begin_token"][
                                "neighbor_idx"
                            ],
                        )
                        self.assertEqual(
                            token_flip_inputs["selected_begin_token"],
                            input_observation_facts["selected_begin_token"][
                                "token"
                            ],
                        )
                        self.assertEqual(
                            token_flip_inputs["first_emitted_candidate_idx"],
                            input_observation_facts["first_emitted_candidate"][
                                "neighbor_idx"
                            ],
                        )
                        self.assertEqual(
                            token_flip_inputs["rdkit_token_flip_adjustment"],
                            input_observation_facts["rdkit_token_flip_adjustment"][
                                "value"
                            ],
                        )
                        self.assertEqual(
                            component["inferred_token_flip"],
                            token_flip_inputs["inferred_token_flip"],
                        )
                        self.assertIn(
                            token_flip_inputs["inference_branch"],
                            TOKEN_FLIP_INFERENCE_BRANCHES,
                        )
                        self.assertTrue(token_flip_inputs["has_required_inputs"])
                        self.assertEqual([], token_flip_inputs["missing_input_facts"])
                        self.assertLessEqual(
                            {
                                "component_phase",
                                "component_begin_atom",
                                "begin_side",
                                "rdkit_token_flip_adjustment",
                            },
                            required_input_facts,
                        )
                        self.assertIn(
                            token_flip_inputs["effective_phase"],
                            ("stored", "flipped"),
                        )
                        self.assertGreaterEqual(
                            token_flip_inputs["effective_begin_atom_idx"],
                            0,
                        )
                        self.assertIsNotNone(token_flip_inputs["begin_side_idx"])
                        if (
                            token_flip_inputs["inference_branch"]
                            != "isolated_all_single_candidate"
                        ):
                            self.assertLessEqual(
                                {"selected_begin_neighbor", "selected_begin_token"},
                                required_input_facts,
                            )
                            self.assertIsNotNone(
                                token_flip_inputs["selected_begin_neighbor_idx"]
                            )
                            self.assertIn(
                                token_flip_inputs["selected_begin_token"],
                                ("/", "\\"),
                            )
                        if (
                            token_flip_inputs["inference_branch"]
                            == "coupled_two_candidate_begin_side"
                        ):
                            self.assertIn(
                                "first_emitted_candidate_or_adjustment_fallback",
                                required_input_facts,
                            )
                        if (
                            token_flip_inputs["inference_branch"]
                            in supported_observation_branches
                        ):
                            token_observation_facts = component[
                                "token_observation_facts"
                            ]
                            self.assertTrue(
                                component["token_observation_supported_branch"]
                            )
                            self.assertIsNone(
                                component["token_observation_unsupported_reason"]
                            )
                            self.assertEqual(1, len(token_observation_facts))
                            token_observation = token_observation_facts[0]
                            if (
                                component["token_constraint_kind"]
                                == "inferred_token_observation"
                            ):
                                self.assertEqual(
                                    [token_observation],
                                    [
                                        fact
                                        for fact in runtime_inferred_token_observation_facts
                                        if fact["runtime_component_idx"]
                                        == component["component_idx"]
                                    ],
                                )
                            else:
                                self.assertEqual(
                                    "known_token_flip",
                                    component["token_constraint_kind"],
                                )
                                self.assertEqual(
                                    [],
                                    [
                                        fact
                                        for fact in runtime_inferred_token_observation_facts
                                        if fact["runtime_component_idx"]
                                        == component["component_idx"]
                                    ],
                                )
                            self.assertEqual(
                                component["component_idx"],
                                token_observation["runtime_component_idx"],
                            )
                            self.assertEqual(
                                token_flip_inputs["effective_phase"],
                                token_observation["component_phase"],
                            )
                            self.assertEqual(
                                TOKEN_OBSERVATION_KIND_BY_SUPPORTED_BRANCH[
                                    token_flip_inputs["inference_branch"]
                                ],
                                token_observation["observation_kind"],
                            )
                            if (
                                token_flip_inputs["inference_branch"]
                                == "isolated_all_single_candidate"
                            ):
                                self.assertIsNone(
                                    token_observation["selected_begin_token"]
                                )
                            else:
                                self.assertEqual(
                                    token_flip_inputs["selected_begin_token"],
                                    token_observation["selected_begin_token"],
                                )
                            if (
                                token_flip_inputs["inference_branch"]
                                == "coupled_two_candidate_begin_side"
                            ):
                                expected_selected_is_first = None
                                if (
                                    token_flip_inputs["first_emitted_candidate_idx"]
                                    is not None
                                ):
                                    expected_selected_is_first = (
                                        token_flip_inputs[
                                            "first_emitted_candidate_idx"
                                        ]
                                        == token_flip_inputs[
                                            "selected_begin_neighbor_idx"
                                        ]
                                    )
                                self.assertEqual(
                                    expected_selected_is_first,
                                    token_observation[
                                        "selected_begin_neighbor_is_first_emitted"
                                    ],
                                )
                            else:
                                self.assertIsNone(
                                    token_observation[
                                        "selected_begin_neighbor_is_first_emitted"
                                    ]
                                )
                            self.assertEqual(
                                token_flip_inputs["rdkit_token_flip_adjustment"],
                                token_observation["rdkit_token_flip_adjustment"],
                            )
                            adjustment_inputs = input_observation_facts[
                                "rdkit_token_flip_adjustment"
                            ]
                            expected_adjustment_observations = []
                            if adjustment_inputs["root_begin_side_adjustment"]:
                                expected_adjustment_observations.append(
                                    "root_begin_side_orientation"
                                )
                            if adjustment_inputs["adjacent_two_candidate_adjustment"]:
                                expected_adjustment_observations.append(
                                    "adjacent_two_candidate_first_emitted"
                                )
                            self.assertEqual(
                                expected_adjustment_observations,
                                token_observation[
                                    "rdkit_token_flip_adjustment_observations"
                                ],
                            )
                            self.assertEqual(
                                component["inferred_token_flip"],
                                token_observation["implied_token_flip"],
                            )
                            self.assertEqual(
                                component["token_phase_assignment_count_before_token"],
                                component[
                                    "token_observation_assignment_count_before"
                                ],
                            )
                            self.assertEqual(
                                component["shadow_debug"][
                                    "token_flip_assignment_count_after_token"
                                ],
                                component["token_observation_assignment_count_after"],
                            )
                            self.assertEqual(
                                component["inferred_token_flip"],
                                component["token_observation_forced_flip"],
                            )
                            self.assertTrue(
                                component["token_observation_matches_inferred_flip"]
                            )
                            saw_supported_token_observation = True
                        else:
                            self.assertFalse(
                                component["token_observation_supported_branch"]
                            )
                            self.assertEqual([], component["token_observation_facts"])
                            self.assertIsNotNone(
                                component["token_observation_unsupported_reason"]
                            )
                        self.assertEqual(
                            component["needs_token_phase_assignment_dimension"],
                            component["carrier_assignment_singleton"],
                        )
                        saw_stored_token_flip |= (
                            component["inferred_token_flip"] == "stored"
                        )
                        saw_flipped_token_flip |= (
                            component["inferred_token_flip"] == "flipped"
                        )
                        saw_rdkit_token_flip_adjustment |= component[
                            "rdkit_token_flip_adjustment"
                        ]
                        saw_phase_dimension_needed |= component[
                            "needs_token_phase_assignment_dimension"
                        ]

        self.assertTrue(saw_stored_token_flip)
        self.assertTrue(saw_flipped_token_flip)
        self.assertTrue(saw_rdkit_token_flip_adjustment)
        self.assertTrue(saw_phase_dimension_needed)
        self.assertTrue(saw_supported_token_observation)
        self.assertTrue(seen_token_flip_inference_branches)
        self.assertEqual(
            TOKEN_FLIP_INFERENCE_BRANCHES,
            seen_token_flip_inference_branches,
        )

    def test_token_flip_adjustment_reason_counts_match_pinned_witnesses(self) -> None:
        saw_pinned_adjustment_reasons = False
        for case in self.cases:
            mol = parse_smiles(case.smiles)
            prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
            rows = _core._stereo_constraint_output_facts(prepared)
            counts: Counter[str] = Counter()

            with self.subTest(case_id=case.case_id, source=case.source):
                for row in rows:
                    for component in row["component_token_phase"]:
                        input_observation_facts = {
                            fact["fact"]: fact
                            for fact in component["token_flip_inference_inputs"][
                                "input_observation_facts"
                            ]
                        }
                        adjustment = input_observation_facts[
                            "rdkit_token_flip_adjustment"
                        ]
                        self.assertEqual(
                            adjustment["value"],
                            adjustment["root_begin_side_adjustment"]
                            ^ adjustment["adjacent_two_candidate_adjustment"],
                        )
                        if adjustment["value"]:
                            counts["value_true"] += 1
                        if adjustment["root_begin_side_adjustment"]:
                            counts["root_begin_side_orientation"] += 1
                            self.assertTrue(adjustment["begin_atom_is_root"])
                            self.assertEqual(
                                "after_atom",
                                adjustment["begin_side_orientation"],
                            )
                        if adjustment["adjacent_two_candidate_adjustment"]:
                            counts["adjacent_two_candidate_first_emitted"] += 1
                            self.assertIsNotNone(
                                adjustment["adjacent_two_candidate_side_idx"]
                            )
                            self.assertTrue(adjustment["selected_neighbor_is_root"])
                            self.assertTrue(
                                adjustment["adjacent_first_emitted_is_not_begin"]
                            )
                            self.assertIsNotNone(
                                adjustment["adjacent_first_emitted_candidate_idx"]
                            )

                if case.expected_token_flip_adjustment_reason_counts:
                    saw_pinned_adjustment_reasons = True
                    self.assertLessEqual(
                        set(dict(case.expected_token_flip_adjustment_reason_counts)),
                        TOKEN_FLIP_ADJUSTMENT_REASON_COUNT_KEYS,
                    )
                    self.assertEqual(
                        dict(case.expected_token_flip_adjustment_reason_counts),
                        dict(sorted(counts.items())),
                    )

        self.assertTrue(saw_pinned_adjustment_reasons)

    def test_marker_placement_state_filters_marker_events(self) -> None:
        for case in self.cases:
            mol = parse_smiles(case.smiles)
            prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
            rows = _core._stereo_constraint_output_facts(prepared)

            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertTrue(rows)
                for row in rows:
                    marker_event_counts = Counter(
                        event["component_idx"] for event in row["marker_event_facts"]
                    )
                    for layer_name, components in row["marker_placement_state"].items():
                        for component in components:
                            self.assertEqual(
                                marker_event_counts[component["component_idx"]],
                                component["marker_event_count"],
                            )
                            self.assertEqual(
                                component["row_count_after_marker_events"],
                                len(component["rows_after_marker_events"]),
                            )
                            self.assertEqual(
                                component[
                                    "token_phase_assignment_count_after_marker_events"
                                ],
                                len(
                                    component[
                                        "token_phase_assignment_ids_after_marker_events"
                                    ]
                                ),
                            )
                            self.assertEqual(
                                component[
                                    "neighbor_assignment_count_after_marker_events"
                                ],
                                len(
                                    component[
                                        "neighbor_assignment_ids_after_marker_events"
                                    ]
                                ),
                            )
                            self.assertEqual(
                                set(component["side_ids"]),
                                {
                                    domain["side_idx"]
                                    for domain in component["survivor_side_domains"]
                                },
                            )
                            for domain in component["survivor_side_domains"]:
                                self.assertIsInstance(
                                    domain["carrier_neighbors"],
                                    list,
                                )
                                self.assertIsInstance(
                                    domain["marker_neighbor_sets"],
                                    list,
                                )
                            self.assertLessEqual(
                                component["row_count_after_marker_events"],
                                component["row_count_before_marker_events"],
                            )
                            self.assertEqual(
                                component["is_empty_after_marker_events"],
                                component["row_count_after_marker_events"] == 0,
                            )
                            if component["token_phase_assignment_count"] == 0:
                                self.assertEqual(
                                    0,
                                    component["row_count_before_marker_events"],
                                    layer_name,
                                )
                    self.assertTrue(
                        all(
                            component["row_count_after_marker_events"] > 0
                            for component in row["marker_placement_state"]["semantic"]
                        )
                    )

    def test_support_boundary_marker_placement_state_tracks_no_marker_events(self) -> None:
        saw_boundary_marker_event = False
        saw_boundary_no_marker_event = False

        for case in self.cases:
            mol = parse_smiles(case.smiles)
            prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
            rows = _core._stereo_constraint_output_facts(prepared)

            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertTrue(rows)
                for row in rows:
                    self.assertTrue(
                        all(
                            event["event"] == "marker_placed"
                            and isinstance(event["slot"], int)
                            for event in row["marker_event_facts"]
                        )
                    )

                    support_boundary = row["support_boundary"]
                    boundary_marker_event_counts = Counter(
                        event["component_idx"]
                        for event in support_boundary["marker_event_facts"]
                    )
                    saw_boundary_marker_event |= any(
                        event["event"] == "marker_placed"
                        for event in support_boundary["marker_event_facts"]
                    )
                    saw_boundary_no_marker_event |= any(
                        event["event"] == "no_marker"
                        for event in support_boundary["marker_event_facts"]
                    )
                    self.assertTrue(
                        all(
                            isinstance(event["slot"], int)
                            for event in support_boundary["marker_event_facts"]
                        )
                    )
                    self.assertLessEqual(
                        len(support_boundary["marker_obligation_facts"]),
                        len(support_boundary["marker_event_facts"]),
                    )
                    self.assertTrue(
                        all(
                            isinstance(event["slot"], int)
                            for event in support_boundary["marker_obligation_facts"]
                        )
                    )
                    self.assertTrue(
                        all(
                            isinstance(domain["slot"], int)
                            and isinstance(domain["same_edge_future_marker_slots"], list)
                            and isinstance(
                                domain["same_side_other_edge_future_markers"], list
                            )
                            for domain in support_boundary["marker_obligation_domains"]
                        )
                    )

                    for components in support_boundary["marker_placement_state"].values():
                        for component in components:
                            self.assertEqual(
                                boundary_marker_event_counts[component["component_idx"]],
                                component["marker_event_count"],
                            )
                            self.assertEqual(
                                component["row_count_after_marker_events"],
                                len(component["rows_after_marker_events"]),
                            )
                            self.assertEqual(
                                component[
                                    "token_phase_assignment_count_after_marker_events"
                                ],
                                len(
                                    component[
                                        "token_phase_assignment_ids_after_marker_events"
                                    ]
                                ),
                            )
                            self.assertEqual(
                                component[
                                    "neighbor_assignment_count_after_marker_events"
                                ],
                                len(
                                    component[
                                        "neighbor_assignment_ids_after_marker_events"
                                    ]
                                ),
                            )
                            self.assertEqual(
                                set(component["side_ids"]),
                                {
                                    domain["side_idx"]
                                    for domain in component["survivor_side_domains"]
                                },
                            )
                            self.assertLessEqual(
                                component["row_count_after_marker_events"],
                                component["row_count_before_marker_events"],
                            )
                            self.assertEqual(
                                component["is_empty_after_marker_events"],
                                component["row_count_after_marker_events"] == 0,
                            )
                    self.assertTrue(
                        all(
                            component["row_count_after_marker_events"] > 0
                            for component in support_boundary[
                                "marker_placement_state"
                            ]["semantic"]
                        )
                    )
                    for components in support_boundary["marker_obligation_state"].values():
                        for component in components:
                            self.assertEqual(
                                component["row_count_after_marker_events"],
                                len(component["rows_after_marker_events"]),
                            )
                            self.assertLessEqual(
                                component["row_count_after_marker_events"],
                                component["row_count_before_marker_events"],
                            )
                            self.assertEqual(
                                component["is_empty_after_marker_events"],
                                component["row_count_after_marker_events"] == 0,
                            )

        self.assertTrue(saw_boundary_marker_event)
        self.assertTrue(saw_boundary_no_marker_event)

    def test_deferred_marker_obligation_witness_scanner_is_bounded(self) -> None:
        case = next(
            case
            for case in self.cases
            if case.case_id == "minimal_nonstereo_double_hazard"
        )
        assert case.expected_marker_row_diagnostics is not None
        mol = parse_smiles(case.smiles)
        prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)

        full_scan = _core._stereo_deferred_marker_obligation_witnesses(prepared)
        self.assertFalse(full_scan["truncated"])
        self.assertEqual(
            case.expected_marker_row_diagnostics.output_row_count,
            full_scan["terminal_state_count"],
        )
        self.assertEqual([], full_scan["witnesses"])

        bounded_scan = _core._stereo_deferred_marker_obligation_witnesses(
            prepared,
            max_terminal_states=5,
        )
        self.assertTrue(bounded_scan["truncated"])
        self.assertEqual(5, bounded_scan["terminal_state_count"])

    def test_reduced_porphyrin_terminal_rows_keep_marker_boundary_survivors(self) -> None:
        case = next(
            case
            for case in self.cases
            if case.case_id == "reduced_porphyrin_traversal_coupling"
        )
        mol = parse_smiles(case.smiles)
        prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
        rows = _core._stereo_constraint_output_facts(prepared)

        boundary_event_counts = Counter(
            event["event"]
            for row in rows
            for event in row["support_boundary"]["marker_event_facts"]
        )
        self.assertGreater(boundary_event_counts["marker_placed"], 0)
        self.assertGreater(boundary_event_counts["no_marker"], 0)

        top_level_empty_counts = Counter(
            layer_name
            for row in rows
            for layer_name, components in row["marker_placement_state"].items()
            for component in components
            if component["is_empty_after_marker_events"]
        )
        self.assertFalse(top_level_empty_counts)

        boundary_empty_counts = Counter(
            layer_name
            for row in rows
            for layer_name, components in row["support_boundary"][
                "marker_placement_state"
            ].items()
            for component in components
            if component["is_empty_after_marker_events"]
        )
        self.assertFalse(boundary_empty_counts)
        boundary_obligation_empty_counts = Counter(
            layer_name
            for row in rows
            for layer_name, components in row["support_boundary"][
                "marker_obligation_state"
            ].items()
            for component in components
            if component["is_empty_after_marker_events"]
        )
        self.assertFalse(boundary_obligation_empty_counts)

    def test_marker_obligations_do_not_coalesce_different_edges(self) -> None:
        case = next(
            case
            for case in self.cases
            if case.case_id == "minimal_nonstereo_double_hazard"
        )
        mol = parse_smiles(case.smiles)
        prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
        rows = _core._stereo_constraint_output_facts(prepared)

        saw_same_side_different_edge_future_marker = False
        for row in rows:
            domains = row["support_boundary"]["marker_obligation_domains"]
            obligations = row["support_boundary"]["marker_obligation_facts"]

            retained_obligations = {
                (
                    event["event"],
                    event["side_idx"],
                    event["slot"],
                    tuple(sorted((event["begin_idx"], event["end_idx"]))),
                )
                for event in obligations
            }
            for domain in domains:
                other_edge_future_markers = domain[
                    "same_side_other_edge_future_markers"
                ]
                if not other_edge_future_markers:
                    continue
                if domain["same_edge_future_marker_slots"]:
                    continue

                saw_same_side_different_edge_future_marker = True
                edge = tuple(domain["canonical_edge"])
                self.assertIn(
                    ("no_marker", domain["side_idx"], domain["slot"], edge),
                    retained_obligations,
                )
                self.assertFalse(domain["is_deferred"])

        self.assertTrue(saw_same_side_different_edge_future_marker)

    def test_sampled_rdkit_outputs_avoid_local_invalid_exact_spellings(self) -> None:
        cases_with_sampled_expectations = tuple(
            case
            for case in self.cases
            if case.expected_rdkit_sampled_support_count is not None
        )
        self.assertTrue(cases_with_sampled_expectations)

        for case in cases_with_sampled_expectations:
            mol = parse_smiles(case.smiles)
            source_identity = Chem.MolToSmiles(
                Chem.Mol(mol),
                canonical=True,
                isomericSmiles=True,
            )
            prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)

            rows = _core._stereo_constraint_output_facts(prepared)
            current_exact_support = frozenset(row["smiles"] for row in rows)
            local_invalid_exact_outputs = frozenset(
                row["smiles"]
                for row in rows
                if not row["resolved_layer_completions"]["rdkit_local_writer"]
            )
            rdkit_sampled_outputs = _rdkit_sampled_outputs(mol)
            sampled_outside_current_exact_support = (
                rdkit_sampled_outputs - current_exact_support
            )
            sampled_outside_current_exact_identities = tuple(
                _canonical_isomeric_smiles(smiles)
                for smiles in sampled_outside_current_exact_support
            )
            ring_digit_direction_erased_current_keys = frozenset(
                _erase_ring_digit_adjacent_direction_markers(smiles)
                for smiles in current_exact_support
            )
            direction_erased_current_keys = frozenset(
                _erase_direction_markers(smiles)
                for smiles in current_exact_support
            )
            current_exact_outputs_with_ring_digit_direction = sum(
                row["directional_spelling"]["ring_digit_adjacent"] > 0
                for row in {row["smiles"]: row for row in rows}.values()
            )
            rdkit_sampled_outputs_with_ring_digit_direction = sum(
                _directional_spelling_summary(smiles)["ring_digit_adjacent"] > 0
                for smiles in rdkit_sampled_outputs
            )
            sampled_outside_current_exact_with_ring_digit_direction = sum(
                _directional_spelling_summary(smiles)["ring_digit_adjacent"] > 0
                for smiles in sampled_outside_current_exact_support
            )

            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertEqual(
                    case.expected_rdkit_sampled_support_count,
                    len(rdkit_sampled_outputs),
                )
                self.assertEqual(
                    case.expected_rdkit_sampled_exact_support_overlap_count,
                    len(rdkit_sampled_outputs & current_exact_support),
                )
                self.assertEqual(
                    case.expected_rdkit_sampled_exact_local_invalid_overlap_count,
                    len(rdkit_sampled_outputs & local_invalid_exact_outputs),
                )
                self.assertEqual(
                    case.expected_rdkit_sampled_outside_current_exact_support_count,
                    len(sampled_outside_current_exact_support),
                )
                self.assertEqual(
                    case.expected_rdkit_sampled_outside_current_exact_identity_equivalent_count,
                    sum(
                        identity == source_identity
                        for identity in sampled_outside_current_exact_identities
                    ),
                )
                self.assertEqual(
                    case.expected_rdkit_sampled_outside_current_exact_parse_failure_count,
                    sampled_outside_current_exact_identities.count(None),
                )
                self.assertEqual(
                    case.expected_rdkit_sampled_outside_current_exact_ring_digit_direction_erased_overlap_count,
                    sum(
                        _erase_ring_digit_adjacent_direction_markers(smiles)
                        in ring_digit_direction_erased_current_keys
                        for smiles in sampled_outside_current_exact_support
                    ),
                )
                self.assertEqual(
                    case.expected_rdkit_sampled_outside_current_exact_direction_erased_overlap_count,
                    sum(
                        _erase_direction_markers(smiles) in direction_erased_current_keys
                        for smiles in sampled_outside_current_exact_support
                    ),
                )
                self.assertEqual(
                    case.expected_grimace_runtime_outputs_with_ring_digit_direction_count,
                    current_exact_outputs_with_ring_digit_direction,
                )
                self.assertEqual(
                    case.expected_rdkit_sampled_outputs_with_ring_digit_direction_count,
                    rdkit_sampled_outputs_with_ring_digit_direction,
                )
                self.assertEqual(
                    case.expected_rdkit_sampled_outside_current_exact_with_ring_digit_direction_count,
                    sampled_outside_current_exact_with_ring_digit_direction,
                )

    def test_rdkit_respelling_candidate_covers_sampled_spelling_family(self) -> None:
        cases_with_sampled_expectations = tuple(
            case
            for case in self.cases
            if case.expected_rdkit_sampled_support_count is not None
        )
        self.assertTrue(cases_with_sampled_expectations)

        for case in cases_with_sampled_expectations:
            mol = parse_smiles(case.smiles)
            prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
            rows = _core._stereo_constraint_output_facts(prepared)

            current_exact_support = frozenset(row["smiles"] for row in rows)
            candidate_respelling_family = _rdkit_respelling_family(current_exact_support)
            rdkit_sampled_outputs = _rdkit_sampled_outputs(mol)
            source_identity = Chem.MolToSmiles(
                Chem.Mol(mol),
                canonical=True,
                isomericSmiles=True,
            )

            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertEqual(rdkit_sampled_outputs, candidate_respelling_family)
                self.assertEqual(
                    case.expected_rdkit_sampled_support_count,
                    len(candidate_respelling_family),
                )
                self.assertEqual(
                    case.expected_rdkit_sampled_outside_current_exact_support_count,
                    len(candidate_respelling_family - current_exact_support),
                )
                self.assertTrue(
                    all(
                        _canonical_isomeric_smiles(smiles) == source_identity
                        for smiles in candidate_respelling_family
                    )
                )

    def test_direction_erased_skeleton_family_covers_sampled_spelling_family(self) -> None:
        cases_with_sampled_expectations = tuple(
            case
            for case in self.cases
            if case.expected_rdkit_sampled_support_count is not None
        )
        self.assertTrue(cases_with_sampled_expectations)

        for case in cases_with_sampled_expectations:
            mol = parse_smiles(case.smiles)
            prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
            rows = _core._stereo_constraint_output_facts(prepared)

            current_skeletons = frozenset(
                row["directional_spelling"]["direction_erased_skeleton"]
                for row in rows
            )
            rdkit_sampled_outputs = _rdkit_sampled_outputs(mol)
            rdkit_sampled_skeletons = frozenset(
                _directional_spelling_summary(smiles)["direction_erased_skeleton"]
                for smiles in rdkit_sampled_outputs
            )
            rdkit_sampled_outside_current = rdkit_sampled_outputs - frozenset(
                row["smiles"] for row in rows
            )
            rdkit_sampled_outside_current_skeletons = frozenset(
                _directional_spelling_summary(smiles)["direction_erased_skeleton"]
                for smiles in rdkit_sampled_outside_current
            )
            current_marker_sequences_by_skeleton = {
                row["directional_spelling"]["direction_erased_skeleton"]: tuple(
                    row["directional_spelling"]["ordered_markers"]
                )
                for row in rows
            }
            current_marker_slots_by_skeleton = {}
            projected_marker_slots_by_skeleton = {}
            current_marker_contexts_by_skeleton = {}
            current_marker_provenance_by_skeleton = {}
            for row in rows:
                skeleton, slots = _direction_marker_slots_from_summary(
                    row["directional_spelling"]
                )
                self.assertEqual(
                    row["directional_spelling"]["direction_erased_skeleton"],
                    skeleton,
                )
                current_marker_slots_by_skeleton[skeleton] = slots
                current_marker_contexts_by_skeleton[skeleton] = _marker_slots_by_slot(
                    row["directional_spelling"]["marker_slots"]
                )
                provenance = tuple(row["directional_marker_provenance"])
                traversal_facts = tuple(row["traversal_facts"])
                self.assertEqual(len(slots), len(provenance))
                self.assertEqual(
                    _marker_slot_pairs(slots),
                    tuple(
                        (int(marker["slot"]), str(marker["marker"]))
                        for marker in provenance
                    ),
                )
                self.assertTrue(
                    all(
                        "component_idx" in marker
                        and "side_idx" in marker
                        and "bond_idx" in marker
                        and "canonical_edge" in marker
                        for marker in provenance
                    )
                )
                self.assertTrue(
                    all(
                        int(marker["component_idx"]) >= 0
                        and int(marker["side_idx"]) >= 0
                        and int(marker["bond_idx"]) >= 0
                        for marker in provenance
                    )
                )
                self.assertTrue(
                    all(
                        marker["trace_role"]
                        == {
                            "after_ring_label": "ring_open",
                            "before_ring_label": "ring_close",
                            "branch_edge": "branch",
                        }.get(marker["local_role"], "tree_or_chain")
                        for marker in provenance
                    )
                )
                self.assertEqual(
                    len(row["resolved_facts"]) + 2 * len(provenance),
                    len(traversal_facts),
                )
                self.assertEqual(
                    len(provenance),
                    sum(
                        fact["fact"] == "carrier_edge_emitted"
                        for fact in traversal_facts
                    ),
                )
                self.assertEqual(
                    len(provenance),
                    sum(
                        fact["fact"] == "directional_marker_placed"
                        for fact in traversal_facts
                    ),
                )
                for layer in (
                    "semantic",
                    "rdkit_local_writer",
                    "rdkit_traversal_writer",
                ):
                    layer_state = row["traversal_assignment_state"][layer]
                    self.assertEqual(1, len(layer_state))
                    component_state = layer_state[0]
                    self.assertEqual([0, 1, 2, 3], component_state["side_ids"])
                    self.assertEqual(
                        component_state["remaining_count"],
                        len(component_state["remaining_assignment_ids"]),
                    )
                projection = row["ring_closure_marker_projection"]
                projected_slots = tuple(
                    _DirectionMarkerSlot(
                        slot=int(marker["slot"]),
                        marker=str(marker["marker"]),
                    )
                    for marker in projection["marker_slots"]
                )
                self.assertEqual(skeleton, projection["direction_erased_skeleton"])
                self.assertEqual(
                    _rdkit_ring_closure_marker_slots(skeleton, slots),
                    projected_slots,
                )
                self.assertEqual(
                    projection["smiles"],
                    _smiles_from_direction_marker_slots(skeleton, projected_slots),
                )
                projected_marker_slots_by_skeleton[skeleton] = projected_slots
                current_marker_provenance_by_skeleton[skeleton] = {
                    int(marker["slot"]): marker for marker in provenance
                }
            rdkit_summaries = tuple(
                _directional_spelling_summary(smiles)
                for smiles in rdkit_sampled_outputs
            )
            rdkit_marker_sequences_by_skeleton = {
                summary["direction_erased_skeleton"]: tuple(summary["ordered_markers"])
                for summary in rdkit_summaries
            }
            rdkit_marker_slots_by_skeleton = {
                skeleton: slots
                for skeleton, slots in (
                    _direction_marker_slots(smiles) for smiles in rdkit_sampled_outputs
                )
            }
            same_marker_sequence_skeletons = frozenset(
                skeleton
                for skeleton, marker_sequence in current_marker_sequences_by_skeleton.items()
                if rdkit_marker_sequences_by_skeleton[skeleton] == marker_sequence
            )
            different_marker_sequence_skeletons = (
                current_skeletons - same_marker_sequence_skeletons
            )
            actual_transitions = tuple(
                sorted(
                    (
                        skeleton,
                        current_marker_sequences_by_skeleton[skeleton],
                        rdkit_marker_sequences_by_skeleton[skeleton],
                    )
                    for skeleton in different_marker_sequence_skeletons
                )
            )
            expected_transitions = tuple(
                (
                    transition.direction_erased_skeleton,
                    transition.grimace_ordered_markers,
                    transition.rdkit_ordered_markers,
                )
                for transition in case.expected_marker_sequence_transitions
            )
            transformed_marker_sequences_by_skeleton = {
                skeleton: _ordered_markers_from_slots(
                    projected_marker_slots_by_skeleton[skeleton]
                )
                for skeleton in current_marker_slots_by_skeleton
            }
            transformed_skeletons = frozenset(
                skeleton
                for skeleton, marker_sequence in transformed_marker_sequences_by_skeleton.items()
                if marker_sequence != current_marker_sequences_by_skeleton[skeleton]
            )
            expected_transformed_skeletons = frozenset(
                transition.direction_erased_skeleton
                for transition in case.expected_marker_sequence_transitions
            )
            transformed_exact_support = frozenset(
                _smiles_from_direction_marker_slots(
                    skeleton,
                    projected_marker_slots_by_skeleton[skeleton],
                )
                for skeleton in current_marker_slots_by_skeleton
            )
            residual_slot_transitions = tuple(
                sorted(
                    (
                        skeleton,
                        _marker_slot_pairs(projected_marker_slots_by_skeleton[skeleton]),
                        _marker_slot_pairs(rdkit_marker_slots_by_skeleton[skeleton]),
                    )
                    for skeleton in current_marker_slots_by_skeleton
                    if projected_marker_slots_by_skeleton[skeleton]
                    != rdkit_marker_slots_by_skeleton[skeleton]
                )
            )
            residual_slot_transition_map = {
                (skeleton, transformed_marker_slots): rdkit_marker_slots
                for (
                    skeleton,
                    transformed_marker_slots,
                    rdkit_marker_slots,
                ) in residual_slot_transitions
            }
            projected_rdkit_support = frozenset(
                _smiles_from_direction_marker_slots(
                    skeleton,
                    _apply_residual_slot_transition(
                        skeleton=skeleton,
                        markers=projected_marker_slots_by_skeleton[skeleton],
                        residual_slot_transition_map=residual_slot_transition_map,
                    ),
                )
                for skeleton in current_marker_slots_by_skeleton
            )
            residual_context_counts = Counter()
            residual_provenance_classes = Counter()
            for skeleton, transformed_slots, rdkit_slots in residual_slot_transitions:
                transformed_slots_by_slot = dict(transformed_slots)
                rdkit_slots_by_slot = dict(rdkit_slots)
                removed_slots = tuple(
                    slot
                    for slot, marker in transformed_slots
                    if rdkit_slots_by_slot.get(slot) != marker
                )
                added_slots = tuple(
                    slot
                    for slot, marker in rdkit_slots
                    if transformed_slots_by_slot.get(slot) != marker
                )
                self.assertEqual(len(removed_slots), len(added_slots))
                for removed_slot, added_slot in zip(
                    removed_slots, added_slots, strict=True
                ):
                    removed_context = current_marker_contexts_by_skeleton[skeleton][removed_slot]
                    removed_provenance = current_marker_provenance_by_skeleton[skeleton][
                        removed_slot
                    ]
                    source_role = str(removed_provenance["trace_role"])
                    target_context = _target_slot_context(skeleton, added_slot)
                    self.assertEqual(
                        str(removed_provenance["local_role"]),
                        _slot_local_role(skeleton, removed_slot),
                    )
                    residual_provenance_classes[
                        _residual_marker_provenance_class(
                            source_role=source_role,
                            target_context=target_context,
                        )
                    ] += 1
                    residual_context_counts[
                        (
                            bool(removed_context["after_ring_label"]),
                            bool(removed_context["after_branch_open"]),
                            target_context,
                        )
                    ] += 1

            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertLessEqual(current_skeletons, rdkit_sampled_skeletons)
                self.assertEqual(
                    case.expected_rdkit_sampled_support_count,
                    len(rdkit_sampled_skeletons),
                )
                self.assertEqual(
                    case.expected_rdkit_sampled_outside_current_exact_support_count,
                    len(rdkit_sampled_outside_current_skeletons),
                )
                self.assertEqual(
                    case.expected_direction_erased_skeletons_with_same_marker_sequence_count,
                    len(same_marker_sequence_skeletons),
                )
                self.assertEqual(
                    case.expected_direction_erased_skeletons_with_different_marker_sequence_count,
                    len(different_marker_sequence_skeletons),
                )
                self.assertEqual(
                    current_skeletons,
                    same_marker_sequence_skeletons | different_marker_sequence_skeletons,
                )
                self.assertEqual(expected_transitions, actual_transitions)
                self.assertEqual(expected_transformed_skeletons, transformed_skeletons)
                self.assertEqual(
                    {
                        skeleton: rdkit_marker_sequences_by_skeleton[skeleton]
                        for skeleton in current_marker_slots_by_skeleton
                    },
                    transformed_marker_sequences_by_skeleton,
                )
                self.assertEqual(
                    case.expected_ring_closure_marker_transform_support_count,
                    len(transformed_exact_support),
                )
                self.assertEqual(
                    case.expected_ring_closure_marker_transform_exact_overlap_count,
                    len(transformed_exact_support & rdkit_sampled_outputs),
                )
                self.assertEqual(
                    case.expected_ring_closure_marker_transform_outside_rdkit_count,
                    len(transformed_exact_support - rdkit_sampled_outputs),
                )
                self.assertEqual(
                    projected_rdkit_support,
                    projected_rdkit_support & rdkit_sampled_outputs,
                )
                self.assertEqual(
                    case.expected_ring_closure_marker_transform_outside_rdkit_count,
                    len(residual_slot_transitions),
                )
                self.assertEqual(
                    Counter(
                        dict(
                            case.expected_ring_closure_marker_transform_residual_provenance_classes
                        )
                    ),
                    residual_provenance_classes,
                )
                self.assertEqual(
                    Counter(
                        {
                            (False, False, ("before_ring_label",)): 2,
                            (False, True, ("before_ring_label",)): 2,
                            (False, False, ("before_bracket_atom",)): 1,
                            (False, False, ("before_bracket_atom", "after_branch_open")): 1,
                            (True, False, ("before_bracket_atom",)): 2,
                            (True, False, ("before_bracket_atom", "after_branch_open")): 2,
                        }
                    ),
                    residual_context_counts,
                )


if __name__ == "__main__":
    unittest.main()
