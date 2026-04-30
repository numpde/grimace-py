from __future__ import annotations

import math
import unittest
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
    for idx, char in enumerate(smiles):
        if char not in ("/", "\\"):
            direction_erased_skeleton.append(char)
            continue
        total_count += 1
        ordered_markers.append(char)

        previous_char = smiles[idx - 1] if idx > 0 else ""
        next_char = smiles[idx + 1] if idx + 1 < len(smiles) else ""
        if (
            previous_char.isdigit()
            or previous_char == "%"
            or next_char.isdigit()
            or next_char == "%"
        ):
            ring_digit_adjacent_count += 1

    return {
        "total": total_count,
        "ring_digit_adjacent": ring_digit_adjacent_count,
        "non_ring": total_count - ring_digit_adjacent_count,
        "direction_erased_skeleton": "".join(direction_erased_skeleton),
        "ordered_markers": ordered_markers,
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
            for row in rows:
                skeleton, slots = _direction_marker_slots(row["smiles"])
                self.assertEqual(
                    row["directional_spelling"]["direction_erased_skeleton"],
                    skeleton,
                )
                current_marker_slots_by_skeleton[skeleton] = slots
            rdkit_summaries = tuple(
                _directional_spelling_summary(smiles)
                for smiles in rdkit_sampled_outputs
            )
            rdkit_marker_sequences_by_skeleton = {
                summary["direction_erased_skeleton"]: tuple(summary["ordered_markers"])
                for summary in rdkit_summaries
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
                    _rdkit_ring_closure_marker_slots(
                        skeleton,
                        marker_slots,
                    )
                )
                for skeleton, marker_slots in current_marker_slots_by_skeleton.items()
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
                    _rdkit_ring_closure_marker_slots(skeleton, marker_slots),
                )
                for skeleton, marker_slots in current_marker_slots_by_skeleton.items()
            )

            with self.subTest(case_id=case.case_id, source=case.source):
                self.assertEqual(current_skeletons, rdkit_sampled_skeletons)
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
                    rdkit_marker_sequences_by_skeleton,
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


if __name__ == "__main__":
    unittest.main()
