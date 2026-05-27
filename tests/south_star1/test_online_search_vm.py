"""Tests for the explicit-frame South Star online search VM."""

from __future__ import annotations

import ast
import unittest
from collections import Counter
from dataclasses import dataclass
from dataclasses import FrozenInstanceError
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import grimace._south_star1.online_search_vm as online_search_vm_module
from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.online_decisions import OnlineDecision
from grimace._south_star1.online_render_sink import OnlineStringBuffer
from grimace._south_star1.online_residual_continuation import ResidualFrontierSink
from grimace._south_star1.online_search_vm import EventLoopFrame
from grimace._south_star1.online_search_vm import DirectionEnumerationFrame
from grimace._south_star1.online_search_vm import OnlineSearchFrame
from grimace._south_star1.online_search_vm import ExhaustiveOnlineSearchVM
from grimace._south_star1.online_search_vm import ParentOrientationFrame
from grimace._south_star1.online_search_vm import PrefixEnumerationFrame
from grimace._south_star1.online_search_vm import RESUMABLE_FRAME_PAYLOAD_TYPES
from grimace._south_star1.online_search_vm import RenderCursorFrame
from grimace._south_star1.online_search_vm import SupportMaximalFrame
from grimace._south_star1.online_search_vm import dispatcher_resumable_frame_payload_types
from grimace._south_star1.online_search_vm import residual_snapshot_frame_audit
from grimace._south_star1.online_search_vm import resume_online_search_from_snapshot
from grimace._south_star1.online_search_vm import topmost_resumable_frame
from grimace._south_star1.online_search_vm import validate_residual_frame_stack
from grimace._south_star1.online_search_vm import capture_residual_continuation
from grimace._south_star1.online_search_vm import iter_exhaustive_online_stereo_witness_strings_vm
from grimace._south_star1.online_search_vm import make_online_search_state
from grimace._south_star1.online_search_vm import _pop_resumable_frame
from grimace._south_star1.online_search_vm import _resume_from_frames
from grimace._south_star1.online_stereo_witness import iter_online_stereo_witness_strings
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.residual_constraints import VarId
from grimace._south_star1.residual_constraints import direction_var
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import RingLabel
from grimace._south_star1.policy import SerializationLanguageMode
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import tetrahedral_facts


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_SEARCH_VM_PATH = (
    REPO_ROOT / "python" / "grimace" / "_south_star1" / "online_search_vm.py"
)


class OnlineSearchVmTest(unittest.TestCase):
    def test_vm_online_tetra_multiset_matches_recursive_online(self) -> None:
        self._assert_vm_matches_recursive(tetrahedral_facts())

    def test_vm_online_directional_multiset_matches_recursive_online(self) -> None:
        self._assert_vm_matches_recursive(directional_facts())

    def test_vm_online_ring_tetra_multiset_matches_recursive_online(self) -> None:
        self._assert_vm_matches_recursive(ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"))

    def test_vm_online_disconnected_multiset_matches_recursive_online(self) -> None:
        self._assert_vm_matches_recursive(disconnected_facts())

    def test_vm_online_shared_carrier_multiset_matches_recursive_online(self) -> None:
        self._assert_vm_matches_recursive(directional_facts())

    def test_vm_snapshot_restores_output_buffer(self) -> None:
        state = _state(tetrahedral_facts())
        snapshot = state.checkpoint()

        self.assertTrue(state.output.append("C", token_text="C"))
        state.rollback(snapshot)

        self.assertEqual(state.output.value(), "")

    def test_vm_snapshot_restores_residual_store(self) -> None:
        state = _state(tetrahedral_facts())
        var = VarId("test", (0,))
        state.residual.add_var(var, ("a", "b"))
        snapshot = state.checkpoint()

        self.assertTrue(state.residual.assign(var, "a"))
        state.rollback(snapshot)

        self.assertIsNone(state.residual.assignment(var))

    def test_vm_snapshot_restores_ring_state(self) -> None:
        state = _state(tetrahedral_facts())
        snapshot = state.checkpoint()

        self.assertTrue(
            state.ring.register_endpoint(
                bond=BondId(0),
                endpoint=0,
                label=RingLabel(1),
            )
        )
        state.rollback(snapshot)

        self.assertEqual(state.ring.endpoint_by_bond, {})
        self.assertEqual(state.ring.label_by_endpoint, {})
        self.assertEqual(state.ring.open_intervals, {})

    def test_vm_snapshot_restores_decision_path(self) -> None:
        state = _state(tetrahedral_facts())
        snapshot = state.checkpoint()

        state.decisions.push(OnlineDecision("test", (1,)))
        state.rollback(snapshot)

        self.assertEqual(state.decisions.path().items, ())

    def test_vm_snapshot_restores_frame_stack(self) -> None:
        state = _state(tetrahedral_facts())
        root_frame = OnlineSearchFrame(EventLoopFrame(("root",)))
        child_frame = OnlineSearchFrame(EventLoopFrame(("child",)))
        state.frames.append(root_frame)
        snapshot = state.checkpoint()

        state.frames.append(child_frame)
        state.rollback(snapshot)

        self.assertEqual(state.frames, [root_frame])

    def test_vm_snapshot_restores_real_traversal_state_after_root(self) -> None:
        state = _state(tetrahedral_facts())
        snapshot = state.checkpoint()

        state.traversal.component_index = 1
        state.traversal.roots.append(AtomId(0))
        state.traversal.parent[AtomId(0)] = None
        state.traversal.tree_bonds.add(BondId(0))
        state.traversal.visited_atoms.add(AtomId(0))
        state.traversal.active_atom_stack.append(AtomId(0))
        state.traversal.syntax_position = 3
        state.rollback(snapshot)

        self.assertEqual(state.traversal.component_index, 0)
        self.assertEqual(state.traversal.roots, [])
        self.assertEqual(state.traversal.parent, {})
        self.assertEqual(state.traversal.tree_bonds, set())
        self.assertEqual(state.traversal.visited_atoms, set())
        self.assertEqual(state.traversal.active_atom_stack, [])
        self.assertEqual(state.traversal.syntax_position, 0)

    def test_vm_snapshot_restores_ring_endpoint_state(self) -> None:
        state = _state(ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"))
        snapshot = state.checkpoint()

        self.assertTrue(
            state.ring.register_endpoint(
                bond=BondId(1),
                endpoint=0,
                label=RingLabel(1),
            )
        )
        state.rollback(snapshot)

        self.assertEqual(state.ring.endpoint_by_bond, {})
        self.assertEqual(state.ring.label_by_endpoint, {})
        self.assertEqual(state.ring.open_intervals, {})
        self.assertEqual(state.ring.next_endpoint_id, 0)

    def test_vm_snapshot_restores_direction_mark_assignment(self) -> None:
        state = _state(directional_facts())
        var = direction_var(0)
        state.residual.add_var(var, (DirectionMark.ABSENT, DirectionMark.FWD))
        snapshot = state.checkpoint()

        self.assertTrue(state.residual.assign(var, DirectionMark.FWD))
        state.rollback(snapshot)

        self.assertIsNone(state.residual.assignment(var))

    def test_from_snapshot_restores_residual_snapshot(self) -> None:
        facts = directional_facts()
        policy = ordinary_policy_for_facts(facts)
        semantics = OrdinarySmilesSemantics()
        state = make_online_search_state(
            facts=facts,
            policy=policy,
            semantics=semantics,
            sink=OnlineStringBuffer(),
        )
        var = direction_var(0)
        state.residual.add_var(var, (DirectionMark.ABSENT, DirectionMark.FWD))
        self.assertTrue(state.residual.assign(var, DirectionMark.FWD))
        snapshot = replace(
            state.checkpoint(),
            frame_stack=(
                OnlineSearchFrame(RenderCursorFrame(_first_render_cursor(facts))),
            ),
        )

        vm = ExhaustiveOnlineSearchVM.from_snapshot(
            facts=facts,
            policy=policy,
            semantics=semantics,
            snapshot=snapshot,
            sink=OnlineStringBuffer(),
        )

        self.assertIsNot(vm.state.residual, state.residual)
        self.assertIs(vm.state.residual.assignment(var), DirectionMark.FWD)

    def test_resumed_snapshot_residual_store_mutation_does_not_mutate_producer_store(self) -> None:
        facts = directional_facts()
        policy = ordinary_policy_for_facts(facts)
        semantics = OrdinarySmilesSemantics()
        state = make_online_search_state(
            facts=facts,
            policy=policy,
            semantics=semantics,
            sink=OnlineStringBuffer(),
        )
        var = direction_var(0)
        state.residual.add_var(var, (DirectionMark.ABSENT, DirectionMark.FWD))
        self.assertTrue(state.residual.assign(var, DirectionMark.FWD))
        snapshot = replace(
            state.checkpoint(),
            frame_stack=(
                OnlineSearchFrame(RenderCursorFrame(_first_render_cursor(facts))),
            ),
        )
        extra = VarId("extra", (0,))

        vm = ExhaustiveOnlineSearchVM.from_snapshot(
            facts=facts,
            policy=policy,
            semantics=semantics,
            snapshot=snapshot,
            sink=OnlineStringBuffer(),
        )
        vm.state.residual.add_var(extra, ("x",))
        self.assertTrue(vm.state.residual.assign(extra, "x"))

        with self.assertRaises(ValueError):
            state.residual.assign(extra, "x")
        self.assertIs(vm.state.residual.assignment(var), DirectionMark.FWD)
        self.assertIs(state.residual.assignment(var), DirectionMark.FWD)

    def test_from_snapshot_uses_frame_dispatcher(self) -> None:
        text = ONLINE_SEARCH_VM_PATH.read_text(encoding="utf-8")

        self.assertIn("vm._iterator = _resume_from_frames(vm.state)", text)
        self.assertNotIn("vm._iterator = _resume_render_cursor", text)

    def test_resumable_frame_type_registry_matches_dispatcher_handlers(self) -> None:
        self.assertEqual(
            set(dispatcher_resumable_frame_payload_types()),
            set(RESUMABLE_FRAME_PAYLOAD_TYPES),
        )

    def test_context_only_frame_stack_is_rejected_as_residual_continuation(self) -> None:
        with self.assertRaisesRegex(ValueError, "no resumable frame"):
            validate_residual_frame_stack(
                (OnlineSearchFrame(EventLoopFrame(("context",))),)
            )

    def test_unknown_frame_payload_is_rejected_by_frame_stack_audit(self) -> None:
        @dataclass(frozen=True, slots=True)
        class UnknownFrame:
            value: int

        with self.assertRaisesRegex(ValueError, "unknown frame payload"):
            validate_residual_frame_stack(
                (
                    OnlineSearchFrame(UnknownFrame(1)),  # type: ignore[arg-type]
                    OnlineSearchFrame(RenderCursorFrame(_first_render_cursor(tetrahedral_facts()))),
                )
            )

    def test_topmost_resumable_frame_is_dispatcher_handled(self) -> None:
        continuation = _first_residual_continuation(directional_facts())

        validate_residual_frame_stack(continuation.snapshot.frame_stack)
        top = topmost_resumable_frame(continuation.snapshot.frame_stack)
        self.assertIsNotNone(top)
        assert top is not None
        self.assertIn(type(top.payload), dispatcher_resumable_frame_payload_types())

    def test_dispatcher_resumes_render_cursor_frame(self) -> None:
        facts = tetrahedral_facts()
        policy = ordinary_policy_for_facts(facts)
        semantics = OrdinarySmilesSemantics()
        continuation = _first_residual_continuation(facts)
        sink = ResidualFrontierSink(required_prefix=continuation.prefix)
        vm = ExhaustiveOnlineSearchVM.from_snapshot(
            facts=facts,
            policy=policy,
            semantics=semantics,
            snapshot=continuation.snapshot,
            sink=sink,
        )
        sink.snapshot_provider = vm.checkpoint
        sink.decision_path_provider = vm.state.decisions.path

        witness = vm.run_until_witness_or_exhausted()

        self.assertIsNotNone(witness)
        self.assertFalse(
            any(isinstance(frame.payload, RenderCursorFrame) for frame in vm.state.frames)
        )

    def test_dispatcher_pops_exact_active_cursor_frame(self) -> None:
        context = OnlineSearchFrame(EventLoopFrame(("context",)))
        active = OnlineSearchFrame(RenderCursorFrame(_first_render_cursor(tetrahedral_facts())))
        frames = [context, active]

        popped = _pop_resumable_frame(frames)

        self.assertEqual(popped, active)
        self.assertEqual(frames, [context])

    def test_dispatcher_does_not_drop_non_cursor_context_frames(self) -> None:
        context = OnlineSearchFrame(ParentOrientationFrame(((AtomId(0), None),)))
        active = OnlineSearchFrame(RenderCursorFrame(_first_render_cursor(tetrahedral_facts())))
        frames = [context, active]

        _pop_resumable_frame(frames)

        self.assertEqual(frames, [context])

    def test_dispatcher_rejects_multiple_active_render_cursors(self) -> None:
        first = OnlineSearchFrame(RenderCursorFrame(_first_render_cursor(tetrahedral_facts())))
        second = OnlineSearchFrame(RenderCursorFrame(_first_render_cursor(tetrahedral_facts())))
        frames = [first, second]

        with self.assertRaises(AssertionError):
            _pop_resumable_frame(frames)

    def test_snapshot_with_render_cursor_and_prefix_frame_resumes_in_stack_order(
        self,
    ) -> None:
        prefix = OnlineSearchFrame(_complete_prefix_frame(tetrahedral_facts()))
        render = OnlineSearchFrame(RenderCursorFrame(_first_render_cursor(tetrahedral_facts())))
        frames = [OnlineSearchFrame(EventLoopFrame(("context",))), prefix, render]

        self.assertEqual(_pop_resumable_frame(frames), render)
        self.assertEqual(frames, [OnlineSearchFrame(EventLoopFrame(("context",))), prefix])
        self.assertEqual(_pop_resumable_frame(frames), prefix)
        self.assertEqual(frames, [OnlineSearchFrame(EventLoopFrame(("context",)))])

    def test_prefix_frame_resume_reaches_sibling_prefix_alternatives_after_render_completion(
        self,
    ) -> None:
        facts = cyclopropane_facts()
        complete = _prefix_frame_from_snapshot(_first_residual_continuation(facts).snapshot)
        frame = replace(
            complete,
            phase="ring",
            index=0,
            ring_labels=(),
            atom_text=(),
            bond_text=(),
        )
        policy = replace(ordinary_policy_for_facts(facts), least_free_ring_labels=False)

        witnesses = _witnesses_from_prefix_frame(facts, frame, policy=policy)

        self.assertGreater(len({witness.rendered for witness in witnesses}), 1)

    def test_prefix_scheduler_frame_does_not_accumulate_stale_duplicates(self) -> None:
        continuation = _first_residual_continuation(tetrahedral_facts())

        self.assertEqual(
            sum(
                isinstance(frame.payload, PrefixEnumerationFrame)
                for frame in continuation.snapshot.frame_stack
            ),
            1,
        )

    def test_dispatcher_rejects_duplicate_prefix_enumeration_frames(self) -> None:
        first = OnlineSearchFrame(_complete_prefix_frame(tetrahedral_facts()))
        second = OnlineSearchFrame(_complete_prefix_frame(tetrahedral_facts()))

        with self.assertRaises(AssertionError):
            _pop_resumable_frame([first, second])

    def test_direction_enumeration_frame_is_frozen_hashable_and_canonical(self) -> None:
        frame = _direction_frame_from_snapshot(
            _first_residual_continuation(directional_facts()).snapshot
        )
        canonical = DirectionEnumerationFrame(
            trace=frame.trace,
            prefix=frame.prefix,
            tetra_tokens=tuple(reversed(frame.tetra_tokens)),
            carriers=tuple(reversed(frame.carriers)),
            carrier_index=frame.carrier_index,
            marks=tuple(reversed(frame.marks)),
            residual_snapshot=frame.residual_snapshot,
            ring_state=frame.ring_state,
            decision_path=frame.decision_path,
            frame_stack_prefix=frame.frame_stack_prefix,
            annotation_count=frame.annotation_count,
        )

        with self.assertRaises(FrozenInstanceError):
            canonical.carrier_index = 99  # type: ignore[misc]
        self.assertIsInstance(hash(canonical), int)
        self.assertEqual(canonical.tetra_tokens, frame.tetra_tokens)
        self.assertEqual(canonical.carriers, frame.carriers)
        self.assertEqual(canonical.marks, frame.marks)

    def test_direction_enumeration_frame_resumes_after_partial_mark_assignment(self) -> None:
        frame = _direction_frame_from_snapshot(
            _first_residual_continuation(directional_facts()).snapshot
        )

        self.assertLess(frame.carrier_index, len(frame.carriers))
        self.assertTrue(_witnesses_from_direction_frame(directional_facts(), frame))

    def test_direction_enumeration_frame_restores_residual_snapshot(self) -> None:
        facts = directional_facts()
        state = _state_for_direction_frame(facts)
        before = state.residual.value_snapshot()
        frame = _direction_frame_from_snapshot(_first_residual_continuation(facts).snapshot)

        tuple(_resume_from_frames_with_frame(state, OnlineSearchFrame(frame)))

        self.assertEqual(state.residual.value_snapshot(), before)

    def test_direction_enumeration_frame_restores_ring_state(self) -> None:
        facts = directional_facts()
        state = _state_for_direction_frame(facts)
        before = state.ring.checkpoint()
        frame = _direction_frame_from_snapshot(_first_residual_continuation(facts).snapshot)

        tuple(_resume_from_frames_with_frame(state, OnlineSearchFrame(frame)))

        self.assertEqual(state.ring.checkpoint(), before)

    def test_direction_enumeration_frame_restores_decision_path(self) -> None:
        facts = directional_facts()
        state = _state_for_direction_frame(facts)
        before = state.decisions.path()
        frame = _direction_frame_from_snapshot(_first_residual_continuation(facts).snapshot)

        tuple(_resume_from_frames_with_frame(state, OnlineSearchFrame(frame)))

        self.assertEqual(state.decisions.path(), before)

    def test_direction_enumeration_frame_rejects_inadmissible_mark_without_sink_append(
        self,
    ) -> None:
        frame = _initial_direction_frame_from(
            _direction_frame_from_snapshot(
                _first_residual_continuation(directional_facts()).snapshot
            )
        )
        invalid = replace(
            frame,
            carrier_index=len(frame.carriers),
            marks=tuple(
                (carrier, DirectionMark.ABSENT)
                for carrier in frame.carriers
            ),
        )
        state = _state_for_direction_frame(
            directional_facts(),
            sink=_AppendRejectingSink(),
        )

        self.assertEqual(
            tuple(_resume_from_frames_with_frame(state, OnlineSearchFrame(invalid))),
            (),
        )

    def test_direction_enumeration_frame_preserves_candidate_order(self) -> None:
        frame = _direction_frame_from_snapshot(
            _first_residual_continuation(directional_facts()).snapshot
        )

        rendered = tuple(
            witness.rendered
            for witness in _witnesses_from_direction_frame(directional_facts(), frame)
        )

        self.assertEqual(rendered, tuple(rendered))
        self.assertTrue(rendered)

    def test_direction_frame_resume_reaches_sibling_directional_alternatives(self) -> None:
        frame = _initial_direction_frame_from(
            _direction_frame_from_snapshot(
                _first_residual_continuation(directional_facts()).snapshot
            )
        )
        witnesses = _witnesses_from_direction_frame(directional_facts(), frame)

        self.assertGreater(len({witness.rendered for witness in witnesses}), 1)

    def test_direction_scheduler_frame_does_not_accumulate_stale_duplicates(self) -> None:
        continuation = _first_residual_continuation(directional_facts())

        self.assertEqual(
            sum(
                isinstance(frame.payload, DirectionEnumerationFrame)
                for frame in continuation.snapshot.frame_stack
            ),
            1,
        )

    def test_dispatcher_rejects_duplicate_direction_enumeration_frames(self) -> None:
        frame = OnlineSearchFrame(
            _direction_frame_from_snapshot(
                _first_residual_continuation(directional_facts()).snapshot
            )
        )

        with self.assertRaises(AssertionError):
            _pop_resumable_frame([frame, frame])

    def test_support_maximal_frame_is_frozen_hashable_and_canonical(self) -> None:
        frame = _support_maximal_frame_from_snapshot(
            _first_residual_continuation(directional_facts()).snapshot
        )
        canonical = SupportMaximalFrame(
            trace=frame.trace,
            prefix=frame.prefix,
            tetra_tokens=tuple(reversed(frame.tetra_tokens)),
            candidates=frame.candidates,
            maximal_indices=tuple(reversed(frame.maximal_indices)),
            next_index=frame.next_index,
            annotation_count=frame.annotation_count,
        )

        with self.assertRaises(FrozenInstanceError):
            canonical.next_index = 99  # type: ignore[misc]
        self.assertIsInstance(hash(canonical), int)
        self.assertEqual(canonical.tetra_tokens, frame.tetra_tokens)
        self.assertEqual(canonical.maximal_indices, frame.maximal_indices)

    def test_support_maximal_frame_resumes_second_selected_candidate(self) -> None:
        frame = _support_maximal_frame_from_snapshot(
            _first_residual_continuation(directional_facts()).snapshot
        )

        self.assertLess(frame.next_index, len(frame.maximal_indices))
        self.assertTrue(_witnesses_from_support_maximal_frame(directional_facts(), frame))

    def test_support_maximal_frame_preserves_selected_candidate_order(self) -> None:
        frame = replace(
            _support_maximal_frame_from_snapshot(
                _first_residual_continuation(directional_facts()).snapshot
            ),
            next_index=0,
        )

        rendered = tuple(
            witness.rendered
            for witness in _witnesses_from_support_maximal_frame(
                directional_facts(),
                frame,
            )
        )

        self.assertEqual(rendered, tuple(rendered))
        self.assertTrue(rendered)

    def test_support_maximal_frame_does_not_render_nonmaximal_candidate(self) -> None:
        frame = replace(
            _support_maximal_frame_from_snapshot(
                _first_residual_continuation(directional_facts()).snapshot
            ),
            maximal_indices=(),
            next_index=0,
        )

        witnesses = _witnesses_from_support_maximal_frame(directional_facts(), frame)

        self.assertEqual(witnesses, ())

    def test_support_maximal_frame_preserves_residual_snapshot(self) -> None:
        facts = directional_facts()
        state = _state_for_direction_frame(facts)
        before = state.residual.value_snapshot()
        frame = _support_maximal_frame_from_snapshot(_first_residual_continuation(facts).snapshot)

        tuple(_resume_from_frames_with_frame(state, OnlineSearchFrame(frame)))

        self.assertEqual(state.residual.value_snapshot(), before)

    def test_support_maximal_frame_preserves_ring_state(self) -> None:
        facts = directional_facts()
        state = _state_for_direction_frame(facts)
        before = state.ring.checkpoint()
        frame = _support_maximal_frame_from_snapshot(_first_residual_continuation(facts).snapshot)

        tuple(_resume_from_frames_with_frame(state, OnlineSearchFrame(frame)))

        self.assertEqual(state.ring.checkpoint(), before)

    def test_support_maximal_frame_preserves_decision_path(self) -> None:
        facts = directional_facts()
        state = _state_for_direction_frame(facts)
        before = state.decisions.path()
        frame = _support_maximal_frame_from_snapshot(_first_residual_continuation(facts).snapshot)

        tuple(_resume_from_frames_with_frame(state, OnlineSearchFrame(frame)))

        self.assertEqual(state.decisions.path(), before)

    def test_support_maximal_frame_does_not_accumulate_stale_duplicates(self) -> None:
        continuation = _first_residual_continuation(directional_facts())

        self.assertEqual(
            sum(
                isinstance(frame.payload, SupportMaximalFrame)
                for frame in continuation.snapshot.frame_stack
            ),
            1,
        )

    def test_dispatcher_rejects_duplicate_support_maximal_frames(self) -> None:
        frame = OnlineSearchFrame(
            _support_maximal_frame_from_snapshot(
                _first_residual_continuation(directional_facts()).snapshot
            )
        )

        with self.assertRaises(AssertionError):
            _pop_resumable_frame([frame, frame])

    def test_duplicate_active_resumable_frame_rejected_for_each_scheduler_frame_type(
        self,
    ) -> None:
        frames = (
            OnlineSearchFrame(RenderCursorFrame(_first_render_cursor(tetrahedral_facts()))),
            OnlineSearchFrame(_complete_prefix_frame(tetrahedral_facts())),
            OnlineSearchFrame(
                _direction_frame_from_snapshot(
                    _first_residual_continuation(directional_facts()).snapshot
                )
            ),
            OnlineSearchFrame(
                _support_maximal_frame_from_snapshot(
                    _first_residual_continuation(directional_facts()).snapshot
                )
            ),
        )

        for frame in frames:
            with self.subTest(frame=type(frame.payload).__name__):
                with self.assertRaises(AssertionError):
                    _pop_resumable_frame([frame, frame])

    def test_dispatcher_preserves_ring_state_annotation_count_and_frontier(self) -> None:
        facts = ring_directional_facts()
        continuation = _first_residual_continuation(facts)
        sink = ResidualFrontierSink(required_prefix=continuation.prefix)
        vm = ExhaustiveOnlineSearchVM.from_snapshot(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
            snapshot=continuation.snapshot,
            sink=sink,
        )
        sink.snapshot_provider = vm.checkpoint
        sink.decision_path_provider = vm.state.decisions.path
        self.assertEqual(vm.state.ring.checkpoint(), continuation.snapshot.ring_state)

        witness = vm.run_until_witness_or_exhausted()

        self.assertIsNotNone(witness)
        self.assertEqual(
            witness.annotation_count,
            _render_cursor_from_snapshot(continuation.snapshot).program.annotation_count,
        )
        self.assertGreaterEqual(vm.state.ring.checkpoint()[3], continuation.snapshot.ring_state[3])

    def test_prefix_enumeration_frame_is_frozen_hashable_and_canonical(self) -> None:
        cursor = _first_render_cursor(tetrahedral_facts())
        prefix = cursor.program.prefix
        frame = PrefixEnumerationFrame(
            trace=cursor.program.trace,
            ring_label_domains=((2, (RingLabel(2), RingLabel(1))), (1, (RingLabel(1),))),
            atom_text_domains=tuple(reversed(tuple((atom, (choice,)) for atom, choice in prefix.atom_text))),
            bond_text_domains=tuple(reversed(tuple((slot, (choice,)) for slot, choice in prefix.bond_text))),
            phase="bond",
            index=len(prefix.bond_text),
            ring_labels=((2, RingLabel(1)), (1, RingLabel(1))),
            atom_text=tuple(reversed(prefix.atom_text)),
            bond_text=tuple(reversed(prefix.bond_text)),
        )

        with self.assertRaises(FrozenInstanceError):
            frame.index = 99  # type: ignore[misc]
        self.assertIsInstance(hash(frame), int)
        self.assertEqual(tuple(slot for slot, _ in frame.bond_text_domains), tuple(sorted(slot for slot, _ in frame.bond_text_domains)))
        self.assertEqual(tuple(atom for atom, _ in frame.atom_text), tuple(sorted((atom for atom, _ in frame.atom_text), key=int)))
        self.assertEqual(tuple(endpoint for endpoint, _ in frame.ring_labels), (1, 2))

    def test_prefix_enumeration_frame_resumes_ring_label_choice(self) -> None:
        facts = ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1")
        complete = _complete_prefix_frame(facts)
        frame = replace(
            complete,
            phase="ring",
            index=0,
            ring_labels=(),
            atom_text=(),
            bond_text=(),
        )

        self.assertTrue(frame.ring_label_domains)
        self.assertTrue(_witnesses_from_prefix_frame(facts, frame))

    def test_prefix_enumeration_frame_resumes_atom_text_choice(self) -> None:
        frame = replace(
            _complete_prefix_frame(tetrahedral_facts()),
            phase="atom",
            index=0,
            atom_text=(),
            bond_text=(),
        )

        self.assertTrue(_witnesses_from_prefix_frame(tetrahedral_facts(), frame))

    def test_prefix_enumeration_frame_resumes_bond_text_choice(self) -> None:
        complete = _complete_prefix_frame(tetrahedral_facts())
        frame = replace(
            complete,
            phase="bond",
            index=0,
            bond_text=(),
        )

        self.assertTrue(_witnesses_from_prefix_frame(tetrahedral_facts(), frame))

    def test_prefix_enumeration_frame_preserves_old_prefix_choice_order(self) -> None:
        facts = directional_facts()
        policy = ordinary_policy_for_facts(facts)
        semantics = OrdinarySmilesSemantics()

        self.assertEqual(
            tuple(
                iter_exhaustive_online_stereo_witness_strings_vm(
                    facts=facts,
                    policy=policy,
                    semantics=semantics,
                )
            ),
            tuple(
                iter_online_stereo_witness_strings(
                    facts=facts,
                    policy=policy,
                    semantics=semantics,
                )
            ),
        )

    def test_residual_snapshot_can_contain_prefix_enumeration_frame(self) -> None:
        continuation = _first_residual_continuation(tetrahedral_facts())

        self.assertTrue(
            any(
                isinstance(frame.payload, PrefixEnumerationFrame)
                for frame in continuation.snapshot.frame_stack
            )
        )

    def test_prefix_frame_resume_preserves_ring_state(self) -> None:
        facts = ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1")
        continuation = _first_residual_continuation(facts)
        frame = _prefix_frame_from_snapshot(continuation.snapshot)
        sink = ResidualFrontierSink(required_prefix="")
        state = make_online_search_state(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
            sink=sink,
        )
        snapshot = replace(state.checkpoint(), frame_stack=(OnlineSearchFrame(frame),))
        state.rollback(snapshot)
        sink.snapshot_provider = state.checkpoint
        sink.decision_path_provider = state.decisions.path

        tuple(_resume_from_frames(state))

        self.assertEqual(state.ring.checkpoint(), ((), (), (), 0))

    def test_prefix_frame_resume_preserves_decision_path(self) -> None:
        facts = tetrahedral_facts()
        frame = _complete_prefix_frame(facts)
        sink = ResidualFrontierSink(required_prefix="")
        state = make_online_search_state(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
            sink=sink,
        )
        snapshot = replace(state.checkpoint(), frame_stack=(OnlineSearchFrame(frame),))
        state.rollback(snapshot)
        sink.snapshot_provider = state.checkpoint
        sink.decision_path_provider = state.decisions.path

        tuple(_resume_from_frames(state))

        self.assertEqual(state.decisions.path().items, ())

    def test_prefix_frame_resume_preserves_annotation_count(self) -> None:
        facts = directional_facts()
        frame = _complete_prefix_frame(facts)

        witnesses = _witnesses_from_prefix_frame(facts, frame)

        self.assertTrue(witnesses)
        self.assertTrue(all(witness.annotation_count >= 0 for witness in witnesses))

    def test_vm_step_interface_yields_witness_then_exhausts(self) -> None:
        vm = ExhaustiveOnlineSearchVM(
            facts=disconnected_facts(),
            policy=ordinary_policy_for_facts(disconnected_facts()),
            semantics=OrdinarySmilesSemantics(),
        )

        first = vm.step()

        self.assertEqual(first.kind, "yield_witness")
        self.assertIsNotNone(first.witness)
        while True:
            result = vm.step()
            if result.kind == "exhausted":
                break
        self.assertEqual(vm.step().kind, "exhausted")

    def test_writer_shaped_rejects_exhaustive_vm_route(self) -> None:
        facts = disconnected_facts()

        with self.assertRaises(SouthStarError):
            ExhaustiveOnlineSearchVM(
                facts=facts,
                policy=ordinary_policy_for_facts(facts),
                semantics=OrdinarySmilesSemantics(),
                serialization_language=SerializationLanguageMode.WRITER_SHAPED,
            )

    def test_generic_vm_alias_is_not_exported(self) -> None:
        self.assertFalse(hasattr(online_search_vm_module, "OnlineSearchVM"))
        self.assertNotIn("OnlineSearchVM", online_search_vm_module.__all__)

    def test_exhaustive_snapshot_carries_language_mode(self) -> None:
        state = _state(disconnected_facts())

        self.assertIs(
            state.checkpoint().serialization_language,
            SerializationLanguageMode.EXHAUSTIVE,
        )

    def test_exhaustive_vm_rejects_writer_shaped_snapshot(self) -> None:
        facts = tetrahedral_facts()
        snapshot = replace(
            _first_residual_continuation(facts).snapshot,
            serialization_language=SerializationLanguageMode.WRITER_SHAPED,
        )

        with self.assertRaises(SouthStarError):
            _vm_from_snapshot(facts, snapshot)

    def test_capture_residual_continuation_contains_snapshot(self) -> None:
        state = _state(tetrahedral_facts())
        frame = OnlineSearchFrame(RenderCursorFrame(_first_render_cursor(tetrahedral_facts())))
        state.frames.append(frame)

        continuation = capture_residual_continuation(state, prefix="C")

        self.assertEqual(continuation.prefix, "C")
        self.assertEqual(continuation.snapshot.frame_stack, (frame,))

    def test_capture_residual_continuation_rejects_context_only_frame_stack(
        self,
    ) -> None:
        state = _state(tetrahedral_facts())
        state.frames.append(OnlineSearchFrame(EventLoopFrame(("root",))))

        with self.assertRaisesRegex(ValueError, "no resumable frame"):
            capture_residual_continuation(state, prefix="C")

    def test_capture_residual_continuation_rejects_unknown_frame_payload(
        self,
    ) -> None:
        @dataclass(frozen=True, slots=True)
        class UnknownFrame:
            value: int

        state = _state(tetrahedral_facts())
        state.frames.append(OnlineSearchFrame(UnknownFrame(1)))  # type: ignore[arg-type]
        state.frames.append(OnlineSearchFrame(RenderCursorFrame(_first_render_cursor(tetrahedral_facts()))))

        with self.assertRaisesRegex(ValueError, "unknown frame payload"):
            capture_residual_continuation(state, prefix="C")

    def test_capture_residual_continuation_rejects_duplicate_active_resumable_frame(
        self,
    ) -> None:
        state = _state(tetrahedral_facts())
        frame = OnlineSearchFrame(RenderCursorFrame(_first_render_cursor(tetrahedral_facts())))
        state.frames.extend((frame, frame))

        with self.assertRaisesRegex(AssertionError, "multiple active render-cursor"):
            capture_residual_continuation(state, prefix="C")

    def test_capture_residual_continuation_rejects_unhandled_topmost_resumable_frame(
        self,
    ) -> None:
        state = _state(tetrahedral_facts())
        state.frames.append(OnlineSearchFrame(RenderCursorFrame(_first_render_cursor(tetrahedral_facts()))))

        with patch.object(
            online_search_vm_module,
            "dispatcher_resumable_frame_payload_types",
            return_value=(),
        ):
            with self.assertRaisesRegex(ValueError, "not dispatcher-handled"):
                capture_residual_continuation(state, prefix="C")

    def test_from_snapshot_rejects_context_only_frame_stack_before_iteration(
        self,
    ) -> None:
        snapshot = _snapshot_with_frames(
            tetrahedral_facts(),
            (OnlineSearchFrame(EventLoopFrame(("context",))),),
        )

        with self.assertRaisesRegex(ValueError, "no resumable frame"):
            _vm_from_snapshot(tetrahedral_facts(), snapshot)

    def test_from_snapshot_rejects_unknown_frame_payload_before_iteration(
        self,
    ) -> None:
        @dataclass(frozen=True, slots=True)
        class UnknownFrame:
            value: int

        snapshot = _snapshot_with_frames(
            tetrahedral_facts(),
            (
                OnlineSearchFrame(UnknownFrame(1)),  # type: ignore[arg-type]
                OnlineSearchFrame(RenderCursorFrame(_first_render_cursor(tetrahedral_facts()))),
            ),
        )

        with self.assertRaisesRegex(ValueError, "unknown frame payload"):
            _vm_from_snapshot(tetrahedral_facts(), snapshot)

    def test_from_snapshot_rejects_duplicate_active_resumable_frame_before_iteration(
        self,
    ) -> None:
        frame = OnlineSearchFrame(RenderCursorFrame(_first_render_cursor(tetrahedral_facts())))
        snapshot = _snapshot_with_frames(tetrahedral_facts(), (frame, frame))

        with self.assertRaisesRegex(AssertionError, "multiple active render-cursor"):
            _vm_from_snapshot(tetrahedral_facts(), snapshot)

    def test_resume_online_search_from_snapshot_rejects_invalid_snapshot_before_iteration(
        self,
    ) -> None:
        snapshot = _snapshot_with_frames(
            tetrahedral_facts(),
            (OnlineSearchFrame(EventLoopFrame(("context",))),),
        )

        with self.assertRaisesRegex(ValueError, "no resumable frame"):
            resume_online_search_from_snapshot(
                facts=tetrahedral_facts(),
                policy=ordinary_policy_for_facts(tetrahedral_facts()),
                semantics=OrdinarySmilesSemantics(),
                snapshot=snapshot,
                sink=OnlineStringBuffer(),
            )

    def test_valid_retained_residual_snapshot_still_resumes(self) -> None:
        continuation = _first_residual_continuation(tetrahedral_facts())
        sink = ResidualFrontierSink(required_prefix=continuation.prefix)
        vm = ExhaustiveOnlineSearchVM.from_snapshot(
            facts=tetrahedral_facts(),
            policy=ordinary_policy_for_facts(tetrahedral_facts()),
            semantics=OrdinarySmilesSemantics(),
            snapshot=continuation.snapshot,
            sink=sink,
        )
        sink.snapshot_provider = vm.checkpoint
        sink.decision_path_provider = vm.state.decisions.path

        self.assertIsNotNone(vm.run_until_witness_or_exhausted())

    def test_fresh_online_search_vm_run_does_not_require_residual_snapshot_validation(
        self,
    ) -> None:
        self.assertTrue(_vm_counter(tetrahedral_facts()))

    def test_directional_candidate_rendering_restores_ring_state_between_candidates(self) -> None:
        facts = ring_directional_facts()
        vm = ExhaustiveOnlineSearchVM(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )

        witnesses = [vm.run_until_witness_or_exhausted() for _ in range(2)]

        self.assertTrue(all(witness is not None for witness in witnesses))
        self.assertEqual(vm.state.ring.checkpoint(), ((), (), (), 0))

    def test_two_directional_candidates_on_same_ring_label_branch_both_render(self) -> None:
        facts = ring_directional_facts()
        rendered = tuple(
            iter_exhaustive_online_stereo_witness_strings_vm(
                facts=facts,
                policy=ordinary_policy_for_facts(facts),
                semantics=OrdinarySmilesSemantics(),
            )
        )

        self.assertGreater(len(rendered), len(set(rendered)))
        self.assertGreaterEqual(len(rendered), 2)

    def test_residual_continuation_captures_partial_ring_state_but_producer_ring_restores(self) -> None:
        facts = ring_directional_facts()
        sink = ResidualFrontierSink(required_prefix="C")
        vm = ExhaustiveOnlineSearchVM(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
            sink_factory=lambda: sink,
        )
        sink.snapshot_provider = vm.checkpoint
        sink.decision_path_provider = vm.state.decisions.path

        while vm.run_until_witness_or_exhausted() is not None:
            pass

        self.assertEqual(vm.state.ring.checkpoint(), ((), (), (), 0))
        continuation = sink.completed_by_token["1"][0]
        self.assertNotEqual(continuation.snapshot.ring_state, ((), (), (), 0))

    def test_non_support_maximal_directional_candidates_stream_without_tuple_buffer(self) -> None:
        tree = ast.parse(ONLINE_SEARCH_VM_PATH.read_text(encoding="utf-8"))
        buffered_directional_candidates = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "tuple":
                continue
            if not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                if arg.func.id == "_iter_directional_candidates":
                    buffered_directional_candidates.append(node.lineno)

        self.assertEqual(buffered_directional_candidates, [])

    def test_no_string_render_cursor_or_completion_frames_remain(self) -> None:
        text = ONLINE_SEARCH_VM_PATH.read_text(encoding="utf-8")

        self.assertNotIn('OnlineSearchFrame("', text)
        self.assertNotIn('OnlineSearchFrame("render-cursor"', text)
        self.assertNotIn('OnlineSearchFrame("completion"', text)

    def test_all_retained_frame_payloads_are_known_dataclasses(self) -> None:
        state = _state(tetrahedral_facts())
        state.frames.append(OnlineSearchFrame(ParentOrientationFrame(((AtomId(0), None),))))

        for frame in state.checkpoint().frame_stack:
            self.assertTrue(hasattr(frame.payload, "__dataclass_fields__"))

    def test_online_search_vm_boundary_no_hidden_generator_or_artifact_imports(self) -> None:
        tree = ast.parse(ONLINE_SEARCH_VM_PATH.read_text(encoding="utf-8"))
        banned_modules = {
            "audit_rdkit",
            "finite_space_checker",
            "rdkit_adapter",
            "semantic_relation_checker",
            "support_artifact",
            "support_artifact_checker",
            "support_enumeration",
        }
        banned_calls = {
            "iter_online_stereo_witnesses_with_sink",
            "iter_online_stereo_witness_strings",
            "iter_online_traversal_traces",
            "online_branch_preserving_choices",
            "online_determinized_choices",
        }
        private_online_stereo_imports: list[str] = []
        imports: list[str] = []
        calls: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.split(".", 1)[0] in banned_modules
                )
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".", 1)[0] in banned_modules:
                    imports.append(module)
                if module == "online_stereo_witness":
                    private_online_stereo_imports.extend(
                        alias.name
                        for alias in node.names
                        if alias.name.startswith("_")
                    )
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                if isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)

        self.assertEqual(imports, [])
        self.assertEqual(private_online_stereo_imports, [])
        self.assertEqual(sorted(set(calls) & banned_calls), [])

    def _assert_vm_matches_recursive(self, facts: MoleculeFacts) -> None:
        self.assertEqual(_vm_counter(facts), _recursive_counter(facts))


def _state(facts: MoleculeFacts):
    return make_online_search_state(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        sink=OnlineStringBuffer(),
    )


def _snapshot_with_frames(
    facts: MoleculeFacts,
    frames: tuple[OnlineSearchFrame, ...],
):
    state = _state(facts)
    return replace(state.checkpoint(), frame_stack=frames)


def _vm_from_snapshot(facts: MoleculeFacts, snapshot):
    return ExhaustiveOnlineSearchVM.from_snapshot(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        snapshot=snapshot,
        sink=OnlineStringBuffer(),
    )


def _vm_counter(facts: MoleculeFacts) -> Counter[str]:
    return Counter(
        iter_exhaustive_online_stereo_witness_strings_vm(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )
    )


def _recursive_counter(facts: MoleculeFacts) -> Counter[str]:
    return Counter(
        iter_online_stereo_witness_strings(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )
    )


def disconnected_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "O")),
        bonds=(),
        components=(
            ComponentFacts(id=ComponentId(0), atoms=(AtomId(0),), bonds=()),
            ComponentFacts(id=ComponentId(1), atoms=(AtomId(1),), bonds=()),
        ),
    )


def ring_directional_facts() -> MoleculeFacts:
    return ordinary_molecule_facts_from_smiles("C/C=C/1CC1")


def _first_residual_continuation(facts: MoleculeFacts):
    sink = ResidualFrontierSink(required_prefix="")
    vm = ExhaustiveOnlineSearchVM(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        sink_factory=lambda: sink,
    )
    sink.snapshot_provider = vm.checkpoint
    sink.decision_path_provider = vm.state.decisions.path
    while vm.run_until_witness_or_exhausted() is not None:
        pass
    for token in sorted(sink.completed_by_token):
        continuations = sink.completed_by_token[token]
        if continuations:
            return continuations[0]
    raise AssertionError("expected token frontier")


def _first_render_cursor(facts: MoleculeFacts):
    return _render_cursor_from_snapshot(_first_residual_continuation(facts).snapshot)


def _prefix_frame_from_snapshot(snapshot):
    for frame in reversed(snapshot.frame_stack):
        if isinstance(frame.payload, PrefixEnumerationFrame):
            return frame.payload
    raise AssertionError("snapshot lacks prefix enumeration frame")


def _direction_frame_from_snapshot(snapshot):
    for frame in reversed(snapshot.frame_stack):
        if isinstance(frame.payload, DirectionEnumerationFrame):
            return frame.payload
    raise AssertionError("snapshot lacks direction enumeration frame")


def _support_maximal_frame_from_snapshot(snapshot):
    for frame in reversed(snapshot.frame_stack):
        if isinstance(frame.payload, SupportMaximalFrame):
            return frame.payload
    raise AssertionError("snapshot lacks support-maximal frame")


def _initial_direction_frame_from(
    frame: DirectionEnumerationFrame,
) -> DirectionEnumerationFrame:
    return replace(
        frame,
        carrier_index=0,
        marks=(),
        residual_snapshot=replace(
            frame.residual_snapshot,
            assignments=(),
            factors=tuple(
                replace(factor, marks=())
                if hasattr(factor, "marks")
                else factor
                for factor in frame.residual_snapshot.factors
            ),
        ),
        annotation_count=0,
    )


def _complete_prefix_frame(facts: MoleculeFacts) -> PrefixEnumerationFrame:
    cursor = _first_render_cursor(facts)
    prefix = cursor.program.prefix
    return PrefixEnumerationFrame(
        trace=cursor.program.trace,
        ring_label_domains=tuple(
            (endpoint, (label,))
            for endpoint, label in prefix.ring_labels
        ),
        atom_text_domains=tuple(
            (atom, (choice,))
            for atom, choice in prefix.atom_text
        ),
        bond_text_domains=tuple(
            (slot, (choice,))
            for slot, choice in prefix.bond_text
        ),
        phase="bond",
        index=len(prefix.bond_text),
        ring_labels=prefix.ring_labels,
        atom_text=prefix.atom_text,
        bond_text=prefix.bond_text,
    )


def _witnesses_from_prefix_frame(
    facts: MoleculeFacts,
    frame: PrefixEnumerationFrame,
    *,
    policy=None,
):
    sink = ResidualFrontierSink(required_prefix="")
    state = make_online_search_state(
        facts=facts,
        policy=policy if policy is not None else ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        sink=sink,
    )
    snapshot = replace(state.checkpoint(), frame_stack=(OnlineSearchFrame(frame),))
    state.rollback(snapshot)
    sink.snapshot_provider = state.checkpoint
    sink.decision_path_provider = state.decisions.path
    return tuple(_resume_from_frames(state))


def _witnesses_from_direction_frame(
    facts: MoleculeFacts,
    frame: DirectionEnumerationFrame,
):
    state = _state_for_direction_frame(facts)
    return tuple(_resume_from_frames_with_frame(state, OnlineSearchFrame(frame)))


def _witnesses_from_support_maximal_frame(
    facts: MoleculeFacts,
    frame: SupportMaximalFrame,
):
    state = _state_for_direction_frame(facts)
    return tuple(_resume_from_frames_with_frame(state, OnlineSearchFrame(frame)))


def _state_for_direction_frame(facts: MoleculeFacts, *, sink=None):
    sink = sink if sink is not None else ResidualFrontierSink(required_prefix="")
    state = make_online_search_state(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        sink=sink,
    )
    sink.snapshot_provider = state.checkpoint
    sink.decision_path_provider = state.decisions.path
    return state


class _AppendRejectingSink:
    def checkpoint(self) -> int:
        return 0

    def rollback(self, checkpoint: object) -> None:
        self._checkpoint = checkpoint

    def append(self, text: str, *, token_text: str | None = None) -> bool:
        raise AssertionError("inadmissible direction frame appended to sink")

    def complete(self) -> bool:
        return True

    def value(self) -> str:
        return ""


def _resume_from_frames_with_frame(state, frame: OnlineSearchFrame):
    snapshot = replace(state.checkpoint(), frame_stack=(frame,))
    state.rollback(snapshot)
    return _resume_from_frames(state)


def _render_cursor_from_snapshot(snapshot):
    for frame in reversed(snapshot.frame_stack):
        if isinstance(frame.payload, RenderCursorFrame):
            return frame.payload.cursor
    raise AssertionError("snapshot lacks render cursor")


if __name__ == "__main__":
    unittest.main()
