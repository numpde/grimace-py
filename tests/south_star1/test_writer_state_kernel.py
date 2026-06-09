"""Tests for the writer-shaped state/frontier MVP."""

from __future__ import annotations

import ast
import contextlib
import inspect
import unittest
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import grimace._south_star1.writer_frontier as writer_frontier_module
import grimace._south_star1.writer_state as writer_state_module
import grimace._south_star1.writer_transitions as writer_transitions
from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.policy import AtomTextChoice
from grimace._south_star1.policy import AtomTextDomain
from grimace._south_star1.policy import RingLabel
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.policy import SmilesPolicy
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import enumerate_prepared_stereo_support
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_frontier import WriterFrontierCursor
from grimace._south_star1.writer_frontier import count_writer_cursor_completions
from grimace._south_star1.writer_frontier import count_writer_frontier_support
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_frontier import initial_writer_transition_frontier_cursor
from grimace._south_star1.writer_frontier import iter_writer_frontier_support
from grimace._south_star1.writer_frontier import writer_frontier_choices
from grimace._south_star1.writer_graph_obligations import WriterBoundaryOwnerKind
from grimace._south_star1.writer_graph_obligations import WriterEdgeObligationKind
from grimace._south_star1.writer_graph_obligations import WriterResidualAttachmentActionKind
from grimace._south_star1.writer_state import ComponentCursor
from grimace._south_star1.writer_state import ObligationState
from grimace._south_star1.writer_state import PendingWriterEntry
from grimace._south_star1.writer_state import WriterAtomFrame
from grimace._south_star1.writer_state import WriterClosedClosure
from grimace._south_star1.writer_state import WriterClosureLabel
from grimace._south_star1.writer_state import WriterOpenClosureEndpoint
from grimace._south_star1.writer_state import WriterPolicyState
from grimace._south_star1.writer_state import WriterRingLabelState
from grimace._south_star1.writer_state import WriterRingState
from grimace._south_star1.writer_state import WriterState
from grimace._south_star1.writer_state import WriterStateKey
from grimace._south_star1.writer_state import writer_state_from_key
from grimace._south_star1.writer_state import writer_state_key
from grimace._south_star1.writer_state import writer_state_key_sort_tuple
from grimace._south_star1.writer_stereo import empty_writer_stereo_state
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import bond
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


REPO_ROOT = Path(__file__).resolve().parents[2]
SOUTH_STAR1_ROOT = REPO_ROOT / "python" / "grimace" / "_south_star1"


class WriterStateKernelTest(unittest.TestCase):
    def _closure_policy_for_outcome(
        self,
        active_atom: AtomId,
    ) -> tuple[
        writer_transitions._WriterActiveEmittedGraphPolicyDecision,
        writer_transitions._WriterActiveEmittedScheduleDecision,
    ]:
        action = writer_transitions._finish_active_action(active_atom)
        transition = SimpleNamespace(emitted_text="")
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        empty_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=batch,
            open_batch=empty_batch,
            surviving_emissions=(emission,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
        )
        active_decision = writer_transitions._active_emitted_closure_decision(
            closure_decision,
            graph_policy_decision=policy,
        )

        return policy, active_decision

    def _blocked_child_active_emitted_outcome(
        self,
        active_atom: AtomId,
    ) -> writer_transitions._WriterActiveEmittedScheduleOutcome:
        child_blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(child_blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        empty_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=empty_batch,
            open_batch=empty_batch,
            surviving_emissions=(),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        return writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.BLOCKED,
            graph_policy_decision=policy,
        )

    def test_writer_shaped_acyclic_support_uses_writer_frontier(self) -> None:
        prepared = _prepare(cco_facts())

        with _forbidden_exhaustive_routes():
            support = enumerate_prepared_stereo_support(
                prepared=prepared,
                runtime_options=_writer_options(),
            )

        self.assertEqual(
            support.strings,
            ("C(C)O", "C(O)C", "CCO", "OCC"),
        )
        self.assertEqual(support.distinct_count, 4)
        self.assertEqual(support.witness_count, 4)

    def test_writer_frontier_groups_same_emitted_text(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        choices = writer_frontier_choices(prepared, cursor)

        self.assertIsNone(choices.terminal)
        self.assertEqual(tuple(choice.emitted_text for choice in choices.choices), ("C",))
        self.assertEqual(choices.choices[0].immediate_multiplicity, 2)
        self.assertEqual(choices.choices[0].support_count, 1)
        self.assertEqual(choices.choices[0].completion_count, 2)
        self.assertEqual(len(choices.choices[0].successor.support_state.states), 2)
        self.assertEqual(
            sum(weight for _, weight in choices.choices[0].successor.weighted_states),
            2,
        )

    def test_writer_frontier_choices_use_next_token_frontier_not_flattened_transitions(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        with patch.object(
            writer_frontier_module,
            "legal_writer_transitions",
            side_effect=AssertionError("writer_frontier used flattened transitions"),
            create=True,
        ), patch(
            "grimace._south_star1.writer_transitions.legal_writer_transitions",
            side_effect=AssertionError(
                "writer_frontier used public flattened transitions"
            ),
        ), patch(
            "grimace._south_star1.writer_frontier._legal_writer_next_token_frontier",
            wraps=writer_frontier_module._legal_writer_next_token_frontier,
        ) as frontier:
            choices = writer_frontier_choices(prepared, cursor)

        self.assertEqual(tuple(choice.emitted_text for choice in choices.choices), ("C",))
        self.assertGreater(frontier.call_count, 0)

    def test_writer_frontier_counts_duplicate_token_paths_to_same_state(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            chain_facts(("C",)),
            writer_surface=SouthStarWriterSurface(),
            policy=duplicate_single_atom_policy(),
        )
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        choices = writer_frontier_choices(prepared, cursor)

        self.assertEqual(tuple(choice.emitted_text for choice in choices.choices), ("C",))
        choice = choices.choices[0]
        self.assertEqual(choice.immediate_multiplicity, 2)
        self.assertEqual(len(choice.successor.support_state.states), 1)
        self.assertEqual(choice.successor.weighted_states[0][1], 2)
        self.assertEqual(choice.support_count, 1)
        self.assertEqual(choice.completion_count, 2)

    def test_writer_frontier_terminal_counts_weighted_cursor(self) -> None:
        prepared = _prepare(chain_facts(("C",)))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        after_atom = writer_frontier_choices(prepared, cursor).choices[0].successor
        terminal_key = after_atom.weighted_states[0][0]
        weighted_terminal = WriterFrontierCursor(
            weighted_states=((terminal_key, 3),)
        )

        choices = writer_frontier_choices(prepared, weighted_terminal)

        self.assertIsNotNone(choices.terminal)
        assert choices.terminal is not None
        self.assertEqual(choices.terminal.support_count, 1)
        self.assertEqual(choices.terminal.completion_count, 3)
        self.assertEqual(choices.terminal.multiplicity, 3)
        self.assertEqual(
            sum(weight for _, weight in choices.terminal.finalized_cursor.weighted_states),
            3,
        )
        self.assertEqual(choices.choices, ())

    def test_writer_support_image_keeps_witness_count_separate(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))

        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        self.assertEqual(support.strings, ("CC",))
        self.assertEqual(support.distinct_count, 1)
        self.assertEqual(support.witness_count, 2)

    def test_writer_witness_completions_can_exceed_support_count(self) -> None:
        prepared = _prepare(chain_facts(("C", "C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        self.assertEqual(count_writer_frontier_support(prepared, cursor.support_state), 2)
        self.assertEqual(count_writer_cursor_completions(prepared, cursor), 4)

    def test_writer_support_count_does_not_call_streaming_support(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        with patch(
            "grimace._south_star1.writer_frontier.iter_writer_frontier_support",
            side_effect=AssertionError("count-only path streamed support strings"),
        ):
            self.assertEqual(count_writer_frontier_support(prepared, cursor.support_state), 4)

    def test_streaming_support_does_not_compute_counted_choices(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())

        with patch(
            "grimace._south_star1.writer_frontier.writer_frontier_choices",
            side_effect=AssertionError("streaming used counted choices"),
        ), patch(
            "grimace._south_star1.writer_frontier.count_writer_frontier_support",
            side_effect=AssertionError("streaming computed support count"),
        ), patch(
            "grimace._south_star1.writer_frontier.count_writer_cursor_completions",
            side_effect=AssertionError("streaming computed completion count"),
        ):
            self.assertEqual(
                tuple(iter_writer_frontier_support(prepared, cursor)),
                ("C(C)O", "C(O)C", "CCO", "OCC"),
            )

    def test_unique_child_is_inline_for_rooted_chain(self) -> None:
        prepared = _prepare(chain_facts(("C", "C", "C")))

        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(rooted_at_atom=0),
        )

        self.assertEqual(support.strings, ("CCC",))
        self.assertNotIn("(", support.strings[0])

    def test_true_side_branches_remain_expressible(self) -> None:
        prepared = _prepare(cco_facts())

        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(rooted_at_atom=1),
        )

        self.assertEqual(support.strings, ("C(C)O", "C(O)C"))

    def test_double_bond_child_entry_is_token_granular(self) -> None:
        prepared = _prepare(two_atom_facts("C", "O", BondOrder.DOUBLE))
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )

        first = writer_frontier_choices(prepared, cursor).choices[0]
        second = writer_frontier_choices(prepared, first.successor).choices[0]
        third = writer_frontier_choices(prepared, second.successor).choices[0]
        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(rooted_at_atom=0),
        )

        self.assertEqual(first.emitted_text, "C")
        self.assertEqual(second.emitted_text, "=")
        self.assertEqual(third.emitted_text, "O")
        self.assertEqual(support.strings, ("C=O",))

    def test_triple_bond_child_entry_is_token_granular(self) -> None:
        prepared = _prepare(two_atom_facts("C", "C", BondOrder.TRIPLE))
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )

        first = writer_frontier_choices(prepared, cursor).choices[0]
        second = writer_frontier_choices(prepared, first.successor).choices[0]
        third = writer_frontier_choices(prepared, second.successor).choices[0]

        self.assertEqual(first.emitted_text, "C")
        self.assertEqual(second.emitted_text, "#")
        self.assertEqual(third.emitted_text, "C")

    def test_writer_shaped_disconnected_components_emit_dot(self) -> None:
        prepared = _prepare(disconnected_co_facts())

        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(),
        )

        self.assertEqual(support.strings, ("C.O",))

    def test_component_boundary_emits_for_graph_complete_component(self) -> None:
        prepared = _prepare(cyclopropane_plus_singleton_facts())
        state = _with_next_component_root(_cyclopropane_terminal_closed_closure_state())
        context = writer_transitions.build_writer_transition_expansion_context(
            prepared,
            state,
        )

        transitions = writer_transitions._finish_active_transitions(
            prepared,
            state,
            context,
        )

        self.assertEqual(
            tuple(transition.kind for transition in transitions),
            (writer_transitions.WriterTransitionKind.DOT,),
        )
        self.assertEqual(transitions[0].emitted_text, ".")

    def test_component_boundary_rejects_open_closure_endpoint(self) -> None:
        prepared = _prepare(cyclopropane_plus_singleton_facts())
        state = _with_next_component_root(_cyclopropane_terminal_open_closure_state())
        context = writer_transitions.build_writer_transition_expansion_context(
            prepared,
            state,
        )

        transitions = writer_transitions._finish_active_transitions(
            prepared,
            state,
            context,
        )

        self.assertEqual(transitions, ())

    def test_component_boundary_rejects_closure_candidate(self) -> None:
        prepared = _prepare(cyclopropane_plus_singleton_facts())
        state = _with_next_component_root(
            replace(
                _cyclopropane_terminal_closed_closure_state(),
                ring_state=WriterRingState(),
            )
        )
        context = writer_transitions.build_writer_transition_expansion_context(
            prepared,
            state,
        )

        transitions = writer_transitions._finish_active_transitions(
            prepared,
            state,
            context,
        )

        self.assertEqual(transitions, ())

    def test_writer_cursor_after_cc_exposes_weighted_terminal(self) -> None:
        prepared = _prepare(chain_facts(("C", "C")))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        first = writer_frontier_choices(prepared, cursor).choices[0]
        second = writer_frontier_choices(prepared, first.successor).choices[0]

        choices = writer_frontier_choices(prepared, second.successor)

        self.assertIsNotNone(choices.terminal)
        assert choices.terminal is not None
        self.assertEqual(choices.terminal.support_count, 1)
        self.assertEqual(choices.terminal.completion_count, 2)
        self.assertEqual(choices.terminal.multiplicity, 2)
        self.assertEqual(
            sum(weight for _, weight in choices.terminal.finalized_cursor.weighted_states),
            2,
        )
        self.assertEqual(choices.choices, ())

    def test_writer_root_restricts_initial_frontier_without_plan_route(self) -> None:
        prepared = _prepare(cco_facts())

        with _forbidden_exhaustive_routes():
            support = enumerate_prepared_stereo_support(
                prepared=prepared,
                runtime_options=_writer_options(rooted_at_atom=2),
            )

        self.assertEqual(support.strings, ("OCC",))

    def test_writer_shaped_cyclic_fails_closed_before_forbidden_routes(self) -> None:
        prepared = _prepare(cyclopropane_facts())

        with _forbidden_exhaustive_routes():
            with self.assertRaises(SouthStarError) as caught:
                enumerate_prepared_stereo_support(
                    prepared=prepared,
                    runtime_options=_writer_options(),
                )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_public_initial_frontier_still_rejects_cyclic_prepared(self) -> None:
        prepared = _prepare(cyclopropane_facts())

        with self.assertRaises(SouthStarError) as caught:
            initial_writer_frontier_cursor(prepared, _writer_options(rooted_at_atom=0))

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_internal_transition_frontier_accepts_cyclic_prepared(self) -> None:
        prepared = _prepare(cyclopropane_facts())

        cursor = initial_writer_transition_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )

        self.assertEqual(len(cursor.weighted_states), 1)

    def test_internal_transition_frontier_rejects_malformed_components(self) -> None:
        prepared = _prepare(cycle_plus_isolate_component_facts())

        with self.assertRaises(SouthStarError) as caught:
            initial_writer_transition_frontier_cursor(
                prepared,
                _writer_options(rooted_at_atom=0),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_internal_transition_frontier_rejects_unsupported_stereo_surface(self) -> None:
        prepared = _prepare(unsupported_directional_implicit_h_facts())

        with self.assertRaises(SouthStarError) as caught:
            initial_writer_transition_frontier_cursor(
                prepared,
                _writer_options(rooted_at_atom=0),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

    def test_raw_legal_transitions_reject_same_unsupported_stereo_surface(self) -> None:
        prepared = _prepare(unsupported_directional_implicit_h_facts())

        with self.assertRaises(SouthStarError) as caught:
            writer_transitions.legal_writer_transitions(
                prepared,
                _raw_initial_state(AtomId(0)),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

    def test_writer_shaped_cycle_plus_isolate_component_fails_closed(self) -> None:
        prepared = _prepare(cycle_plus_isolate_component_facts())

        with _forbidden_exhaustive_routes():
            with self.assertRaises(SouthStarError) as caught:
                enumerate_prepared_stereo_support(
                    prepared=prepared,
                    runtime_options=_writer_options(),
                )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_raw_legal_transitions_allow_cyclic_root_emission(self) -> None:
        prepared = _prepare(cyclopropane_facts())

        transitions = writer_transitions.legal_writer_transitions(
            prepared,
            _raw_initial_state(AtomId(0)),
        )

        self.assertEqual(tuple(transition.emitted_text for transition in transitions), ("C",))
        self.assertEqual(
            tuple(transition.kind for transition in transitions),
            (writer_transitions.WriterTransitionKind.ATOM,),
        )

    def test_raw_legal_transitions_reject_missing_active_frame(self) -> None:
        prepared = _prepare(chain_facts(("C",)))

        with self.assertRaises(SouthStarError) as caught:
            writer_transitions.legal_writer_transitions(
                prepared,
                replace(_raw_initial_state(AtomId(0)), active=None),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_raw_initial_state_still_emits_atom_transition(self) -> None:
        prepared = _prepare(chain_facts(("C",)))

        transitions = writer_transitions.legal_writer_transitions(
            prepared,
            _raw_initial_state(AtomId(0)),
        )

        self.assertEqual(tuple(transition.emitted_text for transition in transitions), ("C",))

    def test_raw_closure_endpoint_transition_opens_ring_label(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        root = writer_transitions.legal_writer_transitions(
            prepared,
            _raw_initial_state(AtomId(0)),
        )[0].successor

        transitions = writer_transitions.legal_writer_transitions(prepared, root)

        self.assertTrue(transitions)
        self.assertEqual(
            {transition.kind for transition in transitions},
            {writer_transitions.WriterTransitionKind.OPEN_CLOSURE_ENDPOINT},
        )
        self.assertEqual({transition.emitted_text for transition in transitions}, {"1"})
        opened = transitions[0].successor
        self.assertEqual(len(opened.ring_state.open_endpoints), 1)
        endpoint = opened.ring_state.open_endpoints[0]
        self.assertEqual(endpoint.label, WriterClosureLabel(value=1, text="1"))
        self.assertEqual(endpoint.first_endpoint_text, "1")
        self.assertEqual(endpoint.first_endpoint_bond_text, "")
        self.assertTrue(
            any(
                factor.kind == "ring_pair" and not factor.closed
                for factor in opened.stereo_state.delayed_factors
            )
        )

    def test_raw_closure_endpoint_transition_pairs_ring_label(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        root = writer_transitions.legal_writer_transitions(
            prepared,
            _raw_initial_state(AtomId(0)),
        )[0].successor
        opened = next(
            transition.successor
            for transition in writer_transitions.legal_writer_transitions(prepared, root)
            if transition.successor.ring_state.open_endpoints[0].second_atom == AtomId(2)
        )
        after_first_child = writer_transitions.legal_writer_transitions(
            prepared,
            opened,
        )[0].successor
        at_partner = writer_transitions.legal_writer_transitions(
            prepared,
            after_first_child,
        )[0].successor

        transitions = writer_transitions.legal_writer_transitions(prepared, at_partner)

        self.assertEqual(
            tuple(transition.kind for transition in transitions),
            (writer_transitions.WriterTransitionKind.PAIR_CLOSURE_ENDPOINT,),
        )
        self.assertEqual(tuple(transition.emitted_text for transition in transitions), ("1",))
        closed = transitions[0].successor
        self.assertEqual(closed.ring_state.open_endpoints, ())
        self.assertEqual(len(closed.ring_state.closed_closures), 1)
        self.assertEqual(
            closed.ring_state.closed_closures[0].label,
            WriterClosureLabel(value=1, text="1"),
        )
        self.assertEqual(
            closed.ring_state.label_state.reusable,
            (WriterClosureLabel(value=1, text="1"),),
        )
        self.assertTrue(
            any(
                factor.kind == "ring_pair" and factor.closed
                for factor in closed.stereo_state.delayed_factors
            )
        )

    def test_internal_transition_frontier_steps_cyclic_closure_lifecycle(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = initial_writer_transition_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )

        initial = writer_frontier_choices(prepared, cursor)
        self.assertEqual(tuple(choice.emitted_text for choice in initial.choices), ("C",))
        after_root = initial.choices[0].successor

        root_choices = writer_frontier_choices(prepared, after_root)
        self.assertEqual(
            tuple(choice.emitted_text for choice in root_choices.choices),
            ("1",),
        )
        opened = root_choices.choices[0].successor
        self.assertTrue(
            all(
                key.ring_state.open_endpoints
                for key, _ in opened.weighted_states
            )
        )

        after_first_child = _only_choice(prepared, opened, "C").successor
        after_second_child = _only_choice(prepared, after_first_child, "C").successor
        pair_choice = _only_choice(prepared, after_second_child, "1")
        closed = pair_choice.successor

        self.assertTrue(
            all(
                not key.ring_state.open_endpoints
                and len(key.ring_state.closed_closures) == 1
                for key, _ in closed.weighted_states
            )
        )

    def test_internal_cyclic_frontier_counts_and_streams_finitely(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = initial_writer_transition_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )

        support_count = count_writer_frontier_support(prepared, cursor.support_state)
        completion_count = count_writer_cursor_completions(prepared, cursor)
        strings = tuple(iter_writer_frontier_support(prepared, cursor))

        self.assertEqual(support_count, len(set(strings)))
        self.assertEqual(len(strings), len(set(strings)))
        self.assertGreater(support_count, 0)
        self.assertGreaterEqual(completion_count, support_count)
        self.assertTrue(all("1" in string for string in strings))

    def test_internal_cyclic_frontier_terminal_paths_close_closures(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        cursor = initial_writer_transition_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )

        terminal_keys = _terminal_keys(prepared, cursor)

        self.assertTrue(terminal_keys)
        self.assertTrue(
            all(
                not key.ring_state.open_endpoints
                and key.ring_state.closed_closures
                for key in terminal_keys
            )
        )

    def test_raw_closure_label_allocator_uses_least_free_not_reusable_first(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=True,
        )
        reusable = WriterClosureLabel(value=2, text="2")

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(
                label_state=WriterRingLabelState(reusable=(reusable,)),
            ),
        )

        self.assertEqual(labels, (WriterClosureLabel(value=1, text="1"),))

    def test_raw_closure_label_allocator_uses_reusable_when_smaller_label_is_active(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=True,
        )

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(
                label_state=WriterRingLabelState(
                    allocated=(WriterClosureLabel(value=1, text="1"),),
                ),
            ),
        )

        self.assertEqual(labels, (WriterClosureLabel(value=2, text="2"),))

    def test_raw_closure_label_allocator_least_free_uses_label_value_not_policy_order(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=True,
            ring_labels=(RingLabel(2), RingLabel(1)),
        )

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(),
        )

        self.assertEqual(labels, (WriterClosureLabel(value=1, text="1"),))

    def test_raw_closure_label_allocator_enumerates_labels_without_least_free(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=False,
        )

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(),
        )

        self.assertEqual(
            labels,
            (
                WriterClosureLabel(value=1, text="1"),
                WriterClosureLabel(value=2, text="2"),
            ),
        )

    def test_raw_closure_label_allocator_nonleast_free_preserves_policy_order(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=False,
            ring_labels=(RingLabel(2), RingLabel(1)),
        )

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(),
        )

        self.assertEqual(
            labels,
            (
                WriterClosureLabel(value=2, text="2"),
                WriterClosureLabel(value=1, text="1"),
            ),
        )

    def test_raw_closure_label_allocator_enumerates_all_free_labels_without_least_free(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=False,
        )

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(
                label_state=WriterRingLabelState(
                    reusable=(WriterClosureLabel(value=2, text="2"),),
                ),
            ),
        )

        self.assertEqual(
            labels,
            (
                WriterClosureLabel(value=1, text="1"),
                WriterClosureLabel(value=2, text="2"),
            ),
        )

    def test_raw_closure_label_allocator_excludes_active_labels_without_least_free(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=False,
        )

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(
                label_state=WriterRingLabelState(
                    allocated=(WriterClosureLabel(value=1, text="1"),),
                    reusable=(WriterClosureLabel(value=2, text="2"),),
                ),
            ),
        )

        self.assertEqual(labels, (WriterClosureLabel(value=2, text="2"),))

    def test_raw_closure_open_transitions_enumerate_labels_without_least_free(self) -> None:
        prepared = _prepare_with_policy(
            cyclopropane_facts(),
            least_free_ring_labels=False,
        )
        root = writer_transitions.legal_writer_transitions(
            prepared,
            _raw_initial_state(AtomId(0)),
        )[0].successor

        transitions = writer_transitions.legal_writer_transitions(prepared, root)

        self.assertEqual(
            {transition.emitted_text for transition in transitions},
            {"1", "2"},
        )
        self.assertEqual(
            {transition.successor.ring_state.open_endpoints[0].label for transition in transitions},
            {
                WriterClosureLabel(value=1, text="1"),
                WriterClosureLabel(value=2, text="2"),
            },
        )

    def test_raw_closure_label_allocator_returns_none_when_exhausted(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        labels = tuple(
            WriterClosureLabel(value=label.value, text=label.text())
            for label in prepared.policy.ring_labels
        )

        labels = writer_transitions._available_closure_labels_for_open(
            prepared,
            WriterRingState(
                label_state=WriterRingLabelState(allocated=labels),
            ),
        )

        self.assertEqual(labels, ())

    def test_raw_terminal_finalization_allows_cyclic_prepared_but_not_eos(self) -> None:
        prepared = _prepare(cyclopropane_facts())

        terminal = writer_transitions.finalize_writer_terminal_state(
            prepared,
            _raw_emitted_root_state(AtomId(0)),
        )

        self.assertIsNone(terminal)

    def test_raw_terminal_finalization_rejects_missing_active_frame(self) -> None:
        prepared = _prepare(chain_facts(("C",)))

        with self.assertRaises(SouthStarError) as caught:
            writer_transitions.finalize_writer_terminal_state(
                prepared,
                replace(_raw_emitted_root_state(AtomId(0)), active=None),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_terminal_finalization_retains_active_final_atom(self) -> None:
        prepared = _prepare(chain_facts(("C",)))

        terminal = writer_transitions.finalize_writer_terminal_state(
            prepared,
            _raw_emitted_root_state(AtomId(0)),
        )

        self.assertIsNotNone(terminal)
        assert terminal is not None
        self.assertEqual(terminal.active.atom, AtomId(0))

    def test_terminal_finalization_rejects_open_closure_endpoint(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        state = _cyclopropane_terminal_open_closure_state()

        terminal = writer_transitions.finalize_writer_terminal_state(prepared, state)

        self.assertIsNone(terminal)
        self.assertFalse(writer_transitions.writer_state_is_eos(prepared, state))
        choices = writer_frontier_choices(
            prepared,
            WriterFrontierCursor(
                weighted_states=((writer_state_key(state), 1),),
            ),
        )
        self.assertIsNone(choices.terminal)

    def test_terminal_finalization_rejects_closure_candidate(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        state = replace(
            _cyclopropane_terminal_closed_closure_state(),
            ring_state=WriterRingState(),
        )

        terminal = writer_transitions.finalize_writer_terminal_state(prepared, state)

        self.assertIsNone(terminal)
        self.assertFalse(writer_transitions.writer_state_is_eos(prepared, state))

    def test_closed_closure_terminal_state_can_finalize(self) -> None:
        prepared = _prepare(cyclopropane_facts())
        state = _cyclopropane_terminal_closed_closure_state()

        terminal = writer_transitions.finalize_writer_terminal_state(prepared, state)

        self.assertIsNotNone(terminal)

    def test_raw_eos_query_allows_cyclic_prepared_but_remains_false(self) -> None:
        prepared = _prepare(cyclopropane_facts())

        eos = writer_transitions.writer_state_is_eos(
            prepared,
            _raw_emitted_root_state(AtomId(0)),
        )

        self.assertFalse(eos)

    def test_raw_eos_query_rejects_missing_active_frame(self) -> None:
        prepared = _prepare(chain_facts(("C",)))

        with self.assertRaises(SouthStarError) as caught:
            writer_transitions.writer_state_is_eos(
                prepared,
                replace(_raw_emitted_root_state(AtomId(0)), active=None),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_legal_transition_expansion_builds_one_graph_context(self) -> None:
        prepared = _prepare(cco_facts())

        with patch(
            "grimace._south_star1.writer_transitions.build_writer_graph_obligation_context",
            wraps=writer_transitions.build_writer_graph_obligation_context,
        ) as mocked:
            transitions = writer_transitions.legal_writer_transitions(
                prepared,
                _raw_initial_state(AtomId(0)),
            )

        self.assertEqual(mocked.call_count, 1)
        self.assertTrue(transitions)

    def test_terminal_finalization_builds_one_graph_context(self) -> None:
        prepared = _prepare(chain_facts(("C",)))
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        emitted = writer_frontier_choices(prepared, cursor).choices[0].successor
        state = writer_state_from_key(emitted.weighted_states[0][0])

        with patch(
            "grimace._south_star1.writer_transitions.build_writer_graph_obligation_context",
            wraps=writer_transitions.build_writer_graph_obligation_context,
        ) as mocked:
            terminal = writer_transitions.finalize_writer_terminal_state(prepared, state)

        self.assertEqual(mocked.call_count, 1)
        self.assertIsNotNone(terminal)

    def test_child_obligations_from_context_does_not_build_graph_context(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        state = writer_state_from_key(after_root.weighted_states[0][0])
        context = writer_transitions.build_writer_transition_expansion_context(
            prepared,
            state,
        )

        with patch(
            "grimace._south_star1.writer_transitions.build_writer_graph_obligation_context",
            side_effect=AssertionError("child obligations rebuilt graph context"),
        ):
            children = writer_transitions._child_obligations_from_context(
                context,
                state,
                AtomId(0),
            )

        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].bond, BondId(0))
        self.assertEqual(children[0].child, AtomId(1))
        self.assertEqual(children[0].attachment_id, 0)
        self.assertEqual(
            children[0].attachment_action_kind,
            WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        self.assertFalse(children[0].pending_entry)

    def test_child_obligation_blockers_collect_closure_candidate_edges(self) -> None:
        context = SimpleNamespace(
            graph=SimpleNamespace(
                edge_partition=SimpleNamespace(
                    obligations=(
                        SimpleNamespace(
                            kind=WriterEdgeObligationKind.CLOSURE_CANDIDATE,
                            bond=BondId(7),
                        ),
                    ),
                ),
            ),
        )

        blockers = writer_transitions._child_obligation_blockers_from_context(
            context,  # type: ignore[arg-type]
        )

        self.assertEqual(len(blockers), 1)
        self.assertIs(
            blockers[0].kind,
            writer_transitions._WriterChildObligationBlockerKind.CLOSURE_CANDIDATE,
        )
        self.assertEqual(blockers[0].bond, BondId(7))

    def test_child_obligation_blockers_ignore_non_closure_candidate_edges(self) -> None:
        context = SimpleNamespace(
            graph=SimpleNamespace(
                edge_partition=SimpleNamespace(
                    obligations=(
                        SimpleNamespace(
                            kind=WriterEdgeObligationKind.TREE_ENTRY,
                            bond=BondId(1),
                        ),
                        SimpleNamespace(
                            kind=WriterEdgeObligationKind.CLOSED_CLOSURE,
                            bond=BondId(2),
                        ),
                    ),
                ),
            ),
        )

        blockers = writer_transitions._child_obligation_blockers_from_context(
            context,  # type: ignore[arg-type]
        )

        self.assertEqual(blockers, ())

    def test_child_obligation_blockers_raise_existing_unsupported_policy_error(self) -> None:
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=writer_transitions._WriterChildObligationBlockerKind.CLOSURE_CANDIDATE,
            bond=BondId(7),
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._raise_for_child_obligation_blockers((blocker,))

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.UNSUPPORTED_POLICY,
        )
        self.assertIn(
            "WRITER_SHAPED closure-candidate edge obligations are not supported yet",
            str(raised.exception),
        )

    def test_child_obligation_blockers_for_atom_collect_multi_incidence_tree_entries(self) -> None:
        action = SimpleNamespace(
            attachment_id=7,
            kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        attachment = SimpleNamespace(
            attachment_id=7,
            boundary=(
                SimpleNamespace(
                    bond=BondId(1),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(2),
                ),
                SimpleNamespace(
                    bond=BondId(2),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(3),
                ),
            ),
        )
        summary = SimpleNamespace(
            attachments=SimpleNamespace(attachments=(attachment,)),
            attachment_actions=(action,),
        )
        context = SimpleNamespace(
            graph=SimpleNamespace(
                edge_partition=SimpleNamespace(obligations=()),
                residual_summary=summary,
            ),
        )

        blockers = writer_transitions._child_obligation_blockers_for_atom(
            context,  # type: ignore[arg-type]
            AtomId(0),
        )

        self.assertEqual(len(blockers), 1)
        self.assertIs(
            blockers[0].kind,
            (
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
        )
        self.assertEqual(blockers[0].atom, AtomId(0))
        self.assertEqual(blockers[0].attachment_id, 7)
        self.assertIs(
            blockers[0].attachment_action_kind,
            WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )

    def test_child_obligation_blockers_for_atom_ignore_multi_incidence_closure_open_actions(self) -> None:
        action = SimpleNamespace(
            attachment_id=7,
            kind=WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY,
        )
        attachment = SimpleNamespace(
            attachment_id=7,
            boundary=(
                SimpleNamespace(
                    bond=BondId(1),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(2),
                ),
                SimpleNamespace(
                    bond=BondId(2),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(3),
                ),
            ),
        )
        summary = SimpleNamespace(
            attachments=SimpleNamespace(attachments=(attachment,)),
            attachment_actions=(action,),
        )
        context = SimpleNamespace(
            graph=SimpleNamespace(
                edge_partition=SimpleNamespace(obligations=()),
                residual_summary=summary,
            ),
        )

        blockers = writer_transitions._child_obligation_blockers_for_atom(
            context,  # type: ignore[arg-type]
            AtomId(0),
        )

        self.assertEqual(blockers, ())

    def test_child_obligation_blockers_raise_multi_incidence_unsupported_policy_error(self) -> None:
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=AtomId(0),
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._raise_for_child_obligation_blockers((blocker,))

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.UNSUPPORTED_POLICY,
        )
        self.assertIn(
            "WRITER_SHAPED multi-incidence residual attachments are not supported yet",
            str(raised.exception),
        )

    def test_active_child_scheduler_uses_atom_scoped_blockers(self) -> None:
        context = object()
        state = object()
        active_atom = AtomId(0)
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )

        with patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(blocker,),
        ) as child_blockers, patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            side_effect=AssertionError("unblocked child obligations were computed"),
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._active_child_scheduled_actions_from_context(
                    context,  # type: ignore[arg-type]
                    state,  # type: ignore[arg-type]
                    active_atom,
                )

        child_blockers.assert_called_once_with(
            context,
            active_atom,
        )
        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.UNSUPPORTED_POLICY,
        )
        self.assertIn(
            "WRITER_SHAPED multi-incidence residual attachments are not supported yet",
            str(raised.exception),
        )

    def test_active_child_schedule_surface_records_blockers_without_children(self) -> None:
        context = object()
        state = object()
        active_atom = AtomId(0)
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )

        with patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(blocker,),
        ) as blockers, patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            side_effect=AssertionError("unblocked children should not be computed"),
        ) as unblocked:
            surface = writer_transitions._active_child_schedule_surface_from_context(
                context,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertEqual(surface.active_atom, active_atom)
        self.assertTrue(surface.blocked)
        self.assertEqual(surface.blockers, (blocker,))
        self.assertEqual(surface.child_obligations, ())
        self.assertEqual(surface.scheduled_actions, ())
        self.assertEqual(surface.graph_action_surfaces, ())
        blockers.assert_called_once_with(context, active_atom)
        unblocked.assert_not_called()

    def test_active_child_schedule_surface_records_child_actions_and_surfaces(self) -> None:
        context = object()
        state = object()
        active_atom = AtomId(0)
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )

        with patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(),
        ) as blockers, patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            return_value=(child,),
        ) as unblocked:
            surface = writer_transitions._active_child_schedule_surface_from_context(
                context,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertFalse(surface.blocked)
        self.assertEqual(surface.blockers, ())
        self.assertEqual(surface.child_obligations, (child,))
        self.assertEqual(len(surface.scheduled_actions), 1)
        graph_surface = surface.graph_action_surfaces[0]
        self.assertIs(
            graph_surface.kind,
            writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
        )
        self.assertEqual(graph_surface.active_atom, active_atom)
        self.assertEqual(graph_surface.bond, BondId(1))
        self.assertEqual(graph_surface.partner_atom, AtomId(2))
        self.assertEqual(graph_surface.boundary_atom, active_atom)
        self.assertEqual(graph_surface.attachment_id, 9)
        self.assertIs(
            graph_surface.attachment_action_kind,
            WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY,
        )
        self.assertIs(
            graph_surface.owner_kind,
            WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        blockers.assert_called_once_with(context, active_atom)
        unblocked.assert_called_once_with(context, state, active_atom)

    def test_active_child_schedule_surface_records_finish_action_when_no_children(self) -> None:
        context = object()
        state = object()
        active_atom = AtomId(5)

        with patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(),
        ) as blockers, patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            return_value=(),
        ) as unblocked:
            surface = writer_transitions._active_child_schedule_surface_from_context(
                context,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertFalse(surface.blocked)
        self.assertEqual(surface.blockers, ())
        self.assertEqual(surface.child_obligations, ())
        self.assertEqual(len(surface.scheduled_actions), 1)
        graph_surface = surface.graph_action_surfaces[0]
        self.assertIs(
            graph_surface.kind,
            writer_transitions._WriterScheduledActionKind.FINISH_ACTIVE,
        )
        self.assertEqual(graph_surface.active_atom, active_atom)
        blockers.assert_called_once_with(context, active_atom)
        unblocked.assert_called_once_with(context, state, active_atom)

    def test_checked_child_obligations_use_atom_scoped_blockers(self) -> None:
        context = object()
        state = object()
        atom = AtomId(0)
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )

        with patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(blocker,),
        ) as blockers, patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            side_effect=AssertionError("unblocked child obligations were computed"),
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._child_obligations_from_context(
                    context,  # type: ignore[arg-type]
                    state,  # type: ignore[arg-type]
                    atom,
                )

        blockers.assert_called_once_with(
            context,
            atom,
        )
        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.UNSUPPORTED_POLICY,
        )
        self.assertIn(
            "WRITER_SHAPED multi-incidence residual attachments are not supported yet",
            str(raised.exception),
        )

    def test_checked_child_obligations_delegate_when_no_atom_scoped_blockers(self) -> None:
        context = object()
        state = object()
        atom = AtomId(0)
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
        )

        with patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(),
        ) as blockers, patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            return_value=(child,),
        ) as unblocked:
            result = writer_transitions._child_obligations_from_context(
                context,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                atom,
            )

        self.assertEqual(result, (child,))
        blockers.assert_called_once_with(
            context,
            atom,
        )
        unblocked.assert_called_once_with(
            context,
            state,
            atom,
        )

    def test_checked_child_obligations_preserve_multi_incidence_policy_error(self) -> None:
        context = object()
        state = object()
        atom = AtomId(0)
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )

        with patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(blocker,),
        ), patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            side_effect=AssertionError("unblocked builder should not run"),
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._child_obligations_from_context(
                    context,  # type: ignore[arg-type]
                    state,  # type: ignore[arg-type]
                    atom,
                )

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.UNSUPPORTED_POLICY,
        )
        self.assertIn(
            "WRITER_SHAPED multi-incidence residual attachments are not supported yet",
            str(raised.exception),
        )

    def test_unblocked_child_obligations_reject_multi_incidence_as_internal_invariant(self) -> None:
        action = SimpleNamespace(
            attachment_id=7,
            kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        attachment = SimpleNamespace(
            attachment_id=7,
            boundary=(
                SimpleNamespace(
                    bond=BondId(1),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(2),
                ),
                SimpleNamespace(
                    bond=BondId(2),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(3),
                ),
            ),
        )
        summary = SimpleNamespace(
            attachments=SimpleNamespace(attachments=(attachment,)),
            attachment_actions=(action,),
        )
        context = SimpleNamespace(
            graph=SimpleNamespace(
                residual_summary=summary,
            ),
        )
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=None,
            ),
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._unblocked_child_obligations_from_context(
                context,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                AtomId(0),
            )

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.INTERNAL_INVARIANT,
        )
        self.assertIn(
            "unblocked child obligation builder received non-singleton boundary",
            str(raised.exception),
        )

    def test_unblocked_child_obligations_carry_action_incidence_metadata(self) -> None:
        action = SimpleNamespace(
            attachment_id=7,
            kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        attachment = SimpleNamespace(
            attachment_id=7,
            boundary=(
                SimpleNamespace(
                    bond=BondId(1),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(2),
                    owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
                ),
            ),
        )
        summary = SimpleNamespace(
            attachments=SimpleNamespace(attachments=(attachment,)),
            attachment_actions=(action,),
        )
        context = SimpleNamespace(
            graph=SimpleNamespace(
                residual_summary=summary,
            ),
        )
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=None,
            ),
        )

        children = writer_transitions._unblocked_child_obligations_from_context(
            context,  # type: ignore[arg-type]
            state,  # type: ignore[arg-type]
            AtomId(0),
        )

        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].bond, BondId(1))
        self.assertEqual(children[0].child, AtomId(2))
        self.assertEqual(children[0].boundary_atom, AtomId(0))
        self.assertIs(
            children[0].owner_kind,
            WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        self.assertEqual(children[0].attachment_id, 7)
        self.assertIs(
            children[0].attachment_action_kind,
            WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        self.assertFalse(children[0].pending_entry)

    def test_unblocked_child_obligations_carry_pending_parent_as_boundary_atom(self) -> None:
        pending = PendingWriterEntry(
            parent=AtomId(0),
            child=AtomId(2),
            bond=BondId(1),
            branch=False,
        )
        summary = SimpleNamespace(
            attachments=SimpleNamespace(attachments=()),
            attachment_actions=(),
        )
        context = SimpleNamespace(
            graph=SimpleNamespace(
                residual_summary=summary,
            ),
        )
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=pending,
            ),
        )

        children = writer_transitions._unblocked_child_obligations_from_context(
            context,  # type: ignore[arg-type]
            state,  # type: ignore[arg-type]
            AtomId(0),
        )

        self.assertEqual(len(children), 1)
        self.assertEqual(children[0].bond, BondId(1))
        self.assertEqual(children[0].child, AtomId(2))
        self.assertEqual(children[0].boundary_atom, AtomId(0))
        self.assertIsNone(children[0].owner_kind)
        self.assertIsNone(children[0].attachment_id)
        self.assertIsNone(children[0].attachment_action_kind)
        self.assertTrue(children[0].pending_entry)

    def test_scheduled_writer_transitions_dispatches_pending_entry_actions(self) -> None:
        prepared = object()
        context = object()
        pending = PendingWriterEntry(
            parent=AtomId(0),
            child=AtomId(1),
            bond=BondId(0),
            branch=False,
        )
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=pending,
            ),
        )
        action = object()
        emission = object()
        surviving_emission = object()
        transition = object()
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),  # type: ignore[arg-type]
            emissions=(emission,),  # type: ignore[arg-type]
            surviving_emissions=(surviving_emission,),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._pending_entry_scheduled_actions",
            return_value=(action,),
        ) as pending_actions, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=batch,
        ) as emission_batch, patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action_emissions",
            return_value=(transition,),
        ) as flatten_emissions, patch(
            "grimace._south_star1.writer_transitions._root_atom_transitions",
            side_effect=AssertionError("root atom path should not run"),
        ), patch(
            "grimace._south_star1.writer_transitions._active_emitted_transitions",
            side_effect=AssertionError("active-emitted path should not run"),
        ):
            result = writer_transitions._scheduled_writer_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertEqual(result, (transition,))
        pending_actions.assert_called_once_with(state)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            (action,),
        )
        flatten_emissions.assert_called_once_with(
            (surviving_emission,),
        )

    def test_scheduled_writer_transitions_dispatches_root_atom_actions(self) -> None:
        prepared = object()
        context = object()
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=None,
            ),
            active=SimpleNamespace(
                atom=AtomId(0),
                atom_emitted=False,
            ),
        )
        action = object()
        emission = object()
        surviving_emission = object()
        transition = object()
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),  # type: ignore[arg-type]
            emissions=(emission,),  # type: ignore[arg-type]
            surviving_emissions=(surviving_emission,),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._root_atom_scheduled_actions",
            return_value=(action,),
        ) as root_actions, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=batch,
        ) as emission_batch, patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action_emissions",
            return_value=(transition,),
        ) as flatten_emissions, patch(
            "grimace._south_star1.writer_transitions._active_emitted_transitions",
            side_effect=AssertionError("active-emitted path should not run"),
        ):
            result = writer_transitions._scheduled_writer_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertEqual(result, (transition,))
        root_actions.assert_called_once_with(state)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            (action,),
        )
        flatten_emissions.assert_called_once_with(
            (surviving_emission,),
        )

    def test_top_level_scheduled_actions_prefer_pending_entry_over_root(self) -> None:
        pending = PendingWriterEntry(
            parent=AtomId(0),
            child=AtomId(1),
            bond=BondId(0),
            branch=False,
        )
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=pending,
            ),
            active=SimpleNamespace(
                atom=AtomId(9),
                atom_emitted=False,
            ),
        )

        actions = writer_transitions._top_level_scheduled_actions(
            state,  # type: ignore[arg-type]
        )

        self.assertEqual(len(actions), 1)
        self.assertIs(
            actions[0].kind,
            writer_transitions._WriterScheduledActionKind.CONSUME_PENDING_ENTRY,
        )
        self.assertEqual(actions[0].pending_entry, pending)

    def test_top_level_scheduled_actions_return_root_when_no_pending(self) -> None:
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=None,
            ),
            active=SimpleNamespace(
                atom=AtomId(3),
                atom_emitted=False,
            ),
        )

        actions = writer_transitions._top_level_scheduled_actions(
            state,  # type: ignore[arg-type]
        )

        self.assertEqual(len(actions), 1)
        self.assertIs(
            actions[0].kind,
            writer_transitions._WriterScheduledActionKind.EMIT_ROOT_ATOM,
        )
        self.assertEqual(actions[0].parent, AtomId(3))

    def test_top_level_scheduled_actions_empty_for_active_emitted_state(self) -> None:
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=None,
            ),
            active=SimpleNamespace(
                atom=AtomId(3),
                atom_emitted=True,
            ),
        )

        self.assertEqual(
            writer_transitions._top_level_scheduled_actions(
                state,  # type: ignore[arg-type]
            ),
            (),
        )

    def test_active_emitted_schedule_decision_rejects_closure_with_child_batch(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(),
            )
        )
        closure_batch = writer_transitions._closure_endpoint_combined_batch(
            closure_endpoint_decision
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterActiveEmittedScheduleDecision(
                kind=(
                    writer_transitions._WriterActiveEmittedScheduleDecisionKind
                    .CLOSURE_ENDPOINT
                ),
                closure_endpoint_decision=closure_endpoint_decision,
                closure_batch=closure_batch,
                selected_batch=closure_batch,
                child_batch=child_batch,
            )

    def test_active_emitted_schedule_decision_requires_selected_child_batch(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(),
            )
        )
        closure_batch = writer_transitions._closure_endpoint_combined_batch(
            closure_endpoint_decision
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterActiveEmittedScheduleDecision(
                kind=(
                    writer_transitions._WriterActiveEmittedScheduleDecisionKind
                    .ACTIVE_CHILD
                ),
                closure_endpoint_decision=closure_endpoint_decision,
                closure_batch=closure_batch,
                selected_batch=closure_batch,
                child_batch=child_batch,
            )

    def test_top_level_schedule_decision_rejects_active_payload_for_top_level_actions(self) -> None:
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        active_decision = writer_transitions._WriterActiveEmittedScheduleDecision(
            kind=(
                writer_transitions._WriterActiveEmittedScheduleDecisionKind
                .CLOSURE_ENDPOINT
            ),
            closure_endpoint_decision=writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=batch,
                open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                    actions=(),
                    emissions=(),
                    surviving_emissions=(),
                ),
                surviving_emissions=(),
            ),
            closure_batch=batch,
            selected_batch=batch,
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterTopLevelScheduleDecision(
                kind=(
                    writer_transitions._WriterTopLevelScheduleDecisionKind
                    .TOP_LEVEL_ACTIONS
                ),
                selected_batch=batch,
                top_level_batch=batch,
                active_emitted_decision=active_decision,
            )

    def test_top_level_schedule_decision_requires_selected_active_batch(self) -> None:
        active_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        wrong_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        active_decision = writer_transitions._WriterActiveEmittedScheduleDecision(
            kind=(
                writer_transitions._WriterActiveEmittedScheduleDecisionKind
                .CLOSURE_ENDPOINT
            ),
            closure_endpoint_decision=writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=active_batch,
                open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                    actions=(),
                    emissions=(),
                    surviving_emissions=(),
                ),
                surviving_emissions=(),
            ),
            closure_batch=active_batch,
            selected_batch=active_batch,
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterTopLevelScheduleDecision(
                kind=(
                    writer_transitions._WriterTopLevelScheduleDecisionKind
                    .ACTIVE_EMITTED
                ),
                selected_batch=wrong_batch,
                active_emitted_decision=active_decision,
            )

    def test_scheduler_decisions_accept_valid_payloads(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(),
            )
        )
        closure_batch = writer_transitions._closure_endpoint_combined_batch(
            closure_endpoint_decision
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(0),
            blockers=(),
            child_obligations=(),
            scheduled_actions=child_batch.actions,
        )

        closure_decision = writer_transitions._WriterActiveEmittedScheduleDecision(
            kind=(
                writer_transitions._WriterActiveEmittedScheduleDecisionKind
                .CLOSURE_ENDPOINT
            ),
            closure_endpoint_decision=closure_endpoint_decision,
            closure_batch=closure_batch,
            selected_batch=closure_batch,
        )

        child_decision = writer_transitions._WriterActiveEmittedScheduleDecision(
            kind=(
                writer_transitions._WriterActiveEmittedScheduleDecisionKind
                .ACTIVE_CHILD
            ),
            closure_endpoint_decision=closure_endpoint_decision,
            closure_batch=closure_batch,
            child_batch=child_batch,
            child_schedule_surface=child_surface,
            selected_batch=child_batch,
        )

        writer_transitions._WriterTopLevelScheduleDecision(
            kind=(
                writer_transitions._WriterTopLevelScheduleDecisionKind
                .TOP_LEVEL_ACTIONS
            ),
            selected_batch=closure_batch,
            top_level_batch=closure_batch,
        )

        writer_transitions._WriterTopLevelScheduleDecision(
            kind=(
                writer_transitions._WriterTopLevelScheduleDecisionKind
                .ACTIVE_EMITTED
            ),
            selected_batch=closure_decision.selected_batch,
            active_emitted_decision=closure_decision,
        )

        writer_transitions._WriterTopLevelScheduleDecision(
            kind=(
                writer_transitions._WriterTopLevelScheduleDecisionKind
                .ACTIVE_EMITTED
            ),
            selected_batch=child_decision.selected_batch,
            active_emitted_decision=child_decision,
        )

    def test_active_emitted_decision_constructors_preserve_selected_batches(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(),
            )
        )
        closure_batch = writer_transitions._closure_endpoint_combined_batch(
            closure_endpoint_decision
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(0),
            blockers=(),
            child_obligations=(),
            scheduled_actions=child_batch.actions,
        )

        closure_decision = writer_transitions._active_emitted_closure_decision(
            closure_endpoint_decision,
        )
        child_decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision,
            child_surface,
            child_batch,
        )

        self.assertIs(
            closure_decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.CLOSURE_ENDPOINT,
        )
        self.assertEqual(closure_decision.closure_batch, closure_batch)
        self.assertIs(closure_decision.selected_batch, closure_decision.closure_batch)
        self.assertIsNone(closure_decision.child_batch)
        self.assertIs(
            child_decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertEqual(child_decision.closure_batch, closure_batch)
        self.assertIs(child_decision.child_batch, child_batch)
        self.assertIs(child_decision.child_schedule_surface, child_surface)
        self.assertIs(child_decision.selected_batch, child_batch)

    def test_active_emitted_decision_constructors_carry_closure_endpoint_decision(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(),
            )
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(0),
            blockers=(),
            child_obligations=(),
            scheduled_actions=child_batch.actions,
        )

        closure_decision = writer_transitions._active_emitted_closure_decision(
            closure_endpoint_decision,
        )
        child_decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision,
            child_surface,
            child_batch,
        )

        self.assertIs(
            closure_decision.closure_endpoint_decision,
            closure_endpoint_decision,
        )
        self.assertEqual(
            closure_decision.closure_batch,
            writer_transitions._closure_endpoint_combined_batch(
                closure_endpoint_decision
            ),
        )
        self.assertIs(closure_decision.selected_batch, closure_decision.closure_batch)
        self.assertIs(
            child_decision.closure_endpoint_decision,
            closure_endpoint_decision,
        )
        self.assertEqual(
            child_decision.closure_batch,
            writer_transitions._closure_endpoint_combined_batch(
                closure_endpoint_decision
            ),
        )
        self.assertIs(
            child_decision.child_schedule_surface,
            child_surface,
        )
        self.assertIs(child_decision.selected_batch, child_batch)

    def test_active_emitted_child_decision_retains_child_schedule_surface(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(),
            )
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
            boundary_atom=AtomId(0),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            AtomId(0),
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(0),
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),
            surviving_emissions=(child_emission,),
        )

        decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision,
            child_surface,
            child_batch,
        )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertIs(decision.child_schedule_surface, child_surface)
        self.assertIs(decision.child_batch, child_batch)
        self.assertIs(decision.selected_batch, child_batch)
        self.assertEqual(
            decision.considered_graph_action_surfaces,
            child_surface.graph_action_surfaces,
        )
        self.assertEqual(
            decision.selected_graph_action_surfaces,
            child_batch.surviving_graph_action_surfaces,
        )

    def test_active_emitted_closure_decision_considered_surfaces_delegate_to_closure_decision(self) -> None:
        label = WriterClosureLabel(value=1, text="1")
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair = writer_transitions._WriterClosurePairObligation(
            endpoint=endpoint,
            closure=closure,
        )
        pair_action = writer_transitions._pair_closure_endpoint_action(
            AtomId(0),
            pair,
        )
        surface = writer_transitions._WriterClosureEndpointScheduleSurface(
            active_atom=AtomId(0),
            pair_actions=(pair_action,),
            open_actions=(),
        )
        emission = writer_transitions._WriterScheduledActionEmission(
            action=pair_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(pair_action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(emission,),
                schedule_surface=surface,
            )
        )

        decision = writer_transitions._active_emitted_closure_decision(
            closure_endpoint_decision,
        )

        self.assertEqual(
            decision.considered_graph_action_surfaces,
            closure_endpoint_decision.considered_graph_action_surfaces,
        )
        self.assertEqual(
            decision.selected_graph_action_surfaces,
            closure_endpoint_decision.selected_graph_action_surfaces,
        )

    def test_active_emitted_closure_decision_retains_graph_policy_decision(self) -> None:
        action = writer_transitions._finish_active_action(AtomId(0))
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(emission,),
            )
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_endpoint_decision,
        )

        decision = writer_transitions._active_emitted_closure_decision(
            closure_endpoint_decision,
            graph_policy_decision=policy,
        )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.CLOSURE_ENDPOINT,
        )
        self.assertIs(decision.graph_policy_decision, policy)
        self.assertIs(
            decision.closure_endpoint_decision,
            policy.closure_endpoint_decision,
        )
        self.assertIsNone(decision.child_schedule_surface)
        self.assertIsNone(decision.child_batch)

    def test_active_emitted_child_decision_retains_graph_policy_decision(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=pair_batch,
                open_batch=open_batch,
                surviving_emissions=(),
            )
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
        )
        child_action = writer_transitions._enter_inline_child_action(
            AtomId(0),
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(0),
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),
            surviving_emissions=(child_emission,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_endpoint_decision,
            child_schedule_surface=child_surface,
        )

        decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision=closure_endpoint_decision,
            child_schedule_surface=child_surface,
            child_batch=child_batch,
            graph_policy_decision=policy,
        )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertIs(decision.graph_policy_decision, policy)
        self.assertIs(decision.child_schedule_surface, child_surface)
        self.assertIs(decision.child_batch, child_batch)
        self.assertIs(decision.selected_batch, child_batch)

    def test_active_emitted_child_decision_rejects_blocked_graph_policy_decision(self) -> None:
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=AtomId(0),
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        blocked_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(0),
            blockers=(blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        blocked_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=blocked_surface,
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
        )
        child_action = writer_transitions._enter_inline_child_action(
            AtomId(0),
            child,
        )
        unblocked_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(0),
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(),
            surviving_emissions=(),
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._active_emitted_child_decision(
                closure_endpoint_decision=closure_decision,
                child_schedule_surface=unblocked_surface,
                child_batch=child_batch,
                graph_policy_decision=blocked_policy,
            )

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.INTERNAL_INVARIANT,
        )

    def test_active_emitted_child_schedule_considered_surfaces_use_retained_policy(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),
            surviving_emissions=(child_emission,),
        )

        decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            child_batch=child_batch,
            graph_policy_decision=policy,
        )

        self.assertEqual(
            decision.considered_graph_action_surfaces,
            policy.considered_graph_action_surfaces,
        )
        self.assertEqual(
            decision.policy_chosen_graph_action_surfaces,
            policy.chosen_graph_action_surfaces,
        )
        self.assertEqual(
            decision.selected_graph_action_surfaces,
            child_batch.surviving_graph_action_surfaces,
        )

    def test_active_emitted_schedule_decision_filters_chosen_and_selected_policy_families(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(
                    writer_transitions._WriterScheduledActionEmission(
                        action=open_action,
                        transitions=(),
                    ),
                ),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),
            surviving_emissions=(child_emission,),
        )

        decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            child_batch=child_batch,
            graph_policy_decision=policy,
        )

        self.assertEqual(
            decision.policy_chosen_graph_action_surfaces_for_policy_family(
                writer_transitions._WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY,
            ),
            child_surface.graph_action_surfaces,
        )
        self.assertEqual(
            decision.selected_graph_action_surfaces_for_policy_family(
                writer_transitions._WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY,
            ),
            child_batch.surviving_graph_action_surfaces,
        )

    def test_active_emitted_schedule_decision_exposes_selected_residual_policy_emission_groups(self) -> None:
        active_atom = AtomId(0)
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(SimpleNamespace(emitted_text="C"),),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),
            surviving_emissions=(child_emission,),
        )
        decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            child_batch=child_batch,
            graph_policy_decision=policy,
        )

        groups = decision.selected_residual_attachment_policy_emission_groups

        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0].cyclic_tree_entry_emissions,
            (child_emission,),
        )
        self.assertEqual(
            groups[0].surviving_cyclic_tree_entry_emissions,
            (child_emission,),
        )

    def test_top_level_decision_constructors_preserve_selected_batches(self) -> None:
        top_level_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        active_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        active_closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=active_batch,
                open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                    actions=(),
                    emissions=(),
                    surviving_emissions=(),
                ),
                surviving_emissions=(),
            )
        )
        active_decision = writer_transitions._active_emitted_closure_decision(
            active_closure_endpoint_decision,
        )

        top_level_decision = writer_transitions._top_level_actions_decision(
            top_level_batch,
        )
        active_top_level_decision = (
            writer_transitions._top_level_active_emitted_decision(
                active_decision,
            )
        )

        self.assertIs(
            top_level_decision.kind,
            writer_transitions._WriterTopLevelScheduleDecisionKind.TOP_LEVEL_ACTIONS,
        )
        self.assertIs(top_level_decision.selected_batch, top_level_batch)
        self.assertIs(top_level_decision.top_level_batch, top_level_batch)
        self.assertIsNone(top_level_decision.active_emitted_decision)
        self.assertIs(
            active_top_level_decision.kind,
            writer_transitions._WriterTopLevelScheduleDecisionKind.ACTIVE_EMITTED,
        )
        self.assertIs(
            active_top_level_decision.selected_batch,
            active_decision.selected_batch,
        )
        self.assertEqual(active_top_level_decision.selected_batch, active_batch)
        self.assertIs(
            active_top_level_decision.active_emitted_decision,
            active_decision,
        )
        self.assertIsNone(active_top_level_decision.top_level_batch)

    def test_top_level_actions_decision_exposes_selected_frontier_and_surfaces(self) -> None:
        action = writer_transitions._emit_root_atom_action(AtomId(0))
        transition = SimpleNamespace(emitted_text="C")
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )

        decision = writer_transitions._top_level_actions_decision(batch)

        self.assertEqual(
            decision.considered_graph_action_surfaces,
            batch.graph_action_surfaces,
        )
        self.assertEqual(
            decision.selected_graph_action_surfaces,
            batch.surviving_graph_action_surfaces,
        )
        self.assertEqual(decision.selected_transitions, batch.surviving_transitions)
        self.assertEqual(
            decision.selected_next_token_frontier,
            batch.surviving_next_token_frontier,
        )

    def test_top_level_active_emitted_decision_delegates_selected_frontier(self) -> None:
        action = writer_transitions._finish_active_action(AtomId(0))
        transition = SimpleNamespace(emitted_text="")
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=batch,
                open_batch=open_batch,
                surviving_emissions=(emission,),
            )
        )
        active_decision = writer_transitions._active_emitted_closure_decision(
            closure_endpoint_decision,
        )

        top_decision = writer_transitions._top_level_active_emitted_decision(
            active_decision,
        )

        self.assertIs(top_decision.selected_batch, active_decision.selected_batch)
        self.assertEqual(
            top_decision.considered_graph_action_surfaces,
            active_decision.considered_graph_action_surfaces,
        )
        self.assertEqual(
            top_decision.selected_graph_action_surfaces,
            active_decision.selected_graph_action_surfaces,
        )
        self.assertEqual(
            top_decision.selected_next_token_frontier,
            active_decision.selected_next_token_frontier,
        )

    def test_top_level_schedule_outcome_validates_top_level_scheduled_payload(self) -> None:
        action = writer_transitions._emit_root_atom_action(AtomId(0))
        transition = SimpleNamespace(emitted_text="C")
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        decision = writer_transitions._top_level_actions_decision(batch)

        outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=decision,
        )

        self.assertIs(outcome.schedule_decision, decision)
        self.assertIsNone(outcome.active_emitted_outcome)
        self.assertEqual(outcome.selected_transitions, decision.selected_transitions)
        self.assertEqual(
            outcome.selected_next_token_frontier,
            decision.selected_next_token_frontier,
        )
        self.assertEqual(outcome.graph_policy_blockers, ())

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterTopLevelScheduleOutcomeKind
                    .SCHEDULED
                ),
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

        active_policy, active_decision = self._closure_policy_for_outcome(
            AtomId(1),
        )
        active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=active_policy,
            schedule_decision=active_decision,
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterTopLevelScheduleOutcomeKind
                    .SCHEDULED
                ),
                schedule_decision=decision,
                active_emitted_outcome=active_outcome,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_top_level_schedule_outcome_validates_active_emitted_scheduled_payload(self) -> None:
        policy, active_decision = self._closure_policy_for_outcome(AtomId(0))
        active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=policy,
            schedule_decision=active_decision,
        )
        top_decision = writer_transitions._top_level_active_emitted_decision(
            active_decision,
        )

        outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=top_decision,
            active_emitted_outcome=active_outcome,
        )

        self.assertIs(outcome.schedule_decision, top_decision)
        self.assertIs(outcome.active_emitted_outcome, active_outcome)
        self.assertIs(outcome.graph_policy_decision, policy)

        other_policy, other_active_decision = self._closure_policy_for_outcome(
            AtomId(1),
        )
        other_active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=other_policy,
            schedule_decision=other_active_decision,
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterTopLevelScheduleOutcomeKind
                    .SCHEDULED
                ),
                schedule_decision=top_decision,
                active_emitted_outcome=other_active_outcome,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_top_level_schedule_outcome_validates_blocked_active_emitted_payload(self) -> None:
        blocked_outcome = self._blocked_child_active_emitted_outcome(AtomId(0))

        outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
            active_emitted_outcome=blocked_outcome,
        )

        self.assertIsNone(outcome.schedule_decision)
        self.assertIs(outcome.active_emitted_outcome, blocked_outcome)
        self.assertEqual(
            outcome.graph_policy_blockers,
            blocked_outcome.graph_policy_blockers,
        )
        self.assertEqual(outcome.selected_transitions, ())
        self.assertEqual(outcome.selected_next_token_frontier, ())

        action = writer_transitions._emit_root_atom_action(AtomId(0))
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(),
            surviving_emissions=(),
        )
        top_decision = writer_transitions._top_level_actions_decision(batch)

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
                schedule_decision=top_decision,
                active_emitted_outcome=blocked_outcome,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

        policy, active_decision = self._closure_policy_for_outcome(AtomId(1))
        scheduled_active_outcome = (
            writer_transitions._WriterActiveEmittedScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedScheduleOutcomeKind
                    .SCHEDULED
                ),
                graph_policy_decision=policy,
                schedule_decision=active_decision,
            )
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterTopLevelScheduleOutcome(
                kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
                active_emitted_outcome=scheduled_active_outcome,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_top_level_active_emitted_decision_exposes_graph_policy_decision(self) -> None:
        action = writer_transitions._finish_active_action(AtomId(0))
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=batch,
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(emission,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=AtomId(0),
            closure_endpoint_decision=closure_decision,
        )
        active_decision = writer_transitions._active_emitted_closure_decision(
            closure_decision,
            graph_policy_decision=policy,
        )

        top = writer_transitions._top_level_active_emitted_decision(
            active_decision,
        )

        self.assertIs(top.active_emitted_graph_policy_decision, policy)

    def test_top_level_active_emitted_decision_exposes_policy_chosen_surfaces(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(
                    writer_transitions._WriterScheduledActionEmission(
                        action=open_action,
                        transitions=(),
                    ),
                ),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),
            surviving_emissions=(child_emission,),
        )
        active_decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
            child_batch=child_batch,
            graph_policy_decision=policy,
        )

        top = writer_transitions._top_level_active_emitted_decision(
            active_decision,
        )

        self.assertEqual(
            top.policy_chosen_graph_action_surfaces,
            active_decision.policy_chosen_graph_action_surfaces,
        )
        self.assertEqual(
            top.considered_graph_action_surfaces,
            active_decision.considered_graph_action_surfaces,
        )
        self.assertEqual(
            top.selected_graph_action_surfaces,
            active_decision.selected_graph_action_surfaces,
        )

    def test_scheduled_writer_next_token_frontier_returns_selected_frontier(self) -> None:
        action = writer_transitions._emit_root_atom_action(AtomId(0))
        transition = SimpleNamespace(emitted_text="C")
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        decision = writer_transitions._top_level_actions_decision(batch)

        with patch(
            "grimace._south_star1.writer_transitions._top_level_schedule_decision",
            return_value=decision,
        ) as schedule:
            frontier = writer_transitions._scheduled_writer_next_token_frontier(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
            )

        self.assertEqual(frontier, decision.selected_next_token_frontier)
        schedule.assert_called_once()

    def test_legal_writer_next_token_frontier_builds_context_and_delegates(self) -> None:
        scheduled_frontier = (object(),)
        context = object()

        with patch(
            "grimace._south_star1.writer_transitions.build_writer_transition_expansion_context",
            return_value=context,
        ) as build, patch(
            "grimace._south_star1.writer_transitions._scheduled_writer_next_token_frontier",
            return_value=scheduled_frontier,  # type: ignore[arg-type]
        ) as scheduled:
            result = writer_transitions._legal_writer_next_token_frontier(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
            )

        self.assertEqual(result, scheduled_frontier)
        build.assert_called_once()
        scheduled.assert_called_once()

    def test_closure_endpoint_schedule_surface_projects_pair_before_open_actions(self) -> None:
        pair_label = WriterClosureLabel(value=1, text="1")
        open_label = WriterClosureLabel(value=2, text="2")
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=pair_label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=pair_label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair = writer_transitions._WriterClosurePairObligation(
            endpoint=endpoint,
            closure=closure,
        )
        pair_action = writer_transitions._pair_closure_endpoint_action(
            AtomId(0),
            pair,
        )
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(2),
            first_atom=AtomId(0),
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            AtomId(0),
            open_obligation,
            open_label,
        )

        surface = writer_transitions._WriterClosureEndpointScheduleSurface(
            active_atom=AtomId(0),
            pair_actions=(pair_action,),
            open_actions=(open_action,),
        )

        self.assertEqual(surface.scheduled_actions, (pair_action, open_action))
        self.assertEqual(
            surface.graph_action_surfaces,
            (
                surface.pair_graph_action_surfaces[0],
                surface.open_graph_action_surfaces[0],
            ),
        )
        pair_surface = surface.pair_graph_action_surfaces[0]
        open_surface = surface.open_graph_action_surfaces[0]
        self.assertIs(
            pair_surface.kind,
            writer_transitions._WriterScheduledActionKind.PAIR_CLOSURE_ENDPOINT,
        )
        self.assertIs(
            open_surface.kind,
            writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
        )
        self.assertIs(pair_surface.closure_label, pair_label)
        self.assertIs(open_surface.closure_label, open_label)
        self.assertEqual(open_surface.attachment_id, 7)
        self.assertIs(open_surface.owner_kind, WriterBoundaryOwnerKind.ACTIVE_ATOM)

    def test_closure_endpoint_schedule_decision_exposes_considered_and_selected_surfaces(self) -> None:
        label = WriterClosureLabel(value=1, text="1")
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair = writer_transitions._WriterClosurePairObligation(
            endpoint=endpoint,
            closure=closure,
        )
        pair_action = writer_transitions._pair_closure_endpoint_action(
            AtomId(0),
            pair,
        )
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(2),
            first_atom=AtomId(0),
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            AtomId(0),
            open_obligation,
            label,
        )
        surface = writer_transitions._WriterClosureEndpointScheduleSurface(
            active_atom=AtomId(0),
            pair_actions=(pair_action,),
            open_actions=(open_action,),
        )
        pair_emission = writer_transitions._WriterScheduledActionEmission(
            action=pair_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(pair_action,),
            emissions=(pair_emission,),
            surviving_emissions=(pair_emission,),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(open_action,),
            emissions=(open_emission,),
            surviving_emissions=(),
        )
        decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(pair_emission,),
            schedule_surface=surface,
        )

        self.assertIs(decision.schedule_surface, surface)
        self.assertEqual(
            decision.considered_graph_action_surfaces,
            surface.graph_action_surfaces,
        )
        self.assertEqual(
            decision.selected_graph_action_surfaces,
            (pair_emission.graph_action_surface,),
        )

    def test_closure_endpoint_schedule_decision_exposes_residual_policy_emission_groups(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(2),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        surface = writer_transitions._WriterClosureEndpointScheduleSurface(
            active_atom=active_atom,
            pair_actions=(),
            open_actions=(open_action,),
        )
        closure_open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(open_action,),
            emissions=(closure_open_emission,),
            surviving_emissions=(),
        )
        decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
            schedule_surface=surface,
        )

        considered = decision.considered_residual_attachment_policy_emission_groups
        selected = decision.selected_residual_attachment_policy_emission_groups

        self.assertEqual(len(considered), 1)
        self.assertEqual(
            considered[0].closure_open_emissions,
            (closure_open_emission,),
        )
        self.assertEqual(considered[0].surviving_closure_open_emissions, ())
        self.assertEqual(selected, ())

    def test_closure_endpoint_schedule_decision_separates_pair_and_open_batches(self) -> None:
        prepared = object()
        state = SimpleNamespace(
            active=SimpleNamespace(atom=AtomId(0)),
            ring_state=object(),
        )
        context = object()
        label = WriterClosureLabel(value=1, text="1")
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=AtomId(0),
            label=label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair = writer_transitions._WriterClosurePairObligation(
            endpoint=endpoint,
            closure=closure,
        )
        pair_action = writer_transitions._pair_closure_endpoint_action(
            AtomId(0),
            pair,
        )
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(2),
            first_atom=AtomId(0),
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            AtomId(0),
            open_obligation,
            label,
        )
        pair_emission = writer_transitions._WriterScheduledActionEmission(
            action=pair_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(pair_action,),
            emissions=(pair_emission,),
            surviving_emissions=(pair_emission,),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(open_action,),
            emissions=(open_emission,),
            surviving_emissions=(open_emission,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_pair_scheduled_actions",
            return_value=(pair_action,),
        ) as pair_actions, patch(
            "grimace._south_star1.writer_transitions._available_closure_labels_for_open",
            return_value=(label,),
        ) as available_labels, patch(
            "grimace._south_star1.writer_transitions._closure_open_scheduled_actions",
            return_value=(open_action,),
        ) as open_actions, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=(pair_batch, open_batch),
        ) as emission_batch:
            decision = writer_transitions._closure_endpoint_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(decision.pair_batch, pair_batch)
        self.assertIs(decision.open_batch, open_batch)
        self.assertIsNotNone(decision.schedule_surface)
        self.assertEqual(decision.schedule_surface.pair_actions, decision.pair_batch.actions)
        self.assertEqual(decision.schedule_surface.open_actions, decision.open_batch.actions)
        self.assertEqual(
            decision.considered_graph_action_surfaces,
            (
                *decision.pair_batch.graph_action_surfaces,
                *decision.open_batch.graph_action_surfaces,
            ),
        )
        self.assertEqual(
            decision.surviving_emissions,
            (pair_emission, open_emission),
        )
        pair_actions.assert_called_once_with(state, AtomId(0))
        available_labels.assert_called_once_with(prepared, state.ring_state)
        open_actions.assert_called_once_with(context, AtomId(0), (label,))
        self.assertEqual(emission_batch.call_count, 2)
        self.assertEqual(
            emission_batch.call_args_list[0].args,
            (prepared, state, context, (pair_action,)),
        )
        self.assertEqual(
            emission_batch.call_args_list[1].args,
            (prepared, state, context, (open_action,)),
        )

    def test_closure_endpoint_schedule_decision_skips_open_actions_without_labels(self) -> None:
        prepared = object()
        state = SimpleNamespace(
            active=SimpleNamespace(atom=AtomId(0)),
            ring_state=object(),
        )
        context = object()
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_pair_scheduled_actions",
            return_value=(),
        ), patch(
            "grimace._south_star1.writer_transitions._available_closure_labels_for_open",
            return_value=(),
        ), patch(
            "grimace._south_star1.writer_transitions._closure_open_scheduled_actions",
            side_effect=AssertionError("open actions should not be built without labels"),
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=(pair_batch, open_batch),
        ) as emission_batch:
            decision = writer_transitions._closure_endpoint_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(decision.pair_batch, pair_batch)
        self.assertIs(decision.open_batch, open_batch)
        self.assertEqual(decision.surviving_emissions, ())
        self.assertEqual(emission_batch.call_count, 2)
        self.assertEqual(
            emission_batch.call_args_list[1].args,
            (prepared, state, context, ()),
        )

    def test_closure_open_obligations_carry_action_incidence_metadata(self) -> None:
        action = SimpleNamespace(
            attachment_id=7,
            kind=WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY,
        )
        attachment = SimpleNamespace(
            attachment_id=7,
            boundary=(
                SimpleNamespace(
                    bond=BondId(1),
                    written_atom=AtomId(0),
                    residual_atom=AtomId(2),
                    owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
                ),
            ),
        )
        summary = SimpleNamespace(
            attachments=SimpleNamespace(attachments=(attachment,)),
            attachment_actions=(action,),
        )
        context = SimpleNamespace(
            graph=SimpleNamespace(
                residual_summary=summary,
            ),
        )

        obligations = writer_transitions._closure_open_obligations_from_context(
            context,  # type: ignore[arg-type]
            AtomId(0),
        )

        self.assertEqual(len(obligations), 1)
        self.assertEqual(obligations[0].bond, BondId(1))
        self.assertEqual(obligations[0].first_atom, AtomId(0))
        self.assertEqual(obligations[0].second_atom, AtomId(2))
        self.assertEqual(obligations[0].attachment_id, 7)
        self.assertIs(
            obligations[0].attachment_action_kind,
            WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY,
        )
        self.assertIs(
            obligations[0].owner_kind,
            WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )

    def test_closure_endpoint_combined_batch_preserves_pair_before_open(self) -> None:
        pair_action = object()
        open_action = object()
        pair_emission = object()
        open_emission = object()
        pair_survivor = object()
        open_survivor = object()
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(pair_action,),  # type: ignore[arg-type]
            emissions=(pair_emission,),  # type: ignore[arg-type]
            surviving_emissions=(pair_survivor,),  # type: ignore[arg-type]
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(open_action,),  # type: ignore[arg-type]
            emissions=(open_emission,),  # type: ignore[arg-type]
            surviving_emissions=(open_survivor,),  # type: ignore[arg-type]
        )
        decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(pair_survivor, open_survivor),  # type: ignore[arg-type]
        )

        combined = writer_transitions._closure_endpoint_combined_batch(decision)

        self.assertEqual(combined.actions, (pair_action, open_action))
        self.assertEqual(combined.emissions, (pair_emission, open_emission))
        self.assertEqual(
            combined.surviving_emissions,
            (pair_survivor, open_survivor),
        )

    def test_top_level_schedule_decision_selects_top_level_batch(self) -> None:
        prepared = object()
        state = object()
        context = object()
        action = object()
        emission = object()
        survivor = object()

        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),  # type: ignore[arg-type]
            emissions=(emission,),  # type: ignore[arg-type]
            surviving_emissions=(survivor,),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(action,),
        ) as top_level_actions, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=batch,
        ) as emission_batch, patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            side_effect=AssertionError("active-emitted decision should not run"),
        ):
            decision = writer_transitions._top_level_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterTopLevelScheduleDecisionKind.TOP_LEVEL_ACTIONS,
        )
        self.assertIs(decision.selected_batch, batch)
        self.assertIs(decision.top_level_batch, batch)
        self.assertIsNone(decision.active_emitted_decision)
        top_level_actions.assert_called_once_with(state)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            (action,),
        )

    def test_top_level_schedule_decision_keeps_zero_survivor_top_level_batch(self) -> None:
        prepared = object()
        state = object()
        context = object()
        action = object()
        emission = object()

        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),  # type: ignore[arg-type]
            emissions=(emission,),  # type: ignore[arg-type]
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(action,),
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=batch,
        ), patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            side_effect=AssertionError("active-emitted decision should not run"),
        ):
            decision = writer_transitions._top_level_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterTopLevelScheduleDecisionKind.TOP_LEVEL_ACTIONS,
        )
        self.assertIs(decision.selected_batch, batch)
        self.assertEqual(decision.selected_batch.surviving_emissions, ())

    def test_top_level_schedule_decision_selects_active_emitted_when_no_top_level_actions(self) -> None:
        prepared = object()
        context = object()
        state = SimpleNamespace(
            active=SimpleNamespace(
                atom=AtomId(4),
            ),
        )

        selected_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=selected_batch,
                open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                    actions=(),
                    emissions=(),
                    surviving_emissions=(),
                ),
                surviving_emissions=(),
            )
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=AtomId(4),
            blockers=(),
            child_obligations=(),
            scheduled_actions=selected_batch.actions,
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=AtomId(4),
            child_schedule_surface=child_surface,
            closure_endpoint_decision=closure_endpoint_decision,
        )
        active_decision = writer_transitions._active_emitted_child_decision(
            closure_endpoint_decision=closure_endpoint_decision,
            child_schedule_surface=child_surface,
            child_batch=selected_batch,
            graph_policy_decision=policy,
        )
        active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=policy,
            schedule_decision=active_decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(),
        ) as top_level_actions, patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            return_value=active_outcome,
        ) as active_emitted_outcome:
            decision = writer_transitions._top_level_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterTopLevelScheduleDecisionKind.ACTIVE_EMITTED,
        )
        self.assertIs(decision.selected_batch, selected_batch)
        self.assertIsNone(decision.top_level_batch)
        self.assertIs(decision.active_emitted_decision, active_decision)
        top_level_actions.assert_called_once_with(state)
        active_emitted_outcome.assert_called_once_with(
            prepared,
            state,
            context,
            AtomId(4),
        )

    def test_top_level_schedule_outcome_keeps_top_level_action_priority(self) -> None:
        prepared = object()
        state = object()
        context = object()
        action = writer_transitions._emit_root_atom_action(AtomId(0))
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(),
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(action,),
        ) as top_level_actions, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=batch,
        ) as emission_batch, patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            side_effect=AssertionError("active-emitted outcome should not run"),
        ) as active_emitted_outcome:
            outcome = writer_transitions._top_level_schedule_outcome(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(
            outcome.kind,
            writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
        )
        self.assertIs(
            outcome.schedule_decision.kind,
            writer_transitions._WriterTopLevelScheduleDecisionKind.TOP_LEVEL_ACTIONS,
        )
        top_level_actions.assert_called_once_with(state)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            (action,),
        )
        active_emitted_outcome.assert_not_called()

    def test_top_level_schedule_outcome_wraps_active_emitted_scheduled_outcome(self) -> None:
        prepared = object()
        context = object()
        active_atom = AtomId(4)
        state = SimpleNamespace(
            active=SimpleNamespace(
                atom=active_atom,
            ),
        )
        policy, active_decision = self._closure_policy_for_outcome(active_atom)
        active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=policy,
            schedule_decision=active_decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(),
        ), patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            return_value=active_outcome,
        ) as active_emitted_outcome:
            outcome = writer_transitions._top_level_schedule_outcome(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(
            outcome.kind,
            writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
        )
        self.assertIs(outcome.active_emitted_outcome, active_outcome)
        self.assertIs(
            outcome.schedule_decision.kind,
            writer_transitions._WriterTopLevelScheduleDecisionKind.ACTIVE_EMITTED,
        )
        self.assertIs(
            outcome.schedule_decision.active_emitted_decision,
            active_decision,
        )
        active_emitted_outcome.assert_called_once_with(
            prepared,
            state,
            context,
            active_atom,
        )

    def test_top_level_schedule_outcome_wraps_active_emitted_blocked_outcome(self) -> None:
        prepared = object()
        context = object()
        active_atom = AtomId(4)
        state = SimpleNamespace(
            active=SimpleNamespace(
                atom=active_atom,
            ),
        )
        active_outcome = self._blocked_child_active_emitted_outcome(active_atom)

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(),
        ), patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            return_value=active_outcome,
        ):
            outcome = writer_transitions._top_level_schedule_outcome(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertIs(
            outcome.kind,
            writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
        )
        self.assertIsNone(outcome.schedule_decision)
        self.assertIs(outcome.active_emitted_outcome, active_outcome)
        self.assertEqual(
            outcome.graph_policy_blockers,
            active_outcome.graph_policy_blockers,
        )

    def test_top_level_schedule_decision_raises_from_blocked_top_level_outcome(self) -> None:
        active_atom = AtomId(4)
        outcome = self._blocked_child_active_emitted_outcome(active_atom)
        top_outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.BLOCKED,
            active_emitted_outcome=outcome,
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_schedule_outcome",
            return_value=top_outcome,
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._top_level_schedule_decision(
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_top_level_schedule_decision_returns_scheduled_outcome_decision(self) -> None:
        action = writer_transitions._emit_root_atom_action(AtomId(0))
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(),
            surviving_emissions=(),
        )
        decision = writer_transitions._top_level_actions_decision(batch)
        outcome = writer_transitions._WriterTopLevelScheduleOutcome(
            kind=writer_transitions._WriterTopLevelScheduleOutcomeKind.SCHEDULED,
            schedule_decision=decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_schedule_outcome",
            return_value=outcome,
        ):
            result = writer_transitions._top_level_schedule_decision(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
            )

        self.assertIs(result, decision)

    def test_scheduled_writer_transitions_flattens_top_level_decision_survivors(self) -> None:
        prepared = object()
        state = object()
        context = object()
        action = writer_transitions._finish_active_action(AtomId(0))
        transition = SimpleNamespace(emitted_text="")
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(transition,),  # type: ignore[arg-type]
        )

        selected_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        decision = writer_transitions._WriterTopLevelScheduleDecision(
            kind=writer_transitions._WriterTopLevelScheduleDecisionKind.TOP_LEVEL_ACTIONS,
            selected_batch=selected_batch,
            top_level_batch=selected_batch,
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_schedule_decision",
            return_value=decision,
        ) as schedule_decision:
            result = writer_transitions._scheduled_writer_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertEqual(result, decision.selected_transitions)
        schedule_decision.assert_called_once_with(
            prepared,
            state,
            context,
        )

    def test_scheduled_writer_transitions_does_not_fall_through_when_top_level_actions_do_not_survive(self) -> None:
        prepared = object()
        context = object()
        state = SimpleNamespace(
            obligations=SimpleNamespace(
                pending_entry=None,
            ),
            active=SimpleNamespace(
                atom=AtomId(0),
                atom_emitted=False,
            ),
        )
        action = object()
        emission = object()
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),  # type: ignore[arg-type]
            emissions=(emission,),  # type: ignore[arg-type]
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(action,),
        ) as top_level_actions, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=batch,
        ) as emission_batch, patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action_emissions",
            return_value=(),
        ) as flatten_emissions, patch(
            "grimace._south_star1.writer_transitions._active_emitted_transitions",
            side_effect=AssertionError("active-emitted path should not run"),
        ):
            result = writer_transitions._scheduled_writer_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertEqual(result, ())
        top_level_actions.assert_called_once_with(state)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            (action,),
        )
        flatten_emissions.assert_called_once_with(())

    def test_scheduled_writer_transitions_falls_through_after_empty_top_level_actions(self) -> None:
        prepared = object()
        context = object()
        state = SimpleNamespace(
            active=SimpleNamespace(
                atom=AtomId(4),
            ),
        )
        survivor = object()
        transition = object()
        selected_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(survivor,),  # type: ignore[arg-type]
        )
        closure_endpoint_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=selected_batch,
                open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                    actions=(),
                    emissions=(),
                    surviving_emissions=(),
                ),
                surviving_emissions=(survivor,),  # type: ignore[arg-type]
            )
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=AtomId(4),
            closure_endpoint_decision=closure_endpoint_decision,
        )
        active_decision = writer_transitions._active_emitted_closure_decision(
            closure_endpoint_decision,
            graph_policy_decision=policy,
        )
        active_outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=policy,
            schedule_decision=active_decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._top_level_scheduled_actions",
            return_value=(),
        ) as top_level_actions, patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            return_value=active_outcome,
        ) as active_emitted_outcome, patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action_emissions",
            return_value=(transition,),
        ) as flatten_emissions:
            result = writer_transitions._scheduled_writer_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
            )

        self.assertEqual(result, (transition,))
        top_level_actions.assert_called_once_with(state)
        active_emitted_outcome.assert_called_once_with(
            prepared,
            state,
            context,
            AtomId(4),
        )
        flatten_emissions.assert_called_once_with((survivor,))

    def test_scheduled_action_emissions_preserve_action_identity(self) -> None:
        prepared = object()
        state = object()
        context = object()
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        transition = object()

        with patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action",
            side_effect=((transition,), ()),
        ) as emit_action:
            emissions = writer_transitions._scheduled_action_emissions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                (first_action, second_action),
            )

        self.assertEqual(len(emissions), 2)
        self.assertIs(emissions[0].action, first_action)
        self.assertEqual(emissions[0].transitions, (transition,))
        self.assertIs(emissions[1].action, second_action)
        self.assertEqual(emissions[1].transitions, ())
        self.assertEqual(emit_action.call_count, 2)
        self.assertEqual(
            emit_action.call_args_list[0].args,
            (prepared, state, context, first_action),
        )
        self.assertEqual(
            emit_action.call_args_list[1].args,
            (prepared, state, context, second_action),
        )

    def test_scheduled_action_emission_exposes_graph_action_surface(self) -> None:
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
            boundary_atom=AtomId(0),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        action = writer_transitions._enter_inline_child_action(AtomId(0), child)
        transition = object()
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(transition,),  # type: ignore[arg-type]
        )

        surface = emission.graph_action_surface

        self.assertIs(
            surface.kind,
            writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
        )
        self.assertEqual(surface.active_atom, AtomId(0))
        self.assertEqual(surface.bond, BondId(1))
        self.assertEqual(surface.partner_atom, AtomId(2))
        self.assertEqual(surface.boundary_atom, AtomId(0))
        self.assertEqual(surface.attachment_id, 7)
        self.assertTrue(emission.survived)
        self.assertEqual(emission.transitions, (transition,))

    def test_zero_transition_scheduled_action_emission_still_exposes_surface(self) -> None:
        action = writer_transitions._finish_active_action(AtomId(3))
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(),
        )

        self.assertFalse(emission.survived)
        self.assertIs(
            emission.graph_action_surface.kind,
            writer_transitions._WriterScheduledActionKind.FINISH_ACTIVE,
        )
        self.assertEqual(emission.graph_action_surface.active_atom, AtomId(3))

    def test_scheduled_action_emission_batch_exposes_all_and_surviving_surfaces(self) -> None:
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        transition = object()
        first_emission = writer_transitions._WriterScheduledActionEmission(
            action=first_action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        second_emission = writer_transitions._WriterScheduledActionEmission(
            action=second_action,
            transitions=(),
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(first_action, second_action),
            emissions=(first_emission, second_emission),
            surviving_emissions=(first_emission,),
        )

        self.assertEqual(
            batch.graph_action_surfaces,
            (
                first_emission.graph_action_surface,
                second_emission.graph_action_surface,
            ),
        )
        self.assertEqual(
            batch.surviving_graph_action_surfaces,
            (first_emission.graph_action_surface,),
        )

    def test_scheduled_action_batch_exposes_surviving_next_token_frontier(self) -> None:
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        first_transition = SimpleNamespace(emitted_text="C")
        second_transition = SimpleNamespace(emitted_text="N")
        third_transition = SimpleNamespace(emitted_text="C")
        first_emission = writer_transitions._WriterScheduledActionEmission(
            action=first_action,
            transitions=(
                first_transition,  # type: ignore[arg-type]
                second_transition,  # type: ignore[arg-type]
            ),
        )
        second_emission = writer_transitions._WriterScheduledActionEmission(
            action=second_action,
            transitions=(third_transition,),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(first_action, second_action),
            emissions=(first_emission, second_emission),
            surviving_emissions=(first_emission, second_emission),
        )

        frontier = batch.surviving_next_token_frontier

        self.assertEqual(
            tuple(entry.emitted_text for entry in frontier),
            ("C", "N"),
        )
        self.assertEqual(len(frontier[0].supports), 2)
        self.assertEqual(len(frontier[1].supports), 1)
        self.assertIs(frontier[0].supports[0].emission, first_emission)
        self.assertIs(frontier[0].supports[0].transition, first_transition)
        self.assertIs(frontier[0].supports[1].emission, second_emission)
        self.assertIs(frontier[0].supports[1].transition, third_transition)
        self.assertIs(frontier[1].supports[0].emission, first_emission)
        self.assertIs(frontier[1].supports[0].transition, second_transition)
        self.assertEqual(
            frontier[0].supports[0].graph_action_surface,
            first_emission.graph_action_surface,
        )
        self.assertEqual(
            frontier[0].transitions,
            (first_transition, third_transition),
        )
        self.assertEqual(frontier[1].transitions, (second_transition,))

    def test_scheduled_action_batch_frontier_uses_only_surviving_emissions(self) -> None:
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        transition = SimpleNamespace(emitted_text="C")
        first_emission = writer_transitions._WriterScheduledActionEmission(
            action=first_action,
            transitions=(),
        )
        second_emission = writer_transitions._WriterScheduledActionEmission(
            action=second_action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(first_action, second_action),
            emissions=(first_emission, second_emission),
            surviving_emissions=(second_emission,),
        )

        frontier = batch.surviving_next_token_frontier

        self.assertEqual(len(frontier), 1)
        self.assertEqual(frontier[0].emitted_text, "C")
        self.assertEqual(len(frontier[0].supports), 1)
        self.assertIs(frontier[0].supports[0].emission, second_emission)

    def test_scheduled_graph_action_surface_policy_family_classifies_action_families(self) -> None:
        pending = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.CONSUME_PENDING_ENTRY,
            active_atom=AtomId(0),
            pending_entry=True,
        )
        root = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.EMIT_ROOT_ATOM,
            active_atom=AtomId(0),
        )
        finish = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.FINISH_ACTIVE,
            active_atom=AtomId(0),
        )
        acyclic = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
            active_atom=AtomId(0),
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        cyclic = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_BRANCH,
            active_atom=AtomId(0),
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        generic_tree = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
            active_atom=AtomId(0),
        )
        closure_open = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
        )
        closure_pair = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.PAIR_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
        )

        self.assertIs(
            pending.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.PENDING_ENTRY,
        )
        self.assertIs(
            root.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.ROOT_ATOM,
        )
        self.assertIs(
            finish.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.FINISH_ACTIVE,
        )
        self.assertIs(
            acyclic.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.ACYCLIC_TREE_ENTRY,
        )
        self.assertIs(
            cyclic.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY,
        )
        self.assertIs(
            generic_tree.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.TREE_ENTRY,
        )
        self.assertIs(
            closure_open.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.CLOSURE_OPEN,
        )
        self.assertIs(
            closure_pair.policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.CLOSURE_PAIR,
        )

    def test_graph_action_surface_residual_attachment_policy_key_uses_residual_surfaces(self) -> None:
        closure_open = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
            attachment_id=7,
        )
        cyclic = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
            active_atom=AtomId(0),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        acyclic = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_BRANCH,
            active_atom=AtomId(0),
            attachment_id=8,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        finish = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.FINISH_ACTIVE,
            active_atom=AtomId(0),
            attachment_id=7,
        )
        closure_pair = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.PAIR_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
            attachment_id=7,
        )

        self.assertEqual(
            closure_open.residual_attachment_policy_key,
            writer_transitions._WriterResidualAttachmentPolicyKey(AtomId(0), 7),
        )
        self.assertEqual(
            cyclic.residual_attachment_policy_key,
            writer_transitions._WriterResidualAttachmentPolicyKey(AtomId(0), 7),
        )
        self.assertEqual(
            acyclic.residual_attachment_policy_key,
            writer_transitions._WriterResidualAttachmentPolicyKey(AtomId(0), 8),
        )
        self.assertIsNone(finish.residual_attachment_policy_key)
        self.assertIsNone(closure_pair.residual_attachment_policy_key)

    def test_residual_attachment_policy_groups_preserve_order_and_group_by_key(self) -> None:
        closure_open_7 = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
            attachment_id=7,
        )
        finish_7 = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.FINISH_ACTIVE,
            active_atom=AtomId(0),
            attachment_id=7,
        )
        cyclic_tree_7 = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
            active_atom=AtomId(0),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        acyclic_tree_8 = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_BRANCH,
            active_atom=AtomId(0),
            attachment_id=8,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        closure_open_8 = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
            attachment_id=8,
        )

        groups = (
            writer_transitions
            ._residual_attachment_policy_groups_from_graph_action_surfaces(
                (
                    closure_open_7,
                    finish_7,
                    cyclic_tree_7,
                    acyclic_tree_8,
                    closure_open_8,
                ),
            )
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(
            groups[0].key,
            writer_transitions._WriterResidualAttachmentPolicyKey(AtomId(0), 7),
        )
        self.assertEqual(groups[0].surfaces, (closure_open_7, cyclic_tree_7))
        self.assertEqual(
            groups[1].key,
            writer_transitions._WriterResidualAttachmentPolicyKey(AtomId(0), 8),
        )
        self.assertEqual(groups[1].surfaces, (acyclic_tree_8, closure_open_8))
        self.assertEqual(groups[0].closure_open_surfaces, (closure_open_7,))
        self.assertEqual(groups[0].cyclic_tree_entry_surfaces, (cyclic_tree_7,))
        self.assertTrue(groups[0].has_closure_open_vs_cyclic_tree_entry_choice)
        self.assertEqual(groups[1].acyclic_tree_entry_surfaces, (acyclic_tree_8,))
        self.assertFalse(groups[1].has_closure_open_vs_cyclic_tree_entry_choice)

    def test_residual_attachment_policy_group_reports_owner_scope_for_closure_open_cyclic_choice(self) -> None:
        closure_open = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
            attachment_id=7,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        cyclic_tree = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
            active_atom=AtomId(0),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        active_owned = writer_transitions._WriterResidualAttachmentPolicyGroup(
            key=writer_transitions._WriterResidualAttachmentPolicyKey(AtomId(0), 7),
            surfaces=(closure_open, cyclic_tree),
        )

        self.assertEqual(
            active_owned.closure_open_owner_kinds,
            (WriterBoundaryOwnerKind.ACTIVE_ATOM,),
        )
        self.assertEqual(
            active_owned.cyclic_tree_entry_owner_kinds,
            (WriterBoundaryOwnerKind.ACTIVE_ATOM,),
        )
        self.assertEqual(
            active_owned.closure_open_vs_cyclic_tree_entry_owner_kinds,
            (
                WriterBoundaryOwnerKind.ACTIVE_ATOM,
                WriterBoundaryOwnerKind.ACTIVE_ATOM,
            ),
        )
        self.assertTrue(
            active_owned.has_active_atom_owned_closure_open_vs_cyclic_tree_entry_choice
        )
        self.assertFalse(
            active_owned
            .has_unsupported_owner_scope_closure_open_vs_cyclic_tree_entry_choice
        )

        branch_return_cyclic = replace(
            cyclic_tree,
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
        )
        branch_return_group = (
            writer_transitions._WriterResidualAttachmentPolicyGroup(
                key=writer_transitions._WriterResidualAttachmentPolicyKey(
                    AtomId(0),
                    7,
                ),
                surfaces=(closure_open, branch_return_cyclic),
            )
        )

        self.assertFalse(
            branch_return_group
            .has_active_atom_owned_closure_open_vs_cyclic_tree_entry_choice
        )
        self.assertTrue(
            branch_return_group
            .has_unsupported_owner_scope_closure_open_vs_cyclic_tree_entry_choice
        )

        missing_owner_closure = replace(
            closure_open,
            owner_kind=None,
        )
        missing_owner_group = (
            writer_transitions._WriterResidualAttachmentPolicyGroup(
                key=writer_transitions._WriterResidualAttachmentPolicyKey(
                    AtomId(0),
                    7,
                ),
                surfaces=(missing_owner_closure, cyclic_tree),
            )
        )

        self.assertFalse(
            missing_owner_group
            .has_active_atom_owned_closure_open_vs_cyclic_tree_entry_choice
        )
        self.assertTrue(
            missing_owner_group
            .has_unsupported_owner_scope_closure_open_vs_cyclic_tree_entry_choice
        )

    def test_active_emitted_graph_policy_blocker_validates_payload_shape(self) -> None:
        child_blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=AtomId(0),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        surface = writer_transitions._WriterScheduledGraphActionSurface(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
            active_atom=AtomId(0),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
        )
        group = writer_transitions._WriterResidualAttachmentPolicyGroup(
            key=writer_transitions._WriterResidualAttachmentPolicyKey(
                AtomId(0),
                7,
            ),
            surfaces=(surface,),
        )

        writer_transitions._WriterActiveEmittedGraphPolicyBlocker(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .CHILD_OBLIGATION
            ),
            child_blocker=child_blocker,
        )
        writer_transitions._WriterActiveEmittedGraphPolicyBlocker(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
            residual_group=group,
        )
        writer_transitions._WriterActiveEmittedGraphPolicyBlocker(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .MISSING_CLOSURE_OPEN_SUPPORT_EVIDENCE
            ),
            residual_group=group,
        )

        invalid_payloads = (
            {
                "kind": (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyBlockerKind
                    .CHILD_OBLIGATION
                ),
                "residual_group": group,
            },
            {
                "kind": (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyBlockerKind
                    .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
                ),
                "child_blocker": child_blocker,
            },
            {
                "kind": (
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyBlockerKind
                    .MISSING_CLOSURE_OPEN_SUPPORT_EVIDENCE
                ),
            },
        )

        for payload in invalid_payloads:
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._WriterActiveEmittedGraphPolicyBlocker(
                    **payload,
                )

            self.assertIs(
                raised.exception.kind,
                SouthStarErrorKind.INTERNAL_INVARIANT,
            )

    def test_residual_attachment_policy_emission_group_reports_dead_closure_open_support(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        closure_open = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        closure_open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            closure_open,
            label,
        )
        cyclic_child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        cyclic_action = writer_transitions._enter_inline_child_action(
            active_atom,
            cyclic_child,
        )
        closure_open_emission = writer_transitions._WriterScheduledActionEmission(
            action=closure_open_action,
            transitions=(),
        )
        cyclic_emission = writer_transitions._WriterScheduledActionEmission(
            action=cyclic_action,
            transitions=(SimpleNamespace(emitted_text="C"),),  # type: ignore[arg-type]
        )
        group = writer_transitions._WriterResidualAttachmentPolicyEmissionGroup(
            key=writer_transitions._WriterResidualAttachmentPolicyKey(
                active_atom,
                7,
            ),
            emissions=(closure_open_emission, cyclic_emission),
        )

        self.assertTrue(group.closure_open_was_considered)
        self.assertFalse(group.closure_open_support_survived)
        self.assertTrue(group.closure_open_support_dead)
        self.assertEqual(group.surviving_closure_open_emissions, ())
        self.assertEqual(
            group.surviving_cyclic_tree_entry_emissions,
            (cyclic_emission,),
        )

        surviving_closure_open_emission = (
            writer_transitions._WriterScheduledActionEmission(
                action=closure_open_action,
                transitions=(SimpleNamespace(emitted_text="1"),),  # type: ignore[arg-type]
            )
        )
        surviving_group = (
            writer_transitions._WriterResidualAttachmentPolicyEmissionGroup(
                key=writer_transitions._WriterResidualAttachmentPolicyKey(
                    active_atom,
                    7,
                ),
                emissions=(surviving_closure_open_emission, cyclic_emission),
            )
        )

        self.assertTrue(surviving_group.closure_open_was_considered)
        self.assertTrue(surviving_group.closure_open_support_survived)
        self.assertFalse(surviving_group.closure_open_support_dead)

    def test_residual_attachment_policy_emission_groups_preserve_order_and_skip_non_residual(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        closure_open = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        closure_open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            closure_open,
            label,
        )
        finish_action = writer_transitions._finish_active_action(active_atom)
        cyclic_child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        cyclic_action = writer_transitions._enter_inline_child_action(
            active_atom,
            cyclic_child,
        )
        acyclic_child = writer_transitions._WriterChildObligation(
            bond=BondId(3),
            child=AtomId(5),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=8,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        acyclic_action = writer_transitions._enter_inline_child_action(
            active_atom,
            acyclic_child,
        )
        closure_open_emission = writer_transitions._WriterScheduledActionEmission(
            action=closure_open_action,
            transitions=(),
        )
        finish_emission = writer_transitions._WriterScheduledActionEmission(
            action=finish_action,
            transitions=(SimpleNamespace(emitted_text=""),),  # type: ignore[arg-type]
        )
        cyclic_tree_emission = writer_transitions._WriterScheduledActionEmission(
            action=cyclic_action,
            transitions=(SimpleNamespace(emitted_text="C"),),  # type: ignore[arg-type]
        )
        acyclic_tree_emission = writer_transitions._WriterScheduledActionEmission(
            action=acyclic_action,
            transitions=(SimpleNamespace(emitted_text="N"),),  # type: ignore[arg-type]
        )

        groups = (
            writer_transitions
            ._residual_attachment_policy_emission_groups_from_scheduled_action_emissions(
                (
                    closure_open_emission,
                    finish_emission,
                    cyclic_tree_emission,
                    acyclic_tree_emission,
                ),
            )
        )

        self.assertEqual(len(groups), 2)
        self.assertEqual(
            groups[0].key,
            writer_transitions._WriterResidualAttachmentPolicyKey(active_atom, 7),
        )
        self.assertEqual(
            groups[0].emissions,
            (closure_open_emission, cyclic_tree_emission),
        )
        self.assertEqual(groups[0].closure_open_emissions, (closure_open_emission,))
        self.assertEqual(
            groups[0].cyclic_tree_entry_emissions,
            (cyclic_tree_emission,),
        )
        self.assertEqual(groups[0].surviving_emissions, (cyclic_tree_emission,))
        self.assertEqual(groups[0].surviving_closure_open_emissions, ())
        self.assertEqual(
            groups[0].surviving_cyclic_tree_entry_emissions,
            (cyclic_tree_emission,),
        )
        self.assertTrue(groups[0].has_closure_open_vs_cyclic_tree_entry_choice)
        self.assertEqual(
            groups[1].key,
            writer_transitions._WriterResidualAttachmentPolicyKey(active_atom, 8),
        )
        self.assertEqual(groups[1].emissions, (acyclic_tree_emission,))

    def test_scheduled_action_batch_exposes_all_and_surviving_residual_policy_emission_groups(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        closure_open = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        closure_open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            closure_open,
            label,
        )
        cyclic_child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        cyclic_action = writer_transitions._enter_inline_child_action(
            active_atom,
            cyclic_child,
        )
        closure_open_zero_emission = (
            writer_transitions._WriterScheduledActionEmission(
                action=closure_open_action,
                transitions=(),
            )
        )
        cyclic_tree_surviving_emission = (
            writer_transitions._WriterScheduledActionEmission(
                action=cyclic_action,
                transitions=(SimpleNamespace(emitted_text="C"),),  # type: ignore[arg-type]
            )
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(closure_open_action, cyclic_action),
            emissions=(
                closure_open_zero_emission,
                cyclic_tree_surviving_emission,
            ),
            surviving_emissions=(cyclic_tree_surviving_emission,),
        )

        all_groups = batch.residual_attachment_policy_emission_groups
        surviving_groups = batch.surviving_residual_attachment_policy_emission_groups

        self.assertEqual(len(all_groups), 1)
        self.assertEqual(
            all_groups[0].emissions,
            (
                closure_open_zero_emission,
                cyclic_tree_surviving_emission,
            ),
        )
        self.assertEqual(len(surviving_groups), 1)
        self.assertEqual(
            surviving_groups[0].emissions,
            (cyclic_tree_surviving_emission,),
        )
        self.assertEqual(surviving_groups[0].closure_open_emissions, ())
        self.assertEqual(
            surviving_groups[0].cyclic_tree_entry_emissions,
            (cyclic_tree_surviving_emission,),
        )

    def test_next_token_frontier_entry_exposes_policy_families_per_support(self) -> None:
        active_atom = AtomId(0)
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        label = WriterClosureLabel(value=1, text="1")
        closure_open = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(2),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        closure_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            closure_open,
            label,
        )
        child_transition = SimpleNamespace(emitted_text="1")
        closure_transition = SimpleNamespace(emitted_text="1")
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(child_transition,),  # type: ignore[arg-type]
        )
        closure_emission = writer_transitions._WriterScheduledActionEmission(
            action=closure_action,
            transitions=(closure_transition,),  # type: ignore[arg-type]
        )

        frontier = (
            writer_transitions
            ._next_token_frontier_from_scheduled_action_emissions(
                (child_emission, closure_emission),
            )
        )

        self.assertEqual(len(frontier), 1)
        self.assertEqual(frontier[0].emitted_text, "1")
        self.assertEqual(
            frontier[0].policy_families,
            (
                writer_transitions._WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY,
                writer_transitions._WriterGraphPolicyActionFamily.CLOSURE_OPEN,
            ),
        )
        self.assertIs(
            frontier[0].supports[0].policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY,
        )
        self.assertIs(
            frontier[0].supports[1].policy_family,
            writer_transitions._WriterGraphPolicyActionFamily.CLOSURE_OPEN,
        )

    def test_transitions_from_scheduled_actions_flattens_emissions(self) -> None:
        prepared = object()
        state = object()
        context = object()
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        first_transition = object()
        second_transition = object()
        emissions = (
            writer_transitions._WriterScheduledActionEmission(
                action=first_action,
                transitions=(first_transition,),
            ),
            writer_transitions._WriterScheduledActionEmission(
                action=second_action,
                transitions=(second_transition,),
            ),
        )

        with patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emissions",
            return_value=emissions,
        ) as scheduled_emissions:
            result = writer_transitions._transitions_from_scheduled_actions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                (first_action, second_action),
            )

        self.assertEqual(result, (first_transition, second_transition))
        scheduled_emissions.assert_called_once_with(
            prepared,
            state,
            context,
            (first_action, second_action),
        )

    def test_transitions_from_scheduled_action_emissions_flattens_transition_tuples(self) -> None:
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        first_transition = object()
        second_transition = object()
        third_transition = object()
        emissions = (
            writer_transitions._WriterScheduledActionEmission(
                action=first_action,
                transitions=(first_transition, second_transition),
            ),
            writer_transitions._WriterScheduledActionEmission(
                action=second_action,
                transitions=(),
            ),
            writer_transitions._WriterScheduledActionEmission(
                action=second_action,
                transitions=(third_transition,),
            ),
        )

        self.assertEqual(
            writer_transitions._transitions_from_scheduled_action_emissions(
                emissions,  # type: ignore[arg-type]
            ),
            (first_transition, second_transition, third_transition),
        )

    def test_surviving_scheduled_action_emissions_drop_zero_transition_emissions(self) -> None:
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        transition = object()
        first_emission = writer_transitions._WriterScheduledActionEmission(
            action=first_action,
            transitions=(transition,),
        )
        second_emission = writer_transitions._WriterScheduledActionEmission(
            action=second_action,
            transitions=(),
        )

        self.assertEqual(
            writer_transitions._surviving_scheduled_action_emissions(
                (first_emission, second_emission),
            ),
            (first_emission,),
        )

    def test_scheduled_action_emission_batch_preserves_actions_and_survivors(self) -> None:
        prepared = object()
        state = object()
        context = object()
        first_action = writer_transitions._finish_active_action(AtomId(0))
        second_action = writer_transitions._emit_root_atom_action(AtomId(1))
        first_emission = writer_transitions._WriterScheduledActionEmission(
            action=first_action,
            transitions=(object(),),
        )
        second_emission = writer_transitions._WriterScheduledActionEmission(
            action=second_action,
            transitions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emissions",
            return_value=(first_emission, second_emission),
        ) as scheduled_emissions, patch(
            "grimace._south_star1.writer_transitions._surviving_scheduled_action_emissions",
            return_value=(first_emission,),
        ) as surviving_emissions:
            batch = writer_transitions._scheduled_action_emission_batch(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                (first_action, second_action),
            )

        self.assertEqual(batch.actions, (first_action, second_action))
        self.assertEqual(batch.emissions, (first_emission, second_emission))
        self.assertEqual(batch.surviving_emissions, (first_emission,))
        scheduled_emissions.assert_called_once_with(
            prepared,
            state,
            context,
            (first_action, second_action),
        )
        surviving_emissions.assert_called_once_with(
            (first_emission, second_emission),
        )

    def test_active_emitted_graph_policy_selects_surviving_closure_without_children(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        action = writer_transitions._finish_active_action(active_atom)
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(emission,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            side_effect=AssertionError("child surface should not be computed"),
        ) as child_surface:
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
        )
        self.assertIsNone(decision.child_schedule_surface)
        self.assertFalse(decision.blocked)
        self.assertEqual(decision.blockers, ())
        self.assertEqual(decision.graph_policy_blockers, ())
        self.assertFalse(decision.graph_policy_blocked)
        child_surface.assert_not_called()

    def test_active_emitted_graph_policy_selects_child_after_empty_closure_survivors(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(action,),
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
        )
        self.assertIs(decision.child_schedule_surface, child_surface)
        self.assertEqual(
            decision.child_scheduled_actions,
            child_surface.scheduled_actions,
        )
        self.assertEqual(decision.graph_policy_blockers, ())
        self.assertFalse(decision.graph_policy_blocked)
        self.assertEqual(
            decision.graph_action_surfaces,
            child_surface.graph_action_surfaces,
        )
        self.assertEqual(decision.blockers, ())
        self.assertFalse(decision.blocked)

    def test_active_emitted_graph_policy_records_blocked_child_without_raising(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
        )
        self.assertTrue(decision.blocked)
        self.assertEqual(decision.blockers, child_surface.blockers)
        self.assertEqual(decision.child_scheduled_actions, ())
        self.assertEqual(decision.graph_action_surfaces, ())

    def test_active_emitted_graph_policy_exposes_child_graph_policy_blockers(self) -> None:
        active_atom = AtomId(7)
        child_blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(child_blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        blockers = policy.graph_policy_blockers

        self.assertTrue(policy.graph_policy_blocked)
        self.assertEqual(policy.blocked, policy.graph_policy_blocked)
        self.assertEqual(len(blockers), 1)
        self.assertIs(
            blockers[0].kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .CHILD_OBLIGATION
            ),
        )
        self.assertIs(blockers[0].child_blocker, child_blocker)
        self.assertIsNone(blockers[0].residual_group)

    def test_active_emitted_schedule_outcome_validates_scheduled_payload_shape(self) -> None:
        active_atom = AtomId(7)
        action = writer_transitions._finish_active_action(active_atom)
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        empty_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=batch,
            open_batch=empty_batch,
            surviving_emissions=(emission,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
        )
        schedule_decision = writer_transitions._active_emitted_closure_decision(
            closure_decision,
            graph_policy_decision=policy,
        )

        outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=policy,
            schedule_decision=schedule_decision,
        )

        self.assertIs(outcome.graph_policy_decision, policy)
        self.assertIs(outcome.schedule_decision, schedule_decision)

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterActiveEmittedScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedScheduleOutcomeKind
                    .SCHEDULED
                ),
                graph_policy_decision=policy,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

        other_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
        )
        other_schedule_decision = (
            writer_transitions._active_emitted_closure_decision(
                closure_decision,
                graph_policy_decision=other_policy,
            )
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterActiveEmittedScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedScheduleOutcomeKind
                    .SCHEDULED
                ),
                graph_policy_decision=policy,
                schedule_decision=other_schedule_decision,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_active_emitted_schedule_outcome_validates_blocked_payload_shape(self) -> None:
        active_atom = AtomId(7)
        child_blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(child_blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        empty_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        blocked_closure_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=empty_batch,
                open_batch=empty_batch,
                surviving_emissions=(),
            )
        )
        blocked_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=blocked_closure_decision,
            child_schedule_surface=child_surface,
        )

        outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.BLOCKED,
            graph_policy_decision=blocked_policy,
        )

        self.assertIsNone(outcome.schedule_decision)
        self.assertEqual(
            outcome.graph_policy_blockers,
            blocked_policy.graph_policy_blockers,
        )
        self.assertEqual(outcome.selected_transitions, ())
        self.assertEqual(outcome.selected_next_token_frontier, ())

        action = writer_transitions._finish_active_action(active_atom)
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        scheduled_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        scheduled_closure_decision = (
            writer_transitions._WriterClosureEndpointScheduleDecision(
                pair_batch=scheduled_batch,
                open_batch=empty_batch,
                surviving_emissions=(emission,),
            )
        )
        scheduled_policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=scheduled_closure_decision,
        )
        schedule_decision = writer_transitions._active_emitted_closure_decision(
            scheduled_closure_decision,
            graph_policy_decision=scheduled_policy,
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterActiveEmittedScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedScheduleOutcomeKind
                    .BLOCKED
                ),
                graph_policy_decision=blocked_policy,
                schedule_decision=schedule_decision,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterActiveEmittedScheduleOutcome(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedScheduleOutcomeKind
                    .BLOCKED
                ),
                graph_policy_decision=scheduled_policy,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_active_emitted_schedule_outcome_returns_closure_schedule(self) -> None:
        active_atom = AtomId(7)
        action = writer_transitions._finish_active_action(active_atom)
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        closure_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        empty_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=closure_batch,
            open_batch=empty_batch,
            surviving_emissions=(emission,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=AssertionError("closure policy should not emit child batch"),
        ) as emission_batch:
            outcome = writer_transitions._active_emitted_schedule_outcome(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            outcome.kind,
            writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
        )
        self.assertIs(outcome.graph_policy_decision, policy)
        self.assertIs(
            outcome.schedule_decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedScheduleDecisionKind
                .CLOSURE_ENDPOINT
            ),
        )
        emission_batch.assert_not_called()

    def test_active_emitted_schedule_outcome_returns_child_schedule_for_emittable_policy(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        transition = SimpleNamespace(emitted_text="C")
        child_emission = writer_transitions._WriterScheduledActionEmission(
            action=child_action,
            transitions=(transition,),  # type: ignore[arg-type]
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),
            surviving_emissions=(child_emission,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ) as emission_batch:
            outcome = writer_transitions._active_emitted_schedule_outcome(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            outcome.kind,
            writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
        )
        self.assertIs(
            outcome.schedule_decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertIs(outcome.schedule_decision.graph_policy_decision, policy)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            policy.child_scheduled_actions,
        )
        self.assertEqual(
            outcome.selected_transitions,
            outcome.schedule_decision.selected_transitions,
        )

    def test_active_emitted_schedule_outcome_returns_blocked_result_without_child_emission(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=AssertionError("blocked outcome should not emit child batch"),
        ) as emission_batch:
            outcome = writer_transitions._active_emitted_schedule_outcome(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            outcome.kind,
            writer_transitions._WriterActiveEmittedScheduleOutcomeKind.BLOCKED,
        )
        self.assertIsNone(outcome.schedule_decision)
        self.assertIs(outcome.graph_policy_decision, policy)
        self.assertEqual(
            outcome.graph_policy_blockers,
            policy.graph_policy_blockers,
        )
        emission_batch.assert_not_called()

    def test_active_emitted_schedule_decision_raises_from_blocked_schedule_outcome(self) -> None:
        active_atom = AtomId(7)
        child_blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(child_blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        empty_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=empty_batch,
            open_batch=empty_batch,
            surviving_emissions=(),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.BLOCKED,
            graph_policy_decision=policy,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            return_value=outcome,
        ):
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._active_emitted_schedule_decision(
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    active_atom,
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_active_emitted_schedule_decision_returns_scheduled_outcome_decision(self) -> None:
        active_atom = AtomId(7)
        action = writer_transitions._finish_active_action(active_atom)
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        closure_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        empty_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=closure_batch,
            open_batch=empty_batch,
            surviving_emissions=(emission,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
        )
        schedule_decision = writer_transitions._active_emitted_closure_decision(
            closure_decision,
            graph_policy_decision=policy,
        )
        outcome = writer_transitions._WriterActiveEmittedScheduleOutcome(
            kind=writer_transitions._WriterActiveEmittedScheduleOutcomeKind.SCHEDULED,
            graph_policy_decision=policy,
            schedule_decision=schedule_decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_schedule_outcome",
            return_value=outcome,
        ):
            decision = writer_transitions._active_emitted_schedule_decision(
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                object(),  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(decision, schedule_decision)

    def test_active_emitted_graph_policy_closure_surfaces_distinguish_considered_and_chosen(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=active_atom,
            label=label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(1),
            first_atom=AtomId(2),
            second_atom=active_atom,
            label=label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair_action = writer_transitions._pair_closure_endpoint_action(
            active_atom,
            writer_transitions._WriterClosurePairObligation(
                endpoint=endpoint,
                closure=closure,
            ),
        )
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(2),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        surface = writer_transitions._WriterClosureEndpointScheduleSurface(
            active_atom=active_atom,
            pair_actions=(pair_action,),
            open_actions=(open_action,),
        )
        pair_emission = writer_transitions._WriterScheduledActionEmission(
            action=pair_action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(pair_action,),
                emissions=(pair_emission,),
                surviving_emissions=(pair_emission,),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(pair_emission,),
            schedule_surface=surface,
        )

        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
        )

        self.assertEqual(
            policy.considered_graph_action_surfaces,
            closure_decision.considered_graph_action_surfaces,
        )
        self.assertEqual(
            policy.chosen_graph_action_surfaces,
            closure_decision.selected_graph_action_surfaces,
        )
        self.assertEqual(
            policy.graph_action_surfaces,
            policy.chosen_graph_action_surfaces,
        )

    def test_active_emitted_graph_policy_child_considered_surfaces_include_closure_before_child(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        self.assertEqual(
            policy.closure_considered_graph_action_surfaces,
            closure_decision.considered_graph_action_surfaces,
        )
        self.assertEqual(
            policy.child_considered_graph_action_surfaces,
            child_surface.graph_action_surfaces,
        )
        self.assertEqual(
            policy.considered_graph_action_surfaces,
            (
                *closure_decision.considered_graph_action_surfaces,
                *child_surface.graph_action_surfaces,
            ),
        )
        self.assertEqual(
            policy.chosen_graph_action_surfaces,
            child_surface.graph_action_surfaces,
        )

    def test_active_emitted_graph_policy_filters_surfaces_by_policy_family(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        self.assertEqual(
            policy.considered_graph_action_surfaces_for_policy_family(
                writer_transitions._WriterGraphPolicyActionFamily.CLOSURE_OPEN,
            ),
            closure_decision.considered_graph_action_surfaces,
        )
        self.assertEqual(
            policy.chosen_graph_action_surfaces_for_policy_family(
                writer_transitions._WriterGraphPolicyActionFamily.CYCLIC_TREE_ENTRY,
            ),
            child_surface.graph_action_surfaces,
        )
        self.assertEqual(
            policy.chosen_graph_action_surfaces_for_policy_family(
                writer_transitions._WriterGraphPolicyActionFamily.CLOSURE_OPEN,
            ),
            (),
        )

    def test_active_emitted_graph_policy_records_unresolved_residual_attachment_choice(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        groups = policy.unresolved_residual_attachment_policy_groups

        self.assertTrue(policy.blocked)
        self.assertEqual(policy.child_scheduled_actions, ())
        self.assertEqual(policy.chosen_graph_action_surfaces, ())
        self.assertEqual(
            groups,
            policy.considered_closure_open_vs_cyclic_tree_entry_groups,
        )
        self.assertEqual(len(groups), 1)
        self.assertEqual(
            groups[0].key,
            writer_transitions._WriterResidualAttachmentPolicyKey(active_atom, 11),
        )
        self.assertEqual(
            groups[0].closure_open_surfaces,
            closure_decision.considered_graph_action_surfaces,
        )
        self.assertEqual(
            groups[0].cyclic_tree_entry_surfaces,
            child_surface.graph_action_surfaces,
        )
        self.assertTrue(groups[0].has_closure_open_vs_cyclic_tree_entry_choice)
        self.assertEqual(
            policy.considered_closure_open_vs_cyclic_tree_entry_groups,
            (groups[0],),
        )

    def test_active_emitted_graph_policy_records_unsupported_owner_scope_residual_choice(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        self.assertTrue(policy.blocked)
        self.assertFalse(policy.emits_child_actions)
        self.assertEqual(policy.child_scheduled_actions, ())
        self.assertTrue(
            policy.unsupported_owner_scope_residual_attachment_policy_groups
        )
        blockers = policy.graph_policy_blockers
        self.assertTrue(policy.graph_policy_blocked)
        self.assertEqual(policy.blocked, policy.graph_policy_blocked)
        self.assertEqual(len(blockers), 1)
        self.assertIs(
            blockers[0].kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
        )
        self.assertEqual(
            blockers[0].residual_group,
            policy.unsupported_owner_scope_residual_attachment_policy_groups[0],
        )
        self.assertEqual(
            blockers[0].residual_attachment_policy_key,
            writer_transitions._WriterResidualAttachmentPolicyKey(active_atom, 11),
        )
        self.assertEqual(
            policy.unresolved_residual_attachment_policy_groups,
            (),
        )
        self.assertEqual(
            policy.resolved_residual_attachment_policy_groups,
            (),
        )

    def test_active_emitted_graph_policy_selects_unresolved_residual_choice_before_child(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
            ),
        )
        self.assertIs(decision.child_schedule_surface, child_surface)
        self.assertEqual(decision.child_scheduled_actions, ())
        self.assertTrue(decision.unresolved_residual_attachment_policy_groups)

    def test_active_emitted_schedule_decision_raises_for_unresolved_residual_choice_without_child_emission(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=AssertionError("unresolved policy should not emit child batch"),
        ) as emission_batch:
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._active_emitted_schedule_decision(
                    prepared,  # type: ignore[arg-type]
                    state,  # type: ignore[arg-type]
                    context,  # type: ignore[arg-type]
                    active_atom,
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)
        emission_batch.assert_not_called()

    def test_active_emitted_schedule_decision_raises_for_unsupported_owner_scope_without_child_emission(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=AssertionError(
                "unsupported owner scope should not emit child batch"
            ),
        ) as emission_batch:
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._active_emitted_schedule_decision(
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    object(),  # type: ignore[arg-type]
                    active_atom,
                )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)
        emission_batch.assert_not_called()

    def test_active_emitted_graph_policy_allows_child_after_dead_closure_open_support(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
        )
        self.assertTrue(decision.emits_child_actions)
        self.assertIs(decision.child_schedule_surface, child_surface)
        self.assertTrue(
            decision.support_dead_closure_open_vs_cyclic_tree_entry_groups
        )
        self.assertEqual(
            (
                decision
                .unsupported_owner_scope_closure_open_vs_cyclic_tree_entry_groups
            ),
            (),
        )
        self.assertEqual(
            decision.resolved_residual_attachment_policy_groups,
            decision.support_dead_closure_open_vs_cyclic_tree_entry_groups,
        )
        self.assertEqual(
            decision.unresolved_residual_attachment_policy_groups,
            (),
        )
        self.assertEqual(
            decision.unsupported_owner_scope_residual_attachment_policy_groups,
            (),
        )
        self.assertEqual(
            decision.missing_closure_open_support_evidence_groups,
            (),
        )
        self.assertEqual(
            decision.unresolved_closure_open_vs_cyclic_tree_entry_groups,
            (),
        )
        self.assertEqual(
            decision.child_scheduled_actions,
            child_surface.scheduled_actions,
        )

    def test_active_emitted_graph_policy_records_branch_return_owned_dead_closure_choice_as_unsupported_owner_scope(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
        )
        self.assertEqual(decision.resolved_residual_attachment_policy_groups, ())
        self.assertTrue(
            (
                decision
                .unsupported_owner_scope_closure_open_vs_cyclic_tree_entry_groups
            )
        )
        self.assertTrue(
            decision.unsupported_owner_scope_residual_attachment_policy_groups
        )
        self.assertEqual(
            decision.missing_closure_open_support_evidence_groups,
            (),
        )
        self.assertEqual(decision.unresolved_residual_attachment_policy_groups, ())
        self.assertTrue(decision.graph_policy_blocked)
        self.assertEqual(decision.blocked, decision.graph_policy_blocked)
        self.assertEqual(decision.child_scheduled_actions, ())

    def test_active_emitted_graph_policy_records_mixed_owner_dead_closure_choice_as_unsupported_owner_scope(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.BRANCH_RETURN,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNSUPPORTED_OWNER_SCOPE_RESIDUAL_ATTACHMENT_CHOICE
            ),
        )
        self.assertTrue(
            (
                decision
                .unsupported_owner_scope_closure_open_vs_cyclic_tree_entry_groups
            )
        )
        self.assertTrue(
            decision.unsupported_owner_scope_residual_attachment_policy_groups
        )
        self.assertEqual(
            decision.unresolved_residual_attachment_policy_groups,
            (),
        )
        self.assertEqual(decision.child_scheduled_actions, ())

    def test_active_emitted_graph_policy_remains_unresolved_without_closure_open_emission_evidence(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
            ),
        )
        self.assertTrue(decision.unresolved_residual_attachment_policy_groups)
        self.assertTrue(decision.missing_closure_open_support_evidence_groups)
        blockers = decision.graph_policy_blockers
        self.assertTrue(decision.graph_policy_blocked)
        self.assertEqual(decision.blocked, decision.graph_policy_blocked)
        self.assertEqual(len(blockers), 1)
        self.assertIs(
            blockers[0].kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyBlockerKind
                .MISSING_CLOSURE_OPEN_SUPPORT_EVIDENCE
            ),
        )
        self.assertEqual(
            blockers[0].residual_group,
            decision.unresolved_residual_attachment_policy_groups[0],
        )
        self.assertEqual(
            decision.unsupported_owner_scope_residual_attachment_policy_groups,
            (),
        )
        self.assertEqual(
            decision.support_dead_closure_open_vs_cyclic_tree_entry_groups,
            (),
        )
        self.assertEqual(decision.child_scheduled_actions, ())

    def test_active_emitted_schedule_decision_emits_child_after_dead_closure_open_support(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(),
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ) as emission_batch:
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertIs(
            decision.graph_policy_decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
        )
        self.assertTrue(decision.graph_policy_decision.emits_child_actions)
        self.assertTrue(
            (
                decision.graph_policy_decision
                .support_dead_closure_open_vs_cyclic_tree_entry_groups
            )
        )
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            child_surface.scheduled_actions,
        )

    def test_active_emitted_graph_policy_allows_child_when_closure_open_and_cyclic_tree_use_different_attachments(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=12,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedGraphPolicyDecisionKind.ACTIVE_CHILD,
        )
        self.assertIsNot(
            decision.kind,
            (
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD_AFTER_DEAD_CLOSURE_OPEN
            ),
        )
        self.assertTrue(decision.emits_child_actions)
        self.assertEqual(decision.resolved_residual_attachment_policy_groups, ())
        self.assertEqual(decision.graph_policy_blockers, ())
        self.assertFalse(decision.graph_policy_blocked)
        self.assertEqual(
            decision.child_scheduled_actions,
            child_surface.scheduled_actions,
        )
        self.assertEqual(
            decision.considered_closure_open_vs_cyclic_tree_entry_groups,
            (),
        )

    def test_active_emitted_graph_policy_selects_plain_active_child_without_resolved_residual_choice(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(0)
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=12,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ):
            decision = writer_transitions._active_emitted_graph_policy_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedGraphPolicyDecisionKind.ACTIVE_CHILD,
        )
        self.assertTrue(decision.emits_child_actions)
        self.assertEqual(decision.resolved_residual_attachment_policy_groups, ())

    def test_active_emitted_graph_policy_rejects_plain_active_child_for_dead_closure_open_choice(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._WriterActiveEmittedGraphPolicyDecision(
                kind=(
                    writer_transitions
                    ._WriterActiveEmittedGraphPolicyDecisionKind
                    .ACTIVE_CHILD
                ),
                active_atom=active_atom,
                closure_endpoint_decision=closure_decision,
                child_schedule_surface=child_surface,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_active_emitted_child_decision_rejects_unresolved_residual_policy_metadata(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        blocked_child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            blocked_child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(blocked_child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .UNRESOLVED_RESIDUAL_ATTACHMENT_CHOICE
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(),
            surviving_emissions=(),
        )

        with self.assertRaises(SouthStarError) as raised:
            writer_transitions._active_emitted_child_decision(
                closure_endpoint_decision=closure_decision,
                child_schedule_surface=child_surface,
                child_batch=child_batch,
                graph_policy_decision=policy,
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.INTERNAL_INVARIANT)

    def test_active_emitted_graph_policy_chosen_residual_groups_use_chosen_surfaces(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=11,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        open_emission = writer_transitions._WriterScheduledActionEmission(
            action=open_action,
            transitions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(open_emission,),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        child = writer_transitions._WriterChildObligation(
            bond=BondId(2),
            child=AtomId(4),
            boundary_atom=active_atom,
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=12,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CYCLIC_TREE_ENTRY
            ),
        )
        child_action = writer_transitions._enter_inline_child_action(
            active_atom,
            child,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(child,),
            scheduled_actions=(child_action,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        chosen = policy.chosen_residual_attachment_policy_groups

        self.assertEqual(len(chosen), 1)
        self.assertEqual(chosen[0].closure_open_surfaces, ())
        self.assertEqual(
            chosen[0].cyclic_tree_entry_surfaces,
            child_surface.graph_action_surfaces,
        )

    def test_active_emitted_graph_policy_blocked_child_chooses_no_graph_surfaces(self) -> None:
        active_atom = AtomId(0)
        label = WriterClosureLabel(value=1, text="1")
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=active_atom,
            second_atom=AtomId(3),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        open_action = writer_transitions._open_closure_endpoint_action(
            active_atom,
            open_obligation,
            label,
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(open_action,),
                emissions=(
                    writer_transitions._WriterScheduledActionEmission(
                        action=open_action,
                        transitions=(),
                    ),
                ),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
            schedule_surface=writer_transitions._WriterClosureEndpointScheduleSurface(
                active_atom=active_atom,
                pair_actions=(),
                open_actions=(open_action,),
            ),
        )
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        self.assertTrue(policy.blocked)
        self.assertEqual(policy.chosen_graph_action_surfaces, ())
        self.assertEqual(policy.graph_action_surfaces, ())
        self.assertEqual(
            policy.considered_graph_action_surfaces,
            (
                *closure_decision.considered_graph_action_surfaces,
                *child_surface.graph_action_surfaces,
            ),
        )

    def test_active_emitted_schedule_decision_selects_surviving_closure_batch(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        pair_action = object()
        pair_emission = object()
        pair_survivor = object()
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(pair_action,),  # type: ignore[arg-type]
            emissions=(pair_emission,),  # type: ignore[arg-type]
            surviving_emissions=(pair_survivor,),  # type: ignore[arg-type]
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(pair_survivor,),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ) as closure_schedule, patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            side_effect=AssertionError("child policy should not run"),
        ):
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.CLOSURE_ENDPOINT,
        )
        self.assertEqual(decision.closure_batch.actions, (pair_action,))
        self.assertEqual(decision.closure_batch.emissions, (pair_emission,))
        self.assertEqual(decision.closure_batch.surviving_emissions, (pair_survivor,))
        self.assertIs(decision.selected_batch, decision.closure_batch)
        self.assertIsNone(decision.child_batch)
        self.assertIsNone(decision.child_schedule_surface)
        closure_schedule.assert_called_once_with(
            prepared,
            state,
            context,
        )

    def test_active_emitted_schedule_decision_uses_closure_endpoint_decision(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        pair_action = object()
        open_action = object()
        pair_emission = object()
        open_emission = object()
        open_survivor = object()
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(pair_action,),  # type: ignore[arg-type]
            emissions=(pair_emission,),  # type: ignore[arg-type]
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(open_action,),  # type: ignore[arg-type]
            emissions=(open_emission,),  # type: ignore[arg-type]
            surviving_emissions=(open_survivor,),  # type: ignore[arg-type]
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(open_survivor,),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ) as closure_schedule, patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            side_effect=AssertionError("child policy should not run"),
        ):
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.CLOSURE_ENDPOINT,
        )
        self.assertEqual(
            decision.selected_batch.actions,
            (pair_action, open_action),
        )
        self.assertEqual(
            decision.selected_batch.emissions,
            (pair_emission, open_emission),
        )
        self.assertEqual(
            decision.selected_batch.surviving_emissions,
            (open_survivor,),
        )
        closure_schedule.assert_called_once_with(
            prepared,
            state,
            context,
        )

    def test_active_emitted_schedule_decision_selects_child_batch_after_zero_closure_survivors(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)

        child_action = writer_transitions._finish_active_action(active_atom)
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(),
            scheduled_actions=(child_action,),
        )

        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(object(),),  # type: ignore[arg-type]
            surviving_emissions=(object(),),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ) as closure_schedule, patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ) as child_surface_from_context, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ) as emission_batch:
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertEqual(decision.closure_batch.actions, ())
        self.assertEqual(decision.closure_batch.emissions, ())
        self.assertEqual(decision.closure_batch.surviving_emissions, ())
        self.assertIs(decision.child_batch, child_batch)
        self.assertIs(decision.child_schedule_surface, child_surface)
        self.assertIs(decision.selected_batch, child_batch)
        closure_schedule.assert_called_once_with(
            prepared,
            state,
            context,
        )
        child_surface_from_context.assert_called_once_with(
            context,
            state,
            active_atom,
        )
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            (child_action,),
        )

    def test_active_emitted_schedule_decision_threads_child_surface_after_empty_closure_survivors(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        child_action = writer_transitions._finish_active_action(active_atom)
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(),
            scheduled_actions=(child_action,),
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(object(),),  # type: ignore[arg-type]
            surviving_emissions=(object(),),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ) as emission_batch:
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertIs(decision.child_schedule_surface, child_surface)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            child_surface.scheduled_actions,
        )

    def test_active_emitted_schedule_decision_does_not_compute_child_surface_when_closure_survives(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        survivor = object()
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(survivor,),  # type: ignore[arg-type]
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(survivor,),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            side_effect=AssertionError("child surface should not be computed"),
        ) as child_surface:
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.CLOSURE_ENDPOINT,
        )
        self.assertIsNone(decision.child_schedule_surface)
        child_surface.assert_not_called()

    def test_active_emitted_schedule_decision_raises_from_child_surface_blockers_without_emitting_child_batch(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_schedule_surface_from_context",
            return_value=child_surface,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=AssertionError("child batch should not emit"),
        ) as emission_batch:
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._active_emitted_schedule_decision(
                    prepared,  # type: ignore[arg-type]
                    state,  # type: ignore[arg-type]
                    context,  # type: ignore[arg-type]
                    active_atom,
                )

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.UNSUPPORTED_POLICY,
        )
        emission_batch.assert_not_called()

    def test_active_emitted_schedule_decision_raises_from_blocked_graph_policy_without_child_emission(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        blocker = writer_transitions._WriterChildObligationBlocker(
            kind=(
                writer_transitions._WriterChildObligationBlockerKind
                .MULTI_INCIDENCE_RESIDUAL_ATTACHMENT
            ),
            atom=active_atom,
            attachment_id=7,
            attachment_action_kind=WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(blocker,),
            child_obligations=(),
            scheduled_actions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
        )
        policy_decision = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .BLOCKED_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            side_effect=AssertionError("child batch should not emit"),
        ) as emission_batch:
            with self.assertRaises(SouthStarError) as raised:
                writer_transitions._active_emitted_schedule_decision(
                    prepared,  # type: ignore[arg-type]
                    state,  # type: ignore[arg-type]
                    context,  # type: ignore[arg-type]
                    active_atom,
                )

        self.assertIs(
            raised.exception.kind,
            SouthStarErrorKind.UNSUPPORTED_POLICY,
        )
        emission_batch.assert_not_called()

    def test_active_emitted_schedule_decision_emits_child_actions_from_graph_policy(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        child_action = writer_transitions._finish_active_action(active_atom)
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(),
            scheduled_actions=(child_action,),  # type: ignore[arg-type]
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
        )
        policy_decision = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),  # type: ignore[arg-type]
            emissions=(object(),),  # type: ignore[arg-type]
            surviving_emissions=(object(),),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ) as emission_batch:
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            policy_decision.child_scheduled_actions,
        )
        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertIs(decision.child_schedule_surface, child_surface)
        self.assertIs(decision.child_batch, child_batch)

    def test_active_emitted_schedule_decision_threads_closure_graph_policy_decision(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        action = writer_transitions._finish_active_action(active_atom)
        emission = writer_transitions._WriterScheduledActionEmission(
            action=action,
            transitions=(object(),),  # type: ignore[arg-type]
        )
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(action,),
            emissions=(emission,),
            surviving_emissions=(emission,),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(emission,),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .CLOSURE_ENDPOINT
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy,
        ):
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.CLOSURE_ENDPOINT,
        )
        self.assertIs(decision.graph_policy_decision, policy)

    def test_active_emitted_schedule_decision_threads_child_graph_policy_decision(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        child_action = writer_transitions._finish_active_action(active_atom)
        child_surface = writer_transitions._WriterActiveChildScheduleSurface(
            active_atom=active_atom,
            blockers=(),
            child_obligations=(),
            scheduled_actions=(child_action,),  # type: ignore[arg-type]
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            open_batch=writer_transitions._WriterScheduledActionEmissionBatch(
                actions=(),
                emissions=(),
                surviving_emissions=(),
            ),
            surviving_emissions=(),
        )
        policy = writer_transitions._WriterActiveEmittedGraphPolicyDecision(
            kind=(
                writer_transitions
                ._WriterActiveEmittedGraphPolicyDecisionKind
                .ACTIVE_CHILD
            ),
            active_atom=active_atom,
            closure_endpoint_decision=closure_decision,
            child_schedule_surface=child_surface,
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),  # type: ignore[arg-type]
            emissions=(object(),),  # type: ignore[arg-type]
            surviving_emissions=(object(),),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._active_emitted_graph_policy_decision",
            return_value=policy,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ) as emission_batch:
            decision = writer_transitions._active_emitted_schedule_decision(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertIs(
            decision.kind,
            writer_transitions._WriterActiveEmittedScheduleDecisionKind.ACTIVE_CHILD,
        )
        self.assertIs(decision.graph_policy_decision, policy)
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            policy.child_scheduled_actions,
        )

    def test_active_emitted_scheduler_does_not_compute_children_when_closure_transition_survives(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        surviving_closure_emission = object()
        closure_transition = object()
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(surviving_closure_emission,),  # type: ignore[arg-type]
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(surviving_closure_emission,),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ) as closure_schedule, patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action_emissions",
            return_value=(closure_transition,),
        ) as flatten_emissions, patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            side_effect=AssertionError("child blockers were computed too early"),
        ), patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            side_effect=AssertionError("child obligations were computed too early"),
        ):
            result = writer_transitions._active_emitted_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertEqual(result, (closure_transition,))
        closure_schedule.assert_called_once_with(
            prepared,
            state,
            context,
        )
        flatten_emissions.assert_called_once_with((surviving_closure_emission,))

    def test_active_emitted_scheduler_computes_children_after_empty_closure_transitions(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)
        child_obligation = object()
        child_action = writer_transitions._finish_active_action(active_atom)
        child_emission = object()
        surviving_child_emission = object()
        child_transition = object()
        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(child_emission,),  # type: ignore[arg-type]
            surviving_emissions=(surviving_child_emission,),  # type: ignore[arg-type]
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ) as closure_schedule, patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ) as emission_batch, patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action_emissions",
            return_value=(child_transition,),
        ) as flatten_emissions, patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(),
        ) as child_blockers, patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            return_value=(child_obligation,),
        ) as child_obligations, patch(
            "grimace._south_star1.writer_transitions._active_child_scheduled_actions",
            return_value=(child_action,),
        ) as child_actions:
            result = writer_transitions._active_emitted_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertEqual(result, (child_transition,))
        closure_schedule.assert_called_once_with(
            prepared,
            state,
            context,
        )
        emission_batch.assert_called_once_with(
            prepared,
            state,
            context,
            (child_action,),
        )
        flatten_emissions.assert_called_once_with(
            (surviving_child_emission,),
        )
        child_blockers.assert_called_once_with(
            context,
            active_atom,
        )
        child_obligations.assert_called_once_with(
            context,
            state,
            active_atom,
        )
        child_actions.assert_called_once_with(
            active_atom,
            (child_obligation,),
        )

    def test_active_emitted_child_fallback_returns_empty_when_no_child_emissions_survive(self) -> None:
        prepared = object()
        state = object()
        context = object()
        active_atom = AtomId(7)

        child_obligation = object()
        child_action = writer_transitions._finish_active_action(active_atom)

        pair_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        open_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(),
            emissions=(),
            surviving_emissions=(),
        )
        closure_decision = writer_transitions._WriterClosureEndpointScheduleDecision(
            pair_batch=pair_batch,
            open_batch=open_batch,
            surviving_emissions=(),
        )
        child_batch = writer_transitions._WriterScheduledActionEmissionBatch(
            actions=(child_action,),
            emissions=(),
            surviving_emissions=(),
        )

        with patch(
            "grimace._south_star1.writer_transitions._closure_endpoint_schedule_decision",
            return_value=closure_decision,
        ), patch(
            "grimace._south_star1.writer_transitions._scheduled_action_emission_batch",
            return_value=child_batch,
        ), patch(
            "grimace._south_star1.writer_transitions._transitions_from_scheduled_action_emissions",
            return_value=(),
        ) as flatten_emissions, patch(
            "grimace._south_star1.writer_transitions._child_obligation_blockers_for_atom",
            return_value=(),
        ), patch(
            "grimace._south_star1.writer_transitions._unblocked_child_obligations_from_context",
            return_value=(child_obligation,),
        ), patch(
            "grimace._south_star1.writer_transitions._active_child_scheduled_actions",
            return_value=(child_action,),
        ):
            result = writer_transitions._active_emitted_transitions(
                prepared,  # type: ignore[arg-type]
                state,  # type: ignore[arg-type]
                context,  # type: ignore[arg-type]
                active_atom,
            )

        self.assertEqual(result, ())
        flatten_emissions.assert_called_once_with(())

    def test_scheduled_action_rejects_finish_payload(self) -> None:
        child = writer_transitions._WriterChildObligation(
            bond=BondId(0),
            child=AtomId(1),
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterScheduledAction(
                kind=writer_transitions._WriterScheduledActionKind.FINISH_ACTIVE,
                parent=AtomId(0),
                child_obligation=child,
            )

    def test_scheduled_action_requires_child_payload_for_child_action(self) -> None:
        with self.assertRaises(SouthStarError):
            writer_transitions._WriterScheduledAction(
                kind=writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
                parent=AtomId(0),
            )

    def test_scheduled_action_rejects_root_atom_payload(self) -> None:
        child = writer_transitions._WriterChildObligation(
            bond=BondId(0),
            child=AtomId(1),
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterScheduledAction(
                kind=writer_transitions._WriterScheduledActionKind.EMIT_ROOT_ATOM,
                parent=AtomId(0),
                child_obligation=child,
            )

    def test_scheduled_action_accepts_root_atom_action(self) -> None:
        action = writer_transitions._emit_root_atom_action(AtomId(0))

        self.assertIs(
            action.kind,
            writer_transitions._WriterScheduledActionKind.EMIT_ROOT_ATOM,
        )
        self.assertEqual(action.parent, AtomId(0))

    def test_scheduled_action_rejects_wrong_payload_family(self) -> None:
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(0),
            first_atom=AtomId(0),
            second_atom=AtomId(1),
            label=WriterClosureLabel(value=1, text="1"),
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(0),
            first_atom=AtomId(0),
            second_atom=AtomId(1),
            label=endpoint.label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair = writer_transitions._WriterClosurePairObligation(
            endpoint=endpoint,
            closure=closure,
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterScheduledAction(
                kind=writer_transitions._WriterScheduledActionKind.OPEN_BRANCH,
                parent=AtomId(0),
                closure_pair_obligation=pair,
            )

    def test_scheduled_action_requires_closure_open_label(self) -> None:
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(0),
            first_atom=AtomId(0),
            second_atom=AtomId(1),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )

        with self.assertRaises(SouthStarError):
            writer_transitions._WriterScheduledAction(
                kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
                parent=AtomId(0),
                closure_open_obligation=open_obligation,
            )

    def test_scheduled_action_accepts_valid_payloads(self) -> None:
        child = writer_transitions._WriterChildObligation(
            bond=BondId(0),
            child=AtomId(1),
        )
        open_obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(1),
            first_atom=AtomId(0),
            second_atom=AtomId(2),
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        label = WriterClosureLabel(value=1, text="1")
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(2),
            first_atom=AtomId(0),
            second_atom=AtomId(3),
            label=label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(2),
            first_atom=AtomId(0),
            second_atom=AtomId(3),
            label=label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair = writer_transitions._WriterClosurePairObligation(
            endpoint=endpoint,
            closure=closure,
        )

        writer_transitions._WriterScheduledAction(
            kind=writer_transitions._WriterScheduledActionKind.FINISH_ACTIVE,
            parent=AtomId(0),
        )
        writer_transitions._WriterScheduledAction(
            kind=writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
            parent=AtomId(0),
            child_obligation=child,
        )
        writer_transitions._WriterScheduledAction(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_BRANCH,
            parent=AtomId(0),
            child_obligation=child,
        )
        writer_transitions._WriterScheduledAction(
            kind=writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
            parent=AtomId(0),
            closure_open_obligation=open_obligation,
            closure_open_label=label,
        )
        writer_transitions._WriterScheduledAction(
            kind=writer_transitions._WriterScheduledActionKind.PAIR_CLOSURE_ENDPOINT,
            parent=AtomId(0),
            closure_pair_obligation=pair,
        )

    def test_scheduled_action_requires_pending_entry_payload(self) -> None:
        with self.assertRaises(SouthStarError):
            writer_transitions._WriterScheduledAction(
                kind=writer_transitions._WriterScheduledActionKind.CONSUME_PENDING_ENTRY,
                parent=AtomId(0),
            )

    def test_scheduled_action_accepts_pending_entry_payload(self) -> None:
        pending = PendingWriterEntry(
            parent=AtomId(0),
            child=AtomId(1),
            bond=BondId(0),
            branch=False,
        )

        action = writer_transitions._consume_pending_entry_action(pending)

        self.assertIs(
            action.kind,
            writer_transitions._WriterScheduledActionKind.CONSUME_PENDING_ENTRY,
        )
        self.assertEqual(action.parent, AtomId(0))
        self.assertEqual(action.pending_entry, pending)

    def test_scheduled_graph_action_surface_for_child_action_carries_incidence_metadata(self) -> None:
        child = writer_transitions._WriterChildObligation(
            bond=BondId(1),
            child=AtomId(2),
            boundary_atom=AtomId(0),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
            attachment_id=7,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY
            ),
        )
        action = writer_transitions._enter_inline_child_action(
            AtomId(0),
            child,
        )

        surface = writer_transitions._scheduled_graph_action_surface(action)

        self.assertIs(
            surface.kind,
            writer_transitions._WriterScheduledActionKind.ENTER_INLINE_CHILD,
        )
        self.assertEqual(surface.active_atom, AtomId(0))
        self.assertEqual(surface.bond, BondId(1))
        self.assertEqual(surface.partner_atom, AtomId(2))
        self.assertEqual(surface.boundary_atom, AtomId(0))
        self.assertEqual(surface.attachment_id, 7)
        self.assertIs(
            surface.attachment_action_kind,
            WriterResidualAttachmentActionKind.ACYCLIC_TREE_ENTRY,
        )
        self.assertIs(surface.owner_kind, WriterBoundaryOwnerKind.ACTIVE_ATOM)
        self.assertFalse(surface.pending_entry)

    def test_scheduled_graph_action_surface_for_closure_open_carries_label_and_incidence_metadata(self) -> None:
        label = WriterClosureLabel(value=1, text="1")
        obligation = writer_transitions._WriterClosureOpenObligation(
            bond=BondId(3),
            first_atom=AtomId(0),
            second_atom=AtomId(4),
            attachment_id=9,
            attachment_action_kind=(
                WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY
            ),
            owner_kind=WriterBoundaryOwnerKind.ACTIVE_ATOM,
        )
        action = writer_transitions._open_closure_endpoint_action(
            AtomId(0),
            obligation,
            label,
        )

        surface = writer_transitions._scheduled_graph_action_surface(action)

        self.assertIs(
            surface.kind,
            writer_transitions._WriterScheduledActionKind.OPEN_CLOSURE_ENDPOINT,
        )
        self.assertEqual(surface.active_atom, AtomId(0))
        self.assertEqual(surface.bond, BondId(3))
        self.assertEqual(surface.partner_atom, AtomId(4))
        self.assertEqual(surface.boundary_atom, AtomId(0))
        self.assertEqual(surface.attachment_id, 9)
        self.assertIs(
            surface.attachment_action_kind,
            WriterResidualAttachmentActionKind.CLOSURE_OPEN_READY,
        )
        self.assertIs(surface.owner_kind, WriterBoundaryOwnerKind.ACTIVE_ATOM)
        self.assertIs(surface.closure_label, label)

    def test_scheduled_graph_action_surface_for_pending_entry_marks_pending_source(self) -> None:
        pending = PendingWriterEntry(
            parent=AtomId(0),
            child=AtomId(2),
            bond=BondId(1),
            branch=False,
        )
        action = writer_transitions._consume_pending_entry_action(pending)

        surface = writer_transitions._scheduled_graph_action_surface(action)

        self.assertIs(
            surface.kind,
            writer_transitions._WriterScheduledActionKind.CONSUME_PENDING_ENTRY,
        )
        self.assertEqual(surface.active_atom, AtomId(0))
        self.assertEqual(surface.bond, BondId(1))
        self.assertEqual(surface.partner_atom, AtomId(2))
        self.assertEqual(surface.boundary_atom, AtomId(0))
        self.assertTrue(surface.pending_entry)
        self.assertIsNone(surface.attachment_id)
        self.assertIsNone(surface.attachment_action_kind)
        self.assertIsNone(surface.owner_kind)

    def test_scheduled_graph_action_surface_for_closure_pair_carries_closure_label(self) -> None:
        label = WriterClosureLabel(value=1, text="1")
        endpoint = WriterOpenClosureEndpoint(
            bond=BondId(5),
            first_atom=AtomId(0),
            second_atom=AtomId(3),
            label=label,
            first_endpoint_text="1",
            first_endpoint_bond_text="",
        )
        closure = WriterClosedClosure(
            bond=BondId(5),
            first_atom=AtomId(0),
            second_atom=AtomId(3),
            label=label,
            first_endpoint_text="1",
            second_endpoint_text="1",
            first_endpoint_bond_text="",
            second_endpoint_bond_text="",
        )
        pair = writer_transitions._WriterClosurePairObligation(
            endpoint=endpoint,
            closure=closure,
        )
        action = writer_transitions._pair_closure_endpoint_action(
            AtomId(3),
            pair,
        )

        surface = writer_transitions._scheduled_graph_action_surface(action)

        self.assertIs(
            surface.kind,
            writer_transitions._WriterScheduledActionKind.PAIR_CLOSURE_ENDPOINT,
        )
        self.assertEqual(surface.active_atom, AtomId(3))
        self.assertEqual(surface.bond, BondId(5))
        self.assertEqual(surface.partner_atom, AtomId(0))
        self.assertEqual(surface.boundary_atom, AtomId(3))
        self.assertIs(surface.closure_label, label)

    def test_writer_shaped_acyclic_stereo_uses_writer_frontier(self) -> None:
        for facts in (tetrahedral_facts(), directional_facts()):
            with self.subTest(facts=facts):
                prepared = _prepare(facts)
                with _forbidden_exhaustive_routes():
                    support = enumerate_prepared_stereo_support(
                        prepared=prepared,
                        runtime_options=_writer_options(),
                    )

                self.assertGreater(support.distinct_count, 0)
                self.assertEqual(len(support.strings), support.distinct_count)

    def test_writer_state_key_excludes_rendered_payloads(self) -> None:
        fields = set(WriterState.__dataclass_fields__)

        self.assertNotIn("rendered", fields)
        self.assertNotIn("prefix", fields)
        self.assertNotIn("suffix", fields)
        key = next(
            iter(
                initial_writer_frontier_cursor(
                    _prepare(cco_facts()),
                    _writer_options(),
                ).support_state.states
            )
        )
        self.assertIsInstance(key, WriterStateKey)
        self.assertEqual(writer_state_key(writer_state_from_key(key)), key)

    def test_writer_state_active_frame_is_non_nullable_in_datamodel(self) -> None:
        source = inspect.getsource(writer_state_module)

        self.assertNotIn("active: WriterAtomFrame | None", source)
        self.assertNotIn('return ("none",)', source)

    def test_writer_frontier_cursor_api_deletes_unweighted_entry_points(self) -> None:
        self.assertFalse(hasattr(writer_frontier_module, "initial_writer_frontier"))
        self.assertFalse(hasattr(writer_frontier_module, "count_writer_witness_completions"))
        self.assertFalse(hasattr(writer_frontier_module, "writer_frontier_successors"))
        self.assertNotIn("initial_writer_frontier", writer_frontier_module.__all__)
        self.assertNotIn("count_writer_witness_completions", writer_frontier_module.__all__)
        self.assertNotIn("writer_frontier_successors", writer_frontier_module.__all__)

    def test_writer_frontier_cursor_uses_structural_key_ordering(self) -> None:
        cursor = initial_writer_frontier_cursor(
            _prepare(cco_facts()),
            _writer_options(),
        )
        keys = tuple(key for key, _ in reversed(cursor.weighted_states))
        reordered = WriterFrontierCursor(
            weighted_states=tuple((key, 1) for key in keys),
        )

        self.assertEqual(
            tuple(key for key, _ in reordered.weighted_states),
            tuple(sorted(keys, key=writer_state_key_sort_tuple)),
        )
        self.assertNotIn(
            "repr(",
            inspect.getsource(WriterFrontierCursor.__post_init__),
        )

    def test_initial_writer_frontier_cursor_rejects_exhaustive_options(self) -> None:
        prepared = _prepare(cco_facts())

        with self.assertRaises(SouthStarError) as caught:
            initial_writer_frontier_cursor(prepared, SouthStarRuntimeOptions())

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_initial_writer_frontier_cursor_invalid_root_raises_typed_error(self) -> None:
        prepared = _prepare(cco_facts())

        with self.assertRaises(SouthStarError) as caught:
            initial_writer_frontier_cursor(prepared, _writer_options(rooted_at_atom=99))

        self.assertIs(caught.exception.kind, SouthStarErrorKind.INVALID_FACTS)

    def test_missing_writer_bond_domain_fails_closed(self) -> None:
        facts = chain_facts(("C", "C"))
        prepared = prepare_south_star_mol_from_facts(
            facts,
            writer_surface=SouthStarWriterSurface(),
            policy=missing_bond_domain_policy(facts),
        )

        with self.assertRaises(SouthStarError) as caught:
            enumerate_prepared_stereo_support(
                prepared=prepared,
                runtime_options=_writer_options(),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_POLICY)

    def test_writer_modules_do_not_import_exhaustive_routes(self) -> None:
        forbidden = {
            "skeleton",
            "exhaustive_online_traversal",
            "online_stereo_witness",
            "online_search_vm",
        }

        for module_name in (
            "writer_events.py",
            "writer_graph_obligations.py",
            "writer_state.py",
            "writer_transitions.py",
            "writer_frontier.py",
            "writer_stereo.py",
            "writer_snapshot.py",
            "writer_support.py",
        ):
            with self.subTest(module=module_name):
                tree = ast.parse((SOUTH_STAR1_ROOT / module_name).read_text(encoding="utf-8"))
                imported: list[str] = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        imported.extend(
                            alias.name.rsplit(".", 1)[-1]
                            for alias in node.names
                            if alias.name.rsplit(".", 1)[-1] in forbidden
                        )
                    if isinstance(node, ast.ImportFrom):
                        module = node.module or ""
                        imported.extend(
                            part for part in forbidden if module.endswith(part)
                        )
                self.assertEqual(imported, [])


def _prepare(facts: MoleculeFacts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _prepare_with_policy(
    facts: MoleculeFacts,
    *,
    least_free_ring_labels: bool,
    ring_labels: tuple[RingLabel, ...] = (RingLabel(1), RingLabel(2)),
):
    prepared = _prepare(facts)
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
        policy=replace(
            prepared.policy,
            ring_labels=ring_labels,
            least_free_ring_labels=least_free_ring_labels,
        ),
    )


def _writer_options(*, rooted_at_atom: int = -1) -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def _only_choice(prepared, cursor, emitted_text: str):
    matches = tuple(
        choice
        for choice in writer_frontier_choices(prepared, cursor).choices
        if choice.emitted_text == emitted_text
    )
    assert len(matches) == 1
    return matches[0]


def _terminal_keys(prepared, cursor: WriterFrontierCursor) -> tuple[WriterStateKey, ...]:
    terminals: list[WriterStateKey] = []

    def rec(current: WriterFrontierCursor) -> None:
        choices = writer_frontier_choices(prepared, current)
        if choices.terminal is not None:
            terminals.extend(
                key
                for key, _ in choices.terminal.finalized_cursor.weighted_states
            )
        for choice in choices.choices:
            rec(choice.successor)

    rec(cursor)
    return tuple(terminals)


def _raw_initial_state(root: AtomId) -> WriterState:
    return WriterState(
        component_cursor=ComponentCursor(
            component_index=0,
            component_roots=(root,),
        ),
        active=WriterAtomFrame(
            atom=root,
            parent=None,
            incoming_bond=None,
            atom_emitted=False,
        ),
        branch_stack=(),
        visited_atoms=frozenset(),
        written_bonds=frozenset(),
        obligations=ObligationState(),
        ring_state=WriterRingState(),
        stereo_state=empty_writer_stereo_state(),
        policy_state=WriterPolicyState(),
    )


def _raw_emitted_root_state(root: AtomId) -> WriterState:
    return WriterState(
        component_cursor=ComponentCursor(
            component_index=0,
            component_roots=(root,),
        ),
        active=WriterAtomFrame(
            atom=root,
            parent=None,
            incoming_bond=None,
            atom_emitted=True,
        ),
        branch_stack=(),
        visited_atoms=frozenset((root,)),
        written_bonds=frozenset(),
        obligations=ObligationState(),
        ring_state=WriterRingState(),
        stereo_state=empty_writer_stereo_state(),
        policy_state=WriterPolicyState(),
    )


def _cyclopropane_terminal_open_closure_state() -> WriterState:
    label = WriterClosureLabel(value=1, text="1")
    return WriterState(
        component_cursor=ComponentCursor(
            component_index=0,
            component_roots=(AtomId(0),),
        ),
        active=WriterAtomFrame(
            atom=AtomId(2),
            parent=AtomId(1),
            incoming_bond=BondId(1),
            atom_emitted=True,
        ),
        branch_stack=(),
        visited_atoms=frozenset((AtomId(0), AtomId(1), AtomId(2))),
        written_bonds=frozenset((BondId(0), BondId(1))),
        obligations=ObligationState(),
        ring_state=WriterRingState(
            open_endpoints=(
                WriterOpenClosureEndpoint(
                    bond=BondId(2),
                    first_atom=AtomId(0),
                    second_atom=AtomId(2),
                    label=label,
                    first_endpoint_text="1",
                    first_endpoint_bond_text="",
                ),
            ),
            label_state=WriterRingLabelState(allocated=(label,)),
        ),
        stereo_state=empty_writer_stereo_state(),
        policy_state=WriterPolicyState(),
    )


def _cyclopropane_terminal_closed_closure_state() -> WriterState:
    label = WriterClosureLabel(value=1, text="1")
    return replace(
        _cyclopropane_terminal_open_closure_state(),
        ring_state=WriterRingState(
            closed_closures=(
                WriterClosedClosure(
                    bond=BondId(2),
                    first_atom=AtomId(0),
                    second_atom=AtomId(2),
                    label=label,
                    first_endpoint_text="1",
                    second_endpoint_text="1",
                    first_endpoint_bond_text="",
                    second_endpoint_bond_text="",
                ),
            ),
            label_state=WriterRingLabelState(reusable=(label,)),
        ),
    )


def _with_next_component_root(state: WriterState) -> WriterState:
    return replace(
        state,
        component_cursor=ComponentCursor(
            component_index=0,
            component_roots=(AtomId(0), AtomId(3)),
        ),
    )


@contextlib.contextmanager
def _forbidden_exhaustive_routes():
    paths = (
        "grimace._south_star1.skeleton.enumerate_traversal_skeletons",
        "grimace._south_star1.exhaustive_online_traversal.iter_exhaustive_online_traversal_traces",
        "grimace._south_star1.exhaustive_online_traversal.iter_prepared_exhaustive_online_traversal_traces",
        "grimace._south_star1.online_stereo_witness.iter_exhaustive_online_stereo_witnesses",
        "grimace._south_star1.online_search_vm.ExhaustiveOnlineSearchVM",
        "grimace._south_star1.skeleton.TraversalSkeleton",
        "grimace._south_star1.exhaustive_online_traversal.ExhaustiveTraversalTrace",
        "grimace._south_star1.skeleton._component_spanning_trees",
        "grimace._south_star1.exhaustive_online_traversal._iter_spanning_forest_choices_lazy",
    )
    with contextlib.ExitStack() as stack:
        for path in paths:
            stack.enter_context(
                patch(
                    path,
                    side_effect=AssertionError(f"writer-shaped called {path}"),
                )
            )
        yield


def chain_facts(symbols: tuple[str, ...]) -> MoleculeFacts:
    return MoleculeFacts(
        atoms=tuple(atom(index, symbol) for index, symbol in enumerate(symbols)),
        bonds=tuple(
            single_bond(index, index, index + 1)
            for index in range(len(symbols) - 1)
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(len(symbols))),
                bonds=tuple(BondId(index) for index in range(len(symbols) - 1)),
            ),
        ),
    )


def disconnected_co_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "O")),
        bonds=(),
        components=(
            ComponentFacts(id=ComponentId(0), atoms=(AtomId(0),), bonds=()),
            ComponentFacts(id=ComponentId(1), atoms=(AtomId(1),), bonds=()),
        ),
    )


def cyclopropane_plus_singleton_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C"), atom(3, "O")),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            single_bond(2, 2, 0),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
            ComponentFacts(
                id=ComponentId(1),
                atoms=(AtomId(3),),
                bonds=(),
            ),
        ),
    )


def duplicate_single_atom_policy() -> SmilesPolicy:
    return SmilesPolicy(
        ring_labels=(RingLabel(1),),
        annotation_mode=AnnotationMode.HARD,
        atom_text_domains=(
            AtomTextDomain(
                atom=AtomId(0),
                choices=(
                    AtomTextChoice(
                        name="carbon_a",
                        text_by_tetra=((TetraToken.NONE, "C"),),
                    ),
                    AtomTextChoice(
                        name="carbon_b",
                        text_by_tetra=((TetraToken.NONE, "C"),),
                    ),
                ),
            ),
        ),
        bond_text_domains=(),
    )


def missing_bond_domain_policy(facts: MoleculeFacts) -> SmilesPolicy:
    return SmilesPolicy(
        ring_labels=(RingLabel(1),),
        annotation_mode=AnnotationMode.HARD,
        atom_text_domains=tuple(
            AtomTextDomain(
                atom=item.id,
                choices=(
                    AtomTextChoice(
                        name=f"atom_{int(item.id)}",
                        text_by_tetra=((TetraToken.NONE, item.symbol),),
                    ),
                ),
            )
            for item in facts.atoms
        ),
        bond_text_domains=(),
    )


def unsupported_directional_implicit_h_facts() -> MoleculeFacts:
    facts = directional_facts()
    return replace(
        facts,
        ligand_occurrences=(
            LigandOccurrence(
                id=OccurrenceId(0),
                site=facts.ligand_occurrences[0].site,
                kind=LigandKind.IMPLICIT_H,
                atom=AtomId(0),
                bond=None,
            ),
            facts.ligand_occurrences[1],
        ),
    )


def two_atom_facts(
    left: str,
    right: str,
    order: BondOrder,
) -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, left), atom(1, right)),
        bonds=(bond(0, 0, 1, order),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1)),
                bonds=(BondId(0),),
            ),
        ),
    )


def cycle_plus_isolate_component_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C"), atom(3, "C")),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            single_bond(2, 2, 0),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2), AtomId(3)),
                bonds=(BondId(0), BondId(1), BondId(2)),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
