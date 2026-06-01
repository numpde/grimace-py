"""Tests for the writer-shaped state/frontier MVP."""

from __future__ import annotations

import ast
import contextlib
import inspect
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import grimace._south_star1.writer_frontier as writer_frontier_module
import grimace._south_star1.writer_state as writer_state_module
import grimace._south_star1.writer_transitions as writer_transitions
from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
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
from grimace._south_star1.writer_state import ComponentCursor
from grimace._south_star1.writer_state import ObligationState
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

        self.assertEqual(children, ((BondId(0), AtomId(1)),))

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
