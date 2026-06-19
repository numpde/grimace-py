"""Writer-owned residual stereo tests."""

from __future__ import annotations

from dataclasses import replace
import inspect
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import enumerate_prepared_stereo_support
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.policy import AtomTextChoice
from grimace._south_star1.policy import AtomTextDomain
from grimace._south_star1.policy import BondTextChoice
from grimace._south_star1.policy import BondTextDomain
from grimace._south_star1.policy import RingLabel
from grimace._south_star1.policy import SmilesPolicy
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.writer_capabilities import _WriterExecutionCapabilityKind
from grimace._south_star1.residual_constraints import ResidualStore
from grimace._south_star1.residual_constraints import DirectionalCarrierResidual
from grimace._south_star1.residual_constraints import DirectionalResidualFactor
from grimace._south_star1.residual_constraints import ResidualPropagationKind
from grimace._south_star1.residual_constraints import TetraResidualFactor
from grimace._south_star1.residual_constraints import add_factor_and_propagate
from grimace._south_star1.residual_constraints import direction_var
from grimace._south_star1.residual_constraints import tetra_var
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import StereoFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.facts import TetrahedralSiteFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_frontier import _initial_writer_transition_frontier_cursor
from grimace._south_star1.writer_frontier import _writer_frontier_choice_snapshot
from grimace._south_star1.writer_frontier import writer_frontier_choices
from grimace._south_star1.writer_events import WriterRingEndpointEmitted
from grimace._south_star1.writer_events import WriterRingEndpointPaired
from grimace._south_star1.writer_events import WriterAtomEmitted
from grimace._south_star1.writer_state import WriterClosureLabel
from grimace._south_star1.writer_state import WriterStereoState
from grimace._south_star1.writer_stereo import advance_writer_stereo_state
from grimace._south_star1.writer_stereo import advance_writer_stereo_state_with_evidence
from grimace._south_star1.writer_stereo import empty_writer_stereo_state
from grimace._south_star1.writer_stereo import initial_writer_stereo_state
from grimace._south_star1.writer_stereo import terminal_writer_stereo_state
import grimace._south_star1.writer_stereo as writer_stereo_module
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


class WriterStereoResidualTest(unittest.TestCase):
    def test_tetrahedral_stereo_prunes_invalid_atom_tokens(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(rooted_at_atom=1),
        )

        self.assertEqual(
            support.strings,
            ("F[C@@H](Br)Cl", "F[C@H](Cl)Br"),
        )
        self.assertEqual(support.distinct_count, 2)
        self.assertEqual(support.witness_count, 2)

    def test_tetra_token_emission_reports_residual_capabilities(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        state = initial_writer_stereo_state(prepared)

        outcome = advance_writer_stereo_state_with_evidence(
            prepared,
            state,
            (
                WriterAtomEmitted(
                    atom=AtomId(0),
                    text="[C@H]",
                    parent=AtomId(1),
                    tetra_token=TetraToken.AT,
                ),
            ),
        )

        self.assertIsNotNone(outcome.state)
        self.assertIn(
            _WriterExecutionCapabilityKind.TETRA_TOKEN_RESTRICTION,
            outcome.execution_capabilities,
        )
        self.assertIn(
            _WriterExecutionCapabilityKind.RESIDUAL_PROPAGATION,
            outcome.execution_capabilities,
        )

    def test_tetra_local_order_closure_reports_residual_capabilities(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        cursor = _initial_writer_transition_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=1),
        )
        for emitted_text in ("F", "[C@@H]", "(", "Br", ")"):
            choices = _writer_frontier_choice_snapshot(
                prepared,
                cursor,
                include_counts=False,
            )
            cursor = next(
                choice.successor
                for choice in choices.choices
                if choice.emitted_text == emitted_text
            )
        choices = _writer_frontier_choice_snapshot(
            prepared,
            cursor,
            include_counts=False,
        )
        choice = next(
            choice for choice in choices.choices if choice.emitted_text == "Cl"
        )

        self.assertIn(
            _WriterExecutionCapabilityKind.TETRA_LOCAL_ORDER_RESTRICTION,
            choice.execution_capabilities,
        )
        self.assertIn(
            _WriterExecutionCapabilityKind.RESIDUAL_PROPAGATION,
            choice.execution_capabilities,
        )
        self.assertIn(
            _WriterExecutionCapabilityKind.RESIDUAL_FACTOR_DISCHARGE,
            choice.execution_capabilities,
        )

    def test_ordinary_atom_emission_reports_no_residual_capabilities(self) -> None:
        prepared = _prepare(cco_facts())

        outcome = advance_writer_stereo_state_with_evidence(
            prepared,
            initial_writer_stereo_state(prepared),
            (
                WriterAtomEmitted(
                    atom=AtomId(1),
                    text="C",
                    parent=None,
                    tetra_token=TetraToken.NONE,
                ),
            ),
        )

        self.assertIsNotNone(outcome.state)
        self.assertFalse(outcome.execution_capabilities)

    def test_directional_carrier_emission_reports_residual_capabilities(self) -> None:
        prepared = _prepare(directional_facts())
        cursor = _initial_writer_transition_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=2),
        )
        choices = _writer_frontier_choice_snapshot(
            prepared,
            cursor,
            include_counts=False,
        )
        cursor = next(
            choice.successor
            for choice in choices.choices
            if choice.emitted_text == "F"
        )
        choices = _writer_frontier_choice_snapshot(
            prepared,
            cursor,
            include_counts=False,
        )
        choice = next(
            choice for choice in choices.choices if choice.emitted_text == "/"
        )

        self.assertIn(
            _WriterExecutionCapabilityKind.DIRECTIONAL_CARRIER_RESTRICTION,
            choice.execution_capabilities,
        )
        self.assertIn(
            _WriterExecutionCapabilityKind.DIRECTIONAL_SITE_COMPATIBILITY,
            choice.execution_capabilities,
        )
        self.assertIn(
            _WriterExecutionCapabilityKind.RESIDUAL_PROPAGATION,
            choice.execution_capabilities,
        )
        self.assertIn(
            _WriterExecutionCapabilityKind.RESIDUAL_FACTOR_DISCHARGE,
            choice.execution_capabilities,
        )

    def test_initial_writer_state_accepts_independent_tetra_sites(self) -> None:
        prepared = _prepare(_two_independent_tetra_facts())
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )
        key = cursor.weighted_states[0][0]

        self.assertEqual(
            sum(
                factor.key.kind == "tetra_site"
                for factor in key.stereo_state.residual_snapshot.factors
            ),
            2,
        )

    def test_tetra_frontier_counts_are_pruned_per_token(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=1),
        )
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        choices = writer_frontier_choices(prepared, after_f)

        self.assertEqual(
            tuple(
                (choice.emitted_text, choice.support_count, choice.completion_count)
                for choice in choices.choices
            ),
            (("[C@@H]", 1, 1), ("[C@H]", 1, 1)),
        )
        successor_key = choices.choices[0].successor.weighted_states[0][0]
        factors = successor_key.stereo_state.residual_snapshot.factors
        self.assertTrue(
            any(factor.key.kind == "tetra_site" for factor in factors)
        )

    def test_directional_stereo_prunes_invalid_carrier_marks(self) -> None:
        prepared = _prepare(directional_facts())
        support = enumerate_prepared_stereo_support(
            prepared=prepared,
            runtime_options=_writer_options(rooted_at_atom=2),
        )

        self.assertEqual(support.strings, ("F/C=C/Cl", "F\\C=C\\Cl"))
        self.assertEqual(support.distinct_count, 2)
        self.assertEqual(support.witness_count, 2)

    def test_directional_frontier_drops_zero_completion_mark_choice(self) -> None:
        prepared = _prepare(directional_facts())
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=2),
        )
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        choices = writer_frontier_choices(prepared, after_f)

        self.assertEqual(
            tuple(choice.emitted_text for choice in choices.choices),
            ("/", "\\"),
        )
        self.assertEqual(
            tuple(choice.completion_count for choice in choices.choices),
            (1, 1),
        )
        successor_key = choices.choices[0].successor.weighted_states[0][0]
        factors = successor_key.stereo_state.residual_snapshot.factors
        self.assertTrue(
            any(factor.key.kind == "directional_site" for factor in factors)
        )

    def test_ring_endpoint_event_creates_pending_ring_pair_factor(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        label = WriterClosureLabel(value=1, text="1")

        state = advance_writer_stereo_state(
            prepared,
            empty_writer_stereo_state(),
            (
                WriterRingEndpointEmitted(
                    bond=BondId(2),
                    endpoint_atom=AtomId(0),
                    partner_atom=AtomId(2),
                    label=label,
                    endpoint_text="1",
                    bond_text="",
                ),
            ),
        )

        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state.residual_snapshot, empty_writer_stereo_state().residual_snapshot)

    def test_ring_endpoint_event_rejects_label_value_text_mismatch(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        label = WriterClosureLabel(value=1, text="7")

        state = advance_writer_stereo_state(
            prepared,
            empty_writer_stereo_state(),
            (
                WriterRingEndpointEmitted(
                    bond=BondId(2),
                    endpoint_atom=AtomId(0),
                    partner_atom=AtomId(2),
                    label=label,
                    endpoint_text="7",
                    bond_text="",
                ),
            ),
        )

        self.assertIsNone(state)

    def test_ring_endpoint_event_rejects_label_outside_policy(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        label = WriterClosureLabel(value=10, text="%10")

        state = advance_writer_stereo_state(
            prepared,
            empty_writer_stereo_state(),
            (
                WriterRingEndpointEmitted(
                    bond=BondId(2),
                    endpoint_atom=AtomId(0),
                    partner_atom=AtomId(2),
                    label=label,
                    endpoint_text="%10",
                    bond_text="",
                ),
            ),
        )

        self.assertIsNone(state)

    def test_ring_endpoint_event_accepts_policy_domain_nonleast_label(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        label = WriterClosureLabel(value=2, text="2")

        state = advance_writer_stereo_state(
            prepared,
            empty_writer_stereo_state(),
            (
                WriterRingEndpointEmitted(
                    bond=BondId(2),
                    endpoint_atom=AtomId(0),
                    partner_atom=AtomId(2),
                    label=label,
                    endpoint_text="2",
                    bond_text="",
                ),
            ),
        )

        self.assertIsNotNone(state)

    def test_ring_endpoint_event_rejects_endpoint_text_mismatch(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        label = WriterClosureLabel(value=1, text="1")

        state = advance_writer_stereo_state(
            prepared,
            empty_writer_stereo_state(),
            (
                WriterRingEndpointEmitted(
                    bond=BondId(2),
                    endpoint_atom=AtomId(0),
                    partner_atom=AtomId(2),
                    label=label,
                    endpoint_text="9",
                    bond_text="",
                ),
            ),
        )

        self.assertIsNone(state)

    def test_ring_endpoint_event_rejects_directional_bond_text(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        label = WriterClosureLabel(value=1, text="1")

        state = advance_writer_stereo_state(
            prepared,
            empty_writer_stereo_state(),
            (
                WriterRingEndpointEmitted(
                    bond=BondId(2),
                    endpoint_atom=AtomId(0),
                    partner_atom=AtomId(2),
                    label=label,
                    endpoint_text="1",
                    bond_text="/",
                ),
            ),
        )

        self.assertIsNone(state)

    def test_ring_endpoint_pair_closes_ring_pair_factor(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        label = WriterClosureLabel(value=1, text="1")
        pending = advance_writer_stereo_state(
            prepared,
            empty_writer_stereo_state(),
            (
                WriterRingEndpointEmitted(
                    bond=BondId(2),
                    endpoint_atom=AtomId(0),
                    partner_atom=AtomId(2),
                    label=label,
                    endpoint_text="1",
                    bond_text="",
                ),
            ),
        )
        assert pending is not None

        closed = advance_writer_stereo_state(
            prepared,
            pending,
            (
                WriterRingEndpointPaired(
                    bond=BondId(2),
                    endpoint_atom=AtomId(2),
                    partner_atom=AtomId(0),
                    label=label,
                    endpoint_text="1",
                    bond_text="",
                ),
            ),
        )

        self.assertIsNotNone(closed)
        assert closed is not None
        self.assertEqual(closed.residual_snapshot, empty_writer_stereo_state().residual_snapshot)

    def test_ring_endpoint_pair_rejects_pending_evidence_with_wrong_side(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        label = WriterClosureLabel(value=1, text="1")

        closed = advance_writer_stereo_state(
            prepared,
            empty_writer_stereo_state(),
            (
                WriterRingEndpointPaired(
                    bond=BondId(2),
                    endpoint_atom=AtomId(2),
                    partner_atom=AtomId(0),
                    label=label,
                    endpoint_text="1",
                    bond_text="",
                ),
            ),
        )

        self.assertIsNotNone(closed)

    def test_ring_endpoint_pair_rejects_pending_evidence_with_wrong_partner(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        label = WriterClosureLabel(value=1, text="1")

        closed = advance_writer_stereo_state(
            prepared,
            empty_writer_stereo_state(),
            (
                WriterRingEndpointPaired(
                    bond=BondId(2),
                    endpoint_atom=AtomId(2),
                    partner_atom=AtomId(0),
                    label=label,
                    endpoint_text="1",
                    bond_text="",
                ),
            ),
        )

        self.assertIsNotNone(closed)

    def test_ring_endpoint_pair_rejects_endpoint_text_mismatch(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        label = WriterClosureLabel(value=1, text="1")
        pending = advance_writer_stereo_state(
            prepared,
            empty_writer_stereo_state(),
            (
                WriterRingEndpointEmitted(
                    bond=BondId(2),
                    endpoint_atom=AtomId(0),
                    partner_atom=AtomId(2),
                    label=label,
                    endpoint_text="1",
                    bond_text="",
                ),
            ),
        )
        assert pending is not None

        closed = advance_writer_stereo_state(
            prepared,
            pending,
            (
                WriterRingEndpointPaired(
                    bond=BondId(2),
                    endpoint_atom=AtomId(2),
                    partner_atom=AtomId(0),
                    label=label,
                    endpoint_text="9",
                    bond_text="",
                ),
            ),
        )

        self.assertIsNone(closed)

    def test_ring_endpoint_pair_rejects_label_outside_policy(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        label = WriterClosureLabel(value=1, text="1")
        pending = advance_writer_stereo_state(
            prepared,
            empty_writer_stereo_state(),
            (
                WriterRingEndpointEmitted(
                    bond=BondId(2),
                    endpoint_atom=AtomId(0),
                    partner_atom=AtomId(2),
                    label=label,
                    endpoint_text="1",
                    bond_text="",
                ),
            ),
        )
        assert pending is not None
        outside = WriterClosureLabel(value=10, text="%10")

        closed = advance_writer_stereo_state(
            prepared,
            pending,
            (
                WriterRingEndpointPaired(
                    bond=BondId(2),
                    endpoint_atom=AtomId(2),
                    partner_atom=AtomId(0),
                    label=outside,
                    endpoint_text="%10",
                    bond_text="",
                ),
            ),
        )

        self.assertIsNone(closed)

    def test_ring_endpoint_pair_accepts_policy_domain_nonleast_label(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        label = WriterClosureLabel(value=2, text="2")

        closed = advance_writer_stereo_state(
            prepared,
            empty_writer_stereo_state(),
            (
                WriterRingEndpointPaired(
                    bond=BondId(2),
                    endpoint_atom=AtomId(2),
                    partner_atom=AtomId(0),
                    label=label,
                    endpoint_text="2",
                    bond_text="",
                ),
            ),
        )

        self.assertIsNotNone(closed)

    def test_ring_endpoint_pair_rejects_directional_bond_text(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        label = WriterClosureLabel(value=1, text="1")
        pending = advance_writer_stereo_state(
            prepared,
            empty_writer_stereo_state(),
            (
                WriterRingEndpointEmitted(
                    bond=BondId(2),
                    endpoint_atom=AtomId(0),
                    partner_atom=AtomId(2),
                    label=label,
                    endpoint_text="1",
                    bond_text="",
                ),
            ),
        )
        assert pending is not None

        closed = advance_writer_stereo_state(
            prepared,
            pending,
            (
                WriterRingEndpointPaired(
                    bond=BondId(2),
                    endpoint_atom=AtomId(2),
                    partner_atom=AtomId(0),
                    label=label,
                    endpoint_text="1",
                    bond_text="\\",
                ),
            ),
        )

        self.assertIsNone(closed)

    def test_tetra_ring_endpoint_open_records_local_order_occurrence(self) -> None:
        prepared = _prepare(ring_core_tetra_facts())
        label = WriterClosureLabel(value=1, text="1")

        outcome = advance_writer_stereo_state_with_evidence(
            prepared,
            initial_writer_stereo_state(prepared),
            (
                WriterRingEndpointEmitted(
                    bond=BondId(2),
                    endpoint_atom=AtomId(0),
                    partner_atom=AtomId(2),
                    label=label,
                    endpoint_text="1",
                    bond_text="",
                ),
            ),
        )

        self.assertIsNotNone(outcome.state)
        assert outcome.state is not None
        self.assertIn(
            _WriterExecutionCapabilityKind
            .TETRA_RING_ENDPOINT_ORDER_OCCURRENCE,
            outcome.execution_capabilities,
        )
        self.assertEqual(
            outcome.state.local_orders,
            (
                writer_stereo_module.WriterLocalOrderRecord(
                    atom=AtomId(0),
                    order=(OccurrenceId(1),),
                ),
            ),
        )

    def test_tetra_ring_endpoint_pair_records_local_order_occurrence(self) -> None:
        prepared = _prepare(ring_core_tetra_facts())
        label = WriterClosureLabel(value=1, text="1")

        outcome = advance_writer_stereo_state_with_evidence(
            prepared,
            initial_writer_stereo_state(prepared),
            (
                WriterRingEndpointPaired(
                    bond=BondId(0),
                    endpoint_atom=AtomId(0),
                    partner_atom=AtomId(1),
                    label=label,
                    endpoint_text="1",
                    bond_text="",
                ),
            ),
        )

        self.assertIsNotNone(outcome.state)
        assert outcome.state is not None
        self.assertIn(
            _WriterExecutionCapabilityKind
            .TETRA_RING_ENDPOINT_ORDER_OCCURRENCE,
            outcome.execution_capabilities,
        )
        self.assertEqual(
            outcome.state.local_orders,
            (
                writer_stereo_module.WriterLocalOrderRecord(
                    atom=AtomId(0),
                    order=(OccurrenceId(0),),
                ),
            ),
        )

    def test_tetra_ring_endpoint_rejects_wrong_partner(self) -> None:
        prepared = _prepare(ring_core_tetra_facts())
        label = WriterClosureLabel(value=1, text="1")

        with self.assertRaises(SouthStarError) as caught:
            advance_writer_stereo_state_with_evidence(
                prepared,
                initial_writer_stereo_state(prepared),
                (
                    WriterRingEndpointEmitted(
                        bond=BondId(2),
                        endpoint_atom=AtomId(0),
                        partner_atom=AtomId(1),
                        label=label,
                        endpoint_text="1",
                        bond_text="",
                    ),
                ),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

    def test_tetra_ring_endpoint_rejects_second_distinct_incidence(self) -> None:
        prepared = _prepare(ring_core_tetra_facts())
        label = WriterClosureLabel(value=1, text="1")
        pending = advance_writer_stereo_state(
            prepared,
            initial_writer_stereo_state(prepared),
            (
                WriterRingEndpointEmitted(
                    bond=BondId(2),
                    endpoint_atom=AtomId(0),
                    partner_atom=AtomId(2),
                    label=label,
                    endpoint_text="1",
                    bond_text="",
                ),
            ),
        )
        assert pending is not None

        with self.assertRaises(SouthStarError) as caught:
            advance_writer_stereo_state(
                prepared,
                pending,
                (
                    WriterRingEndpointPaired(
                        bond=BondId(0),
                        endpoint_atom=AtomId(0),
                        partner_atom=AtomId(1),
                        label=label,
                        endpoint_text="1",
                        bond_text="",
                    ),
                ),
            )

        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

    def test_ring_endpoint_event_on_directional_carrier_fails_closed(self) -> None:
        prepared = _prepare(directional_facts())
        label = WriterClosureLabel(value=1, text="1")

        with self.assertRaises(SouthStarError) as caught:
            advance_writer_stereo_state(
                prepared,
                empty_writer_stereo_state(),
                (
                    WriterRingEndpointEmitted(
                        bond=BondId(1),
                        endpoint_atom=AtomId(0),
                        partner_atom=AtomId(2),
                        label=label,
                        endpoint_text="1",
                        bond_text="",
                    ),
                ),
            )
        self.assertIs(caught.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

    def test_terminal_eos_persists_final_stereo_closure(self) -> None:
        facts = terminal_tetra_center_facts()
        prepared = prepare_south_star_mol_from_facts(
            facts,
            writer_surface=SouthStarWriterSurface(),
            policy=terminal_tetra_center_policy(),
        )
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        center_choice = writer_frontier_choices(prepared, after_f).choices[0]
        pre_terminal_key = center_choice.successor.weighted_states[0][0]
        self.assertTrue(
            any(
                factor.key.kind == "tetra_site"
                for factor in pre_terminal_key.stereo_state.residual_snapshot.factors
            )
        )

        terminal = writer_frontier_choices(prepared, center_choice.successor).terminal
        choices = _writer_frontier_choice_snapshot(
            prepared,
            center_choice.successor,
            include_counts=False,
        )

        self.assertIsNotNone(terminal)
        assert terminal is not None
        self.assertEqual(
            choices.terminal_execution_capabilities,
            frozenset((
                _WriterExecutionCapabilityKind
                .TETRA_LOCAL_ORDER_RESTRICTION,
                _WriterExecutionCapabilityKind.RESIDUAL_PROPAGATION,
                _WriterExecutionCapabilityKind.RESIDUAL_FACTOR_DISCHARGE,
            )),
        )
        finalized_key = terminal.finalized_cursor.weighted_states[0][0]
        self.assertEqual(finalized_key.stereo_state.residual_snapshot.factors, ())

    def test_non_stereo_terminal_eos_reports_no_execution_capabilities(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = _initial_writer_transition_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=0),
        )
        pending = (cursor,)
        seen = set()
        terminal_count = 0

        while pending:
            current = pending[0]
            pending = pending[1:]
            if current in seen:
                continue
            seen.add(current)

            choices = _writer_frontier_choice_snapshot(
                prepared,
                current,
                include_counts=False,
            )
            if choices.terminal is not None:
                terminal_count += 1
                self.assertFalse(choices.terminal_execution_capabilities)

            pending = (
                *pending,
                *(choice.successor for choice in choices.choices),
            )

        self.assertGreater(terminal_count, 0)

    def test_add_factor_and_propagate_rolls_back_rejected_factor(self) -> None:
        store = ResidualStore()
        var = tetra_var(("test", 0))
        store.add_var(var, (TetraToken.AT, TetraToken.ATAT))
        self.assertIs(
            store.restrict_to_value(var, TetraToken.ATAT).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )
        factor = TetraResidualFactor(
            scope=(var,),
            status=SiteStatus.SPECIFIED,
            target=TetraValue.PLUS,
            reference_order=_occurrences(0, 1, 2, 3),
            local_order=_occurrences(0, 1, 2, 3),
        )

        self.assertIs(
            add_factor_and_propagate(store, factor).kind,
            ResidualPropagationKind.CONTRADICTION,
        )
        self.assertEqual(store.value_snapshot().factors, ())
        self.assertEqual(
            ResidualStore.from_value_snapshot(store.value_snapshot()).value_snapshot(),
            store.value_snapshot(),
        )

    def test_add_factor_and_propagate_accepted_factor_rolls_back_to_checkpoint(self) -> None:
        store = ResidualStore()
        var = tetra_var(("test", 1))
        store.add_var(var, (TetraToken.AT, TetraToken.ATAT))
        self.assertIs(
            store.restrict_to_value(var, TetraToken.AT).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )
        checkpoint = store.checkpoint()
        factor = TetraResidualFactor(
            scope=(var,),
            status=SiteStatus.SPECIFIED,
            target=TetraValue.PLUS,
            reference_order=_occurrences(0, 1, 2, 3),
            local_order=_occurrences(0, 1, 2, 3),
        )

        self.assertIs(
            add_factor_and_propagate(store, factor).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )
        self.assertEqual(len(store.value_snapshot().factors), 1)
        store.rollback(checkpoint)

        self.assertEqual(store.value_snapshot().factors, ())

    def test_empty_event_batch_accepts_supported_residual_state(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        state = empty_writer_stereo_state()

        self.assertEqual(
            advance_writer_stereo_state(prepared, state, ()),
            state,
        )

    def test_writer_stereo_does_not_call_global_residual_support_query(self) -> None:
        source = inspect.getsource(writer_stereo_module)
        self.assertNotIn(
            "residual_store_assignments_have_support",
            source,
        )

    def test_residual_contradiction_is_detected_by_propagation(self) -> None:
        left = direction_var(("left", 0))
        right = direction_var(("right", 0))
        store = ResidualStore()
        store.add_var(left, (DirectionMark.FWD,))
        store.add_var(right, (DirectionMark.ABSENT,))
        factor = DirectionalResidualFactor(
            scope=(left, right),
            status=SiteStatus.SPECIFIED,
            target=DirectionalValue.OPPOSITE,
            carrier_models={
                left: DirectionalCarrierResidual(left, "left", 1, 1),
                right: DirectionalCarrierResidual(right, "right", 1, 1),
            },
        )

        self.assertIs(
            add_factor_and_propagate(store, factor).kind,
            ResidualPropagationKind.CONTRADICTION,
        )

    def test_terminal_stereo_closure_accepts_supported_residual_state(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())

        self.assertIsNotNone(
            terminal_writer_stereo_state(
                prepared,
                empty_writer_stereo_state(),
                AtomId(0),
            )
        )

    def test_empty_event_batch_is_identity_for_residual_snapshot(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        left = direction_var(("left", 0))
        right = direction_var(("right", 0))
        store = ResidualStore()
        store.add_var(left, (DirectionMark.FWD,))
        store.add_var(right, (DirectionMark.REV,))
        factor = DirectionalResidualFactor(
            scope=(left, right),
            status=SiteStatus.SPECIFIED,
            target=DirectionalValue.OPPOSITE,
            carrier_models={
                left: DirectionalCarrierResidual(left, "left", 1, 1),
                right: DirectionalCarrierResidual(right, "right", 1, 1),
            },
        )
        self.assertIs(
            add_factor_and_propagate(store, factor).kind,
            ResidualPropagationKind.CERTIFIED_CONSISTENT,
        )
        state = WriterStereoState(
            residual_snapshot=store.value_snapshot(),
            atom_occurrences=(),
            bond_occurrences=(),
            local_orders=(),
        )

        self.assertEqual(
            advance_writer_stereo_state(prepared, state, ()),
            state,
        )


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _writer_options(*, rooted_at_atom: int = -1) -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def triangle_no_stereo_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C")),
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
        ),
    )


def _two_independent_tetra_facts() -> MoleculeFacts:
    left_site = SiteId(0)
    right_site = SiteId(1)
    return MoleculeFacts(
        atoms=(
            atom(0, "C"),
            atom(1, "C"),
            atom(2, "F"),
            atom(3, "Cl"),
            atom(4, "Br"),
            atom(5, "O"),
        ),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 0, 2),
            single_bond(2, 0, 3),
            single_bond(3, 1, 4),
            single_bond(4, 1, 5),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(6)),
                bonds=tuple(BondId(index) for index in range(5)),
            ),
        ),
        stereo=StereoFacts(
            tetrahedral=(
                TetrahedralSiteFacts(
                    id=left_site,
                    center=AtomId(0),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    ligand_occurrences=_occurrences(0, 1, 2, 3),
                    reference_order=_occurrences(0, 1, 2, 3),
                ),
                TetrahedralSiteFacts(
                    id=right_site,
                    center=AtomId(1),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    ligand_occurrences=_occurrences(4, 5, 6, 7),
                    reference_order=_occurrences(4, 5, 6, 7),
                ),
            ),
        ),
        ligand_occurrences=(
            LigandOccurrence(
                id=OccurrenceId(0),
                site=left_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(1),
                bond=BondId(0),
            ),
            LigandOccurrence(
                id=OccurrenceId(1),
                site=left_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(2),
                bond=BondId(1),
            ),
            LigandOccurrence(
                id=OccurrenceId(2),
                site=left_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(3),
                bond=BondId(2),
            ),
            LigandOccurrence(
                id=OccurrenceId(3),
                site=left_site,
                kind=LigandKind.IMPLICIT_H,
                atom=AtomId(0),
                bond=None,
            ),
            LigandOccurrence(
                id=OccurrenceId(4),
                site=right_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(0),
                bond=BondId(0),
            ),
            LigandOccurrence(
                id=OccurrenceId(5),
                site=right_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(4),
                bond=BondId(3),
            ),
            LigandOccurrence(
                id=OccurrenceId(6),
                site=right_site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(5),
                bond=BondId(4),
            ),
            LigandOccurrence(
                id=OccurrenceId(7),
                site=right_site,
                kind=LigandKind.IMPLICIT_H,
                atom=AtomId(1),
                bond=None,
            ),
        ),
    )


def terminal_tetra_center_facts() -> MoleculeFacts:
    site = SiteId(0)
    return MoleculeFacts(
        atoms=(
            atom(0, "F"),
            replace(atom(1, "C"), implicit_h_count=3),
        ),
        bonds=(single_bond(0, 0, 1),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1)),
                bonds=(BondId(0),),
            ),
        ),
        stereo=StereoFacts(
            tetrahedral=(
                TetrahedralSiteFacts(
                    id=site,
                    center=AtomId(1),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    ligand_occurrences=tuple(OccurrenceId(index) for index in range(4)),
                    reference_order=tuple(OccurrenceId(index) for index in range(4)),
                ),
            ),
        ),
        ligand_occurrences=(
            LigandOccurrence(
                id=OccurrenceId(0),
                site=site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(0),
                bond=BondId(0),
            ),
            LigandOccurrence(
                id=OccurrenceId(1),
                site=site,
                kind=LigandKind.IMPLICIT_H,
                atom=AtomId(1),
                bond=None,
            ),
            LigandOccurrence(
                id=OccurrenceId(2),
                site=site,
                kind=LigandKind.IMPLICIT_H,
                atom=AtomId(1),
                bond=None,
            ),
            LigandOccurrence(
                id=OccurrenceId(3),
                site=site,
                kind=LigandKind.IMPLICIT_H,
                atom=AtomId(1),
                bond=None,
            ),
        ),
    )


def ring_core_tetra_facts() -> MoleculeFacts:
    site = SiteId(0)
    return MoleculeFacts(
        atoms=(
            replace(atom(0, "C"), implicit_h_count=1),
            atom(1, "C"),
            atom(2, "C"),
            atom(3, "F"),
        ),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
            single_bond(2, 2, 0),
            single_bond(3, 0, 3),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=tuple(AtomId(index) for index in range(4)),
                bonds=tuple(BondId(index) for index in range(4)),
            ),
        ),
        stereo=StereoFacts(
            tetrahedral=(
                TetrahedralSiteFacts(
                    id=site,
                    center=AtomId(0),
                    status=SiteStatus.SPECIFIED,
                    target=TetraValue.PLUS,
                    ligand_occurrences=tuple(
                        OccurrenceId(index) for index in range(4)
                    ),
                    reference_order=tuple(
                        OccurrenceId(index) for index in range(4)
                    ),
                ),
            ),
        ),
        ligand_occurrences=(
            LigandOccurrence(
                id=OccurrenceId(0),
                site=site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(1),
                bond=BondId(0),
            ),
            LigandOccurrence(
                id=OccurrenceId(1),
                site=site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(2),
                bond=BondId(2),
            ),
            LigandOccurrence(
                id=OccurrenceId(2),
                site=site,
                kind=LigandKind.NEIGHBOR_ATOM,
                atom=AtomId(3),
                bond=BondId(3),
            ),
            LigandOccurrence(
                id=OccurrenceId(3),
                site=site,
                kind=LigandKind.IMPLICIT_H,
                atom=AtomId(0),
                bond=None,
            ),
        ),
    )


def ring_core_tetra_with_remote_non_single_facts(
    order: BondOrder,
) -> MoleculeFacts:
    assert order in {BondOrder.DOUBLE, BondOrder.TRIPLE}
    facts = ring_core_tetra_facts()
    return replace(
        facts,
        bonds=tuple(
            replace(bond, order=order)
            if bond.id == BondId(1)
            else bond
            for bond in facts.bonds
        ),
    )


def terminal_tetra_center_policy() -> SmilesPolicy:
    return SmilesPolicy(
        ring_labels=(RingLabel(1),),
        annotation_mode=AnnotationMode.HARD,
        atom_text_domains=(
            AtomTextDomain(
                atom=AtomId(0),
                choices=(AtomTextChoice("fluorine", ((TetraToken.NONE, "F"),)),),
            ),
            AtomTextDomain(
                atom=AtomId(1),
                choices=(
                    AtomTextChoice(
                        "terminal_tetra_carbon",
                        (
                            (TetraToken.AT, "[C@H3]"),
                            (TetraToken.ATAT, "[C@@H3]"),
                        ),
                    ),
                ),
            ),
        ),
        bond_text_domains=(
            BondTextDomain(
                bond=BondId(0),
                slot_kind="tree",
                choices=(BondTextChoice("single_elided", "", False),),
            ),
        ),
    )


def _occurrences(*values: int) -> tuple[OccurrenceId, ...]:
    return tuple(OccurrenceId(value) for value in values)


if __name__ == "__main__":
    unittest.main()
