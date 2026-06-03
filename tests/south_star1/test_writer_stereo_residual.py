"""Writer-owned residual stereo tests."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
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
from grimace._south_star1.residual_constraints import ResidualStore
from grimace._south_star1.residual_constraints import DirectionalCarrierResidual
from grimace._south_star1.residual_constraints import DirectionalResidualFactor
from grimace._south_star1.residual_constraints import TetraResidualFactor
from grimace._south_star1.residual_constraints import add_factor_checked
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
from grimace._south_star1.writer_frontier import writer_frontier_choices
from grimace._south_star1.writer_events import WriterRingEndpointEmitted
from grimace._south_star1.writer_events import WriterRingEndpointPaired
from grimace._south_star1.writer_state import WriterClosureLabel
from grimace._south_star1.writer_state import WriterStereoState
from grimace._south_star1.writer_stereo import advance_writer_stereo_state
from grimace._south_star1.writer_stereo import empty_writer_stereo_state
from grimace._south_star1.writer_stereo import terminal_writer_stereo_state
from grimace._south_star1.writer_stereo import WriterDelayedStereoFactor
from tests.south_star1.helpers import atom
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
        pending = successor_key.stereo_state.delayed_factors
        self.assertTrue(any(factor.kind == "tetra" and not factor.closed for factor in pending))

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
        pending = successor_key.stereo_state.delayed_factors
        self.assertTrue(
            any(factor.kind == "directional" and not factor.closed for factor in pending)
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
        self.assertEqual(
            state.delayed_factors,
            (
                WriterDelayedStereoFactor(
                    kind="ring_pair",
                    site=SiteId(2),
                    evidence=(
                        ("ring_endpoint", 2, "open", 0, 2, 1, "1", "1", ""),
                    ),
                    closed=False,
                ),
            ),
        )

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
        self.assertTrue(
            any(
                factor.kind == "ring_pair"
                and factor.closed
                and factor.evidence
                == (("ring_pair", 2, 0, 2, 1, "1", "1", "1", "", ""),)
                for factor in closed.delayed_factors
            )
        )

    def test_ring_endpoint_pair_rejects_pending_evidence_with_wrong_side(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        label = WriterClosureLabel(value=1, text="1")
        pending = replace(
            empty_writer_stereo_state(),
            delayed_factors=(
                WriterDelayedStereoFactor(
                    kind="ring_pair",
                    site=SiteId(2),
                    evidence=(("ring_endpoint", 2, "close", 0, 2, 1, "1", "1", ""),),
                    closed=False,
                ),
            ),
        )

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

        self.assertIsNone(closed)

    def test_ring_endpoint_pair_rejects_pending_evidence_with_wrong_partner(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
        label = WriterClosureLabel(value=1, text="1")
        pending = replace(
            empty_writer_stereo_state(),
            delayed_factors=(
                WriterDelayedStereoFactor(
                    kind="ring_pair",
                    site=SiteId(2),
                    evidence=(("ring_endpoint", 2, "open", 0, 1, 1, "1", "1", ""),),
                    closed=False,
                ),
            ),
        )

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

        self.assertIsNone(closed)

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
        pending = replace(
            empty_writer_stereo_state(),
            delayed_factors=(
                WriterDelayedStereoFactor(
                    kind="ring_pair",
                    site=SiteId(2),
                    evidence=(("ring_endpoint", 2, "open", 0, 2, 2, "2", "2", ""),),
                    closed=False,
                ),
            ),
        )

        closed = advance_writer_stereo_state(
            prepared,
            pending,
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
        self.assertEqual(pre_terminal_key.stereo_state.residual_snapshot.factors, ())
        self.assertTrue(
            any(
                factor.kind == "tetra" and not factor.closed
                for factor in pre_terminal_key.stereo_state.delayed_factors
            )
        )

        terminal = writer_frontier_choices(prepared, center_choice.successor).terminal

        self.assertIsNotNone(terminal)
        assert terminal is not None
        finalized_key = terminal.finalized_cursor.weighted_states[0][0]
        self.assertEqual(len(finalized_key.stereo_state.residual_snapshot.factors), 1)
        self.assertTrue(
            any(
                factor.kind == "tetra" and factor.closed
                for factor in finalized_key.stereo_state.delayed_factors
            )
        )

    def test_add_factor_checked_rolls_back_rejected_factor(self) -> None:
        store = ResidualStore()
        var = tetra_var(("test", 0))
        store.add_var(var, (TetraToken.AT, TetraToken.ATAT))
        self.assertTrue(store.assign(var, TetraToken.ATAT))
        factor = TetraResidualFactor(
            scope=(var,),
            status=SiteStatus.SPECIFIED,
            target=TetraValue.PLUS,
            reference_order=(0, 1, 2, 3),
            local_order=(0, 1, 2, 3),
        )

        self.assertFalse(add_factor_checked(store, factor))
        self.assertEqual(store.value_snapshot().factors, ())
        self.assertEqual(
            ResidualStore.from_value_snapshot(store.value_snapshot()).value_snapshot(),
            store.value_snapshot(),
        )

    def test_add_factor_checked_accepted_factor_rolls_back_to_checkpoint(self) -> None:
        store = ResidualStore()
        var = tetra_var(("test", 1))
        store.add_var(var, (TetraToken.AT, TetraToken.ATAT))
        self.assertTrue(store.assign(var, TetraToken.AT))
        checkpoint = store.checkpoint()
        factor = TetraResidualFactor(
            scope=(var,),
            status=SiteStatus.SPECIFIED,
            target=TetraValue.PLUS,
            reference_order=(0, 1, 2, 3),
            local_order=(0, 1, 2, 3),
        )

        self.assertTrue(add_factor_checked(store, factor))
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

    def test_empty_event_batch_rejects_unsupported_residual_snapshot(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
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
        self.assertTrue(add_factor_checked(store, factor))
        state = WriterStereoState(
            residual_snapshot=store.value_snapshot(),
            atom_occurrences=(),
            bond_occurrences=(),
            local_orders=(),
            delayed_factors=(),
        )

        self.assertIsNone(advance_writer_stereo_state(prepared, state, ()))

    def test_terminal_stereo_closure_accepts_supported_residual_state(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())

        self.assertIsNotNone(
            terminal_writer_stereo_state(
                prepared,
                empty_writer_stereo_state(),
                AtomId(0),
            )
        )

    def test_terminal_stereo_closure_rejects_unsupported_residual_snapshot(self) -> None:
        prepared = _prepare(triangle_no_stereo_facts())
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
        self.assertTrue(add_factor_checked(store, factor))
        state = WriterStereoState(
            residual_snapshot=store.value_snapshot(),
            atom_occurrences=(),
            bond_occurrences=(),
            local_orders=(),
            delayed_factors=(),
        )

        self.assertIsNone(
            terminal_writer_stereo_state(prepared, state, AtomId(0))
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


if __name__ == "__main__":
    unittest.main()
