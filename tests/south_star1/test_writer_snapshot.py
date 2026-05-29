"""Writer frontier snapshot tests."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.residual_constraints import ResidualStore
from grimace._south_star1.residual_constraints import tetra_var
from grimace._south_star1.writer_frontier import count_writer_cursor_completions
from grimace._south_star1.writer_frontier import count_writer_frontier_support
from grimace._south_star1.writer_frontier import WriterFrontierCursor
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_frontier import writer_frontier_choices
from grimace._south_star1.writer_snapshot import WriterDecoderBoundary
from grimace._south_star1.writer_snapshot import WriterFrontierFrame
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot import resume_writer_frontier_choices_from_snapshot
from grimace._south_star1.writer_snapshot import validate_writer_cursor_against_prepared
from grimace._south_star1.writer_snapshot import validate_writer_search_snapshot
from grimace._south_star1.writer_snapshot import writer_frontier_cursor_from_snapshot
from grimace._south_star1.writer_state import ObligationStateKey
from grimace._south_star1.writer_state import PendingEntryPhase
from grimace._south_star1.writer_state import PendingWriterEntry
from grimace._south_star1.writer_state import WriterRingStateKey
from grimace._south_star1.writer_stereo import WriterAtomOccurrenceRecord
from grimace._south_star1.writer_stereo import WriterBondOccurrenceRecord
from grimace._south_star1.writer_stereo import WriterDelayedStereoFactor
from grimace._south_star1.writer_stereo import WriterLocalOrderRecord
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


class WriterSnapshotTest(unittest.TestCase):
    def test_weighted_cursor_snapshot_round_trips_choices_and_counts(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
            decoder_boundary=WriterDecoderBoundary(consumed_token_count=0),
        )

        self.assertEqual(
            writer_frontier_cursor_from_snapshot(snapshot, prepared=prepared),
            cursor,
        )
        self.assertEqual(
            resume_writer_frontier_choices_from_snapshot(snapshot, prepared=prepared),
            writer_frontier_choices(prepared, cursor),
        )
        self.assertEqual(
            count_writer_frontier_support(prepared, snapshot.cursor.support_state),
            count_writer_frontier_support(prepared, cursor.support_state),
        )
        self.assertEqual(
            count_writer_cursor_completions(prepared, snapshot.cursor),
            count_writer_cursor_completions(prepared, cursor),
        )

    def test_stereo_residual_snapshot_round_trips(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
        key = after_center.weighted_states[0][0]
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=after_center,
        )

        validate_writer_search_snapshot(snapshot, prepared=prepared)
        self.assertNotEqual(key.stereo_state.residual_snapshot.domains, ())
        self.assertEqual(
            writer_frontier_cursor_from_snapshot(snapshot, prepared=prepared),
            after_center,
        )

    def test_tampered_mode_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        tampered = replace(
            snapshot,
            serialization_language=SerializationLanguageMode.EXHAUSTIVE,
        )

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(tampered, prepared=prepared)

    def test_structural_prepared_identity_mismatch_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        changed_facts = replace(
            cco_facts(),
            atoms=(atom(0, "C"), atom(1, "C"), atom(2, "C")),
        )
        changed_prepared = _prepare(changed_facts)

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(snapshot, prepared=changed_prepared)

    def test_unknown_frame_payload_is_rejected(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        tampered = replace(snapshot, frame_stack=snapshot.frame_stack + (object(),))

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(tampered, prepared=prepared)

    def test_dormant_snapshot_frame_types_are_not_exported(self) -> None:
        import grimace._south_star1.writer_snapshot as writer_snapshot

        self.assertFalse(hasattr(writer_snapshot, "WriterTransitionFrame"))
        self.assertFalse(hasattr(writer_snapshot, "WriterStereoResidualFrame"))
        self.assertFalse(hasattr(writer_snapshot, "WriterDelayedFactorFrame"))

    def test_extra_context_frame_payload_is_rejected_until_stack_resume_exists(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=after_center,
        )
        tampered = replace(
            snapshot,
            frame_stack=snapshot.frame_stack + (object(),),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(tampered, prepared=prepared)

    def test_cursor_audit_rejects_unknown_active_atom(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        key = cursor.weighted_states[0][0]
        tampered_key = replace(key, active=replace(key.active, atom=AtomId(99)))

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=_writer_options(),
            )

    def test_cursor_audit_rejects_missing_active_frame(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        key = cursor.weighted_states[0][0]
        tampered_key = replace(key, active=None)

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=_writer_options(),
            )

    def test_cursor_audit_rejects_root_frame_mismatch(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_frontier_cursor(prepared, options)
        key = cursor.weighted_states[0][0]
        tampered_key = replace(key, active=replace(key.active, atom=AtomId(1)))

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_invalid_pending_graph_triple(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(
            prepared,
            _writer_options(rooted_at_atom=1),
        )
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        key = after_root.weighted_states[0][0]
        tampered_key = replace(
            key,
            obligations=ObligationStateKey(
                pending_entry=PendingWriterEntry(
                    parent=AtomId(1),
                    child=AtomId(2),
                    bond=BondId(0),
                    branch=False,
                )
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=_writer_options(rooted_at_atom=1),
            )

    def test_cursor_audit_rejects_post_bond_pending_without_bond_record(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=1)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        pending_cursor = writer_frontier_choices(prepared, after_root).choices[0].successor
        key = pending_cursor.weighted_states[0][0]
        assert key.obligations.pending_entry is not None
        tampered_key = replace(
            key,
            obligations=ObligationStateKey(
                pending_entry=replace(
                    key.obligations.pending_entry,
                    phase=PendingEntryPhase.NEEDS_ATOM_AFTER_BOND,
                )
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_pre_bond_pending_with_bond_record(self) -> None:
        prepared = _prepare(directional_facts())
        options = _writer_options(rooted_at_atom=2)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        post_bond = writer_frontier_choices(prepared, after_f).choices[0].successor
        key = post_bond.weighted_states[0][0]
        assert key.obligations.pending_entry is not None
        tampered_key = replace(
            key,
            obligations=ObligationStateKey(
                pending_entry=replace(
                    key.obligations.pending_entry,
                    phase=PendingEntryPhase.NEEDS_BOND_OR_ATOM,
                )
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_accepts_post_bond_pending_with_matching_bond_record(self) -> None:
        prepared = _prepare(directional_facts())
        options = _writer_options(rooted_at_atom=2)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        post_bond = writer_frontier_choices(prepared, after_f).choices[0].successor

        validate_writer_cursor_against_prepared(
            prepared,
            post_bond,
            runtime_options=options,
        )

    def test_cursor_audit_rejects_nonempty_ring_state(self) -> None:
        prepared = _prepare(cco_facts())
        cursor = initial_writer_frontier_cursor(prepared, _writer_options())
        key = cursor.weighted_states[0][0]
        tampered_key = replace(
            key,
            ring_state=WriterRingStateKey(closed_bonds=frozenset((BondId(0),))),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=_writer_options(),
            )

    def test_cursor_audit_rejects_incomplete_completed_component(self) -> None:
        prepared = _prepare(chain_plus_singleton_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_c = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_second_c = writer_frontier_choices(prepared, after_c).choices[0].successor
        after_dot = writer_frontier_choices(prepared, after_second_c).choices[0].successor
        key = after_dot.weighted_states[0][0]
        tampered_key = replace(key, visited_atoms=frozenset((AtomId(0),)))

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_unreachable_current_component_atom(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=1)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        key = after_root.weighted_states[0][0]
        tampered_key = replace(
            key,
            visited_atoms=frozenset((*key.visited_atoms, AtomId(2))),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_future_unemitted_bond_occurrence(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        key = after_root.weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                bond_occurrences=key.stereo_state.bond_occurrences
                + (
                    WriterBondOccurrenceRecord(
                        bond=BondId(1),
                        parent=AtomId(1),
                        child=AtomId(2),
                        mark=DirectionMark.ABSENT,
                        var=None,
                    ),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_future_unvisited_atom_occurrence(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        key = after_root.weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                atom_occurrences=key.stereo_state.atom_occurrences
                + (
                    WriterAtomOccurrenceRecord(
                        atom=AtomId(2),
                        token=TetraToken.NONE,
                        var=None,
                    ),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_future_unvisited_local_order(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_root = writer_frontier_choices(prepared, cursor).choices[0].successor
        key = after_root.weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=key.stereo_state.local_orders
                + (
                    WriterLocalOrderRecord(
                        atom=AtomId(2),
                        order=(),
                    ),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_invalid_local_order_occurrence(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
        key = after_center.weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=(
                    WriterLocalOrderRecord(
                        atom=AtomId(1),
                        order=(OccurrenceId(999),),
                    ),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_wrong_site_local_order_occurrence(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
        key = after_center.weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=(
                    WriterLocalOrderRecord(
                        atom=AtomId(1),
                        order=(OccurrenceId(0),),
                    ),
                ),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_atom_occurrence_assignment_mismatch(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
        key = after_center.weighted_states[0][0]
        records = tuple(
            replace(record, token=TetraToken.NONE)
            if record.var is not None
            else record
            for record in key.stereo_state.atom_occurrences
        )
        tampered_key = replace(
            key,
            stereo_state=replace(key.stereo_state, atom_occurrences=records),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_bond_occurrence_assignment_mismatch(self) -> None:
        prepared = _prepare(directional_facts())
        options = _writer_options(rooted_at_atom=2)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_slash = writer_frontier_choices(prepared, after_f).choices[0].successor
        key = after_slash.weighted_states[0][0]
        records = tuple(
            replace(record, mark=DirectionMark.ABSENT)
            if record.var is not None
            else record
            for record in key.stereo_state.bond_occurrences
        )
        tampered_key = replace(
            key,
            stereo_state=replace(key.stereo_state, bond_occurrences=records),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_duplicate_atom_occurrence(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                atom_occurrences=key.stereo_state.atom_occurrences
                + (key.stereo_state.atom_occurrences[-1],),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_duplicate_bond_occurrence(self) -> None:
        prepared = _prepare(directional_facts())
        options = _writer_options(rooted_at_atom=2)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_slash = writer_frontier_choices(prepared, after_f).choices[0].successor
        key = after_slash.weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                bond_occurrences=key.stereo_state.bond_occurrences
                + (key.stereo_state.bond_occurrences[-1],),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_duplicate_local_order(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                local_orders=key.stereo_state.local_orders
                + (key.stereo_state.local_orders[-1],),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_duplicate_delayed_factor(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                delayed_factors=key.stereo_state.delayed_factors
                + (key.stereo_state.delayed_factors[-1],),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_snapshot_rejects_pending_and_closed_delayed_factor_for_same_site(self) -> None:
        prepared, options, key = _terminal_tetra_key()
        closed_factor = key.stereo_state.delayed_factors[0]
        pending_factor = WriterDelayedStereoFactor(
            kind=closed_factor.kind,
            site=closed_factor.site,
            scope=closed_factor.scope,
            evidence=closed_factor.evidence,
            closed=False,
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                delayed_factors=key.stereo_state.delayed_factors + (pending_factor,),
            ),
        )
        valid_cursor = _cursor_with_key(key)
        tampered_cursor = _cursor_with_key(tampered_key)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=valid_cursor,
        )
        tampered_snapshot = replace(
            snapshot,
            cursor=tampered_cursor,
            frame_stack=(WriterFrontierFrame(tampered_cursor),),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(tampered_snapshot, prepared=prepared)

    def test_snapshot_rejects_pending_tetra_factor_with_closed_local_order(self) -> None:
        prepared, options, key = _terminal_tetra_key()
        closed_factor = key.stereo_state.delayed_factors[0]
        pending_factor = WriterDelayedStereoFactor(
            kind=closed_factor.kind,
            site=closed_factor.site,
            scope=closed_factor.scope,
            evidence=closed_factor.evidence,
            closed=False,
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                delayed_factors=(pending_factor,),
                residual_snapshot=replace(
                    key.stereo_state.residual_snapshot,
                    factors=(),
                ),
            ),
        )
        valid_cursor = _cursor_with_key(key)
        tampered_cursor = _cursor_with_key(tampered_key)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=valid_cursor,
        )
        tampered_snapshot = replace(
            snapshot,
            cursor=tampered_cursor,
            frame_stack=(WriterFrontierFrame(tampered_cursor),),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(tampered_snapshot, prepared=prepared)

    def test_cursor_audit_rejects_closed_delay_without_residual_factor(self) -> None:
        from tests.south_star1.test_writer_stereo_residual import terminal_tetra_center_facts
        from tests.south_star1.test_writer_stereo_residual import terminal_tetra_center_policy

        prepared = prepare_south_star_mol_from_facts(
            terminal_tetra_center_facts(),
            writer_surface=SouthStarWriterSurface(),
            policy=terminal_tetra_center_policy(),
        )
        options = _writer_options(rooted_at_atom=0)
        cursor = initial_writer_frontier_cursor(prepared, options)
        after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
        after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
        terminal = writer_frontier_choices(prepared, after_center).terminal
        assert terminal is not None
        key = terminal.finalized_cursor.weighted_states[0][0]
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=ResidualStore().value_snapshot(),
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_duplicate_residual_factor_snapshot(self) -> None:
        prepared, options, key = _terminal_tetra_key()
        factor = key.stereo_state.residual_snapshot.factors[0]
        tampered_snapshot = replace(
            key.stereo_state.residual_snapshot,
            factors=key.stereo_state.residual_snapshot.factors + (factor,),
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=tampered_snapshot,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_closed_factor_semantic_mismatch(self) -> None:
        prepared, options, key = _terminal_tetra_key()
        factor = key.stereo_state.residual_snapshot.factors[0]
        tampered_snapshot = replace(
            key.stereo_state.residual_snapshot,
            factors=(replace(factor, local_order=tuple(reversed(factor.local_order))),),
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=tampered_snapshot,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_residual_assignment_without_occurrence(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        extra_var = tetra_var(("writer", 999))
        tampered_snapshot = replace(
            key.stereo_state.residual_snapshot,
            domains=key.stereo_state.residual_snapshot.domains
            + ((extra_var, (TetraToken.AT, TetraToken.ATAT)),),
            assignments=key.stereo_state.residual_snapshot.assignments
            + ((extra_var, TetraToken.AT),),
        )
        tampered_key = replace(
            key,
            stereo_state=replace(
                key.stereo_state,
                residual_snapshot=tampered_snapshot,
            ),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_cursor_audit_rejects_occurrence_without_delayed_factor(self) -> None:
        prepared = _prepare(tetrahedral_facts())
        options = _writer_options(rooted_at_atom=1)
        key = _tetra_center_key(prepared, options)
        tampered_key = replace(
            key,
            stereo_state=replace(key.stereo_state, delayed_factors=()),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_cursor_against_prepared(
                prepared,
                _cursor_with_key(tampered_key),
                runtime_options=options,
            )

    def test_frontier_frame_cursor_must_match_snapshot_cursor(self) -> None:
        prepared = _prepare(cco_facts())
        options = _writer_options()
        cursor = initial_writer_frontier_cursor(prepared, options)
        choices = writer_frontier_choices(prepared, cursor)
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
        )
        tampered = replace(
            snapshot,
            frame_stack=(WriterFrontierFrame(choices.choices[0].successor),),
        )

        with self.assertRaises(SouthStarError):
            validate_writer_search_snapshot(tampered, prepared=prepared)


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


def _cursor_with_key(key) -> WriterFrontierCursor:
    return WriterFrontierCursor(weighted_states=((key, 1),))


def _tetra_center_key(prepared, options):
    cursor = initial_writer_frontier_cursor(prepared, options)
    after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
    after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
    return after_center.weighted_states[0][0]


def _terminal_tetra_key():
    from tests.south_star1.test_writer_stereo_residual import terminal_tetra_center_facts
    from tests.south_star1.test_writer_stereo_residual import terminal_tetra_center_policy

    prepared = prepare_south_star_mol_from_facts(
        terminal_tetra_center_facts(),
        writer_surface=SouthStarWriterSurface(),
        policy=terminal_tetra_center_policy(),
    )
    options = _writer_options(rooted_at_atom=0)
    cursor = initial_writer_frontier_cursor(prepared, options)
    after_f = writer_frontier_choices(prepared, cursor).choices[0].successor
    after_center = writer_frontier_choices(prepared, after_f).choices[0].successor
    terminal = writer_frontier_choices(prepared, after_center).terminal
    assert terminal is not None
    return prepared, options, terminal.finalized_cursor.weighted_states[0][0]


def chain_plus_singleton_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "O")),
        bonds=(single_bond(0, 0, 1),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1)),
                bonds=(BondId(0),),
            ),
            ComponentFacts(
                id=ComponentId(1),
                atoms=(AtomId(2),),
                bonds=(),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
