"""Writer frontier snapshot tests."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.residual_constraints import ResidualStore
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
from grimace._south_star1.writer_state import PendingWriterEntry
from grimace._south_star1.writer_state import WriterRingStateKey
from grimace._south_star1.writer_stereo import WriterLocalOrderRecord
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import cco_facts
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


if __name__ == "__main__":
    unittest.main()
