"""Writer frontier snapshot tests."""

from __future__ import annotations

from dataclasses import replace
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_frontier import count_writer_cursor_completions
from grimace._south_star1.writer_frontier import count_writer_frontier_support
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_frontier import writer_frontier_choices
from grimace._south_star1.writer_snapshot import WriterDecoderBoundary
from grimace._south_star1.writer_snapshot import WriterStereoResidualFrame
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot import resume_writer_frontier_choices_from_snapshot
from grimace._south_star1.writer_snapshot import validate_writer_search_snapshot
from grimace._south_star1.writer_snapshot import writer_frontier_cursor_from_snapshot
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
        residual_frame = WriterStereoResidualFrame(
            key.stereo_state.residual_snapshot,
        )
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=after_center,
        )
        snapshot = replace(
            snapshot,
            frame_stack=snapshot.frame_stack + (residual_frame,),
        )

        validate_writer_search_snapshot(snapshot, prepared=prepared)
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


if __name__ == "__main__":
    unittest.main()
