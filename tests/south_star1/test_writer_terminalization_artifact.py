"""Count-free writer terminalization artifact regressions."""

from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import unittest
from unittest.mock import patch

from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot import WriterDecoderBoundary
from grimace._south_star1.writer_terminalization_artifact import verify_writer_terminalization_artifact_envelope
from grimace._south_star1.writer_terminalization_artifact import writer_terminalization_artifact_for_support
from grimace._south_star1.writer_terminalization_artifact_checker import verify_writer_terminalization_artifact_consistency
from grimace._south_star1.writer_terminalization_artifact_fact_verifier import verify_writer_terminalization_artifact_for_facts
from tests.south_star1.helpers import cco_facts
from tests.south_star1.test_writer_stereo_residual import _directional_non_single_ring_carrier_facts
from tests.south_star1.test_writer_stereo_residual import _directional_ring_carrier_facts
from tests.south_star1.test_writer_stereo_residual import _shared_directional_ring_carrier_facts
from tests.south_star1.test_writer_stereo_residual import terminal_tetra_center_facts
from tests.south_star1.test_writer_stereo_residual import terminal_tetra_center_policy
from tests.south_star1.test_writer_support_artifact_fact_verifier import _initial_snapshot
from tests.south_star1.test_writer_support_artifact_fact_verifier import _writer_options


class WriterTerminalizationArtifactTest(unittest.TestCase):
    def test_positive_terminalization_matrix(self) -> None:
        cases = (
            ("ordinary", cco_facts(), _writer_options(), None, "noop", ()),
            (
                "tetra",
                terminal_tetra_center_facts(),
                _writer_options(rooted_at_atom=0),
                terminal_tetra_center_policy(),
                "tetra_local_order_factor_closure",
                ("tetrahedral local-order factor closure",),
            ),
            ("simple_ring", _directional_ring_carrier_facts(), _writer_options(rooted_at_atom=0), None, "noop", ()),
            ("shared_ring", _shared_directional_ring_carrier_facts(), _writer_options(rooted_at_atom=1), None, "noop", ()),
            ("non_single_ring", _directional_non_single_ring_carrier_facts(), _writer_options(rooted_at_atom=0), None, "noop", ()),
        )
        for name, facts, options, policy, mode, operations in cases:
            with self.subTest(name=name):
                prepared, artifact = _terminal_artifact(facts, options, policy)
                structural = verify_writer_terminalization_artifact_consistency(artifact)
                live = verify_writer_terminalization_artifact_envelope(prepared=prepared, artifact=artifact)
                checked = verify_writer_terminalization_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=artifact,
                    policy=policy,
                )
                self.assertTrue(structural.accepted, structural.reason)
                self.assertTrue(live.accepted, live.reason)
                self.assertTrue(checked.accepted, checked.reason)
                self.assertEqual(len(artifact["objects"]), 3)
                self.assertEqual(checked.terminalization_mode, mode)
                self.assertEqual(checked.semantically_replayed_operations, operations)
                self.assertEqual(checked.unchecked_obligation_families, ())

    def test_build_and_live_check_do_not_materialize_counts_or_support(self) -> None:
        facts = cco_facts()
        options = _writer_options()
        prepared, snapshot, support = _terminal_source(facts, options, None)
        patches = (
            patch("grimace._south_star1.writer_frontier_count_envelope.writer_frontier_count_envelope_for_snapshot", side_effect=AssertionError("count path")),
            patch("grimace._south_star1.writer_count_dag_envelope.writer_count_certificate_dag_envelope_for_product", side_effect=AssertionError("dag path")),
            patch("grimace._south_star1.writer_support_string_envelope._iter_writer_snapshot_certified_support_strings", side_effect=AssertionError("support path")),
        )
        with patches[0], patches[1], patches[2]:
            artifact = writer_terminalization_artifact_for_support(
                prepared=prepared, snapshot=snapshot, support=support
            )
            checked = verify_writer_terminalization_artifact_envelope(
                prepared=prepared, artifact=artifact
            )
        self.assertTrue(checked.accepted, checked.reason)

    def test_old_schema_and_count_object_reject(self) -> None:
        _prepared, artifact = _terminal_artifact(cco_facts(), _writer_options(), None)
        old = deepcopy(artifact)
        old["schema_version"] = 0
        self.assertIn(
            "unknown_schema_version",
            verify_writer_terminalization_artifact_consistency(old).reason,
        )
        extra = deepcopy(artifact)
        extra["objects"].append({
            "object_id": "obj:count",
            "kind": "count_dag",
            "payload": {},
            "digest": "count",
        })
        self.assertIn(
            "object_count_mismatch",
            verify_writer_terminalization_artifact_consistency(extra).reason,
        )


@lru_cache(maxsize=None)
def _terminal_artifact(facts, options, policy):
    prepared, snapshot, support = _terminal_source(facts, options, policy)
    return prepared, writer_terminalization_artifact_for_support(
        prepared=prepared, snapshot=snapshot, support=support
    )


def _terminal_source(facts, options, policy):
    prepared = prepare_south_star_mol_from_facts(
        facts, writer_surface=SouthStarWriterSurface(), policy=policy
    )
    snapshot = _initial_snapshot(prepared, options)
    for depth in range(256):
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            snapshot.cursor,
            include_counts=False,
            include_frontier_certificate=True,
            include_count_certificate=False,
        )
        if batch.terminal_supports:
            return prepared, snapshot, batch.terminal_supports[0]
        support = batch.supports[0]
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=support.successor_cursor,
            decoder_boundary=WriterDecoderBoundary(depth + 1),
        )
    raise AssertionError("terminal support not reached")


if __name__ == "__main__":
    unittest.main()
