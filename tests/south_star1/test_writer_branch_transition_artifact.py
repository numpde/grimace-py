"""Count-free writer branch transition artifact tests."""

from __future__ import annotations

from functools import lru_cache
from copy import deepcopy
import unittest
from unittest.mock import patch

from grimace._south_star1.ids import BondId
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.writer_branch_transition_artifact import verify_writer_branch_transition_artifact_envelope
from grimace._south_star1.writer_branch_transition_artifact import branch_transition_artifact_manifest
from grimace._south_star1.writer_branch_transition_artifact import writer_branch_transition_artifact_for_support
from grimace._south_star1.writer_branch_transition_artifact_checker import verify_writer_branch_transition_artifact_consistency
from grimace._south_star1.writer_branch_transition_artifact_fact_verifier import verify_writer_branch_transition_artifact_for_facts
from grimace._south_star1.writer_envelope_terms import _digest_terms_bounded
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_support_artifact_checker import artifact_metrics
from grimace._south_star1.writer_support_artifact_checker import support_artifact_object_identity_term
from grimace._south_star1.writer_events import WriterRingEndpointEmitted
from grimace._south_star1.writer_events import WriterRingEndpointPaired
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_snapshot import WriterDecoderBoundary
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from tests.south_star1.test_writer_stereo_residual import _shared_directional_ring_carrier_facts
from tests.south_star1.test_writer_stereo_residual import _directional_non_single_ring_carrier_facts
from tests.south_star1.test_writer_stereo_residual import _directional_ring_carrier_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import shared_acyclic_directional_facts
from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.test_writer_support_artifact_fact_verifier import _initial_snapshot
from tests.south_star1.test_writer_support_artifact_fact_verifier import _prepare
from tests.south_star1.test_writer_support_artifact_fact_verifier import _writer_options


class WriterBranchTransitionArtifactTest(unittest.TestCase):
    def test_supported_transition_matrix_replays_facts_bound(self) -> None:
        cases = (
            (tetrahedral_facts(), _writer_options(), "tetrahedral atom-token restriction"),
            (tetrahedral_facts(), _writer_options(), "tetrahedral local-order factor closure"),
            (directional_facts(), _writer_options(rooted_at_atom=2), "directional carrier-mark restriction"),
            (shared_acyclic_directional_facts(), _writer_options(rooted_at_atom=0), "directional carrier-mark restriction"),
            (_directional_ring_carrier_facts(), _writer_options(rooted_at_atom=0), "directional ring endpoint projection"),
            (_directional_ring_carrier_facts(), _writer_options(rooted_at_atom=0), "directional ring pair restriction"),
            (_directional_non_single_ring_carrier_facts(), _writer_options(rooted_at_atom=0), "directional ring pair restriction"),
        )
        for facts, options, operation in cases:
            with self.subTest(operation=operation, facts=type(facts).__name__):
                prepared, artifact = _branch_artifact_for_operation(facts, options, operation)
                live = verify_writer_branch_transition_artifact_envelope(prepared=prepared, artifact=artifact)
                facts_bound = verify_writer_branch_transition_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=artifact,
                )
                self.assertTrue(live.accepted, live.reason)
                self.assertTrue(facts_bound.accepted, facts_bound.reason)
                self.assertEqual(facts_bound.unchecked_obligation_families, ())
                self.assertIn(operation, facts_bound.semantically_replayed_operations)

    def test_shared_ring_opening_and_pair_branches_are_typed_incomplete(self) -> None:
        for phase in ("opening", "pair"):
            for mark in (DirectionMark.ABSENT, DirectionMark.FWD, DirectionMark.REV):
                with self.subTest(phase=phase, mark=mark):
                    facts, options, prepared, artifact = _shared_ring_branch_artifact(phase, mark)

                    structural = verify_writer_branch_transition_artifact_consistency(artifact)
                    live = verify_writer_branch_transition_artifact_envelope(
                        prepared=prepared,
                        artifact=artifact,
                    )
                    facts_bound = verify_writer_branch_transition_artifact_for_facts(
                        facts=facts,
                        runtime_options=options,
                        artifact=artifact,
                    )

                    self.assertTrue(structural.accepted, structural.reason)
                    self.assertTrue(live.accepted, live.reason)
                    self.assertTrue(facts_bound.accepted, facts_bound.reason)
                    self.assertEqual(len(artifact["objects"]), 3)
                    self.assertLess(
                        artifact["metrics"]["largest_object_identity_input_bytes"],
                        WriterEnvelopeWorkBudget().max_digest_term_bytes,
                    )
                    self.assertEqual(
                        facts_bound.unchecked_obligation_families,
                        ("shared_directional_ring_transition_replay",),
                    )

    def test_build_and_live_verification_do_not_enter_count_or_support_paths(self) -> None:
        facts, options, prepared, snapshot, support = _shared_ring_branch_sources()[
            ("opening", DirectionMark.FWD)
        ]
        del facts, options
        blockers = (
            patch(
                "grimace._south_star1.writer_frontier_count_envelope."
                "writer_frontier_count_envelope_for_snapshot",
                side_effect=AssertionError("count envelope invoked"),
            ),
            patch(
                "grimace._south_star1.writer_count_dag_envelope."
                "writer_count_certificate_dag_envelope_for_product",
                side_effect=AssertionError("count DAG invoked"),
            ),
            patch(
                "grimace._south_star1.writer_snapshot."
                "_iter_writer_snapshot_certified_support_strings",
                side_effect=AssertionError("support enumeration invoked"),
                create=True,
            ),
        )
        with blockers[0], blockers[1], blockers[2]:
            artifact = writer_branch_transition_artifact_for_support(
                prepared=prepared,
                snapshot=snapshot,
                support=support,
            )
            live = verify_writer_branch_transition_artifact_envelope(
                prepared=prepared,
                artifact=artifact,
            )
        self.assertTrue(live.accepted, live.reason)

    def test_coherently_redigested_emitted_text_forgery_is_live_rejected(self) -> None:
        _facts, _options, prepared, artifact = _shared_ring_branch_artifact(
            "opening",
            DirectionMark.FWD,
        )
        forged = deepcopy(artifact)
        objects = {item["kind"]: item for item in forged["objects"]}
        objects["branch_support"]["payload"]["emitted_text"] += "X"
        objects["text_projection"]["payload"]["emitted_text"] += "X"
        _redigest_branch_artifact(forged)

        structural = verify_writer_branch_transition_artifact_consistency(forged)
        live = verify_writer_branch_transition_artifact_envelope(
            prepared=prepared,
            artifact=forged,
        )

        self.assertTrue(structural.accepted, structural.reason)
        self.assertFalse(live.accepted)
        self.assertIn("live_branch_artifact_mismatch", live.reason)

    def test_count_object_and_duplicate_object_are_structurally_rejected(self) -> None:
        _facts, _options, _prepared, artifact = _shared_ring_branch_artifact(
            "opening",
            DirectionMark.ABSENT,
        )
        duplicate = deepcopy(artifact)
        duplicate["objects"].append(deepcopy(duplicate["objects"][0]))
        self.assertFalse(
            verify_writer_branch_transition_artifact_consistency(duplicate).accepted
        )

        count_object = deepcopy(artifact)
        count_object["objects"][0]["kind"] = "count_dag"
        self.assertFalse(
            verify_writer_branch_transition_artifact_consistency(count_object).accepted
        )

        old_schema = deepcopy(artifact)
        old_schema["schema_version"] = 0
        rejected = verify_writer_branch_transition_artifact_consistency(old_schema)
        self.assertFalse(rejected.accepted)
        self.assertIn("unknown_schema_version", rejected.reason)


@lru_cache(maxsize=6)
def _shared_ring_branch_artifact(phase: str, mark: DirectionMark):
    facts, options, prepared, snapshot, support = _shared_ring_branch_sources()[(phase, mark)]
    artifact = writer_branch_transition_artifact_for_support(
        prepared=prepared,
        snapshot=snapshot,
        support=support,
    )
    return facts, options, prepared, artifact


@lru_cache(maxsize=1)
def _shared_ring_branch_sources():
    facts = _shared_directional_ring_carrier_facts()
    options = _writer_options(rooted_at_atom=1)
    prepared = _prepare(facts)
    initial = _initial_snapshot(prepared, options)
    pending = [(initial.cursor, 0)]
    seen = set()
    found = {}
    while pending and len(found) < 6:
        cursor, depth = pending.pop()
        key = repr(cursor)
        if key in seen:
            continue
        seen.add(key)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=True,
            include_count_certificate=False,
        )
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=cursor,
            decoder_boundary=WriterDecoderBoundary(consumed_token_count=depth),
        )
        for support in batch.supports:
            for event in support.events:
                if isinstance(event, WriterRingEndpointEmitted) and event.bond == BondId(1):
                    found.setdefault(("opening", event.direction_mark), (facts, options, prepared, snapshot, support))
                if isinstance(event, WriterRingEndpointPaired) and event.bond == BondId(1):
                    found.setdefault(("pair", event.first_endpoint_direction_mark), (facts, options, prepared, snapshot, support))
            pending.append((support.successor_cursor, depth + 1))
    if len(found) != 6:
        raise AssertionError(f"missing shared-ring branch sources: {sorted(found)}")
    return found


def _branch_artifact_for_operation(facts, options, operation):
    prepared = _prepare(facts)
    initial = _initial_snapshot(prepared, options)
    pending = [(initial.cursor, 0)]
    seen = set()
    while pending:
        cursor, depth = pending.pop()
        key = repr(cursor)
        if key in seen:
            continue
        seen.add(key)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=True,
            include_count_certificate=False,
        )
        for support in batch.supports:
            if any(item.operation == operation for item in support.residual_work_evidence):
                snapshot = capture_writer_frontier_snapshot(
                    prepared=prepared,
                    runtime_options=options,
                    cursor=cursor,
                    decoder_boundary=WriterDecoderBoundary(consumed_token_count=depth),
                )
                return prepared, writer_branch_transition_artifact_for_support(
                    prepared=prepared,
                    snapshot=snapshot,
                    support=support,
                )
            pending.append((support.successor_cursor, depth + 1))
    raise AssertionError(f"missing branch operation {operation!r}")


def _redigest_branch_artifact(artifact) -> None:
    budget = WriterEnvelopeWorkBudget()
    by_kind = {item["kind"]: item for item in artifact["objects"]}
    branch = by_kind["branch_support"]
    branch_digest = _identity_digest(
        support_artifact_object_identity_term(branch["kind"], branch["payload"]),
        budget=budget,
        operation="test.branch_transition.branch_object",
    )
    branch["digest"] = branch_digest
    branch["object_id"] = f"obj:{branch_digest}"
    projection = by_kind["text_projection"]
    projection["payload"]["branch_support_refs"] = [branch["object_id"]]
    identity = {
        key: value
        for key, value in projection["payload"].items()
        if key not in ("digest", "branch_support_refs")
    }
    projection["payload"]["digest"] = _identity_digest(
        identity,
        budget=budget,
        operation="test.branch_transition.projection_identity",
    )
    projection_digest = _identity_digest(
        support_artifact_object_identity_term(projection["kind"], projection["payload"]),
        budget=budget,
        operation="test.branch_transition.projection_object",
    )
    projection["digest"] = projection_digest
    projection["object_id"] = f"obj:{projection_digest}"
    artifact["roots"]["branch_support_ref"] = branch["object_id"]
    artifact["roots"]["text_projection_ref"] = projection["object_id"]
    artifact["objects"] = sorted(artifact["objects"], key=lambda item: item["object_id"])
    metrics = artifact_metrics(artifact["objects"])
    artifact["metrics"] = {**metrics, "reachable_object_count": 3, "unreferenced_object_count": 0}
    artifact["digest"] = _digest_terms_bounded(
        branch_transition_artifact_manifest(artifact),
        budget=budget,
        operation="test.branch_transition.artifact",
    )


if __name__ == "__main__":
    unittest.main()
