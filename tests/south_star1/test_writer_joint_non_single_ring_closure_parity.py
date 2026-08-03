"""RDKit-derived parity coverage for default joint non-single ring closures."""

from __future__ import annotations

import unittest

from grimace._south_star1.fact_isomorphism import facts_are_isomorphic
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_closure_bond_text_lifecycle import (
    WriterClosureBondTextLifecycleEvidence,
)
from grimace._south_star1.ids import BondId
from grimace._south_star1.slots import BondSlotKind
from grimace._south_star1.writer_frontier import (
    _checked_writer_frontier_branch_supports,
)
from grimace._south_star1.writer_frontier import (
    _snapshot_advance_writer_frontier_product,
)
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_runtime import count_writer_runtime_completions
from grimace._south_star1.writer_runtime import count_writer_runtime_support
from grimace._south_star1.writer_runtime import initial_writer_runtime_state
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot import (
    _writer_snapshot_advance_outcome_by_emitted_text,
)
from grimace._south_star1.writer_support import (
    enumerate_prepared_writer_shaped_support,
)
from grimace._south_star1.writer_support_artifact_checker import (
    verify_writer_support_artifact_consistency,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    verify_writer_support_artifact_envelope,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_snapshot,
)
from grimace._south_star1.writer_support_artifact_fact_verifier import (
    verify_writer_support_artifact_for_facts,
)
from grimace._south_star1.errors import SouthStarError
from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options
from tests.south_star1.default_writer_capability_ledger import default_writer_capability_case
from tests.south_star1.qualification_assertions import assert_offline_case_matches_ledger


_DOUBLE_RING_CLOSURE_SMILES = "C1=CC1"
_TRIPLE_RING_CLOSURE_SMILES = "C1#CC1"
_RDKit_GRAPH_ONLY_OPTIONS = RdkitOrdinaryExtractionOptions(
    include_potential_sites=False,
)
_RDKit_WITH_POTENTIAL_STEREO_OPTIONS = RdkitOrdinaryExtractionOptions(
    include_potential_sites=True,
)


class WriterJointNonSingleRingClosureParityTest(unittest.TestCase):
    def test_default_policy_accepts_rdkit_double_ring_closure_candidate(
        self,
    ) -> None:
        facts = _rdkit_graph_facts(_DOUBLE_RING_CLOSURE_SMILES)

        prepared = prepare_writer_facts(facts)

        ring_choices = prepared.policy.bond_text_domain(
            facts,
            BondId(0),
            slot_kind=BondSlotKind.RING_ENDPOINT.value,
        )
        self.assertEqual({choice.base_text for choice in ring_choices}, {"", "="})

    def test_explicit_unsupported_policy_rejects_rdkit_double_ring_closure_candidate(
        self,
    ) -> None:
        facts = _rdkit_graph_facts(_DOUBLE_RING_CLOSURE_SMILES)

        with self.assertRaisesRegex(SouthStarError, "non-single ring closures"):
            ordinary_policy_for_facts(
                facts,
                OrdinaryPolicyOptions(non_single_ring_closures="unsupported"),
            )

    def test_joint_policy_accepts_rdkit_double_ring_closure_candidate(
        self,
    ) -> None:
        facts = _rdkit_graph_facts(_DOUBLE_RING_CLOSURE_SMILES)

        prepared = prepare_joint_non_single_ring_facts(facts)

        ring_choices = prepared.policy.bond_text_domain(
            facts,
            BondId(0),
            slot_kind=BondSlotKind.RING_ENDPOINT.value,
        )
        self.assertEqual({choice.base_text for choice in ring_choices}, {"", "="})

    def test_joint_policy_support_count_matches_materialized_support(
        self,
    ) -> None:
        prepared = prepare_writer_facts(_rdkit_graph_facts(_DOUBLE_RING_CLOSURE_SMILES))
        options = writer_runtime_options(rooted_at_atom=0)
        image = enumerate_prepared_writer_shaped_support(
            prepared=prepared,
            runtime_options=options,
        )
        state = initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=options,
        )

        self.assertEqual(image.distinct_count, 3)
        self.assertEqual(image.witness_count, 3)
        self.assertEqual(len(set(image.strings)), image.distinct_count)
        self.assertEqual(
            count_writer_runtime_support(prepared=prepared, state=state),
            image.distinct_count,
        )
        self.assertEqual(
            count_writer_runtime_completions(prepared=prepared, state=state),
            image.witness_count,
        )

    def test_default_policy_artifact_structural_and_live_verifiers_accept(
        self,
    ) -> None:
        facts = _rdkit_graph_facts(_DOUBLE_RING_CLOSURE_SMILES)
        prepared = prepare_writer_facts(facts)
        options = writer_runtime_options(rooted_at_atom=0)
        artifact = _snapshot_artifact(prepared, options)

        structural = verify_writer_support_artifact_consistency(artifact)
        live = verify_writer_support_artifact_envelope(
            prepared=prepared,
            envelope=artifact,
        )

        self.assertTrue(structural.accepted, structural.reason)
        self.assertTrue(live.accepted, live.reason)
        self.assertEqual(structural.support_count, 3)
        self.assertEqual(live.witness_count, 3)

    def test_default_policy_facts_bound_verifier_accepts(self) -> None:
        case = default_writer_capability_case("cyclopropene_double_closure")
        facts = _rdkit_graph_facts(_DOUBLE_RING_CLOSURE_SMILES)
        prepared = prepare_writer_facts(facts)
        options = writer_runtime_options(rooted_at_atom=0)
        artifact = _snapshot_artifact(prepared, options)

        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=artifact,
        )

        self.assertTrue(verification.accepted, verification.reason)
        self.assertTrue(verification.structurally_checked)
        self.assertTrue(verification.facts_identity_checked)
        assert_offline_case_matches_ledger(
            self,
            case=case,
            verification=verification,
        )

    def test_explicit_unsupported_policy_fails_before_fact_verification(
        self,
    ) -> None:
        facts = _rdkit_graph_facts(_DOUBLE_RING_CLOSURE_SMILES)

        with self.assertRaisesRegex(SouthStarError, "non-single ring closures"):
            ordinary_policy_for_facts(
                facts,
                OrdinaryPolicyOptions(non_single_ring_closures="unsupported"),
            )

    def test_joint_policy_generated_strings_round_trip_through_rdkit_audit(
        self,
    ) -> None:
        facts = _rdkit_graph_facts(_DOUBLE_RING_CLOSURE_SMILES)
        image = enumerate_prepared_writer_shaped_support(
            prepared=prepare_writer_facts(facts),
            runtime_options=writer_runtime_options(rooted_at_atom=0),
        )

        for text in image.strings:
            with self.subTest(text=text):
                reparsed = _rdkit_graph_facts(text)
                self.assertTrue(
                    facts_are_isomorphic(facts, reparsed).isomorphic,
                    text,
                )

    def test_joint_policy_branch_evidence_contains_closure_bond_text_lifecycle(
        self,
    ) -> None:
        prepared = prepare_writer_facts(_rdkit_graph_facts(_DOUBLE_RING_CLOSURE_SMILES))

        evidence = _closure_bond_text_lifecycle_evidence(prepared)

        self.assertTrue(evidence)
        self.assertTrue(
            all(
                isinstance(item, WriterClosureBondTextLifecycleEvidence)
                for item in evidence
            )
        )
        self.assertEqual({item.bond_order for item in evidence}, {"double"})
        self.assertEqual(
            {item.marker_side for item in evidence},
            {"opening", "closing"},
        )
        self.assertTrue(
            all(
                (item.opening_marker == "=") ^ (item.closing_marker == "=")
                for item in evidence
            )
        )

    def test_joint_policy_artifact_contains_required_support_objects(
        self,
    ) -> None:
        artifact = _snapshot_artifact(
            prepare_writer_facts(_rdkit_graph_facts(_DOUBLE_RING_CLOSURE_SMILES)),
            writer_runtime_options(rooted_at_atom=0),
        )
        kind_counts = artifact["metrics"]["object_kind_counts"]
        root = _artifact_object(artifact, artifact["roots"]["support_image_root"])

        self.assertEqual(kind_counts["count_envelope"], 1)
        self.assertEqual(kind_counts["frontier_product"], 1)
        self.assertGreater(kind_counts["support_string"], 0)
        self.assertGreater(kind_counts["terminal_projection"], 0)
        self.assertGreater(kind_counts["terminal_support"], 0)
        self.assertEqual(root["kind"], "support_image")
        self.assertEqual(root["payload"]["distinct_count"], 3)
        self.assertEqual(root["payload"]["witness_count"], 3)

    def test_joint_policy_triple_rdkit_fixture_parity(self) -> None:
        facts = _rdkit_graph_facts(_TRIPLE_RING_CLOSURE_SMILES)
        prepared = prepare_writer_facts(facts)
        image = enumerate_prepared_writer_shaped_support(
            prepared=prepared,
            runtime_options=writer_runtime_options(rooted_at_atom=0),
        )

        self.assertEqual(image.distinct_count, 3)
        self.assertEqual(image.witness_count, 3)
        for text in image.strings:
            with self.subTest(text=text):
                reparsed = _rdkit_graph_facts(text)
                self.assertTrue(
                    facts_are_isomorphic(facts, reparsed).isomorphic,
                    text,
                )

    def test_potential_stereo_enabled_fixture_blocks_with_typed_stereo_evidence(
        self,
    ) -> None:
        facts = ordinary_molecule_facts_from_smiles(
            _DOUBLE_RING_CLOSURE_SMILES,
            _RDKit_WITH_POTENTIAL_STEREO_OPTIONS,
        )
        prepared = prepare_writer_facts(facts)

        blockers = _reachable_stereo_policy_blockers(prepared)

        self.assertTrue(blockers)
        self.assertEqual(
            {blocker.kind for blocker in blockers},
            {"unsupported_directional_non_neighbor_ligand"},
        )
        self.assertEqual(
            {blocker.operation for blocker in blockers},
            {"directional carrier-mark restriction"},
        )


def _rdkit_graph_facts(smiles: str):
    return ordinary_molecule_facts_from_smiles(
        smiles,
        _RDKit_GRAPH_ONLY_OPTIONS,
    )


def prepare_joint_non_single_ring_facts(facts):
    return prepare_writer_facts(
        facts,
        policy=ordinary_policy_for_facts(
            facts,
            OrdinaryPolicyOptions(non_single_ring_closures="joint"),
        ),
    )


def _snapshot_artifact(prepared, options):
    return writer_support_artifact_envelope_for_snapshot(
        prepared=prepared,
        snapshot=initial_writer_snapshot(prepared, options),
    )


def _artifact_object(artifact, object_id):
    for item in artifact["objects"]:
        if item["object_id"] == object_id:
            return item
    raise AssertionError(f"missing artifact object: {object_id}")


def _closure_bond_text_lifecycle_evidence(prepared):
    pending = [initial_writer_frontier_cursor(prepared, writer_runtime_options(rooted_at_atom=0))]
    seen = set()
    evidence = []
    while pending:
        cursor = pending.pop(0)
        if cursor in seen:
            continue
        seen.add(cursor)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
        )
        for support in batch.supports:
            ring = support.successor_state_certificate.ring_replay_certificate
            if ring is not None:
                evidence.extend(ring.closure_bond_text_lifecycle_evidence)
            pending.append(support.successor_cursor)
    return tuple(evidence)


def _reachable_stereo_policy_blockers(prepared):
    pending = [(initial_writer_runtime_state(
        prepared=prepared,
        runtime_options=writer_runtime_options(rooted_at_atom=0),
    ).snapshot, ())]
    seen = set()
    blockers = []
    while pending:
        snapshot, emitted_texts = pending.pop(0)
        if snapshot.cursor in seen:
            continue
        seen.add(snapshot.cursor)
        product = _snapshot_advance_writer_frontier_product(
            prepared,
            snapshot.cursor,
        )
        if product.blocked:
            blockers.extend(
                item.blocker
                for item in (
                    product
                    .blocked_frontier_certificate
                    .stereo_policy_blocker_certificates
                )
            )
            continue
        projection = product.projection_certificate
        if projection.terminal_projection_certificate is not None:
            continue
        for text_projection in projection.text_choice_projection_certificates:
            outcome = _writer_snapshot_advance_outcome_by_emitted_text(
                snapshot,
                prepared=prepared,
                emitted_text=text_projection.emitted_text,
            )
            if outcome.advanced_snapshot is not None:
                pending.append(
                    (
                        outcome.advanced_snapshot,
                        emitted_texts + (text_projection.emitted_text,),
                    )
                )
    return tuple(blockers)


if __name__ == "__main__":
    unittest.main()
