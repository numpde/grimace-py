"""Physically owned rich support-artifact tetra-lifecycle contracts.Facts-bound writer support artifact verifier tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import os
from types import SimpleNamespace
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import StereoFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_envelope_terms import _digest_terms_bounded
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_events import WriterRingEndpointEmitted
from grimace._south_star1.writer_events import WriterRingEndpointPaired
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import WriterDecoderBoundary
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot_prefix_envelope import (
    writer_snapshot_prefix_read_envelope_for_emitted_texts,
)
import grimace._south_star1.writer_stereo as writer_stereo_module
import grimace._south_star1.writer_support_artifact_offline_verifier as offline_verifier_module
from grimace._south_star1.writer_support_artifact_checker import (
    verify_writer_support_artifact_consistency,
)
from grimace._south_star1.writer_support_artifact_fact_verifier import (
    OBJECT_KIND_OFFLINE_COVERAGE,
)
from grimace._south_star1.writer_support_artifact_fact_verifier import (
    verify_writer_support_artifact_for_facts,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    classify_residual_stereo_obligations_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    validate_writer_bracket_atom_text_against_facts,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_branch_projection_identities_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_count_dag_arithmetic,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_graph_ring_branch_deltas_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_local_branch_successor_evidence_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_support_image_coverage_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_support_string_replay_paths_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_terminal_support_identities_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_writer_support_artifact_offline_replay,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    verify_writer_support_artifact_envelope,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_prefix_read,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_snapshot,
)
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import shared_acyclic_directional_facts
from tests.south_star1.writer_artifact_resealing import reseal_support_artifact
from tests.south_star1.writer_artifact_test_support import artifact_object_by_id
from tests.south_star1.writer_artifact_test_support import closed_term_digest
from tests.south_star1.writer_artifact_test_support import closed_term_field
from tests.south_star1.writer_artifact_test_support import set_closed_term_field
from tests.south_star1.writer_artifact_test_support import refresh_cursor_digest
from tests.south_star1.writer_artifact_test_support import refresh_closed_term_digest_field
from tests.south_star1.writer_artifact_test_support import refresh_kind_manifest_digest
from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options
from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.writer_test_fixtures import (
    directional_non_single_ring_carrier_facts,
)
from tests.south_star1.writer_test_fixtures import (
    directional_ring_carrier_facts,
)
from tests.south_star1.writer_test_fixtures import (
    shared_directional_ring_carrier_facts,
)
from tests.south_star1.helpers import two_atom_facts

RUN_SLOW_ENV = "SOUTH_STAR1_RUN_SLOW"



from tests.south_star1.writer_support_artifact_fixtures import completed_prefix_support_artifact_fixture, rdkit_graph_facts, rdkit_support_artifact_fixture, rdkit_support_artifact_verification, support_artifact_fixture, tetra_support_artifact_fixture
from tests.south_star1.writer_support_artifact_queries import (
    classify_obligation_replay, coverage_object, cursor_state_by_digest, first_branch_support_object,
    first_graph_ring_delta_branch, first_graph_ring_delta_event, first_local_evidence, first_residual_work_branch,
    first_support_string_object, first_terminal_projection_object, first_terminal_support_object, first_text_projection_object,
    require_structurally_valid_support_artifact, single_cursor_state, support_strings, text_projection_for_branch,
    verify_branch_projection_relation, verify_graph_ring_delta_relation, verify_local_branch_evidence_relation,
    verify_support_image_coverage_relation, verify_support_string_replay_relation, verify_terminal_identity_relation,
)
from tests.south_star1.writer_support_artifact_transition_test_support import (linked_tetra_lifecycle_manifest, refresh_linked_raw_lifecycle_residual_digest, text_projection_identity_digest, propagate_text_projection_cursor_change)
from tests.south_star1.writer_support_artifact_graph_test_support import (first_directional_bond_delta_branch, first_closure_evidence_item, tetra_facts_with_implicit_h_only_outside_specified_site)
from tests.south_star1.writer_support_artifact_tetra_test_support import (append_unrelated_raw_lifecycle, different_local_order_digest, refresh_local_order_event_identity_digest)
from tests.south_star1.writer_support_artifact_directional_test_support import (directional_transition_branch_and_manifest, directional_transition_manifest, directional_discharge_key_pairs, bond_occurrence_terms_for_branch, mutate_directional_restriction_sign, mutate_directional_canonical_orientation, mutate_directional_model_field, remove_directional_model, remove_directional_restriction, duplicate_directional_model_site, mutate_directional_successor_snapshot, set_directional_discharges, set_directional_discharges_by_keys, remove_raw_lifecycle_capability, mutate_directional_term_mark, mutate_directional_term_bond, remove_directional_successor_bond_occurrence, duplicate_directional_successor_bond_occurrence, mutate_directional_successor_snapshot_unrelated)

class WriterSupportArtifactTetraLifecycleTest(unittest.TestCase):
    def test_specified_tetra_no_second_authority_from_atom_token_text(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        event = first_graph_ring_delta_event(branch, "atom_emitted")
        self.assertIn(event["tetra_token"]["value"], ("@", "@@"))
        self.assertTrue(
            any(event["tetra_token"]["value"] in text for text in support_strings(artifact))
        )
        manifest["linked_lifecycle_digests"] = []

        classification = classify_obligation_replay(facts=facts, artifact=artifact)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )
        self.assertFalse(verification.accepted)
        self.assertIn("object_digest_mismatch", verification.reason)

    def test_specified_tetra_no_second_authority_from_final_support_strings(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = linked_tetra_lifecycle_manifest(
            branch=branch,
            manifest=manifest,
            lifecycle_kind="certificate",
            certificate_kind="tetra_token_restricted",
        )
        self.assertTrue(any("@" in text for text in support_strings(artifact)))
        lifecycle["certificate_capability"] = "tetra_local_order_restriction"

        classification = classify_obligation_replay(facts=facts, artifact=artifact)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_certificate_capability_mismatch",
            classification.reason,
        )
        self.assertFalse(verification.accepted)
        self.assertIn("object_digest_mismatch", verification.reason)

    def test_specified_tetra_no_second_authority_from_stale_lifecycle_digest(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = linked_tetra_lifecycle_manifest(
            branch=branch,
            manifest=manifest,
            lifecycle_kind="raw",
            certificate_kind="tetra_token_restricted",
        )
        lifecycle["source_digest"] = "stale_source_digest"

        classification = classify_obligation_replay(facts=facts, artifact=artifact)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_lifecycle_source_mismatch",
            classification.reason,
        )
        self.assertFalse(verification.accepted)
        self.assertIn("object_digest_mismatch", verification.reason)

    def test_specified_tetra_residual_lifecycle_provenance_mutations_are_rejected(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        cases = (
            (
                "raw_event_kind",
                "raw",
                "lifecycle_event_kind",
                "local_order_closed",
                "tetra_residual_lifecycle_event_kind_mismatch",
            ),
            (
                "raw_missing_capability",
                "raw",
                "lifecycle_capabilities",
                ["tetra_token_restriction"],
                "tetra_residual_lifecycle_capabilities_mismatch",
            ),
            (
                "raw_extra_capability",
                "raw",
                "lifecycle_capabilities",
                [
                    "residual_factor_discharge",
                    "residual_propagation",
                    "tetra_token_restriction",
                ],
                "tetra_residual_lifecycle_capabilities_mismatch",
            ),
            (
                "raw_outcome",
                "raw",
                "lifecycle_outcome_kind",
                "event_recorded",
                "tetra_residual_lifecycle_outcome_kind_mismatch",
            ),
            (
                "raw_change_flag",
                "raw",
                "residual_snapshot_changed",
                False,
                "tetra_residual_lifecycle_change_flag_mismatch",
            ),
            (
                "raw_work_digests",
                "raw",
                "residual_work_digests",
                [],
                "residual_lifecycle_reverse_link_provenance_mismatch",
            ),
            (
                "raw_work_operations",
                "raw",
                "residual_work_operations",
                ["wrong"],
                "tetra_residual_lifecycle_work_operation_mismatch",
            ),
            (
                "certificate_kind",
                "certificate",
                "certificate_kind",
                "tetra_local_order_restricted",
                "tetra_residual_lifecycle_evidence_missing",
            ),
            (
                "certificate_capability",
                "certificate",
                "certificate_capability",
                "tetra_local_order_restriction",
                "tetra_residual_certificate_capability_mismatch",
            ),
            (
                "certificate_lifecycle_digest",
                "certificate",
                "certificate_lifecycle_digest",
                "wrong",
                "tetra_residual_certificate_lifecycle_digest_mismatch",
            ),
        )
        for name, lifecycle_kind, field, value, reason in cases:
            with self.subTest(name=name):
                mutated = deepcopy(artifact)
                branch = first_residual_work_branch(
                    mutated,
                    operation="tetrahedral atom-token restriction",
                )
                manifest = next(
                    item
                    for item in (
                        branch["payload"]["obligation_manifests"]["residual_work"]
                    )
                    if item["operation"] == "tetrahedral atom-token restriction"
                )
                lifecycle = linked_tetra_lifecycle_manifest(
                    branch=branch,
                    manifest=manifest,
                    lifecycle_kind=lifecycle_kind,
                    certificate_kind="tetra_token_restricted",
                )
                lifecycle[field] = value

                classification = classify_obligation_replay(facts=facts, artifact=mutated)

                self.assertFalse(classification.accepted)
                self.assertIn(reason, classification.reason)

    def test_specified_tetra_residual_reciprocal_extra_link_is_rejected(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = append_unrelated_raw_lifecycle(branch, manifest=manifest)
        manifest["linked_lifecycle_digests"].append(unrelated["evidence_digest"])
        unrelated["linked_residual_work_digests"].append(manifest["evidence_digest"])

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_bogus_reverse_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = next(
            item
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if item["evidence_digest"] in manifest["linked_lifecycle_digests"]
        )
        lifecycle["linked_residual_work_digests"].append("bogus")

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_coherently_forged_link_projection(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = append_unrelated_raw_lifecycle(branch, manifest=manifest)
        unrelated["linked_residual_work_digests"].append(manifest["evidence_digest"])
        unrelated["residual_work_digests"].append(manifest["evidence_digest"])
        manifest["linked_lifecycle_digests"] = [
            item["evidence_digest"]
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if manifest["evidence_digest"] in item["residual_work_digests"]
        ]

        # Public verification rejects this hand-edited artifact structurally via
        # stale object digests; this isolates the offline classifier precondition.
        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_lifecycle_evidence_missing",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_extra_lifecycle_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = append_unrelated_raw_lifecycle(branch, manifest=manifest)
        manifest["linked_lifecycle_digests"].append(unrelated["evidence_digest"])

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_reverse_only_lifecycle_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = next(
            item
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if item["evidence_digest"] in manifest["linked_lifecycle_digests"]
        )
        manifest["linked_lifecycle_digests"].remove(lifecycle["evidence_digest"])

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_unreciprocated_reverse_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        target = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        other = deepcopy(target)
        other["evidence_digest"] = "other_residual"
        other["linked_lifecycle_digests"] = []
        branch["payload"]["obligation_manifests"]["residual_work"].append(other)
        lifecycle = next(
            item
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if item["evidence_digest"] in target["linked_lifecycle_digests"]
        )
        lifecycle["linked_residual_work_digests"].append(other["evidence_digest"])

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_unrelated_lifecycle_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = append_unrelated_raw_lifecycle(branch, manifest=manifest)
        manifest["linked_lifecycle_digests"] = [unrelated["evidence_digest"]]

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_wrong_lifecycle_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        manifest["linked_lifecycle_digests"] = ["wrong"]

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_requires_exact_lifecycle_operations(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        lifecycle = branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
        branch_certificate = next(
            item
            for item in lifecycle
            if item["operation"] == "WriterStereoBranchCertificate"
        )
        branch_certificate["operation"] = "UnexpectedStereoCertificate"

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_lifecycle_operation_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_requires_lifecycle_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        manifest["linked_lifecycle_digests"] = []

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_requires_reverse_lifecycle_link(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = next(
            item
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if item["evidence_digest"] in manifest["linked_lifecycle_digests"]
        )
        lifecycle["linked_residual_work_digests"].remove(manifest["evidence_digest"])

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )
