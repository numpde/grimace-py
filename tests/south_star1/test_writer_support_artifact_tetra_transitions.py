"""Physically owned rich support-artifact tetra-transitions contracts.Facts-bound writer support artifact verifier tests."""

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

class WriterSupportArtifactTetraTransitionTest(unittest.TestCase):
    def test_specified_tetra_atom_token_rejects_coherent_detached_residual_snapshots(
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
        for field in ("source_snapshot", "successor_snapshot"):
            snapshot = closed_term_field(manifest["transition_term"], field)
            closed_term_field(snapshot, "domains").append(
                [
                    {
                        "__dataclass__": "grimace._south_star1.residual_constraints.VarId",
                        "fields": [["kind", "detached_test_component"], ["key", [123]]],
                    },
                    [False, True],
                ]
            )
            digest = closed_term_digest(
                snapshot,
                operation="test.tetra.detached_snapshot_record",
            )
            set_closed_term_field(
                manifest["transition_term"],
                f"{field}_digest",
                digest,
            )
            lifecycle_field = (
                "source_residual_snapshot_digest"
                if field == "source_snapshot"
                else "successor_residual_snapshot_digest"
            )
            refresh_linked_raw_lifecycle_residual_digest(
                branch,
                manifest=manifest,
                field=lifecycle_field,
                digest=digest,
            )
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        require_structurally_valid_support_artifact(artifact)

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_source_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_atom_token_residual_event_atom_is_bound(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        event = first_graph_ring_delta_event(branch, "atom_emitted")
        event["atom"] = 1

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_atom_token_residual_atom_mismatch",
            classification.reason,
        )

    def test_specified_tetra_atom_token_residual_token_matches_text(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        event = first_graph_ring_delta_event(branch, "atom_emitted")
        token = event["tetra_token"]
        token["value"] = "@@" if token["value"] == "@" else "@"

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_atom_token_residual_token_mismatch",
            classification.reason,
        )

    def test_specified_tetra_local_order_closed_atom_is_center(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        event = first_graph_ring_delta_event(branch, "local_order_closed")
        event["atom"] = 1

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_residual_center_mismatch",
            classification.reason,
        )

    def test_specified_tetra_local_order_event_bond_connects_ligand(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        event = first_graph_ring_delta_event(branch, "atom_emitted")
        event["incoming_bond"] = 99

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_residual_bond_mismatch",
            classification.reason,
        )

    def test_specified_tetra_local_order_event_parent_is_center(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        event = first_graph_ring_delta_event(branch, "atom_emitted")
        event["parent"] = event["atom"]

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_residual_parent_mismatch",
            classification.reason,
        )

    def test_specified_tetra_local_order_rejects_coherent_detached_record_identity(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        event = first_graph_ring_delta_event(branch, "local_order_closed")
        event["successor_local_order_record_digest"] = different_local_order_digest(
            artifact,
            branch=branch,
            cursor_name="successor_cursor",
            atom=event["atom"],
        )
        refresh_local_order_event_identity_digest(event)
        refresh_kind_manifest_digest(branch["payload"]["graph_ring_delta"], operation="test.graph_ring_delta.digest")
        reseal_support_artifact(artifact)
        require_structurally_valid_support_artifact(artifact)

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_successor_record_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_local_order_rejects_coherent_detached_source_record_identity(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        event = first_graph_ring_delta_event(branch, "local_order_closed")
        event["source_local_order_record_digest"] = different_local_order_digest(
            artifact,
            branch=branch,
            cursor_name="source_cursor",
            atom=event["atom"],
        )
        refresh_local_order_event_identity_digest(event)
        refresh_kind_manifest_digest(branch["payload"]["graph_ring_delta"], operation="test.graph_ring_delta.digest")
        reseal_support_artifact(artifact)
        require_structurally_valid_support_artifact(artifact)

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_source_record_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_raw_smiles_blocks_without_potential_sites(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            SouthStarError,
            "raw tetrahedral stereo has no ordinary potential site",
        ) as raised:
            ordinary_molecule_facts_from_smiles(
                "[C@H](F)(Cl)Br",
                RdkitOrdinaryExtractionOptions(include_potential_sites=False),
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

    def test_specified_tetra_residual_manifest_digest_mismatch_is_rejected(
        self,
    ) -> None:
        facts = tetrahedral_facts()
        prepared = prepare_writer_facts(facts)
        options = writer_runtime_options()
        artifact = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, options),
        )
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = branch["payload"]["obligation_manifests"]["residual_work"][0]
        manifest["source_digest"] = "wrong"

        classification = classify_obligation_replay(facts=facts, artifact=artifact)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=artifact,
        )

        self.assertFalse(classification.accepted)
        self.assertIn("tetra_residual_source_digest_mismatch", classification.reason)
        self.assertFalse(verification.accepted)
        self.assertIn("object_digest_mismatch", verification.reason)

    def test_specified_tetra_transition_rejects_branch_detached_snapshots(
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
        successor = closed_term_field(manifest["transition_term"], "successor_snapshot")
        set_closed_term_field(manifest["transition_term"], "source_snapshot", successor)
        set_closed_term_field(
            manifest["transition_term"],
            "source_snapshot_digest",
            closed_term_digest(
                successor,
                operation="test.tetra.detached_snapshot",
            ),
        )
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        require_structurally_valid_support_artifact(artifact)

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_source_lifecycle_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_event_detached_local_order(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral local-order factor closure"
        )
        local_order = closed_term_field(manifest["transition_term"], "local_order")
        local_order[:] = list(reversed(local_order))
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        require_structurally_valid_support_artifact(artifact)

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_event_order_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_flipped_local_order_parity(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral local-order factor closure"
        )
        target = closed_term_field(manifest["transition_term"], "target_parity")
        target["value"] = "odd" if target["value"] == "even" else "even"
        constraint = closed_term_field(manifest["transition_term"], "constraint_value")
        constraint["value"] = target["value"]
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        require_structurally_valid_support_artifact(artifact)

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_target_parity_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_flipped_reference_order(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral local-order factor closure"
        )
        reference_order = closed_term_field(manifest["transition_term"], "reference_order")
        reference_order[:] = list(reversed(reference_order))
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        require_structurally_valid_support_artifact(artifact)

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_reference_order_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_missing_factor_discharge(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral local-order factor closure"
        )
        set_closed_term_field(manifest["transition_term"], "discharged_factor_keys", [])
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        require_structurally_valid_support_artifact(artifact)

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_discharge_factor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_state_detached_source_projection(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral local-order factor closure"
        )
        source = closed_term_field(manifest["transition_term"], "source_snapshot")
        constraint_var = closed_term_field(manifest["transition_term"], "constraint_var")
        domains = closed_term_field(source, "domains")
        domains[:] = [
            item
            for item in domains
            if item[0] != constraint_var
        ]
        source_digest = closed_term_digest(
            source,
            operation="test.tetra.source_projection",
        )
        set_closed_term_field(
            manifest["transition_term"],
            "source_snapshot_digest",
            source_digest,
        )
        refresh_linked_raw_lifecycle_residual_digest(
            branch,
            manifest=manifest,
            field="source_residual_snapshot_digest",
            digest=source_digest,
        )
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        require_structurally_valid_support_artifact(artifact)

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_source_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_state_detached_successor_component(
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
        successor = closed_term_field(manifest["transition_term"], "successor_snapshot")
        domains = closed_term_field(successor, "domains")
        domains.append(
            [
                {
                    "__dataclass__": "grimace._south_star1.residual_constraints.VarId",
                    "fields": [["kind", "unrelated_test_component"], ["key", [99]]],
                },
                [False, True],
            ]
        )
        successor_digest = closed_term_digest(
            successor,
            operation="test.tetra.successor_component",
        )
        set_closed_term_field(
            manifest["transition_term"],
            "successor_snapshot_digest",
            successor_digest,
        )
        refresh_linked_raw_lifecycle_residual_digest(
            branch,
            manifest=manifest,
            field="successor_residual_snapshot_digest",
            digest=successor_digest,
        )
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        require_structurally_valid_support_artifact(artifact)

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_successor_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_state_detached_successor_discharge(
        self,
    ) -> None:
        fixture = tetra_support_artifact_fixture()
        facts, artifact = fixture.facts, fixture.artifact
        branch = first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral local-order factor closure"
        )
        transition = manifest["transition_term"]
        source = closed_term_field(transition, "source_snapshot")
        successor = closed_term_field(transition, "successor_snapshot")
        discharged = closed_term_field(transition, "discharged_factor_keys")
        source_factor = next(
            factor
            for factor in closed_term_field(source, "factors")
            if closed_term_field(factor, "key") == discharged[0]
        )
        closed_term_field(successor, "factors").append(source_factor)
        successor_digest = closed_term_digest(
            successor,
            operation="test.tetra.successor_discharge",
        )
        set_closed_term_field(
            transition,
            "successor_snapshot_digest",
            successor_digest,
        )
        refresh_linked_raw_lifecycle_residual_digest(
            branch,
            manifest=manifest,
            field="successor_residual_snapshot_digest",
            digest=successor_digest,
        )
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        require_structurally_valid_support_artifact(artifact)

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_successor_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_state_detached_successor_token_domain(
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
        successor = closed_term_field(manifest["transition_term"], "successor_snapshot")
        domains = closed_term_field(successor, "domains")
        token_domain = next(
            domain
            for var, domain in domains
            if closed_term_field(var, "kind") == "tetra_token"
        )
        token = closed_term_field(manifest["transition_term"], "token")
        wrong_value = "@" if token["value"] == "@@" else "@@"
        token_domain[:] = [
            {
                "__enum__": "grimace._south_star1.policy.TetraToken",
                "value": wrong_value,
            }
        ]
        successor_digest = closed_term_digest(
            successor,
            operation="test.tetra.successor_token_domain",
        )
        set_closed_term_field(
            manifest["transition_term"],
            "successor_snapshot_digest",
            successor_digest,
        )
        refresh_linked_raw_lifecycle_residual_digest(
            branch,
            manifest=manifest,
            field="successor_residual_snapshot_digest",
            digest=successor_digest,
        )
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        require_structurally_valid_support_artifact(artifact)

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_successor_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_state_detached_wrong_successor(
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
        successor = closed_term_field(manifest["transition_term"], "successor_snapshot")
        assignments = closed_term_field(successor, "assignments")
        assignments[:] = []
        successor_digest = closed_term_digest(
            successor,
            operation="test.tetra.successor_wrong_successor",
        )
        set_closed_term_field(
            manifest["transition_term"],
            "successor_snapshot_digest",
            successor_digest,
        )
        refresh_linked_raw_lifecycle_residual_digest(
            branch,
            manifest=manifest,
            field="successor_residual_snapshot_digest",
            digest=successor_digest,
        )
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        require_structurally_valid_support_artifact(artifact)

        classification = classify_obligation_replay(facts=facts, artifact=artifact)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_successor_state_anchor_mismatch",
            classification.reason,
        )

    def test_supported_specified_tetra_artifact_is_offline_complete(
        self,
    ) -> None:
        facts = tetrahedral_facts()
        site = facts.stereo.tetrahedral[0]
        self.assertIs(site.status, SiteStatus.SPECIFIED)
        self.assertIs(site.target, TetraValue.PLUS)
        prepared = prepare_writer_facts(facts)
        options = writer_runtime_options()
        artifact = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, options),
        )

        structural = verify_writer_support_artifact_consistency(artifact)
        live = verify_writer_support_artifact_envelope(
            prepared=prepared,
            envelope=artifact,
        )
        classification = classify_obligation_replay(facts=facts, artifact=artifact)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=artifact,
        )

        self.assertTrue(structural.accepted, structural.reason)
        self.assertTrue(live.accepted, live.reason)
        self.assertEqual(structural.support_count, 12)
        self.assertEqual(
            tuple(support_strings(artifact)),
            (
                "Br[C@@H](Cl)F",
                "Br[C@H](F)Cl",
                "Cl[C@@H](F)Br",
                "Cl[C@H](Br)F",
                "F[C@@H](Br)Cl",
                "F[C@H](Cl)Br",
                "[C@@H](Br)(Cl)F",
                "[C@@H](Cl)(F)Br",
                "[C@@H](F)(Br)Cl",
                "[C@H](Br)(F)Cl",
                "[C@H](Cl)(Br)F",
                "[C@H](F)(Cl)Br",
            ),
        )
        self.assertTrue(classification.accepted, classification.reason)
        self.assertTrue(classification.residual_obligations_present)
        self.assertTrue(classification.stereo_obligations_present)
        self.assertEqual(classification.unchecked_families, ())
        self.assertNotIn("residual_work", classification.unchecked_families)
        self.assertIn("residual_work", classification.checked_families)
        self.assertIn("stereo_lifecycle", classification.checked_families)
        self.assertIn("terminal_stereo_lifecycle", classification.checked_families)
        self.assertTrue(verification.accepted, verification.reason)
        self.assertTrue(verification.structurally_checked)
        self.assertTrue(verification.facts_identity_checked)
        self.assertTrue(verification.offline_replay_complete)
        self.assertEqual(verification.offline_unchecked_object_kinds, ())
        self.assertEqual(verification.offline_unchecked_obligation_families, ())
        self.assertIn(
            "residual_work",
            verification.offline_checked_obligation_families,
        )
        self.assertIn(
            "bracket_atom_text",
            verification.offline_checked_relation_families,
        )
