"""Facts-bound writer support artifact verifier tests."""

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


class WriterSupportArtifactDomainMethods:

























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
        classification = _obligation_classification(artifact, facts=facts)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=artifact,
        )

        self.assertTrue(structural.accepted, structural.reason)
        self.assertTrue(live.accepted, live.reason)
        self.assertEqual(structural.support_count, 12)
        self.assertEqual(
            tuple(_support_strings(artifact)),
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
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = branch["payload"]["obligation_manifests"]["residual_work"][0]
        manifest["source_digest"] = "wrong"

        classification = _obligation_classification(artifact, facts=facts)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=artifact,
        )

        self.assertFalse(classification.accepted)
        self.assertIn("tetra_residual_source_digest_mismatch", classification.reason)
        self.assertFalse(verification.accepted)
        self.assertIn("object_digest_mismatch", verification.reason)

    def test_specified_tetra_transition_rejects_state_detached_successor_token_domain(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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
        _refresh_linked_raw_lifecycle_residual_digest(
            branch,
            manifest=manifest,
            field="successor_residual_snapshot_digest",
            digest=successor_digest,
        )
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_successor_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_flipped_local_order_parity(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_target_parity_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_flipped_reference_order(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_reference_order_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_event_detached_local_order(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_event_order_mismatch",
            classification.reason,
        )

    def test_specified_tetra_local_order_rejects_coherent_detached_record_identity(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        event = _first_graph_ring_delta_event(branch, "local_order_closed")
        event["successor_local_order_record_digest"] = _different_local_order_digest(
            artifact,
            branch=branch,
            cursor_name="successor_cursor",
            atom=event["atom"],
        )
        _refresh_local_order_event_identity_digest(event)
        refresh_kind_manifest_digest(branch["payload"]["graph_ring_delta"], operation="test.graph_ring_delta.digest")
        reseal_support_artifact(artifact)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_successor_record_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_local_order_rejects_coherent_detached_source_record_identity(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        event = _first_graph_ring_delta_event(branch, "local_order_closed")
        event["source_local_order_record_digest"] = _different_local_order_digest(
            artifact,
            branch=branch,
            cursor_name="source_cursor",
            atom=event["atom"],
        )
        _refresh_local_order_event_identity_digest(event)
        refresh_kind_manifest_digest(branch["payload"]["graph_ring_delta"], operation="test.graph_ring_delta.digest")
        reseal_support_artifact(artifact)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_source_record_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_missing_factor_discharge(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_discharge_factor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_state_detached_successor_discharge(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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
        _refresh_linked_raw_lifecycle_residual_digest(
            branch,
            manifest=manifest,
            field="successor_residual_snapshot_digest",
            digest=successor_digest,
        )
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_successor_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_state_detached_source_projection(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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
        _refresh_linked_raw_lifecycle_residual_digest(
            branch,
            manifest=manifest,
            field="source_residual_snapshot_digest",
            digest=source_digest,
        )
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_source_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_state_detached_successor_component(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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
        _refresh_linked_raw_lifecycle_residual_digest(
            branch,
            manifest=manifest,
            field="successor_residual_snapshot_digest",
            digest=successor_digest,
        )
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_successor_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_state_detached_wrong_successor(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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
        _refresh_linked_raw_lifecycle_residual_digest(
            branch,
            manifest=manifest,
            field="successor_residual_snapshot_digest",
            digest=successor_digest,
        )
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_successor_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_branch_detached_snapshots(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_source_lifecycle_mismatch",
            classification.reason,
        )

    def test_specified_tetra_atom_token_rejects_coherent_detached_residual_snapshots(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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
            _refresh_linked_raw_lifecycle_residual_digest(
                branch,
                manifest=manifest,
                field=lifecycle_field,
                digest=digest,
            )
        refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
        reseal_support_artifact(artifact)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_source_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_atom_token_residual_event_atom_is_bound(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        event = _first_graph_ring_delta_event(branch, "atom_emitted")
        event["atom"] = 1

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_atom_token_residual_atom_mismatch",
            classification.reason,
        )

    def test_specified_tetra_atom_token_residual_token_matches_text(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        event = _first_graph_ring_delta_event(branch, "atom_emitted")
        token = event["tetra_token"]
        token["value"] = "@@" if token["value"] == "@" else "@"

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_atom_token_residual_token_mismatch",
            classification.reason,
        )

    def test_specified_tetra_local_order_closed_atom_is_center(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        event = _first_graph_ring_delta_event(branch, "local_order_closed")
        event["atom"] = 1

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_residual_center_mismatch",
            classification.reason,
        )

    def test_specified_tetra_local_order_event_parent_is_center(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        event = _first_graph_ring_delta_event(branch, "atom_emitted")
        event["parent"] = event["atom"]

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_residual_parent_mismatch",
            classification.reason,
        )

    def test_specified_tetra_local_order_event_bond_connects_ligand(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        event = _first_graph_ring_delta_event(branch, "atom_emitted")
        event["incoming_bond"] = 99

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_residual_bond_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_requires_exact_lifecycle_operations(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_lifecycle_operation_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_requires_lifecycle_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        manifest["linked_lifecycle_digests"] = []

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_no_second_authority_from_atom_token_text(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        event = _first_graph_ring_delta_event(branch, "atom_emitted")
        self.assertIn(event["tetra_token"]["value"], ("@", "@@"))
        self.assertTrue(
            any(event["tetra_token"]["value"] in text for text in _support_strings(artifact))
        )
        manifest["linked_lifecycle_digests"] = []

        classification = _obligation_classification(artifact, facts=facts)
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

    def test_specified_tetra_residual_rejects_wrong_lifecycle_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        manifest["linked_lifecycle_digests"] = ["wrong"]

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_unrelated_lifecycle_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = _append_unrelated_raw_lifecycle(branch, manifest=manifest)
        manifest["linked_lifecycle_digests"] = [unrelated["evidence_digest"]]

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_extra_lifecycle_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = _append_unrelated_raw_lifecycle(branch, manifest=manifest)
        manifest["linked_lifecycle_digests"].append(unrelated["evidence_digest"])

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_requires_reverse_lifecycle_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_reverse_only_lifecycle_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_bogus_reverse_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_unreciprocated_reverse_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
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

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_reciprocal_extra_link_is_rejected(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = _append_unrelated_raw_lifecycle(branch, manifest=manifest)
        manifest["linked_lifecycle_digests"].append(unrelated["evidence_digest"])
        unrelated["linked_residual_work_digests"].append(manifest["evidence_digest"])

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_lifecycle_provenance_mutations_are_rejected(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
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
                branch = _first_residual_work_branch(
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
                lifecycle = _linked_tetra_lifecycle_manifest(
                    branch=branch,
                    manifest=manifest,
                    lifecycle_kind=lifecycle_kind,
                    certificate_kind="tetra_token_restricted",
                )
                lifecycle[field] = value

                classification = _obligation_classification(mutated, facts=facts)

                self.assertFalse(classification.accepted)
                self.assertIn(reason, classification.reason)

    def test_specified_tetra_no_second_authority_from_stale_lifecycle_digest(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = _linked_tetra_lifecycle_manifest(
            branch=branch,
            manifest=manifest,
            lifecycle_kind="raw",
            certificate_kind="tetra_token_restricted",
        )
        lifecycle["source_digest"] = "stale_source_digest"

        classification = _obligation_classification(artifact, facts=facts)
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

    def test_specified_tetra_no_second_authority_from_final_support_strings(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = _linked_tetra_lifecycle_manifest(
            branch=branch,
            manifest=manifest,
            lifecycle_kind="certificate",
            certificate_kind="tetra_token_restricted",
        )
        self.assertTrue(any("@" in text for text in _support_strings(artifact)))
        lifecycle["certificate_capability"] = "tetra_local_order_restriction"

        classification = _obligation_classification(artifact, facts=facts)
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

    def test_specified_tetra_residual_rejects_coherently_forged_link_projection(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = _append_unrelated_raw_lifecycle(branch, manifest=manifest)
        unrelated["linked_residual_work_digests"].append(manifest["evidence_digest"])
        unrelated["residual_work_digests"].append(manifest["evidence_digest"])
        manifest["linked_lifecycle_digests"] = [
            item["evidence_digest"]
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if manifest["evidence_digest"] in item["residual_work_digests"]
        ]

        # Public verification rejects this hand-edited artifact structurally via
        # stale object digests; this isolates the offline classifier precondition.
        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_lifecycle_evidence_missing",
            classification.reason,
        )

























































































def _manual_tetra_artifact():
    facts = tetrahedral_facts()
    prepared = prepare_writer_facts(facts)
    return (
        facts,
        writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, writer_runtime_options()),
        ),
    )








































































def _bond_occurrence_terms_for_branch(
    artifact,
    branch,
    *,
    cursor_name: str,
    bond: int,
):
    projection = _text_projection_for_branch(artifact, branch)
    state = _single_cursor_state(projection["payload"][cursor_name])
    stereo = closed_term_field(state, "stereo_state")
    return tuple(
        occurrence
        for occurrence in closed_term_field(stereo, "bond_occurrences")
        if int(closed_term_field(occurrence, "bond")) == bond
    )


def _mutate_directional_restriction_sign(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    sign = closed_term_field(manifest["transition_term"], "restrictions")[0][1]
    sign["value"] = "negative" if sign["value"] == "positive" else "positive"
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def _mutate_directional_canonical_orientation(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    value = closed_term_field(manifest["transition_term"], "canonical_orientation")
    set_closed_term_field(manifest["transition_term"], "canonical_orientation", -value)
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def _mutate_directional_model_field(
    artifact,
    *,
    bond: int,
    field: str,
    value,
    model_index: int = 0,
) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    model = closed_term_field(manifest["transition_term"], "carrier_models")[model_index]
    set_closed_term_field(model, field, value)
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def _remove_directional_model(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    models = closed_term_field(manifest["transition_term"], "carrier_models")
    del models[-1]
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def _remove_directional_restriction(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    restrictions = closed_term_field(manifest["transition_term"], "restrictions")
    del restrictions[-1]
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def _duplicate_directional_model_site(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    models = closed_term_field(manifest["transition_term"], "carrier_models")
    duplicate_site = closed_term_field(models[0], "site")
    set_closed_term_field(models[1], "site", duplicate_site)
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def _mutate_directional_successor_snapshot(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    successor = closed_term_field(manifest["transition_term"], "successor_snapshot")
    domains = closed_term_field(successor, "domains")
    domains[:] = list(reversed(domains))
    digest = closed_term_digest(successor, operation="test.directional.successor_snapshot")
    set_closed_term_field(manifest["transition_term"], "successor_snapshot_digest", digest)
    _refresh_linked_raw_lifecycle_residual_digest(
        branch,
        manifest=manifest,
        field="successor_residual_snapshot_digest",
        digest=digest,
    )
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def _set_directional_discharges(
    artifact,
    *,
    bond: int,
    kinds: tuple[str, ...],
) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    source = closed_term_field(manifest["transition_term"], "source_snapshot")
    factor_by_kind = {
        closed_term_field(closed_term_field(factor, "key"), "kind"): closed_term_field(factor, "key")
        for factor in closed_term_field(source, "factors")
    }
    set_closed_term_field(
        manifest["transition_term"],
        "discharged_factor_keys",
        [factor_by_kind[kind] for kind in kinds],
    )
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def _set_directional_discharges_by_keys(
    artifact,
    *,
    bond: int,
    key_pairs: tuple[tuple[str, tuple[int, ...]], ...],
) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    source = closed_term_field(manifest["transition_term"], "source_snapshot")
    factor_by_pair = {
        (
            closed_term_field(closed_term_field(factor, "key"), "kind"),
            tuple(closed_term_field(closed_term_field(factor, "key"), "key")),
        ): closed_term_field(factor, "key")
        for factor in closed_term_field(source, "factors")
    }
    set_closed_term_field(
        manifest["transition_term"],
        "discharged_factor_keys",
        [factor_by_pair[key_pair] for key_pair in key_pairs],
    )
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def _remove_raw_lifecycle_capability(
    artifact,
    *,
    bond: int,
    capability: str,
) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    lifecycle = _linked_tetra_lifecycle_manifest(
        branch=branch,
        manifest=manifest,
        lifecycle_kind="raw",
        certificate_kind="directional_carrier_restricted",
    )
    capabilities = lifecycle["lifecycle_capabilities"]
    capabilities.remove(capability)
    reseal_support_artifact(artifact)


def _mutate_directional_term_mark(artifact, *, bond: int, value: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    mark = closed_term_field(manifest["transition_term"], "direction_mark")
    mark["value"] = value if mark["value"] != value else -value
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def _mutate_directional_term_bond(artifact, *, bond: int, value: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    set_closed_term_field(manifest["transition_term"], "bond", value)
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def _remove_directional_successor_bond_occurrence(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    projection = _text_projection_for_branch(artifact, branch)
    cursor = projection["payload"]["successor_cursor"]
    state = _single_cursor_state(cursor)
    stereo = closed_term_field(state, "stereo_state")
    occurrences = closed_term_field(stereo, "bond_occurrences")
    kept = [
        occurrence
        for occurrence in occurrences
        if int(closed_term_field(occurrence, "bond")) != bond
    ]
    if len(kept) == len(occurrences):
        raise AssertionError(f"missing successor bond occurrence for bond {bond}")
    occurrences[:] = kept
    old_state_digest = branch["payload"]["successor_state_digest"]
    refresh_cursor_digest(cursor, operation="test.cursor.digest")
    successor_state_digest = closed_term_digest(state, operation="test.directional.successor_bond_occurrence")
    _propagate_text_projection_cursor_change(
        artifact,
        old_cursor_digest=branch["payload"]["successor_cursor_digest"],
        new_cursor=cursor,
        old_state_digest=old_state_digest,
        new_state_digest=successor_state_digest,
    )
    branch["payload"]["successor_state_digest"] = successor_state_digest
    branch["payload"]["graph_ring_delta"]["manifest"]["successor_state_digest"] = (
        successor_state_digest
    )
    manifest["successor_digest"] = successor_state_digest
    for lifecycle in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]:
        if manifest["evidence_digest"] in lifecycle["linked_residual_work_digests"]:
            lifecycle["successor_digest"] = successor_state_digest
    branch["payload"]["successor_cursor_digest"] = cursor["digest"]
    branch["payload"]["graph_ring_delta"]["manifest"]["successor_cursor_digest"] = (
        cursor["digest"]
    )
    refresh_kind_manifest_digest(branch["payload"]["graph_ring_delta"], operation="test.graph_ring_delta.digest")
    projection["payload"]["digest"] = _text_projection_identity_digest(
        projection["payload"]
    )
    reseal_support_artifact(artifact)


def _duplicate_directional_successor_bond_occurrence(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    projection = _text_projection_for_branch(artifact, branch)
    cursor = projection["payload"]["successor_cursor"]
    state = _single_cursor_state(cursor)
    stereo = closed_term_field(state, "stereo_state")
    occurrences = closed_term_field(stereo, "bond_occurrences")
    matches = [
        occurrence
        for occurrence in occurrences
        if int(closed_term_field(occurrence, "bond")) == bond
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one successor bond occurrence for bond {bond}")
    occurrences.append(deepcopy(matches[0]))
    old_state_digest = branch["payload"]["successor_state_digest"]
    refresh_cursor_digest(cursor, operation="test.cursor.digest")
    successor_state_digest = closed_term_digest(state, operation="test.directional.duplicate_bond_occurrence")
    _propagate_text_projection_cursor_change(
        artifact,
        old_cursor_digest=branch["payload"]["successor_cursor_digest"],
        new_cursor=cursor,
        old_state_digest=old_state_digest,
        new_state_digest=successor_state_digest,
    )
    branch["payload"]["successor_state_digest"] = successor_state_digest
    branch["payload"]["graph_ring_delta"]["manifest"]["successor_state_digest"] = (
        successor_state_digest
    )
    manifest["successor_digest"] = successor_state_digest
    for lifecycle in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]:
        if manifest["evidence_digest"] in lifecycle["linked_residual_work_digests"]:
            lifecycle["successor_digest"] = successor_state_digest
    branch["payload"]["successor_cursor_digest"] = cursor["digest"]
    branch["payload"]["graph_ring_delta"]["manifest"]["successor_cursor_digest"] = (
        cursor["digest"]
    )
    refresh_kind_manifest_digest(branch["payload"]["graph_ring_delta"], operation="test.graph_ring_delta.digest")
    projection["payload"]["digest"] = _text_projection_identity_digest(
        projection["payload"]
    )
    reseal_support_artifact(artifact)


def _mutate_directional_successor_snapshot_unrelated(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    successor = closed_term_field(manifest["transition_term"], "successor_snapshot")
    closed_term_field(successor, "domains").append(
        [
            {
                "__dataclass__": "grimace._south_star1.residual_constraints.VarId",
                "fields": [["kind", "unrelated_directional_test"], ["key", [99]]],
            },
            [False, True],
        ]
    )
    digest = closed_term_digest(successor, operation="test.directional.successor_snapshot_unrelated")
    set_closed_term_field(manifest["transition_term"], "successor_snapshot_digest", digest)
    _refresh_linked_raw_lifecycle_residual_digest(
        branch,
        manifest=manifest,
        field="successor_residual_snapshot_digest",
        digest=digest,
    )
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def _linked_tetra_lifecycle_manifest(
    *,
    branch,
    manifest,
    lifecycle_kind: str,
    certificate_kind: str,
):
    operation = (
        "WriterStereoLifecycleEvidence"
        if lifecycle_kind == "raw"
        else "WriterStereoBranchCertificate"
    )
    for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]:
        if item["evidence_digest"] not in manifest["linked_lifecycle_digests"]:
            continue
        if item["operation"] != operation:
            continue
        if lifecycle_kind == "certificate" and item["certificate_kind"] != certificate_kind:
            continue
        return item
    raise AssertionError(f"missing linked tetra lifecycle manifest: {lifecycle_kind}")


def _append_unrelated_raw_lifecycle(branch, *, manifest):
    linked = _linked_tetra_lifecycle_manifest(
        branch=branch,
        manifest=manifest,
        lifecycle_kind="raw",
        certificate_kind="",
    )
    unrelated = deepcopy(linked)
    unrelated["evidence_digest"] = f"unrelated:{linked['evidence_digest']}"
    unrelated["linked_residual_work_digests"] = []
    unrelated["residual_work_digests"] = []
    unrelated["residual_work_operations"] = []
    branch["payload"]["obligation_manifests"]["stereo_lifecycle"].append(unrelated)
    return unrelated






















def _refresh_linked_raw_lifecycle_residual_digest(
    branch,
    *,
    manifest,
    field: str,
    digest: str,
) -> None:
    lifecycle = _linked_tetra_lifecycle_manifest(
        branch=branch,
        manifest=manifest,
        lifecycle_kind="raw",
        certificate_kind="",
    )
    lifecycle[field] = digest




def _different_local_order_digest(artifact, *, branch, cursor_name: str, atom: int) -> str:
    projection = _text_projection_for_branch(artifact, branch)
    cursor = projection["payload"][cursor_name]
    state = cursor["terms"]["fields"][0][1][0][0]
    stereo = closed_term_field(state, "stereo_state")
    for record in closed_term_field(stereo, "local_orders"):
        if closed_term_field(record, "atom") != atom:
            return closed_term_digest(record, operation="test.tetra.local_order_alternate")
    raise AssertionError("missing alternate local-order record")




def _refresh_local_order_event_identity_digest(event) -> None:
    identity = {
        "site": event["site"],
        "atom": event["atom"],
        "local_order": event["local_order"],
        "reference_order": event["reference_order"],
        "source_local_order_record_digest": event[
            "source_local_order_record_digest"
        ],
        "successor_local_order_record_digest": event[
            "successor_local_order_record_digest"
        ],
    }
    event["local_order_identity_digest"] = _identity_digest(identity)











__all__ = [name for name in globals() if not name.startswith("__")]
