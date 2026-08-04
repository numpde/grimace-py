"""Physically owned rich support-artifact integration contracts.Facts-bound writer support artifact verifier tests."""

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

class WriterSupportArtifactFactVerifierTest(unittest.TestCase):
    def test_facts_bound_verifier_reports_bracket_atom_offline_check(self) -> None:
        for smiles in ("[N+]", "[NH+]", "[NH2+]", "[NH3+]", "[NH4+]", "[O-]", "[OH-]"):
            with self.subTest(smiles=smiles):
                verification = rdkit_support_artifact_verification(smiles)

                self.assertTrue(verification.accepted, verification.reason)
                self.assertIn(
                    "bracket_atom_text",
                    verification.offline_checked_relation_families,
                )
                self.assertIn("text_projection", verification.offline_checked_object_kinds)
                self.assertTrue(verification.offline_replay_complete)

    def test_facts_bound_verifier_reports_isotope_atom_offline_check(self) -> None:
        verification = rdkit_support_artifact_verification("[13CH4]")

        self.assertTrue(verification.accepted, verification.reason)
        self.assertIn(
            "bracket_atom_text",
            verification.offline_checked_relation_families,
        )
        self.assertIn("text_projection", verification.offline_checked_object_kinds)
        self.assertTrue(verification.offline_replay_complete)

    def test_facts_bound_verifier_reports_joint_double_closure_offline_check(
        self,
    ) -> None:
        verification = rdkit_support_artifact_verification("C1=CC1")

        self.assertTrue(verification.accepted, verification.reason)
        self.assertIn(
            "closure_bond_text",
            verification.offline_checked_relation_families,
        )
        self.assertTrue(verification.offline_replay_complete)

    def test_facts_bound_verifier_reports_joint_triple_closure_offline_check(
        self,
    ) -> None:
        verification = rdkit_support_artifact_verification("C1#CC1")

        self.assertTrue(verification.accepted, verification.reason)
        self.assertIn(
            "closure_bond_text",
            verification.offline_checked_relation_families,
        )
        self.assertTrue(verification.offline_replay_complete)

    def test_mutated_prepared_identity_is_rejected(self) -> None:
        artifact = support_artifact_fixture(cco_facts()).artifact
        artifact["prepared_identity"] = deepcopy(artifact["prepared_identity"])
        artifact["prepared_identity"]["digest"] = "0" * 64
        reseal_support_artifact(artifact)

        verification = verify_writer_support_artifact_for_facts(
            facts=cco_facts(),
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("prepared_identity_mismatch", verification.reason)

    def test_mutated_source_prepared_identity_is_rejected(self) -> None:
        artifact = support_artifact_fixture(cco_facts()).artifact
        source = artifact_object_by_id(artifact, artifact["roots"]["source_ref"])
        source["payload"]["prepared_identity_digest"] = "0" * 64

        verification = verify_writer_support_artifact_for_facts(
            facts=cco_facts(),
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)

    def test_offline_coverage_ledger_classifies_partial_replay(self) -> None:
        self.assertEqual(
            OBJECT_KIND_OFFLINE_COVERAGE,
            {
                "source_snapshot": "identity_checked",
                "count_envelope": "arithmetic_checked",
                "count_dag": "arithmetic_checked",
                "frontier_product": "structurally_checked",
                "replay_path": "partially_offline_checked",
                "branch_support": "partially_offline_checked",
                "text_projection": "partially_offline_checked",
                "terminal_projection": "partially_offline_checked",
                "terminal_support": "partially_offline_checked",
                "support_string": "partially_offline_checked",
                "support_image_coverage": "structurally_checked",
                "support_image": "structurally_checked",
            },
        )

    def test_offline_coverage_ledger_covers_artifact_object_kinds(self) -> None:
        artifact = rdkit_support_artifact_fixture("C1=CC1").artifact
        kinds = {item["kind"] for item in artifact["objects"]}

        self.assertLessEqual(kinds, set(OBJECT_KIND_OFFLINE_COVERAGE))

    def test_prefix_artifact_verifies_against_matching_facts(self) -> None:
        facts = two_atom_facts()
        prepared = prepare_writer_facts(facts)
        options = writer_runtime_options()
        prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, options),
            emitted_texts=("C", "C"),
        )
        artifact = writer_support_artifact_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
        )

        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=artifact,
        )

        self.assertTrue(verification.accepted, verification.reason)
        self.assertTrue(verification.structurally_checked)
        self.assertTrue(verification.facts_identity_checked)
        self.assertTrue(verification.offline_replay_complete)

    def test_snapshot_artifact_verifies_against_matching_facts(self) -> None:
        facts = cco_facts()
        prepared = prepare_writer_facts(facts)
        options = writer_runtime_options()
        artifact = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, options),
        )

        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=artifact,
        )

        self.assertTrue(verification.accepted, verification.reason)
        self.assertTrue(verification.structurally_checked)
        self.assertTrue(verification.facts_identity_checked)
        self.assertTrue(
            verification.offline_replay_complete,
            verification.offline_unchecked_obligation_families,
        )
        self.assertIn("support_string", verification.offline_checked_object_kinds)
        self.assertIn("replay_path", verification.offline_checked_object_kinds)
        self.assertIn("branch_support", verification.offline_checked_object_kinds)
        self.assertIn("count_envelope", verification.offline_checked_object_kinds)
        self.assertIn("count_dag", verification.offline_checked_object_kinds)
        self.assertIn(
            "count_dag_arithmetic",
            verification.offline_checked_relation_families,
        )
        self.assertIn(
            "support_image_coverage",
            verification.offline_checked_relation_families,
        )
        self.assertIn(
            "support_string_replay_path",
            verification.offline_checked_relation_families,
        )
        self.assertIn(
            "branch_projection_identity",
            verification.offline_checked_relation_families,
        )
        self.assertIn(
            "graph_ring_branch_delta",
            verification.offline_checked_relation_families,
        )
        self.assertIn(
            "local_branch_successor_evidence",
            verification.offline_checked_relation_families,
        )
        self.assertIn(
            "terminal_support_identity",
            verification.offline_checked_relation_families,
        )
        self.assertIn("support_image", verification.offline_checked_object_kinds)
        self.assertIn(
            "support_image_coverage",
            verification.offline_checked_object_kinds,
        )
        self.assertEqual(verification.support_count, 4)

    def test_structurally_invalid_artifact_is_rejected(self) -> None:
        artifact = support_artifact_fixture(cco_facts()).artifact
        artifact["objects"].pop()

        verification = verify_writer_support_artifact_for_facts(
            facts=cco_facts(),
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertFalse(verification.structurally_checked)

    def test_unknown_object_kind_is_rejected_by_structural_checker(self) -> None:
        artifact = support_artifact_fixture(cco_facts()).artifact
        root = artifact_object_by_id(artifact, artifact["roots"]["support_image_root"])
        root["kind"] = "unknown"

        verification = verify_writer_support_artifact_for_facts(
            facts=cco_facts(),
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertFalse(verification.structurally_checked)

    def test_wrong_explicit_policy_is_rejected(self) -> None:
        facts = two_atom_facts()
        artifact = support_artifact_fixture(facts).artifact
        wrong_policy = ordinary_policy_for_facts(
            facts,
            OrdinaryPolicyOptions(single_bond_mode="both"),
        )

        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=writer_runtime_options(),
            artifact=artifact,
            policy=wrong_policy,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("prepared_identity_mismatch", verification.reason)

    def test_wrong_facts_are_rejected(self) -> None:
        artifact = support_artifact_fixture(cco_facts()).artifact

        verification = verify_writer_support_artifact_for_facts(
            facts=two_atom_facts(),
            runtime_options=writer_runtime_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("prepared_identity_mismatch", verification.reason)

    def test_wrong_runtime_options_are_rejected(self) -> None:
        facts = cco_facts()
        artifact = support_artifact_fixture(facts).artifact
        wrong_options = writer_runtime_options(rooted_at_atom=0)

        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=wrong_options,
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("prepared_identity_mismatch", verification.reason)
