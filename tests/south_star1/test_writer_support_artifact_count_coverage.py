"""Physically owned rich support-artifact count-coverage contracts.Facts-bound writer support artifact verifier tests."""

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

class WriterSupportArtifactCountCoverageTest(unittest.TestCase):
    def test_count_dag_arithmetic_accepts_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "CC(C)O", "C1CC1", "C1=CC1", "[NH4+]", "[13CH4]"):
            with self.subTest(smiles=smiles):
                artifact = rdkit_support_artifact_fixture(smiles).artifact
                count = artifact_object_by_id(artifact, artifact["roots"]["count_ref"])
                count_dag = artifact_object_by_id(artifact, count["payload"]["count_dag_ref"])

                verification = verify_count_dag_arithmetic(
                    count_dag=count_dag["payload"],
                    count_object=count["payload"],
                )

                self.assertTrue(verification.accepted, verification.reason)
                self.assertEqual(verification.support_count, count["payload"]["support_count"])
                self.assertEqual(
                    verification.completion_count,
                    count["payload"]["completion_count"],
                )

    def test_count_dag_arithmetic_rejects_changed_count_object_totals(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        count = deepcopy(artifact_object_by_id(artifact, artifact["roots"]["count_ref"]))
        count_dag = artifact_object_by_id(artifact, count["payload"]["count_dag_ref"])
        count["payload"]["support_count"] += 1

        verification = verify_count_dag_arithmetic(
            count_dag=count_dag["payload"],
            count_object=count["payload"],
        )

        self.assertFalse(verification.accepted)
        self.assertIn("count_dag_support_count_mismatch", verification.reason)

    def test_count_dag_arithmetic_rejects_changed_root_node_count(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        count = artifact_object_by_id(artifact, artifact["roots"]["count_ref"])
        count_dag = deepcopy(artifact_object_by_id(artifact, count["payload"]["count_dag_ref"]))
        root_id = count_dag["payload"]["roots"]["support_count_root"]
        root = next(
            node
            for node in count_dag["payload"]["nodes"]
            if node["node_id"] == root_id
        )
        root["support_count"] += 1

        verification = verify_count_dag_arithmetic(
            count_dag=count_dag["payload"],
            count_object=count["payload"],
        )

        self.assertFalse(verification.accepted)

    def test_count_dag_arithmetic_rejects_missing_child_and_cycle(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        count = artifact_object_by_id(artifact, artifact["roots"]["count_ref"])
        count_dag = deepcopy(artifact_object_by_id(artifact, count["payload"]["count_dag_ref"]))
        child_id = next(
            node["children"][0]
            for node in count_dag["payload"]["nodes"]
            if node["children"]
        )
        count_dag["payload"]["nodes"] = [
            node for node in count_dag["payload"]["nodes"] if node["node_id"] != child_id
        ]

        missing = verify_count_dag_arithmetic(
            count_dag=count_dag["payload"],
            count_object=count["payload"],
        )

        self.assertFalse(missing.accepted)

        count_dag = deepcopy(artifact_object_by_id(artifact, count["payload"]["count_dag_ref"]))
        count_dag["payload"]["nodes"][0]["children"].append(
            count_dag["payload"]["nodes"][0]["node_id"]
        )
        cycle = verify_count_dag_arithmetic(
            count_dag=count_dag["payload"],
            count_object=count["payload"],
        )

        self.assertFalse(cycle.accepted)

    def test_coverage_missing_or_extra_text_bucket_is_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        coverage = coverage_object(artifact)
        coverage["payload"]["text_buckets"] = []

        missing = verify_support_image_coverage_relation(artifact)

        self.assertFalse(missing.accepted)
        self.assertIn("coverage_partition_mismatch", missing.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        coverage = coverage_object(artifact)
        coverage["payload"]["text_buckets"].append(
            deepcopy(coverage["payload"]["text_buckets"][0])
        )

        extra = verify_support_image_coverage_relation(artifact)

        self.assertFalse(extra.accepted)
        self.assertIn("coverage_duplicate_assignment", extra.reason)

    def test_coverage_support_and_witness_totals_are_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        root = artifact_object_by_id(artifact, artifact["roots"]["support_image_root"])
        root["payload"]["distinct_count"] += 1

        distinct = verify_support_image_coverage_relation(artifact)

        self.assertFalse(distinct.accepted)
        self.assertIn("support_image_distinct_count_mismatch", distinct.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        root = artifact_object_by_id(artifact, artifact["roots"]["support_image_root"])
        root["payload"]["witness_count"] += 1

        witness = verify_support_image_coverage_relation(artifact)

        self.assertFalse(witness.accepted)
        self.assertIn("coverage_count_completion_total_mismatch", witness.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        count = artifact_object_by_id(artifact, artifact["roots"]["count_ref"])
        count["payload"]["support_count"] += 1

        count_support = verify_support_image_coverage_relation(artifact)

        self.assertFalse(count_support.accepted)
        self.assertIn("coverage_count_support_total_mismatch", count_support.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        count = artifact_object_by_id(artifact, artifact["roots"]["count_ref"])
        count["payload"]["completion_count"] += 1

        count_completion = verify_support_image_coverage_relation(artifact)

        self.assertFalse(count_completion.accepted)
        self.assertIn(
            "coverage_count_completion_total_mismatch",
            count_completion.reason,
        )

    def test_coverage_terminal_bucket_mutations_are_rejected(self) -> None:
        artifact = completed_prefix_support_artifact_fixture().artifact
        coverage = coverage_object(artifact)
        coverage["payload"]["terminal_bucket"] = None

        missing = verify_support_image_coverage_relation(artifact)

        self.assertFalse(missing.accepted)
        self.assertIn("coverage_terminal_bucket_missing", missing.reason)

        artifact = completed_prefix_support_artifact_fixture().artifact
        coverage = coverage_object(artifact)
        coverage["payload"]["text_buckets"].append(
            {
                "text_projection": {},
                "support_count": 1,
                "string_refs": [
                    artifact_object_by_id(
                        artifact,
                        artifact["roots"]["support_image_root"],
                    )["payload"]["support_string_refs"][0]
                ],
            }
        )

        text_bucket = verify_support_image_coverage_relation(artifact)

        self.assertFalse(text_bucket.accepted)
        self.assertIn("coverage_empty_string_in_text_bucket", text_bucket.reason)

        artifact = completed_prefix_support_artifact_fixture().artifact
        coverage = coverage_object(artifact)
        coverage["payload"]["terminal_bucket"]["terminal_projection"] = {
            **coverage["payload"]["terminal_bucket"]["terminal_projection"],
            "digest": "0" * 64,
        }

        wrong_projection = verify_support_image_coverage_relation(artifact)

        self.assertFalse(wrong_projection.accepted)
        self.assertIn("coverage_terminal_projection_mismatch", wrong_projection.reason)

    def test_coverage_wrong_duplicate_and_unassigned_refs_are_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        coverage = coverage_object(artifact)
        coverage["payload"]["text_buckets"][0]["string_refs"] = ["missing"]
        coverage["payload"]["text_buckets"][0]["support_count"] = 1

        wrong = verify_support_image_coverage_relation(artifact)

        self.assertFalse(wrong.accepted)
        self.assertIn("coverage_text_bucket_unknown_ref", wrong.reason)

        artifact = rdkit_support_artifact_fixture("CC(C)O").artifact
        root = artifact_object_by_id(artifact, artifact["roots"]["support_image_root"])
        coverage = coverage_object(artifact)
        first_ref = root["payload"]["support_string_refs"][0]
        coverage["payload"]["text_buckets"][0]["string_refs"] = [first_ref, first_ref]
        coverage["payload"]["text_buckets"][0]["support_count"] = 2

        duplicate = verify_support_image_coverage_relation(artifact)

        self.assertFalse(duplicate.accepted)
        self.assertIn("coverage_duplicate_assignment", duplicate.reason)

    def test_coverage_wrong_text_projection_ref_is_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        coverage = coverage_object(artifact)
        coverage["payload"]["text_buckets"][0]["text_projection"] = {
            **coverage["payload"]["text_buckets"][0]["text_projection"],
            "emitted_text": "N",
        }

        verification = verify_support_image_coverage_relation(artifact)

        self.assertFalse(verification.accepted)
        self.assertIn("coverage_text_projection_mismatch", verification.reason)

    def test_support_image_coverage_accepts_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "C1CC1", "C1=CC1", "[NH4+]", "[13CH4]"):
            with self.subTest(smiles=smiles):
                artifact = rdkit_support_artifact_fixture(smiles).artifact

                verification = verify_support_image_coverage_relation(artifact)

                self.assertTrue(verification.accepted, verification.reason)
                root = artifact_object_by_id(artifact, artifact["roots"]["support_image_root"])
                self.assertEqual(verification.support_count, root["payload"]["distinct_count"])
                self.assertEqual(verification.witness_count, root["payload"]["witness_count"])

    def test_support_image_coverage_accepts_terminal_bucket(self) -> None:
        artifact = completed_prefix_support_artifact_fixture().artifact

        verification = verify_support_image_coverage_relation(artifact)

        self.assertTrue(verification.accepted, verification.reason)
        self.assertEqual(verification.support_count, 1)
        self.assertEqual(verification.witness_count, 2)
