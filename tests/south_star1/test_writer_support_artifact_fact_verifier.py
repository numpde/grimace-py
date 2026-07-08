"""Facts-bound writer support artifact verifier tests."""

from __future__ import annotations

from copy import deepcopy
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_envelope_terms import _digest_terms_bounded
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot_prefix_envelope import (
    writer_snapshot_prefix_read_envelope_for_emitted_texts,
)
from grimace._south_star1.writer_support_artifact_checker import artifact_manifest
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
    writer_support_artifact_envelope_for_prefix_read,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_snapshot,
)
from tests.south_star1.helpers import cco_facts
from tests.south_star1.test_writer_snapshot import two_atom_facts


class WriterSupportArtifactFactVerifierTest(unittest.TestCase):
    def test_snapshot_artifact_verifies_against_matching_facts(self) -> None:
        facts = cco_facts()
        prepared = _prepare(facts)
        options = _writer_options()
        artifact = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared, options),
        )

        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=artifact,
        )

        self.assertTrue(verification.accepted, verification.reason)
        self.assertTrue(verification.structurally_checked)
        self.assertTrue(verification.facts_identity_checked)
        self.assertFalse(verification.offline_replay_complete)
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

    def test_prefix_artifact_verifies_against_matching_facts(self) -> None:
        facts = two_atom_facts()
        prepared = _prepare(facts)
        options = _writer_options()
        prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared, options),
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
        self.assertFalse(verification.offline_replay_complete)

    def test_facts_bound_verifier_reports_bracket_atom_offline_check(self) -> None:
        verification = _rdkit_artifact_verification("[NH4+]")

        self.assertTrue(verification.accepted, verification.reason)
        self.assertIn(
            "bracket_atom_text",
            verification.offline_checked_relation_families,
        )
        self.assertIn("text_projection", verification.offline_checked_object_kinds)
        self.assertFalse(verification.offline_replay_complete)

    def test_facts_bound_verifier_reports_isotope_atom_offline_check(self) -> None:
        verification = _rdkit_artifact_verification("[13CH4]")

        self.assertTrue(verification.accepted, verification.reason)
        self.assertIn(
            "bracket_atom_text",
            verification.offline_checked_relation_families,
        )
        self.assertIn("text_projection", verification.offline_checked_object_kinds)
        self.assertFalse(verification.offline_replay_complete)

    def test_facts_bound_verifier_reports_joint_double_closure_offline_check(
        self,
    ) -> None:
        verification = _rdkit_artifact_verification("C1=CC1")

        self.assertTrue(verification.accepted, verification.reason)
        self.assertIn(
            "closure_bond_text",
            verification.offline_checked_relation_families,
        )
        self.assertFalse(verification.offline_replay_complete)

    def test_facts_bound_verifier_reports_joint_triple_closure_offline_check(
        self,
    ) -> None:
        verification = _rdkit_artifact_verification("C1#CC1")

        self.assertTrue(verification.accepted, verification.reason)
        self.assertIn(
            "closure_bond_text",
            verification.offline_checked_relation_families,
        )
        self.assertFalse(verification.offline_replay_complete)

    def test_offline_bracket_atom_replay_rejects_wrong_facts(self) -> None:
        with self.assertRaisesRegex(SouthStarError, "bracket_atom_text_facts_mismatch"):
            validate_writer_bracket_atom_text_against_facts(
                facts=_rdkit_facts("[NH3+]"),
                rendered_text="[NH4+]",
            )

        with self.assertRaisesRegex(SouthStarError, "bracket_atom_text_facts_mismatch"):
            validate_writer_bracket_atom_text_against_facts(
                facts=_rdkit_facts("[12CH4]"),
                rendered_text="[13CH4]",
            )

    def test_offline_joint_closure_replay_rejects_wrong_facts(self) -> None:
        artifact = _rdkit_artifact("C1=CC1")
        verification = verify_writer_support_artifact_offline_replay(
            facts=_rdkit_facts("C1CC1"),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("graph_ring_bond_marker_mismatch", verification.reason)

    def test_graph_ring_branch_deltas_accept_default_relation_fixtures(self) -> None:
        for smiles in (
            "CCO",
            "CC(C)O",
            "C1CC1",
            "C1CCC1",
            "C1=CC1",
            "C1#CC1",
            "[NH4+]",
            "[13CH4]",
        ):
            with self.subTest(smiles=smiles):
                artifact = _rdkit_artifact(smiles)

                verification = _graph_ring_delta_verification(
                    _rdkit_facts(smiles),
                    artifact,
                )

                self.assertTrue(verification.accepted, verification.reason)
                self.assertGreater(verification.checked_branches, 0)

    def test_graph_ring_branch_delta_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        branch = _first_graph_ring_delta_branch(artifact, "bond_advance")
        event = _first_graph_ring_delta_event(branch, "bond_emitted")
        event["bond"] = "missing"
        _refresh_graph_ring_delta_digest(branch["payload"]["graph_ring_delta"])

        wrong_bond = _graph_ring_delta_verification(_rdkit_facts("CCO"), artifact)

        self.assertFalse(wrong_bond.accepted)
        self.assertIn("local_closure_bond_missing", wrong_bond.reason)

        artifact = _rdkit_artifact("CCO")
        branch = _first_graph_ring_delta_branch(artifact, "atom_start")
        event = _first_graph_ring_delta_event(branch, "atom_emitted")
        event["atom"] = "missing"
        _refresh_graph_ring_delta_digest(branch["payload"]["graph_ring_delta"])

        wrong_atom = _graph_ring_delta_verification(_rdkit_facts("CCO"), artifact)

        self.assertFalse(wrong_atom.accepted)
        self.assertIn("local_atom_text_atom_missing", wrong_atom.reason)

        artifact = _rdkit_artifact("C1=CC1")
        branch = _first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        event = _first_graph_ring_delta_event(branch, "ring_endpoint_paired")
        event["label"] = "wrong"
        _refresh_graph_ring_delta_digest(branch["payload"]["graph_ring_delta"])

        wrong_label = _graph_ring_delta_verification(_rdkit_facts("C1=CC1"), artifact)

        self.assertFalse(wrong_label.accepted)
        self.assertIn("graph_ring_endpoint_label_mismatch", wrong_label.reason)

        artifact = _rdkit_artifact("C1=CC1")
        branch = _first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        branch["payload"]["successor_state_digest"] = "wrong"

        wrong_state = _graph_ring_delta_verification(_rdkit_facts("C1=CC1"), artifact)

        self.assertFalse(wrong_state.accepted)
        self.assertIn("graph_ring_delta_successor_state_digest_mismatch", wrong_state.reason)

    def test_default_corpus_obligations_are_classified(self) -> None:
        cases = {
            "CCO": (
                "terminal_graph_obligation_work",
                "terminal_stereo_lifecycle",
            ),
            "CC(C)O": (
                "terminal_graph_obligation_work",
                "terminal_stereo_lifecycle",
            ),
            "C1CC1": (
                "finite_relation_work",
                "graph_obligation_work",
                "terminal_graph_obligation_work",
                "terminal_stereo_lifecycle",
            ),
            "C1CCC1": (
                "finite_relation_work",
                "graph_obligation_work",
                "terminal_graph_obligation_work",
                "terminal_stereo_lifecycle",
            ),
            "C1=CC1": (
                "finite_relation_work",
                "graph_obligation_work",
                "terminal_graph_obligation_work",
                "terminal_stereo_lifecycle",
            ),
            "C1#CC1": (
                "finite_relation_work",
                "graph_obligation_work",
                "terminal_graph_obligation_work",
                "terminal_stereo_lifecycle",
            ),
            "[NH4+]": (
                "terminal_graph_obligation_work",
                "terminal_stereo_lifecycle",
            ),
            "[13CH4]": (
                "terminal_graph_obligation_work",
                "terminal_stereo_lifecycle",
            ),
        }
        for smiles, unchecked_families in cases.items():
            with self.subTest(smiles=smiles):
                artifact = _rdkit_artifact(smiles)

                classification = _obligation_classification(artifact)
                verification = verify_writer_support_artifact_for_facts(
                    facts=_rdkit_facts(smiles),
                    runtime_options=_writer_options(),
                    artifact=artifact,
                )

                self.assertTrue(classification.accepted, classification.reason)
                self.assertEqual(classification.unchecked_families, unchecked_families)
                self.assertTrue(classification.stereo_obligations_present)
                self.assertEqual(
                    classification.graph_obligations_present,
                    "terminal_graph_obligation_work" in unchecked_families
                    or "graph_obligation_work" in unchecked_families,
                )
                self.assertEqual(
                    classification.residual_obligations_present,
                    smiles.startswith("C1"),
                )
                self.assertIn("stereo_lifecycle", classification.checked_families)
                self.assertIn(
                    "residual_work_checked_empty",
                    classification.checked_empty_families,
                )
                self.assertTrue(verification.accepted, verification.reason)
                self.assertFalse(verification.offline_replay_complete)
                self.assertEqual(verification.offline_unchecked_object_kinds, ())
                self.assertEqual(
                    verification.offline_unchecked_obligation_families,
                    unchecked_families,
                )
                self.assertIn(
                    "stereo_lifecycle",
                    verification.offline_checked_obligation_families,
                )
                self.assertIn(
                    "residual_work_checked_empty",
                    verification.offline_empty_obligation_families,
                )

    def test_synthetic_stereo_obligation_is_reported_unchecked(self) -> None:
        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        manifest = branch["payload"]["obligation_manifests"]["stereo_lifecycle"][0]
        manifest["is_discharged"] = False
        manifest["is_noop"] = False
        manifest["is_empty"] = False

        classification = _obligation_classification(artifact)

        self.assertTrue(classification.accepted, classification.reason)
        self.assertTrue(classification.stereo_obligations_present)
        self.assertIn("stereo_lifecycle", classification.unchecked_families)

    def test_obligation_summary_mutation_is_structurally_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        branch["payload"]["obligation_summary"]["stereo_lifecycle_count"] += 1

        verification = verify_writer_support_artifact_for_facts(
            facts=_rdkit_facts("CCO"),
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("object_digest_mismatch", verification.reason)

    def test_count_dag_arithmetic_accepts_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "CC(C)O", "C1CC1", "C1=CC1", "[NH4+]", "[13CH4]"):
            with self.subTest(smiles=smiles):
                artifact = _rdkit_artifact(smiles)
                count = _object(artifact, artifact["roots"]["count_ref"])
                count_dag = _object(artifact, count["payload"]["count_dag_ref"])

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
        artifact = _rdkit_artifact("CCO")
        count = deepcopy(_object(artifact, artifact["roots"]["count_ref"]))
        count_dag = _object(artifact, count["payload"]["count_dag_ref"])
        count["payload"]["support_count"] += 1

        verification = verify_count_dag_arithmetic(
            count_dag=count_dag["payload"],
            count_object=count["payload"],
        )

        self.assertFalse(verification.accepted)
        self.assertIn("count_dag_support_count_mismatch", verification.reason)

    def test_count_dag_arithmetic_rejects_changed_root_node_count(self) -> None:
        artifact = _rdkit_artifact("CCO")
        count = _object(artifact, artifact["roots"]["count_ref"])
        count_dag = deepcopy(_object(artifact, count["payload"]["count_dag_ref"]))
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
        artifact = _rdkit_artifact("CCO")
        count = _object(artifact, artifact["roots"]["count_ref"])
        count_dag = deepcopy(_object(artifact, count["payload"]["count_dag_ref"]))
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

        count_dag = deepcopy(_object(artifact, count["payload"]["count_dag_ref"]))
        count_dag["payload"]["nodes"][0]["children"].append(
            count_dag["payload"]["nodes"][0]["node_id"]
        )
        cycle = verify_count_dag_arithmetic(
            count_dag=count_dag["payload"],
            count_object=count["payload"],
        )

        self.assertFalse(cycle.accepted)

    def test_support_image_coverage_accepts_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "C1CC1", "C1=CC1", "[NH4+]", "[13CH4]"):
            with self.subTest(smiles=smiles):
                artifact = _rdkit_artifact(smiles)

                verification = _coverage_verification(artifact)

                self.assertTrue(verification.accepted, verification.reason)
                root = _object(artifact, artifact["roots"]["support_image_root"])
                self.assertEqual(verification.support_count, root["payload"]["distinct_count"])
                self.assertEqual(verification.witness_count, root["payload"]["witness_count"])

    def test_support_image_coverage_accepts_terminal_bucket(self) -> None:
        artifact = _two_atom_completed_prefix_artifact()

        verification = _coverage_verification(artifact)

        self.assertTrue(verification.accepted, verification.reason)
        self.assertEqual(verification.support_count, 1)
        self.assertEqual(verification.witness_count, 2)

    def test_coverage_missing_or_extra_text_bucket_is_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        coverage = _coverage_object(artifact)
        coverage["payload"]["text_buckets"] = []

        missing = _coverage_verification(artifact)

        self.assertFalse(missing.accepted)
        self.assertIn("coverage_partition_mismatch", missing.reason)

        artifact = _rdkit_artifact("CCO")
        coverage = _coverage_object(artifact)
        coverage["payload"]["text_buckets"].append(
            deepcopy(coverage["payload"]["text_buckets"][0])
        )

        extra = _coverage_verification(artifact)

        self.assertFalse(extra.accepted)
        self.assertIn("coverage_duplicate_assignment", extra.reason)

    def test_coverage_wrong_text_projection_ref_is_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        coverage = _coverage_object(artifact)
        coverage["payload"]["text_buckets"][0]["text_projection"] = {
            **coverage["payload"]["text_buckets"][0]["text_projection"],
            "emitted_text": "N",
        }

        verification = _coverage_verification(artifact)

        self.assertFalse(verification.accepted)
        self.assertIn("coverage_text_projection_mismatch", verification.reason)

    def test_coverage_wrong_duplicate_and_unassigned_refs_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        coverage = _coverage_object(artifact)
        coverage["payload"]["text_buckets"][0]["string_refs"] = ["missing"]
        coverage["payload"]["text_buckets"][0]["support_count"] = 1

        wrong = _coverage_verification(artifact)

        self.assertFalse(wrong.accepted)
        self.assertIn("coverage_text_bucket_unknown_ref", wrong.reason)

        artifact = _rdkit_artifact("CC(C)O")
        root = _object(artifact, artifact["roots"]["support_image_root"])
        coverage = _coverage_object(artifact)
        first_ref = root["payload"]["support_string_refs"][0]
        coverage["payload"]["text_buckets"][0]["string_refs"] = [first_ref, first_ref]
        coverage["payload"]["text_buckets"][0]["support_count"] = 2

        duplicate = _coverage_verification(artifact)

        self.assertFalse(duplicate.accepted)
        self.assertIn("coverage_duplicate_assignment", duplicate.reason)

    def test_coverage_support_and_witness_totals_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        root = _object(artifact, artifact["roots"]["support_image_root"])
        root["payload"]["distinct_count"] += 1

        distinct = _coverage_verification(artifact)

        self.assertFalse(distinct.accepted)
        self.assertIn("support_image_distinct_count_mismatch", distinct.reason)

        artifact = _rdkit_artifact("CCO")
        root = _object(artifact, artifact["roots"]["support_image_root"])
        root["payload"]["witness_count"] += 1

        witness = _coverage_verification(artifact)

        self.assertFalse(witness.accepted)
        self.assertIn("coverage_count_completion_total_mismatch", witness.reason)

        artifact = _rdkit_artifact("CCO")
        count = _object(artifact, artifact["roots"]["count_ref"])
        count["payload"]["support_count"] += 1

        count_support = _coverage_verification(artifact)

        self.assertFalse(count_support.accepted)
        self.assertIn("coverage_count_support_total_mismatch", count_support.reason)

        artifact = _rdkit_artifact("CCO")
        count = _object(artifact, artifact["roots"]["count_ref"])
        count["payload"]["completion_count"] += 1

        count_completion = _coverage_verification(artifact)

        self.assertFalse(count_completion.accepted)
        self.assertIn(
            "coverage_count_completion_total_mismatch",
            count_completion.reason,
        )

    def test_coverage_terminal_bucket_mutations_are_rejected(self) -> None:
        artifact = _two_atom_completed_prefix_artifact()
        coverage = _coverage_object(artifact)
        coverage["payload"]["terminal_bucket"] = None

        missing = _coverage_verification(artifact)

        self.assertFalse(missing.accepted)
        self.assertIn("coverage_terminal_bucket_missing", missing.reason)

        artifact = _two_atom_completed_prefix_artifact()
        coverage = _coverage_object(artifact)
        coverage["payload"]["text_buckets"].append(
            {
                "text_projection": {},
                "support_count": 1,
                "string_refs": [
                    _object(
                        artifact,
                        artifact["roots"]["support_image_root"],
                    )["payload"]["support_string_refs"][0]
                ],
            }
        )

        text_bucket = _coverage_verification(artifact)

        self.assertFalse(text_bucket.accepted)
        self.assertIn("coverage_empty_string_in_text_bucket", text_bucket.reason)

        artifact = _two_atom_completed_prefix_artifact()
        coverage = _coverage_object(artifact)
        coverage["payload"]["terminal_bucket"]["terminal_projection"] = {
            **coverage["payload"]["terminal_bucket"]["terminal_projection"],
            "digest": "0" * 64,
        }

        wrong_projection = _coverage_verification(artifact)

        self.assertFalse(wrong_projection.accepted)
        self.assertIn("coverage_terminal_projection_mismatch", wrong_projection.reason)

    def test_support_string_replay_paths_accept_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "C1CC1", "C1=CC1", "[NH4+]", "[13CH4]"):
            with self.subTest(smiles=smiles):
                artifact = _rdkit_artifact(smiles)

                verification = _replay_path_verification(artifact)

                self.assertTrue(verification.accepted, verification.reason)
                self.assertGreater(verification.checked_support_strings, 0)
                self.assertIn(
                    "support_string_replay_path",
                    verification.relation_families,
                )

    def test_support_string_replay_paths_accept_empty_terminal_path(self) -> None:
        artifact = _two_atom_completed_prefix_artifact()

        verification = _replay_path_verification(artifact)

        self.assertTrue(verification.accepted, verification.reason)
        self.assertEqual(verification.checked_support_strings, 1)
        self.assertEqual(verification.checked_projection_steps, 0)

    def test_replay_path_wrong_emitted_texts_and_join_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        replay = _object(artifact, support["payload"]["replay_path_ref"])
        replay["payload"]["emitted_texts"] = ["C"]

        wrong_replay = _replay_path_verification(artifact)

        self.assertFalse(wrong_replay.accepted)
        self.assertIn("replay_path_emitted_texts_mismatch", wrong_replay.reason)

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        support["payload"]["string"] = "bad"

        wrong_join = _replay_path_verification(artifact)

        self.assertFalse(wrong_join.accepted)
        self.assertIn("replay_path_support_string_join_mismatch", wrong_join.reason)

    def test_replay_path_missing_and_extra_projection_refs_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        support["payload"]["text_projection_refs"] = (
            support["payload"]["text_projection_refs"][:-1]
        )

        missing = _replay_path_verification(artifact)

        self.assertFalse(missing.accepted)
        self.assertIn("replay_path_text_projection_count_mismatch", missing.reason)

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        support["payload"]["text_projection_refs"] = [
            *support["payload"]["text_projection_refs"],
            support["payload"]["text_projection_refs"][0],
        ]

        extra = _replay_path_verification(artifact)

        self.assertFalse(extra.accepted)
        self.assertIn("replay_path_text_projection_count_mismatch", extra.reason)

    def test_replay_path_projection_chain_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        projection = _first_text_projection_object(artifact)
        projection["payload"]["emitted_text"] = "N"

        wrong_text = _replay_path_verification(artifact)

        self.assertFalse(wrong_text.accepted)
        self.assertIn("replay_path_projection_text_mismatch", wrong_text.reason)

        artifact = _rdkit_artifact("CCO")
        projection = _first_text_projection_object(artifact)
        projection["payload"]["source_cursor"] = (
            projection["payload"]["successor_cursor"]
        )

        wrong_source = _replay_path_verification(artifact)

        self.assertFalse(wrong_source.accepted)
        self.assertIn(
            "replay_path_projection_source_cursor_mismatch",
            wrong_source.reason,
        )

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        second = _object(artifact, support["payload"]["text_projection_refs"][1])
        second["payload"]["source_cursor"] = second["payload"]["successor_cursor"]

        broken_chain = _replay_path_verification(artifact)

        self.assertFalse(broken_chain.accepted)
        self.assertIn(
            "replay_path_projection_source_cursor_mismatch",
            broken_chain.reason,
        )

    def test_replay_path_terminal_and_final_cursor_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        replay = _object(artifact, support["payload"]["replay_path_ref"])
        replay["payload"]["final_cursor_digest"] = "0" * 64

        final_cursor = _replay_path_verification(artifact)

        self.assertFalse(final_cursor.accepted)
        self.assertIn("replay_path_final_cursor_mismatch", final_cursor.reason)

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        terminal = _object(artifact, support["payload"]["terminal_projection_ref"])
        terminal["payload"]["source_cursor"] = terminal["payload"]["finalized_cursor"]

        terminal_source = _replay_path_verification(artifact)

        self.assertFalse(terminal_source.accepted)
        self.assertIn(
            "replay_path_terminal_source_cursor_mismatch",
            terminal_source.reason,
        )

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        support["payload"]["terminal_projection_ref"] = (
            support["payload"]["replay_path_ref"]
        )

        missing_terminal = _replay_path_verification(artifact)

        self.assertFalse(missing_terminal.accepted)
        self.assertIn(
            "replay_path_terminal_projection_ref_kind_mismatch",
            missing_terminal.reason,
        )

    def test_replay_path_empty_and_terminal_support_mutations_are_rejected(self) -> None:
        artifact = _two_atom_completed_prefix_artifact()
        support = _first_support_string_object(artifact)
        support["payload"]["text_projection_refs"] = ["missing"]

        empty_text = _replay_path_verification(artifact)

        self.assertFalse(empty_text.accepted)
        self.assertIn("replay_path_text_projection_count_mismatch", empty_text.reason)

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        support["payload"]["terminal_support_refs"] = [
            support["payload"]["terminal_projection_ref"]
        ]

        wrong_support = _replay_path_verification(artifact)

        self.assertFalse(wrong_support.accepted)
        self.assertIn(
            "replay_path_terminal_support_ref_kind_mismatch",
            wrong_support.reason,
        )

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        terminal_support = _object(artifact, support["payload"]["terminal_support_refs"][0])
        terminal_support["payload"]["digest"] = "0" * 64

        stale_support = _replay_path_verification(artifact)

        self.assertFalse(stale_support.accepted)
        self.assertIn(
            "replay_path_terminal_support_identity_mismatch",
            stale_support.reason,
        )

    def test_terminal_support_identities_accept_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "C1CC1", "C1=CC1", "[NH4+]", "[13CH4]"):
            with self.subTest(smiles=smiles):
                artifact = _rdkit_artifact(smiles)

                verification = _terminal_identity_verification(artifact)

                self.assertTrue(verification.accepted, verification.reason)
                self.assertGreater(verification.checked_terminal_projections, 0)
                self.assertGreater(verification.checked_terminal_supports, 0)
                self.assertGreater(verification.checked_terminal_paths, 0)

    def test_terminal_support_identities_accept_empty_terminal_bucket(self) -> None:
        artifact = _two_atom_completed_prefix_artifact()

        verification = _terminal_identity_verification(artifact)

        self.assertTrue(verification.accepted, verification.reason)
        self.assertEqual(verification.checked_terminal_paths, 1)

    def test_terminal_projection_cursor_and_support_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        terminal = _first_terminal_projection_object(artifact)
        terminal["payload"].pop("source_cursor")

        missing_source = _terminal_identity_verification(artifact)

        self.assertFalse(missing_source.accepted)

        artifact = _rdkit_artifact("CCO")
        terminal = _first_terminal_projection_object(artifact)
        terminal["payload"]["source_cursor"] = terminal["payload"]["finalized_cursor"]

        wrong_source = _terminal_identity_verification(artifact)

        self.assertFalse(wrong_source.accepted)
        self.assertIn("terminal_projection_source_cursor_mismatch", wrong_source.reason)

        artifact = _rdkit_artifact("CCO")
        terminal = _first_terminal_projection_object(artifact)
        terminal["payload"]["finalized_cursor"] = terminal["payload"]["source_cursor"]
        terminal["payload"]["terminal_support_identities"][0][
            "terminal_support_key_digest"
        ] = "0" * 64

        wrong_key = _terminal_identity_verification(artifact)

        self.assertFalse(wrong_key.accepted)
        self.assertIn("terminal_support_identity_mismatch", wrong_key.reason)

    def test_terminal_support_ordinal_and_key_mutations_are_rejected(self) -> None:
        artifact = _two_atom_completed_prefix_artifact()
        support = _first_terminal_support_object(artifact)
        support["payload"]["terminal_ordinal"] = -1
        terminal = _first_terminal_projection_object(artifact)
        terminal["payload"]["terminal_support_identities"][0]["terminal_ordinal"] = -1

        wrong_ordinal = _terminal_identity_verification(artifact)

        self.assertFalse(wrong_ordinal.accepted)
        self.assertIn("terminal_support_ordinal_negative", wrong_ordinal.reason)

        artifact = _two_atom_completed_prefix_artifact()
        terminal = _first_terminal_projection_object(artifact)
        identities = terminal["payload"]["terminal_support_identities"]
        identities[1]["terminal_ordinal"] = identities[0]["terminal_ordinal"]
        support = _first_support_string_object(artifact)
        second_support = _object(artifact, support["payload"]["terminal_support_refs"][1])
        second_support["payload"]["terminal_ordinal"] = identities[0]["terminal_ordinal"]

        duplicate_ordinal = _terminal_identity_verification(artifact)

        self.assertFalse(duplicate_ordinal.accepted)
        self.assertIn("terminal_projection_duplicate_ordinal", duplicate_ordinal.reason)

        artifact = _two_atom_completed_prefix_artifact()
        terminal = _first_terminal_projection_object(artifact)
        identities = terminal["payload"]["terminal_support_identities"]
        identities[1]["terminal_support_key_digest"] = identities[0][
            "terminal_support_key_digest"
        ]
        support = _first_support_string_object(artifact)
        second_support = _object(artifact, support["payload"]["terminal_support_refs"][1])
        second_support["payload"]["terminal_support_key_digest"] = identities[0][
            "terminal_support_key_digest"
        ]

        duplicate_key = _terminal_identity_verification(artifact)

        self.assertFalse(duplicate_key.accepted)
        self.assertIn("terminal_projection_duplicate_key_digest", duplicate_key.reason)

        artifact = _rdkit_artifact("CCO")
        support = _first_terminal_support_object(artifact)
        support["payload"]["parent_weight"] = 0
        terminal = _first_terminal_projection_object(artifact)
        terminal["payload"]["terminal_support_identities"][0]["parent_weight"] = 0

        parent_weight = _terminal_identity_verification(artifact)

        self.assertFalse(parent_weight.accepted)
        self.assertIn(
            "terminal_support_parent_weight_nonpositive",
            parent_weight.reason,
        )

    def test_terminal_support_ref_and_bucket_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        support["payload"]["terminal_support_refs"] = []

        missing_ref = _terminal_identity_verification(artifact)

        self.assertFalse(missing_ref.accepted)
        self.assertIn("terminal_support_refs_missing", missing_ref.reason)

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        support["payload"]["terminal_support_refs"] = [
            support["payload"]["terminal_projection_ref"]
        ]

        wrong_ref_kind = _terminal_identity_verification(artifact)

        self.assertFalse(wrong_ref_kind.accepted)
        self.assertIn("terminal_support_ref_kind_mismatch", wrong_ref_kind.reason)

        artifact = _rdkit_artifact("CCO")
        terminal_support = _first_terminal_support_object(artifact)
        terminal_support["payload"]["digest"] = "0" * 64

        stale_ref = _terminal_identity_verification(artifact)

        self.assertFalse(stale_ref.accepted)
        self.assertIn("terminal_support_not_in_projection", stale_ref.reason)

        artifact = _two_atom_completed_prefix_artifact()
        coverage = _coverage_object(artifact)
        coverage["payload"]["terminal_bucket"]["terminal_projection"] = {}

        wrong_bucket_projection = _terminal_identity_verification(artifact)

        self.assertFalse(wrong_bucket_projection.accepted)
        self.assertIn("terminal_bucket_projection_mismatch", wrong_bucket_projection.reason)

        artifact = _two_atom_completed_prefix_artifact()
        support = _first_support_string_object(artifact)
        support["payload"]["terminal_support_refs"] = support["payload"][
            "terminal_support_refs"
        ][:-1]

        wrong_bucket_support = _terminal_identity_verification(artifact)

        self.assertFalse(wrong_bucket_support.accepted)
        self.assertIn("terminal_projection_support_set_mismatch", wrong_bucket_support.reason)

    def test_branch_projection_identities_accept_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "CC(C)O", "C1=CC1"):
            with self.subTest(smiles=smiles):
                artifact = _rdkit_artifact(smiles)

                verification = _branch_projection_verification(artifact)

                self.assertTrue(verification.accepted, verification.reason)
                self.assertGreater(verification.checked_text_projections, 0)
                self.assertGreater(verification.checked_branch_supports, 0)

    def test_branch_projection_identity_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        projection = _first_text_projection_object(artifact)
        projection["payload"]["branch_support_refs"] = []

        missing = _branch_projection_verification(artifact)

        self.assertFalse(missing.accepted)
        self.assertIn("branch_projection_support_refs_missing", missing.reason)

        artifact = _rdkit_artifact("CCO")
        projection = _first_text_projection_object(artifact)
        branch = _first_branch_support_object(artifact)
        branch["payload"]["emitted_text"] = "N"

        wrong_text = _branch_projection_verification(artifact)

        self.assertFalse(wrong_text.accepted)
        self.assertIn("branch_projection_emitted_text_mismatch", wrong_text.reason)

        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        branch["payload"]["source_cursor_digest"] = "0" * 64

        wrong_source = _branch_projection_verification(artifact)

        self.assertFalse(wrong_source.accepted)
        self.assertIn("branch_projection_source_cursor_mismatch", wrong_source.reason)

        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        branch["payload"]["successor_cursor_digest"] = "0" * 64

        wrong_successor = _branch_projection_verification(artifact)

        self.assertFalse(wrong_successor.accepted)
        self.assertIn(
            "branch_projection_successor_cursor_mismatch",
            wrong_successor.reason,
        )

    def test_branch_projection_multiplicity_and_digest_mutations_are_rejected(
        self,
    ) -> None:
        artifact = _rdkit_artifact("CC(C)O")
        projection = _first_text_projection_object(artifact)
        projection["payload"]["branch_support_refs"] = (
            projection["payload"]["branch_support_refs"][:-1]
        )

        count_mismatch = _branch_projection_verification(artifact)

        self.assertFalse(count_mismatch.accepted)
        self.assertIn("branch_projection_multiplicity_mismatch", count_mismatch.reason)

        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        branch["payload"]["checked_branch_certificate_digest"] = "0" * 64

        stale_digest = _branch_projection_verification(artifact)

        self.assertFalse(stale_digest.accepted)
        self.assertIn(
            "branch_projection_certificate_digest_mismatch",
            stale_digest.reason,
        )

        artifact = _rdkit_artifact("CCO")
        projection = _first_text_projection_object(artifact)
        projection["payload"]["branch_support_refs"] = [
            projection["payload"]["branch_support_refs"][0],
            projection["payload"]["branch_support_refs"][0],
        ]

        duplicate_ref = _branch_projection_verification(artifact)

        self.assertFalse(duplicate_ref.accepted)
        self.assertIn("branch_projection_duplicate_support_ref", duplicate_ref.reason)

    def test_local_branch_successor_evidence_accepts_relation_fixtures(self) -> None:
        for smiles, plain_atom_count, bracket_atom_count, closure_count in (
            ("[NH4+]", 0, 1, 0),
            ("[13CH4]", 0, 1, 0),
            ("C1=CC1", 1, 0, 1),
            ("C1#CC1", 1, 0, 1),
            ("CCO", 3, 0, 0),
        ):
            with self.subTest(smiles=smiles):
                artifact = _rdkit_artifact(smiles)

                verification = _local_branch_evidence_verification(
                    _rdkit_facts(smiles),
                    artifact,
                )

                self.assertTrue(verification.accepted, verification.reason)
                self.assertGreaterEqual(
                    verification.checked_plain_atom_text_branches,
                    plain_atom_count,
                )
                self.assertGreaterEqual(
                    verification.checked_bracket_atom_text_branches,
                    bracket_atom_count,
                )
                self.assertGreaterEqual(
                    verification.checked_closure_bond_text_branches,
                    closure_count,
                )

    def test_local_atom_text_evidence_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("[NH4+]")
        evidence = _first_local_evidence(artifact, "bracket_atom_text")
        evidence["manifest"]["rendered_text"] = "[NH3+]"
        _refresh_local_evidence_digest(evidence)

        wrong_text = _local_branch_evidence_verification(_rdkit_facts("[NH4+]"), artifact)

        self.assertFalse(wrong_text.accepted)
        self.assertIn("local_bracket_atom_text_rendered_text_mismatch", wrong_text.reason)

        artifact = _rdkit_artifact("[NH4+]")
        evidence = _first_local_evidence(artifact, "bracket_atom_text")
        evidence["manifest"]["formal_charge"] = 0
        _refresh_local_evidence_digest(evidence)

        wrong_charge = _local_branch_evidence_verification(_rdkit_facts("[NH4+]"), artifact)

        self.assertFalse(wrong_charge.accepted)
        self.assertIn("local_bracket_atom_text_charge_mismatch", wrong_charge.reason)

        artifact = _rdkit_artifact("[13CH4]")
        evidence = _first_local_evidence(artifact, "bracket_atom_text")
        evidence["manifest"]["isotope"] = 12
        _refresh_local_evidence_digest(evidence)

        wrong_isotope = _local_branch_evidence_verification(
            _rdkit_facts("[13CH4]"),
            artifact,
        )

        self.assertFalse(wrong_isotope.accepted)
        self.assertIn("local_bracket_atom_text_isotope_mismatch", wrong_isotope.reason)

        artifact = _rdkit_artifact("[13CH4]")
        evidence = _first_local_evidence(artifact, "bracket_atom_text")
        evidence["manifest"]["hydrogen_count"] = 3
        _refresh_local_evidence_digest(evidence)

        wrong_h_count = _local_branch_evidence_verification(
            _rdkit_facts("[13CH4]"),
            artifact,
        )

        self.assertFalse(wrong_h_count.accepted)
        self.assertIn("local_bracket_atom_text_hydrogen_count_mismatch", wrong_h_count.reason)

        artifact = _rdkit_artifact("CCO")
        evidence = _first_local_evidence(artifact, "plain_atom_text")
        evidence["manifest"]["element"] = "N"
        _refresh_local_evidence_digest(evidence)

        wrong_plain = _local_branch_evidence_verification(_rdkit_facts("CCO"), artifact)

        self.assertFalse(wrong_plain.accepted)
        self.assertIn("local_plain_atom_text_element_mismatch", wrong_plain.reason)

    def test_local_closure_bond_text_evidence_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("C1#CC1")
        item = _first_closure_evidence_item(artifact)
        item["bond_order"] = "double"
        _refresh_local_evidence_digest(_first_local_evidence(artifact, "closure_bond_text"))

        wrong_order = _local_branch_evidence_verification(_rdkit_facts("C1#CC1"), artifact)

        self.assertFalse(wrong_order.accepted)
        self.assertIn("local_closure_bond_order_mismatch", wrong_order.reason)

        artifact = _rdkit_artifact("C1#CC1")
        item = _first_closure_evidence_item(artifact)
        item["opening_marker"] = ""
        item["closing_marker"] = ""
        _refresh_local_evidence_digest(_first_local_evidence(artifact, "closure_bond_text"))

        missing_marker = _local_branch_evidence_verification(
            _rdkit_facts("C1#CC1"),
            artifact,
        )

        self.assertFalse(missing_marker.accepted)
        self.assertIn("local_closure_marker_missing", missing_marker.reason)

        artifact = _rdkit_artifact("C1#CC1")
        item = _first_closure_evidence_item(artifact)
        item["opening_marker"] = "#"
        item["closing_marker"] = "#"
        _refresh_local_evidence_digest(_first_local_evidence(artifact, "closure_bond_text"))

        duplicate_marker = _local_branch_evidence_verification(
            _rdkit_facts("C1#CC1"),
            artifact,
        )

        self.assertFalse(duplicate_marker.accepted)
        self.assertIn("local_closure_marker_duplicate", duplicate_marker.reason)

        artifact = _rdkit_artifact("C1#CC1")
        item = _first_closure_evidence_item(artifact)
        item["opening_marker"] = "="
        item["closing_marker"] = "="
        _refresh_local_evidence_digest(_first_local_evidence(artifact, "closure_bond_text"))

        wrong_marker = _local_branch_evidence_verification(
            _rdkit_facts("C1#CC1"),
            artifact,
        )

        self.assertFalse(wrong_marker.accepted)
        self.assertIn("local_closure_marker_missing", wrong_marker.reason)

        artifact = _rdkit_artifact("C1#CC1")
        item = _first_closure_evidence_item(artifact)
        item["bond"] = "missing"
        _refresh_local_evidence_digest(_first_local_evidence(artifact, "closure_bond_text"))

        wrong_bond = _local_branch_evidence_verification(_rdkit_facts("C1#CC1"), artifact)

        self.assertFalse(wrong_bond.accepted)
        self.assertIn("local_closure_bond_missing", wrong_bond.reason)

    def test_local_branch_evidence_rejects_wrong_facts(self) -> None:
        artifact = _rdkit_artifact("[NH4+]")

        atom = _local_branch_evidence_verification(_rdkit_facts("[13CH4]"), artifact)

        self.assertFalse(atom.accepted)
        self.assertIn("local_bracket_atom_text_element_mismatch", atom.reason)

        artifact = _rdkit_artifact("[13CH4]")

        isotope = _local_branch_evidence_verification(_rdkit_facts("C"), artifact)

        self.assertFalse(isotope.accepted)
        self.assertIn("local_bracket_atom_text_isotope_mismatch", isotope.reason)

        artifact = _rdkit_artifact("C1=CC1")

        closure = _local_branch_evidence_verification(_rdkit_facts("C1CC1"), artifact)

        self.assertFalse(closure.accepted)
        self.assertIn("local_closure_bond_order_unsupported", closure.reason)

    def test_local_branch_evidence_unknown_kind_rejected(self) -> None:
        artifact = _rdkit_artifact("[NH4+]")
        evidence = _first_local_evidence(artifact, "bracket_atom_text")
        evidence["kind"] = "unknown"
        _refresh_local_evidence_digest(evidence)

        verification = _local_branch_evidence_verification(
            _rdkit_facts("[NH4+]"),
            artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("local_branch_unknown_evidence_kind", verification.reason)

    def test_local_branch_identity_fields_are_checked(self) -> None:
        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        branch["payload"]["source_cursor_digest"] = "wrong"

        wrong_source = _branch_projection_verification(artifact)

        self.assertFalse(wrong_source.accepted)
        self.assertIn("branch_projection_source_cursor_mismatch", wrong_source.reason)

        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        branch["payload"]["successor_cursor_digest"] = "wrong"

        wrong_successor = _branch_projection_verification(artifact)

        self.assertFalse(wrong_successor.accepted)
        self.assertIn("branch_projection_successor_cursor_mismatch", wrong_successor.reason)

        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        branch["payload"]["checked_branch_certificate_digest"] = ""

        missing_digest = _local_branch_evidence_verification(_rdkit_facts("CCO"), artifact)

        self.assertFalse(missing_digest.accepted)
        self.assertIn(
            "local_branch_checked_certificate_digest_missing",
            missing_digest.reason,
        )

    def test_wrong_facts_are_rejected(self) -> None:
        artifact = _snapshot_artifact(cco_facts())

        verification = verify_writer_support_artifact_for_facts(
            facts=two_atom_facts(),
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("prepared_identity_mismatch", verification.reason)

    def test_wrong_runtime_options_are_rejected(self) -> None:
        facts = cco_facts()
        artifact = _snapshot_artifact(facts)
        wrong_options = _writer_options(rooted_at_atom=0)

        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=wrong_options,
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("prepared_identity_mismatch", verification.reason)

    def test_wrong_explicit_policy_is_rejected(self) -> None:
        facts = two_atom_facts()
        artifact = _snapshot_artifact(facts)
        wrong_policy = ordinary_policy_for_facts(
            facts,
            OrdinaryPolicyOptions(single_bond_mode="both"),
        )

        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=_writer_options(),
            artifact=artifact,
            policy=wrong_policy,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("prepared_identity_mismatch", verification.reason)

    def test_mutated_prepared_identity_is_rejected(self) -> None:
        artifact = _snapshot_artifact(cco_facts())
        artifact["prepared_identity"] = deepcopy(artifact["prepared_identity"])
        artifact["prepared_identity"]["digest"] = "0" * 64
        _refresh_artifact_digest(artifact)

        verification = verify_writer_support_artifact_for_facts(
            facts=cco_facts(),
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("prepared_identity_mismatch", verification.reason)

    def test_mutated_source_prepared_identity_is_rejected(self) -> None:
        artifact = _snapshot_artifact(cco_facts())
        source = _object(artifact, artifact["roots"]["source_ref"])
        source["payload"]["prepared_identity_digest"] = "0" * 64

        verification = verify_writer_support_artifact_for_facts(
            facts=cco_facts(),
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)

    def test_structurally_invalid_artifact_is_rejected(self) -> None:
        artifact = _snapshot_artifact(cco_facts())
        artifact["objects"].pop()

        verification = verify_writer_support_artifact_for_facts(
            facts=cco_facts(),
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertFalse(verification.structurally_checked)

    def test_unknown_object_kind_is_rejected_by_structural_checker(self) -> None:
        artifact = _snapshot_artifact(cco_facts())
        root = _object(artifact, artifact["roots"]["support_image_root"])
        root["kind"] = "unknown"

        verification = verify_writer_support_artifact_for_facts(
            facts=cco_facts(),
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertFalse(verification.structurally_checked)

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
        artifact = _rdkit_artifact("C1=CC1")
        kinds = {item["kind"] for item in artifact["objects"]}

        self.assertLessEqual(kinds, set(OBJECT_KIND_OFFLINE_COVERAGE))


def _snapshot_artifact(facts):
    options = _writer_options()
    prepared = _prepare(facts)
    return deepcopy(
        writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared, options),
        )
    )


def _rdkit_facts(smiles: str):
    return ordinary_molecule_facts_from_smiles(
        smiles,
        RdkitOrdinaryExtractionOptions(include_potential_sites=False),
    )


def _rdkit_artifact(smiles: str):
    return _snapshot_artifact(_rdkit_facts(smiles))


def _rdkit_artifact_verification(smiles: str):
    facts = _rdkit_facts(smiles)
    return verify_writer_support_artifact_for_facts(
        facts=facts,
        runtime_options=_writer_options(),
        artifact=_snapshot_artifact(facts),
    )


def _two_atom_completed_prefix_artifact():
    facts = two_atom_facts()
    prepared = _prepare(facts)
    options = _writer_options()
    prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
        prepared=prepared,
        snapshot=_initial_snapshot(prepared, options),
        emitted_texts=("C", "C"),
    )
    return deepcopy(
        writer_support_artifact_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
        )
    )


def _coverage_object(artifact):
    root = _object(artifact, artifact["roots"]["support_image_root"])
    return _object(artifact, root["payload"]["coverage_ref"])


def _coverage_verification(artifact):
    return verify_support_image_coverage_offline(
        artifact=artifact,
        objects={item["object_id"]: item for item in artifact["objects"]},
    )


def _replay_path_verification(artifact):
    return verify_support_string_replay_paths_offline(
        artifact=artifact,
        objects={item["object_id"]: item for item in artifact["objects"]},
    )


def _branch_projection_verification(artifact):
    return verify_branch_projection_identities_offline(
        artifact=artifact,
        objects={item["object_id"]: item for item in artifact["objects"]},
    )


def _graph_ring_delta_verification(facts, artifact):
    return verify_graph_ring_branch_deltas_offline(
        facts=facts,
        artifact=artifact,
        objects={item["object_id"]: item for item in artifact["objects"]},
    )


def _obligation_classification(artifact):
    return classify_residual_stereo_obligations_offline(
        artifact=artifact,
        objects={item["object_id"]: item for item in artifact["objects"]},
    )


def _local_branch_evidence_verification(facts, artifact):
    return verify_local_branch_successor_evidence_offline(
        facts=facts,
        artifact=artifact,
        objects={item["object_id"]: item for item in artifact["objects"]},
    )


def _terminal_identity_verification(artifact):
    return verify_terminal_support_identities_offline(
        artifact=artifact,
        objects={item["object_id"]: item for item in artifact["objects"]},
    )


def _first_support_string_object(artifact):
    root = _object(artifact, artifact["roots"]["support_image_root"])
    return _object(artifact, root["payload"]["support_string_refs"][0])


def _first_text_projection_object(artifact):
    support = _first_support_string_object(artifact)
    return _object(artifact, support["payload"]["text_projection_refs"][0])


def _first_branch_support_object(artifact):
    projection = _first_text_projection_object(artifact)
    return _object(artifact, projection["payload"]["branch_support_refs"][0])


def _first_graph_ring_delta_branch(artifact, kind: str):
    for item in artifact["objects"]:
        if item["kind"] != "branch_support":
            continue
        if item["payload"]["graph_ring_delta"]["kind"] == kind:
            return item
    raise AssertionError(f"missing graph/ring delta kind: {kind}")


def _first_graph_ring_delta_event(branch, kind: str):
    for item in branch["payload"]["graph_ring_delta"]["manifest"]["event_manifests"]:
        if item["kind"] == kind:
            return item
    raise AssertionError(f"missing graph/ring event kind: {kind}")


def _first_terminal_projection_object(artifact):
    support = _first_support_string_object(artifact)
    return _object(artifact, support["payload"]["terminal_projection_ref"])


def _first_terminal_support_object(artifact):
    support = _first_support_string_object(artifact)
    return _object(artifact, support["payload"]["terminal_support_refs"][0])


def _first_local_evidence(artifact, kind: str):
    for item in artifact["objects"]:
        if item["kind"] != "branch_support":
            continue
        evidence = item["payload"]["local_evidence"]
        if evidence["kind"] == kind:
            return evidence
    raise AssertionError(f"missing local evidence kind: {kind}")


def _refresh_local_evidence_digest(evidence) -> None:
    evidence["digest"] = _identity_digest(
        {"kind": evidence["kind"], "manifest": evidence["manifest"]},
    )


def _refresh_graph_ring_delta_digest(delta) -> None:
    delta["digest"] = _identity_digest(
        {"kind": delta["kind"], "manifest": delta["manifest"]},
    )


def _first_closure_evidence_item(artifact):
    evidence = _first_local_evidence(artifact, "closure_bond_text")
    return evidence["manifest"]["items"][0]


def _initial_snapshot(prepared, options):
    return capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=initial_writer_frontier_cursor(prepared, options),
    )


def _writer_options(rooted_at_atom=-1):
    return SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        canonical=False,
        do_random=True,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _object(artifact, object_id):
    return next(item for item in artifact["objects"] if item["object_id"] == object_id)


def _refresh_artifact_digest(artifact):
    artifact["digest"] = _digest_terms_bounded(
        artifact_manifest(artifact),
        budget=WriterEnvelopeWorkBudget(),
        operation="test.artifact_manifest.digest",
    )


if __name__ == "__main__":
    unittest.main()
