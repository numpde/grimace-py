"""Physically owned rich support-artifact integration contracts."""

from copy import deepcopy
import unittest
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.writer_support_artifact_fact_verifier import OBJECT_KIND_OFFLINE_COVERAGE
from grimace._south_star1.writer_support_artifact_fact_verifier import verify_writer_support_artifact_for_facts
from tests.south_star1.helpers import cco_facts
from tests.south_star1.writer_artifact_resealing import reseal_support_artifact
from tests.south_star1.writer_artifact_test_support import artifact_object_by_id
from tests.south_star1.writer_test_context import writer_runtime_options
from tests.south_star1.helpers import two_atom_facts
from tests.south_star1.writer_support_artifact_fixtures import (
    rdkit_support_artifact_fixture,
    completed_prefix_support_artifact_fixture,
    support_artifact_fixture,
)





class WriterSupportArtifactFactVerifierTest(unittest.TestCase):
    def test_facts_bound_verifier_reports_bracket_atom_offline_check(self) -> None:
        for smiles in ("[N+]", "[NH+]", "[NH2+]", "[NH3+]", "[NH4+]", "[O-]", "[OH-]"):
            with self.subTest(smiles=smiles):
                fixture = rdkit_support_artifact_fixture(smiles)
                verification = verify_writer_support_artifact_for_facts(
                    facts=fixture.facts,
                    runtime_options=fixture.runtime_options,
                    artifact=fixture.artifact,
                )

                self.assertTrue(verification.accepted, verification.reason)
                self.assertIn(
                    "bracket_atom_text",
                    verification.offline_checked_relation_families,
                )
                self.assertIn("text_projection", verification.offline_checked_object_kinds)
                self.assertTrue(verification.offline_replay_complete)

    def test_facts_bound_verifier_reports_isotope_atom_offline_check(self) -> None:
        fixture = rdkit_support_artifact_fixture("[13CH4]")
        verification = verify_writer_support_artifact_for_facts(
            facts=fixture.facts,
            runtime_options=fixture.runtime_options,
            artifact=fixture.artifact,
        )

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
        fixture = rdkit_support_artifact_fixture("C1=CC1")
        verification = verify_writer_support_artifact_for_facts(
            facts=fixture.facts,
            runtime_options=fixture.runtime_options,
            artifact=fixture.artifact,
        )

        self.assertTrue(verification.accepted, verification.reason)
        self.assertIn(
            "closure_bond_text",
            verification.offline_checked_relation_families,
        )
        self.assertTrue(verification.offline_replay_complete)

    def test_facts_bound_verifier_reports_joint_triple_closure_offline_check(
        self,
    ) -> None:
        fixture = rdkit_support_artifact_fixture("C1#CC1")
        verification = verify_writer_support_artifact_for_facts(
            facts=fixture.facts,
            runtime_options=fixture.runtime_options,
            artifact=fixture.artifact,
        )

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
        fixture = completed_prefix_support_artifact_fixture()
        facts = fixture.facts
        options = fixture.runtime_options
        artifact = fixture.artifact

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
        fixture = support_artifact_fixture(cco_facts())
        facts = fixture.facts
        options = fixture.runtime_options
        artifact = fixture.artifact

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
