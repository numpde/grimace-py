"""Physically owned rich support-artifact graph-relations contracts."""

from dataclasses import replace
import unittest
from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.facts import StereoFacts
from grimace._south_star1.writer_support_artifact_offline_verifier import validate_writer_bracket_atom_text_against_facts
from grimace._south_star1.writer_support_artifact_offline_verifier import verify_writer_support_artifact_offline_replay
from grimace._south_star1.writer_support_artifact_envelope import writer_support_artifact_envelope_for_snapshot
from tests.south_star1.writer_artifact_test_support import refresh_kind_manifest_digest
from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options
from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.writer_test_fixtures import directional_non_single_ring_carrier_facts
from tests.south_star1.writer_support_artifact_fixtures import rdkit_graph_facts, rdkit_support_artifact_fixture
from tests.south_star1.writer_support_artifact_queries import first_branch_support_object, first_graph_ring_delta_branch, first_graph_ring_delta_event, first_local_evidence, first_text_projection_object, verify_branch_projection_relation, verify_graph_ring_delta_relation, verify_local_branch_evidence_relation
from tests.south_star1.writer_support_artifact_graph_test_support import first_directional_bond_delta_branch, first_closure_evidence_item, tetra_facts_with_implicit_h_only_outside_specified_site





class WriterSupportArtifactGraphRelationTest(unittest.TestCase):
    def test_branch_projection_identities_accept_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "CC(C)O", "C1=CC1"):
            with self.subTest(smiles=smiles):
                artifact = rdkit_support_artifact_fixture(smiles).artifact

                verification = verify_branch_projection_relation(artifact)

                self.assertTrue(verification.accepted, verification.reason)
                self.assertGreater(verification.checked_text_projections, 0)
                self.assertGreater(verification.checked_branch_supports, 0)

    def test_branch_projection_identity_mutations_are_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        projection = first_text_projection_object(artifact)
        projection["payload"]["branch_support_refs"] = []

        missing = verify_branch_projection_relation(artifact)

        self.assertFalse(missing.accepted)
        self.assertIn("branch_projection_support_refs_missing", missing.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        projection = first_text_projection_object(artifact)
        branch = first_branch_support_object(artifact)
        branch["payload"]["emitted_text"] = "N"

        wrong_text = verify_branch_projection_relation(artifact)

        self.assertFalse(wrong_text.accepted)
        self.assertIn("branch_projection_emitted_text_mismatch", wrong_text.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        branch = first_branch_support_object(artifact)
        branch["payload"]["source_cursor_digest"] = "0" * 64

        wrong_source = verify_branch_projection_relation(artifact)

        self.assertFalse(wrong_source.accepted)
        self.assertIn("branch_projection_source_cursor_mismatch", wrong_source.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        branch = first_branch_support_object(artifact)
        branch["payload"]["successor_cursor_digest"] = "0" * 64

        wrong_successor = verify_branch_projection_relation(artifact)

        self.assertFalse(wrong_successor.accepted)
        self.assertIn(
            "branch_projection_successor_cursor_mismatch",
            wrong_successor.reason,
        )

    def test_branch_projection_multiplicity_and_digest_mutations_are_rejected(
        self,
    ) -> None:
        artifact = rdkit_support_artifact_fixture("CC(C)O").artifact
        projection = first_text_projection_object(artifact)
        projection["payload"]["branch_support_refs"] = (
            projection["payload"]["branch_support_refs"][:-1]
        )

        count_mismatch = verify_branch_projection_relation(artifact)

        self.assertFalse(count_mismatch.accepted)
        self.assertIn("branch_projection_multiplicity_mismatch", count_mismatch.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        branch = first_branch_support_object(artifact)
        branch["payload"]["checked_branch_certificate_digest"] = "0" * 64

        stale_digest = verify_branch_projection_relation(artifact)

        self.assertFalse(stale_digest.accepted)
        self.assertIn(
            "branch_projection_certificate_digest_mismatch",
            stale_digest.reason,
        )

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        projection = first_text_projection_object(artifact)
        projection["payload"]["branch_support_refs"] = [
            projection["payload"]["branch_support_refs"][0],
            projection["payload"]["branch_support_refs"][0],
        ]

        duplicate_ref = verify_branch_projection_relation(artifact)

        self.assertFalse(duplicate_ref.accepted)
        self.assertIn("branch_projection_duplicate_support_ref", duplicate_ref.reason)

    def test_graph_ring_branch_delta_mutations_are_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        branch = first_graph_ring_delta_branch(artifact, "bond_advance")
        event = first_graph_ring_delta_event(branch, "bond_emitted")
        event["bond"] = "missing"
        refresh_kind_manifest_digest(branch["payload"]["graph_ring_delta"], operation="test.graph_ring_delta.digest")

        wrong_bond = verify_graph_ring_delta_relation(rdkit_graph_facts("CCO"), artifact)

        self.assertFalse(wrong_bond.accepted)
        self.assertIn("local_closure_bond_missing", wrong_bond.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        branch = first_graph_ring_delta_branch(artifact, "atom_start")
        event = first_graph_ring_delta_event(branch, "atom_emitted")
        event["atom"] = "missing"
        refresh_kind_manifest_digest(branch["payload"]["graph_ring_delta"], operation="test.graph_ring_delta.digest")

        wrong_atom = verify_graph_ring_delta_relation(rdkit_graph_facts("CCO"), artifact)

        self.assertFalse(wrong_atom.accepted)
        self.assertIn("local_atom_text_atom_missing", wrong_atom.reason)

        artifact = rdkit_support_artifact_fixture("C1=CC1").artifact
        branch = first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        event = first_graph_ring_delta_event(branch, "ring_endpoint_paired")
        event["label"] = "wrong"
        refresh_kind_manifest_digest(branch["payload"]["graph_ring_delta"], operation="test.graph_ring_delta.digest")

        wrong_label = verify_graph_ring_delta_relation(rdkit_graph_facts("C1=CC1"), artifact)

        self.assertFalse(wrong_label.accepted)
        self.assertIn("graph_ring_endpoint_label_mismatch", wrong_label.reason)

        artifact = rdkit_support_artifact_fixture("C1=CC1").artifact
        branch = first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        branch["payload"]["successor_state_digest"] = "wrong"

        wrong_state = verify_graph_ring_delta_relation(rdkit_graph_facts("C1=CC1"), artifact)

        self.assertFalse(wrong_state.accepted)
        self.assertIn("graph_ring_delta_successor_state_digest_mismatch", wrong_state.reason)

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
                artifact = rdkit_support_artifact_fixture(smiles).artifact

                verification = verify_graph_ring_delta_relation(
                    rdkit_graph_facts(smiles),
                    artifact,
                )

                self.assertTrue(verification.accepted, verification.reason)
                self.assertGreater(verification.checked_branches, 0)

    def test_graph_ring_directional_carrier_text_is_replayed_by_direction_mark(
        self,
    ) -> None:
        facts = directional_non_single_ring_carrier_facts()
        prepared = prepare_writer_facts(facts)
        options = writer_runtime_options(rooted_at_atom=0)
        artifact = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=initial_writer_snapshot(prepared, options),
        )

        accepted = verify_graph_ring_delta_relation(facts, artifact)

        self.assertTrue(accepted.accepted, accepted.reason)

        branch = first_directional_bond_delta_branch(artifact)
        event = first_graph_ring_delta_event(branch, "bond_emitted")
        event["text"] = "="
        refresh_kind_manifest_digest(branch["payload"]["graph_ring_delta"], operation="test.graph_ring_delta.digest")

        wrong_direction_text = verify_graph_ring_delta_relation(facts, artifact)

        self.assertFalse(wrong_direction_text.accepted)
        self.assertIn("graph_ring_bond_marker_mismatch", wrong_direction_text.reason)
        self.assertIn("expected_direction_text", wrong_direction_text.reason)
        self.assertIn("direction_mark", wrong_direction_text.reason)
        self.assertIn("successor_certificate", wrong_direction_text.reason)

    def test_local_atom_text_evidence_mutations_are_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("[NH4+]").artifact
        evidence = first_local_evidence(artifact, "bracket_atom_text")
        evidence["manifest"]["rendered_text"] = "[NH3+]"
        refresh_kind_manifest_digest(evidence, operation="test.local_evidence.digest")

        wrong_text = verify_local_branch_evidence_relation(rdkit_graph_facts("[NH4+]"), artifact)

        self.assertFalse(wrong_text.accepted)
        self.assertIn("local_bracket_atom_text_rendered_text_mismatch", wrong_text.reason)

        artifact = rdkit_support_artifact_fixture("[NH4+]").artifact
        evidence = first_local_evidence(artifact, "bracket_atom_text")
        evidence["manifest"]["formal_charge"] = 0
        refresh_kind_manifest_digest(evidence, operation="test.local_evidence.digest")

        wrong_charge = verify_local_branch_evidence_relation(rdkit_graph_facts("[NH4+]"), artifact)

        self.assertFalse(wrong_charge.accepted)
        self.assertIn("local_bracket_atom_text_charge_mismatch", wrong_charge.reason)

        artifact = rdkit_support_artifact_fixture("[13CH4]").artifact
        evidence = first_local_evidence(artifact, "bracket_atom_text")
        evidence["manifest"]["isotope"] = 12
        refresh_kind_manifest_digest(evidence, operation="test.local_evidence.digest")

        wrong_isotope = verify_local_branch_evidence_relation(
            rdkit_graph_facts("[13CH4]"),
            artifact,
        )

        self.assertFalse(wrong_isotope.accepted)
        self.assertIn("local_bracket_atom_text_isotope_mismatch", wrong_isotope.reason)

        artifact = rdkit_support_artifact_fixture("[13CH4]").artifact
        evidence = first_local_evidence(artifact, "bracket_atom_text")
        evidence["manifest"]["hydrogen_count"] = 3
        refresh_kind_manifest_digest(evidence, operation="test.local_evidence.digest")

        wrong_h_count = verify_local_branch_evidence_relation(
            rdkit_graph_facts("[13CH4]"),
            artifact,
        )

        self.assertFalse(wrong_h_count.accepted)
        self.assertIn("local_bracket_atom_text_hydrogen_count_mismatch", wrong_h_count.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        evidence = first_local_evidence(artifact, "plain_atom_text")
        evidence["manifest"]["element"] = "N"
        refresh_kind_manifest_digest(evidence, operation="test.local_evidence.digest")

        wrong_plain = verify_local_branch_evidence_relation(rdkit_graph_facts("CCO"), artifact)

        self.assertFalse(wrong_plain.accepted)
        self.assertIn("local_plain_atom_text_element_mismatch", wrong_plain.reason)

    def test_local_branch_evidence_rejects_wrong_facts(self) -> None:
        artifact = rdkit_support_artifact_fixture("[NH4+]").artifact

        atom = verify_local_branch_evidence_relation(rdkit_graph_facts("[13CH4]"), artifact)

        self.assertFalse(atom.accepted)
        self.assertIn("local_bracket_atom_text_element_mismatch", atom.reason)

        artifact = rdkit_support_artifact_fixture("[13CH4]").artifact

        isotope = verify_local_branch_evidence_relation(rdkit_graph_facts("C"), artifact)

        self.assertFalse(isotope.accepted)
        self.assertIn("local_bracket_atom_text_isotope_mismatch", isotope.reason)

        artifact = rdkit_support_artifact_fixture("C1=CC1").artifact

        closure = verify_local_branch_evidence_relation(rdkit_graph_facts("C1CC1"), artifact)

        self.assertFalse(closure.accepted)
        self.assertIn("local_closure_bond_order_unsupported", closure.reason)

    def test_local_branch_evidence_unknown_kind_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("[NH4+]").artifact
        evidence = first_local_evidence(artifact, "bracket_atom_text")
        evidence["kind"] = "unknown"
        refresh_kind_manifest_digest(evidence, operation="test.local_evidence.digest")

        verification = verify_local_branch_evidence_relation(
            rdkit_graph_facts("[NH4+]"),
            artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("local_branch_unknown_evidence_kind", verification.reason)

    def test_local_branch_identity_fields_are_checked(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        branch = first_branch_support_object(artifact)
        branch["payload"]["source_cursor_digest"] = "wrong"

        wrong_source = verify_branch_projection_relation(artifact)

        self.assertFalse(wrong_source.accepted)
        self.assertIn("branch_projection_source_cursor_mismatch", wrong_source.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        branch = first_branch_support_object(artifact)
        branch["payload"]["successor_cursor_digest"] = "wrong"

        wrong_successor = verify_branch_projection_relation(artifact)

        self.assertFalse(wrong_successor.accepted)
        self.assertIn("branch_projection_successor_cursor_mismatch", wrong_successor.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        branch = first_branch_support_object(artifact)
        branch["payload"]["checked_branch_certificate_digest"] = ""

        missing_digest = verify_local_branch_evidence_relation(rdkit_graph_facts("CCO"), artifact)

        self.assertFalse(missing_digest.accepted)
        self.assertIn(
            "local_branch_checked_certificate_digest_missing",
            missing_digest.reason,
        )

    def test_local_branch_successor_evidence_accepts_relation_fixtures(self) -> None:
        for smiles, plain_atom_count, bracket_atom_count, closure_count in (
            ("[NH4+]", 0, 1, 0),
            ("[13CH4]", 0, 1, 0),
            ("C1=CC1", 1, 0, 1),
            ("C1#CC1", 1, 0, 1),
            ("CCO", 3, 0, 0),
        ):
            with self.subTest(smiles=smiles):
                artifact = rdkit_support_artifact_fixture(smiles).artifact

                verification = verify_local_branch_evidence_relation(
                    rdkit_graph_facts(smiles),
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

    def test_local_closure_bond_text_evidence_mutations_are_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("C1#CC1").artifact
        item = first_closure_evidence_item(artifact)
        item["bond_order"] = "double"
        refresh_kind_manifest_digest(first_local_evidence(artifact, "closure_bond_text"), operation="test.local_evidence.digest")

        wrong_order = verify_local_branch_evidence_relation(rdkit_graph_facts("C1#CC1"), artifact)

        self.assertFalse(wrong_order.accepted)
        self.assertIn("local_closure_bond_order_mismatch", wrong_order.reason)

        artifact = rdkit_support_artifact_fixture("C1#CC1").artifact
        item = first_closure_evidence_item(artifact)
        item["opening_marker"] = ""
        item["closing_marker"] = ""
        refresh_kind_manifest_digest(first_local_evidence(artifact, "closure_bond_text"), operation="test.local_evidence.digest")

        missing_marker = verify_local_branch_evidence_relation(
            rdkit_graph_facts("C1#CC1"),
            artifact,
        )

        self.assertFalse(missing_marker.accepted)
        self.assertIn("local_closure_marker_missing", missing_marker.reason)

        artifact = rdkit_support_artifact_fixture("C1#CC1").artifact
        item = first_closure_evidence_item(artifact)
        item["opening_marker"] = "#"
        item["closing_marker"] = "#"
        refresh_kind_manifest_digest(first_local_evidence(artifact, "closure_bond_text"), operation="test.local_evidence.digest")

        duplicate_marker = verify_local_branch_evidence_relation(
            rdkit_graph_facts("C1#CC1"),
            artifact,
        )

        self.assertFalse(duplicate_marker.accepted)
        self.assertIn("local_closure_marker_duplicate", duplicate_marker.reason)

        artifact = rdkit_support_artifact_fixture("C1#CC1").artifact
        item = first_closure_evidence_item(artifact)
        item["opening_marker"] = "="
        item["closing_marker"] = "="
        refresh_kind_manifest_digest(first_local_evidence(artifact, "closure_bond_text"), operation="test.local_evidence.digest")

        wrong_marker = verify_local_branch_evidence_relation(
            rdkit_graph_facts("C1#CC1"),
            artifact,
        )

        self.assertFalse(wrong_marker.accepted)
        self.assertIn("local_closure_marker_missing", wrong_marker.reason)

        artifact = rdkit_support_artifact_fixture("C1#CC1").artifact
        item = first_closure_evidence_item(artifact)
        item["bond"] = "missing"
        refresh_kind_manifest_digest(first_local_evidence(artifact, "closure_bond_text"), operation="test.local_evidence.digest")

        wrong_bond = verify_local_branch_evidence_relation(rdkit_graph_facts("C1#CC1"), artifact)

        self.assertFalse(wrong_bond.accepted)
        self.assertIn("local_closure_bond_missing", wrong_bond.reason)

    def test_offline_bracket_atom_replay_rejects_wrong_facts(self) -> None:
        with self.assertRaisesRegex(SouthStarError, "bracket_atom_text_facts_mismatch"):
            validate_writer_bracket_atom_text_against_facts(
                facts=rdkit_graph_facts("[NH3+]"),
                rendered_text="[NH4+]",
            )

        with self.assertRaisesRegex(SouthStarError, "bracket_atom_text_facts_mismatch"):
            validate_writer_bracket_atom_text_against_facts(
                facts=rdkit_graph_facts("[12CH4]"),
                rendered_text="[13CH4]",
            )
        with self.assertRaisesRegex(SouthStarError, "bracket_atom_text_facts_mismatch"):
            validate_writer_bracket_atom_text_against_facts(
                facts=rdkit_graph_facts("[O-]"),
                rendered_text="[OH-]",
            )
        with self.assertRaisesRegex(SouthStarError, "bracket_atom_text_facts_mismatch"):
            validate_writer_bracket_atom_text_against_facts(
                facts=rdkit_graph_facts("[NH4+]"),
                rendered_text="[NH3+]",
            )

    def test_offline_joint_closure_replay_rejects_wrong_facts(self) -> None:
        artifact = rdkit_support_artifact_fixture("C1=CC1").artifact
        verification = verify_writer_support_artifact_offline_replay(
            facts=rdkit_graph_facts("C1CC1"),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("graph_ring_bond_marker_mismatch", verification.reason)

    def test_offline_tetra_bracket_atom_replay_rejects_wrong_facts(self) -> None:
        facts = tetrahedral_facts()
        validate_writer_bracket_atom_text_against_facts(
            facts=facts,
            rendered_text="[C@H]",
        )
        validate_writer_bracket_atom_text_against_facts(
            facts=facts,
            rendered_text="[C@@H]",
        )
        double_recorded_h_facts = replace(
            facts,
            atoms=(
                replace(facts.atoms[0], implicit_h_count=1),
                *facts.atoms[1:],
            ),
        )
        validate_writer_bracket_atom_text_against_facts(
            facts=double_recorded_h_facts,
            rendered_text="[C@H]",
        )

        cases = (
            replace(facts, stereo=StereoFacts()),
            replace(facts, ligand_occurrences=facts.ligand_occurrences[:-1]),
            tetra_facts_with_implicit_h_only_outside_specified_site(facts),
            replace(
                facts,
                atoms=(
                    replace(facts.atoms[0], implicit_h_count=2),
                    *facts.atoms[1:],
                ),
            ),
        )
        for wrong_facts in cases:
            with self.subTest(facts=wrong_facts):
                with self.assertRaisesRegex(
                    SouthStarError,
                    "bracket_atom_text_facts_mismatch",
                ):
                    validate_writer_bracket_atom_text_against_facts(
                        facts=wrong_facts,
                        rendered_text="[C@H]",
                    )
        for rendered_text in ("[N@H]", "[13C@H]", "[C@H+]"):
            with self.subTest(rendered_text=rendered_text):
                with self.assertRaisesRegex(
                    SouthStarError,
                    "bracket_atom_text_facts_mismatch",
                ):
                    validate_writer_bracket_atom_text_against_facts(
                        facts=facts,
                        rendered_text=rendered_text,
                    )
