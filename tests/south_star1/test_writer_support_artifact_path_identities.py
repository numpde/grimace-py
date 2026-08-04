"""Physically owned rich support-artifact path-identities contracts."""

from copy import deepcopy
import unittest
from grimace._south_star1.writer_support_artifact_checker import verify_writer_support_artifact_consistency
from grimace._south_star1.writer_support_artifact_fact_verifier import verify_writer_support_artifact_for_facts
from tests.south_star1.writer_artifact_resealing import reseal_support_artifact
from tests.south_star1.writer_artifact_test_support import artifact_object_by_id
from tests.south_star1.writer_test_context import writer_runtime_options
from tests.south_star1.writer_support_artifact_fixtures import completed_prefix_support_artifact_fixture, rdkit_graph_facts, rdkit_support_artifact_fixture
from tests.south_star1.writer_support_artifact_queries import coverage_object, first_support_string_object, first_terminal_projection_object, first_terminal_support_object, first_text_projection_object, verify_support_string_replay_relation, verify_terminal_identity_relation





class WriterSupportArtifactPathIdentityTest(unittest.TestCase):
    def test_replay_path_empty_and_terminal_support_mutations_are_rejected(self) -> None:
        artifact = completed_prefix_support_artifact_fixture().artifact
        support = first_support_string_object(artifact)
        support["payload"]["text_projection_refs"] = ["missing"]

        empty_text = verify_support_string_replay_relation(artifact)

        self.assertFalse(empty_text.accepted)
        self.assertIn("replay_path_text_projection_count_mismatch", empty_text.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        support = first_support_string_object(artifact)
        support["payload"]["terminal_support_refs"] = [
            support["payload"]["terminal_projection_ref"]
        ]

        wrong_support = verify_support_string_replay_relation(artifact)

        self.assertFalse(wrong_support.accepted)
        self.assertIn(
            "replay_path_terminal_support_ref_kind_mismatch",
            wrong_support.reason,
        )

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        support = first_support_string_object(artifact)
        terminal_support = artifact_object_by_id(artifact, support["payload"]["terminal_support_refs"][0])
        terminal_support["payload"]["digest"] = "0" * 64

        stale_support = verify_support_string_replay_relation(artifact)

        self.assertFalse(stale_support.accepted)
        self.assertIn(
            "replay_path_terminal_support_identity_mismatch",
            stale_support.reason,
        )

    def test_replay_path_missing_and_extra_projection_refs_are_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        support = first_support_string_object(artifact)
        support["payload"]["text_projection_refs"] = (
            support["payload"]["text_projection_refs"][:-1]
        )

        missing = verify_support_string_replay_relation(artifact)

        self.assertFalse(missing.accepted)
        self.assertIn("replay_path_text_projection_count_mismatch", missing.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        support = first_support_string_object(artifact)
        support["payload"]["text_projection_refs"] = [
            *support["payload"]["text_projection_refs"],
            support["payload"]["text_projection_refs"][0],
        ]

        extra = verify_support_string_replay_relation(artifact)

        self.assertFalse(extra.accepted)
        self.assertIn("replay_path_text_projection_count_mismatch", extra.reason)

    def test_replay_path_projection_chain_mutations_are_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        projection = first_text_projection_object(artifact)
        projection["payload"]["emitted_text"] = "N"

        wrong_text = verify_support_string_replay_relation(artifact)

        self.assertFalse(wrong_text.accepted)
        self.assertIn("replay_path_projection_text_mismatch", wrong_text.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        projection = first_text_projection_object(artifact)
        projection["payload"]["source_cursor"] = (
            projection["payload"]["successor_cursor"]
        )

        wrong_source = verify_support_string_replay_relation(artifact)

        self.assertFalse(wrong_source.accepted)
        self.assertIn(
            "replay_path_projection_source_cursor_mismatch",
            wrong_source.reason,
        )

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        support = first_support_string_object(artifact)
        second = artifact_object_by_id(artifact, support["payload"]["text_projection_refs"][1])
        second["payload"]["source_cursor"] = second["payload"]["successor_cursor"]

        broken_chain = verify_support_string_replay_relation(artifact)

        self.assertFalse(broken_chain.accepted)
        self.assertIn(
            "replay_path_projection_source_cursor_mismatch",
            broken_chain.reason,
        )

    def test_replay_path_terminal_and_final_cursor_mutations_are_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        support = first_support_string_object(artifact)
        replay = artifact_object_by_id(artifact, support["payload"]["replay_path_ref"])
        replay["payload"]["final_cursor_digest"] = "0" * 64

        final_cursor = verify_support_string_replay_relation(artifact)

        self.assertFalse(final_cursor.accepted)
        self.assertIn("replay_path_final_cursor_mismatch", final_cursor.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        support = first_support_string_object(artifact)
        terminal = artifact_object_by_id(artifact, support["payload"]["terminal_projection_ref"])
        terminal["payload"]["source_cursor"] = terminal["payload"]["finalized_cursor"]

        terminal_source = verify_support_string_replay_relation(artifact)

        self.assertFalse(terminal_source.accepted)
        self.assertIn(
            "replay_path_terminal_source_cursor_mismatch",
            terminal_source.reason,
        )

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        support = first_support_string_object(artifact)
        support["payload"]["terminal_projection_ref"] = (
            support["payload"]["replay_path_ref"]
        )

        missing_terminal = verify_support_string_replay_relation(artifact)

        self.assertFalse(missing_terminal.accepted)
        self.assertIn(
            "replay_path_terminal_projection_ref_kind_mismatch",
            missing_terminal.reason,
        )

    def test_replay_path_wrong_emitted_texts_and_join_are_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        support = first_support_string_object(artifact)
        replay = artifact_object_by_id(artifact, support["payload"]["replay_path_ref"])
        replay["payload"]["emitted_texts"] = ["C"]

        wrong_replay = verify_support_string_replay_relation(artifact)

        self.assertFalse(wrong_replay.accepted)
        self.assertIn("replay_path_emitted_texts_mismatch", wrong_replay.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        support = first_support_string_object(artifact)
        support["payload"]["string"] = "bad"

        wrong_join = verify_support_string_replay_relation(artifact)

        self.assertFalse(wrong_join.accepted)
        self.assertIn("replay_path_support_string_join_mismatch", wrong_join.reason)

    def test_support_string_replay_paths_accept_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "C1CC1", "C1=CC1", "[NH4+]", "[13CH4]"):
            with self.subTest(smiles=smiles):
                artifact = rdkit_support_artifact_fixture(smiles).artifact

                verification = verify_support_string_replay_relation(artifact)

                self.assertTrue(verification.accepted, verification.reason)
                self.assertGreater(verification.checked_support_strings, 0)
                self.assertIn(
                    "support_string_replay_path",
                    verification.relation_families,
                )

    def test_support_string_replay_paths_accept_empty_terminal_path(self) -> None:
        artifact = completed_prefix_support_artifact_fixture().artifact

        verification = verify_support_string_replay_relation(artifact)

        self.assertTrue(verification.accepted, verification.reason)
        self.assertEqual(verification.checked_support_strings, 1)
        self.assertEqual(verification.checked_projection_steps, 0)

    def test_terminal_projection_cursor_and_support_mutations_are_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        terminal = first_terminal_projection_object(artifact)
        terminal["payload"].pop("source_cursor")

        missing_source = verify_terminal_identity_relation(artifact)

        self.assertFalse(missing_source.accepted)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        terminal = first_terminal_projection_object(artifact)
        terminal["payload"]["source_cursor"] = terminal["payload"]["finalized_cursor"]

        wrong_source = verify_terminal_identity_relation(artifact)

        self.assertFalse(wrong_source.accepted)
        self.assertIn("terminal_projection_source_cursor_mismatch", wrong_source.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        terminal = first_terminal_projection_object(artifact)
        terminal["payload"]["finalized_cursor"] = terminal["payload"]["source_cursor"]
        terminal["payload"]["terminal_support_identities"][0][
            "terminal_support_key_digest"
        ] = "0" * 64

        wrong_key = verify_terminal_identity_relation(artifact)

        self.assertFalse(wrong_key.accepted)
        self.assertIn("terminal_support_identity_mismatch", wrong_key.reason)

    def test_terminal_support_identities_accept_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "C1CC1", "C1=CC1", "[NH4+]", "[13CH4]"):
            with self.subTest(smiles=smiles):
                artifact = rdkit_support_artifact_fixture(smiles).artifact

                verification = verify_terminal_identity_relation(artifact)

                self.assertTrue(verification.accepted, verification.reason)
                self.assertGreater(verification.checked_terminal_projections, 0)
                self.assertGreater(verification.checked_terminal_supports, 0)
                self.assertGreater(verification.checked_terminal_paths, 0)

    def test_terminal_support_identities_accept_empty_terminal_bucket(self) -> None:
        artifact = completed_prefix_support_artifact_fixture().artifact

        verification = verify_terminal_identity_relation(artifact)

        self.assertTrue(verification.accepted, verification.reason)
        self.assertEqual(verification.checked_terminal_paths, 1)

    def test_terminal_support_identity_forgeries_reject_after_redigest(self) -> None:
        facts = rdkit_graph_facts("CCO")
        cases = (
            (
                "support_key",
                lambda payload: payload.__setitem__(
                    "terminal_support_key_digest", "0" * 64
                ),
                "terminal_support_terminal_support_key_digest_mismatch",
            ),
            (
                "certificate",
                lambda payload: payload["terminal_certificate_digests"].reverse(),
                "terminal_support_terminal_certificate_digests_mismatch",
            ),
            (
                "lifecycle_tuple",
                lambda payload: payload.__setitem__(
                    "terminal_stereo_lifecycle_evidence_digest", "0" * 64
                ),
                "terminal_support_terminal_stereo_lifecycle_evidence_digest_mismatch",
            ),
            (
                "discharged_flag",
                lambda payload: payload["obligation_manifests"]
                ["terminal_graph_obligation_work"][0].__setitem__(
                    "is_discharged", True
                ),
                "terminal_graph_manifest_mismatch",
            ),
        )
        for name, mutate, reason in cases:
            with self.subTest(name=name):
                artifact = deepcopy(rdkit_support_artifact_fixture("CCO").artifact)
                terminal = first_terminal_support_object(artifact)
                mutate(terminal["payload"])
                identity = {
                    key: value
                    for key, value in terminal["payload"].items()
                    if key not in (
                        "terminalization_term",
                        "terminalization_term_digest",
                        "obligation_summary",
                        "obligation_manifests",
                    )
                }
                for item in artifact["objects"]:
                    if item["kind"] != "terminal_projection":
                        continue
                    for index, candidate in enumerate(
                        item["payload"]["terminal_support_identities"]
                    ):
                        if candidate["digest"] == identity["digest"]:
                            item["payload"]["terminal_support_identities"][index] = (
                                deepcopy(identity)
                            )
                reseal_support_artifact(artifact)
                structural = verify_writer_support_artifact_consistency(artifact)
                checked = verify_writer_support_artifact_for_facts(
                    facts=facts,
                    runtime_options=writer_runtime_options(),
                    artifact=artifact,
                )
                self.assertTrue(structural.accepted, structural.reason)
                self.assertFalse(checked.accepted)
                self.assertIn(reason, checked.reason)

    def test_terminal_support_ordinal_and_key_mutations_are_rejected(self) -> None:
        artifact = completed_prefix_support_artifact_fixture().artifact
        support = first_terminal_support_object(artifact)
        support["payload"]["terminal_ordinal"] = -1
        terminal = first_terminal_projection_object(artifact)
        terminal["payload"]["terminal_support_identities"][0]["terminal_ordinal"] = -1

        wrong_ordinal = verify_terminal_identity_relation(artifact)

        self.assertFalse(wrong_ordinal.accepted)
        self.assertIn("terminal_support_ordinal_negative", wrong_ordinal.reason)

        artifact = completed_prefix_support_artifact_fixture().artifact
        terminal = first_terminal_projection_object(artifact)
        identities = terminal["payload"]["terminal_support_identities"]
        identities[1]["terminal_ordinal"] = identities[0]["terminal_ordinal"]
        support = first_support_string_object(artifact)
        second_support = artifact_object_by_id(artifact, support["payload"]["terminal_support_refs"][1])
        second_support["payload"]["terminal_ordinal"] = identities[0]["terminal_ordinal"]

        duplicate_ordinal = verify_terminal_identity_relation(artifact)

        self.assertFalse(duplicate_ordinal.accepted)
        self.assertIn("terminal_projection_duplicate_ordinal", duplicate_ordinal.reason)

        artifact = completed_prefix_support_artifact_fixture().artifact
        terminal = first_terminal_projection_object(artifact)
        identities = terminal["payload"]["terminal_support_identities"]
        identities[1]["terminal_support_key_digest"] = identities[0][
            "terminal_support_key_digest"
        ]
        support = first_support_string_object(artifact)
        second_support = artifact_object_by_id(artifact, support["payload"]["terminal_support_refs"][1])
        second_support["payload"]["terminal_support_key_digest"] = identities[0][
            "terminal_support_key_digest"
        ]

        duplicate_key = verify_terminal_identity_relation(artifact)

        self.assertFalse(duplicate_key.accepted)
        self.assertIn("terminal_projection_duplicate_key_digest", duplicate_key.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        support = first_terminal_support_object(artifact)
        support["payload"]["parent_weight"] = 0
        terminal = first_terminal_projection_object(artifact)
        terminal["payload"]["terminal_support_identities"][0]["parent_weight"] = 0

        parent_weight = verify_terminal_identity_relation(artifact)

        self.assertFalse(parent_weight.accepted)
        self.assertIn(
            "terminal_support_parent_weight_nonpositive",
            parent_weight.reason,
        )

    def test_terminal_support_ref_and_bucket_mutations_are_rejected(self) -> None:
        artifact = rdkit_support_artifact_fixture("CCO").artifact
        support = first_support_string_object(artifact)
        support["payload"]["terminal_support_refs"] = []

        missing_ref = verify_terminal_identity_relation(artifact)

        self.assertFalse(missing_ref.accepted)
        self.assertIn("terminal_support_refs_missing", missing_ref.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        support = first_support_string_object(artifact)
        support["payload"]["terminal_support_refs"] = [
            support["payload"]["terminal_projection_ref"]
        ]

        wrong_ref_kind = verify_terminal_identity_relation(artifact)

        self.assertFalse(wrong_ref_kind.accepted)
        self.assertIn("terminal_support_ref_kind_mismatch", wrong_ref_kind.reason)

        artifact = rdkit_support_artifact_fixture("CCO").artifact
        terminal_support = first_terminal_support_object(artifact)
        terminal_support["payload"]["digest"] = "0" * 64

        stale_ref = verify_terminal_identity_relation(artifact)

        self.assertFalse(stale_ref.accepted)
        self.assertIn("terminal_support_not_in_projection", stale_ref.reason)

        artifact = completed_prefix_support_artifact_fixture().artifact
        coverage = coverage_object(artifact)
        coverage["payload"]["terminal_bucket"]["terminal_projection"] = {}

        wrong_bucket_projection = verify_terminal_identity_relation(artifact)

        self.assertFalse(wrong_bucket_projection.accepted)
        self.assertIn("terminal_bucket_projection_mismatch", wrong_bucket_projection.reason)

        artifact = completed_prefix_support_artifact_fixture().artifact
        support = first_support_string_object(artifact)
        support["payload"]["terminal_support_refs"] = support["payload"][
            "terminal_support_refs"
        ][:-1]

        wrong_bucket_support = verify_terminal_identity_relation(artifact)

        self.assertFalse(wrong_bucket_support.accepted)
        self.assertIn("terminal_projection_support_set_mismatch", wrong_bucket_support.reason)
