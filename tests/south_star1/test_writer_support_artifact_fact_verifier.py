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
    validate_writer_bracket_atom_text_against_facts,
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
        self.assertIn("count_envelope", verification.offline_unchecked_object_kinds)
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
        self.assertIn("closure_bond_text_unexpected_marker", verification.reason)

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
                "count_envelope": "structurally_checked",
                "frontier_product": "structurally_checked",
                "replay_path": "partially_offline_checked",
                "text_projection": "partially_offline_checked",
                "terminal_projection": "identity_shape_checked",
                "terminal_support": "structurally_checked",
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
