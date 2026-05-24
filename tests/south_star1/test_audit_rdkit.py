"""Tests for optional RDKit audit helpers."""

from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._south_star1.audit_rdkit import SOUTH_STAR1_SUPPORTED_V0_AUDIT_CASES
from grimace._south_star1.audit_rdkit import SOUTH_STAR1_UNSUPPORTED_V0_AUDIT_CASES
from grimace._south_star1.audit_rdkit import audit_generated_support_with_rdkit
from grimace._south_star1.audit_rdkit import audit_generated_witnesses_with_rdkit
from grimace._south_star1.audit_rdkit import summarize_rdkit_audit
from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_rdkit
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.support_enumeration import enumerate_stereo_support


class RdkitAuditTest(unittest.TestCase):
    def test_supported_audit_matrix_roundtrips_to_isomorphic_facts(self) -> None:
        for case in SOUTH_STAR1_SUPPORTED_V0_AUDIT_CASES:
            with self.subTest(case=case.name):
                results = audit_generated_support_with_rdkit(
                    Chem.MolFromSmiles(case.smiles),
                    policy_options=case.policy_options,
                    adapter_options=case.adapter_options,
                )
                summary = summarize_rdkit_audit(
                    case_name=case.name,
                    input_smiles=case.smiles,
                    results=results,
                )

                self.assertTrue(results)
                self.assertTrue(summary.ok, summary)
                self.assertEqual(summary.parsed_count, summary.support_count)
                self.assertIsNone(summary.first_failure)
                if case.max_support_size is not None:
                    self.assertLessEqual(summary.support_count, case.max_support_size)

    def test_unsupported_audit_matrix_rejects_with_expected_kind(self) -> None:
        for case in SOUTH_STAR1_UNSUPPORTED_V0_AUDIT_CASES:
            with self.subTest(case=case.name):
                with self.assertRaises(SouthStarError) as raised:
                    ordinary_molecule_facts_from_rdkit(
                        Chem.MolFromSmiles(case.smiles),
                        case.adapter_options,
                    )

                self.assertIs(raised.exception.kind, case.expected_error_kind)

    def test_audit_accepts_explicit_skeleton_slice(self) -> None:
        mol = Chem.MolFromSmiles("[C@H](F)(Cl)Br")
        facts = ordinary_molecule_facts_from_rdkit(mol)
        policy = ordinary_policy_for_facts(facts)
        skeletons = tuple(
            skeleton
            for skeleton in enumerate_traversal_skeletons(
                facts,
                build_graph_index(facts),
                policy,
            )
            if skeleton.roots == (AtomId(0),)
        )

        results = audit_generated_support_with_rdkit(mol, skeletons=skeletons)

        self.assertTrue(results)
        self.assertTrue(all(result.ok for result in results))

    def test_small_connected_support_is_stable_under_rdkit_reparse(self) -> None:
        cases = ("[C@H](F)(Cl)Br", "C(/F)=C(\\Cl)")

        for text in cases:
            with self.subTest(text=text):
                original_support = _support_for_smiles(text)
                self.assertTrue(original_support)

                for generated in original_support:
                    reparsed_support = _support_for_smiles(generated)
                    self.assertEqual(reparsed_support, original_support, generated)

    def test_witness_audit_preserves_witness_context(self) -> None:
        mol = Chem.MolFromSmiles("CCO")
        facts = ordinary_molecule_facts_from_rdkit(mol)
        policy = ordinary_policy_for_facts(facts)
        skeletons = enumerate_traversal_skeletons(
            facts,
            build_graph_index(facts),
            policy,
        )[:1]

        results = audit_generated_witnesses_with_rdkit(mol, skeletons=skeletons)

        self.assertTrue(results)
        self.assertTrue(all(result.ok for result in results))
        self.assertTrue(all(result.witness_id for result in results))
        self.assertTrue(all(result.constraints for result in results))


def _support_for_smiles(text: str) -> frozenset[str]:
    facts = ordinary_molecule_facts_from_rdkit(Chem.MolFromSmiles(text))
    image = enumerate_stereo_support(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
    )
    return frozenset(image.strings)


if __name__ == "__main__":
    unittest.main()
