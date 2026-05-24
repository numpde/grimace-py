"""Tests for optional RDKit audit helpers."""

from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._south_star1.audit_rdkit import RdkitAuditCase
from grimace._south_star1.audit_rdkit import audit_generated_support_with_rdkit
from grimace._south_star1.audit_rdkit import audit_generated_witnesses_with_rdkit
from grimace._south_star1.audit_rdkit import summarize_rdkit_audit
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_rdkit
from grimace._south_star1.skeleton import enumerate_traversal_skeletons


SUPPORTED_AUDIT_CASES = (
    RdkitAuditCase(
        name="nonstereo_tree",
        smiles="CCO",
        kind="supported",
        tags=("nonstereo",),
        max_support_size=12,
    ),
    RdkitAuditCase(
        name="simple_directional",
        smiles="C(/F)=C(\\Cl)",
        kind="supported",
        tags=("directional",),
        max_support_size=64,
    ),
    RdkitAuditCase(
        name="ring_tetra",
        smiles="[C@H]1(F)CO1",
        kind="supported",
        tags=("ring", "tetra"),
        max_support_size=168,
    ),
    RdkitAuditCase(
        name="disconnected_stereo",
        smiles="CCO.[C@H](F)(Cl)Br",
        kind="supported",
        tags=("disconnected", "tetra"),
        max_support_size=432,
    ),
    RdkitAuditCase(
        name="mixed_tetra_directional",
        smiles="[C@H](F)(Cl)C(/F)=C(\\Cl)",
        kind="supported",
        tags=("mixed", "tetra", "directional"),
        max_support_size=1024,
    ),
)


class RdkitAuditTest(unittest.TestCase):
    def test_supported_audit_matrix_roundtrips_to_isomorphic_facts(self) -> None:
        for case in SUPPORTED_AUDIT_CASES:
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


if __name__ == "__main__":
    unittest.main()
