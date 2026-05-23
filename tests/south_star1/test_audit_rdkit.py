"""Tests for optional RDKit audit helpers."""

from __future__ import annotations

import unittest

from rdkit import Chem

from grimace._south_star1.audit_rdkit import audit_generated_support_with_rdkit
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_rdkit
from grimace._south_star1.skeleton import enumerate_traversal_skeletons


class RdkitAuditTest(unittest.TestCase):
    def test_audit_reports_nonstereo_support_as_isomorphic(self) -> None:
        results = audit_generated_support_with_rdkit(Chem.MolFromSmiles("CCO"))

        self.assertTrue(results)
        self.assertTrue(all(result.ok for result in results))

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


if __name__ == "__main__":
    unittest.main()
