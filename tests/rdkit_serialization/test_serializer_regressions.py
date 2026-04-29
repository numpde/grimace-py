from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

from tests.helpers.mols import parse_smiles
from tests.helpers.rdkit_serializer_regressions import (
    load_pinned_serializer_regression_cases,
)
from tests.rdkit_serialization._support import (
    RDKIT_PINNED_SAMPLING_SEEDS,
    assert_grimace_support_and_inventory_equal,
    sample_rdkit_random_support,
)


class SerializerRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.cases = load_pinned_serializer_regression_cases(rdBase.rdkitVersion)
        except FileNotFoundError:
            raise unittest.SkipTest(
                f"no pinned serializer-regression corpus for RDKit {rdBase.rdkitVersion}"
            )

    @staticmethod
    def _parse_molblock(molblock: str) -> Chem.Mol:
        mol = Chem.MolFromMolBlock(molblock)
        if mol is None:
            raise AssertionError("RDKit failed to parse regression mol block")
        return mol

    @classmethod
    def _parse_fixture_molecule(cls, case) -> Chem.Mol:
        if case.molblock is not None:
            return cls._parse_molblock(case.molblock)
        return parse_smiles(case.smiles)

    def test_fixture_backed_serializer_regressions(self) -> None:
        for case in self.cases:
            mol = self._parse_fixture_molecule(case)
            with self.subTest(case_id=case.case_id, source=case.source):
                assert_grimace_support_and_inventory_equal(
                    self,
                    mol=mol,
                    expected_support=set(case.expected),
                    expected_inventory=case.expected_inventory,
                    rooted_at_atom=case.rooted_at_atom,
                    isomeric_smiles=case.isomeric_smiles,
                    kekule_smiles=case.kekule_smiles,
                    all_bonds_explicit=case.all_bonds_explicit,
                    all_hs_explicit=case.all_hs_explicit,
                    ignore_atom_map_numbers=case.ignore_atom_map_numbers,
                )

    def test_fixture_backed_rdkit_sampling_when_declared(self) -> None:
        for case in self.cases:
            if case.rdkit_sample_draw_budget is None:
                continue
            mol = self._parse_fixture_molecule(case)
            expected = set(case.expected)
            with self.subTest(case_id=case.case_id, source=case.source):
                for seed in RDKIT_PINNED_SAMPLING_SEEDS:
                    self.assertEqual(
                        expected,
                        sample_rdkit_random_support(
                            mol,
                            root_idx=case.rooted_at_atom,
                            isomeric_smiles=case.isomeric_smiles,
                            draw_budget=case.rdkit_sample_draw_budget,
                            kekule_smiles=case.kekule_smiles,
                            all_bonds_explicit=case.all_bonds_explicit,
                            all_hs_explicit=case.all_hs_explicit,
                            ignore_atom_map_numbers=case.ignore_atom_map_numbers,
                            seed=seed,
                        ),
                    )


if __name__ == "__main__":
    unittest.main()
