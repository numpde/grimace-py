from __future__ import annotations

import unittest

from rdkit import Chem

from smiles_next_token._reference import (
    PreparedSmilesGraph as ReferencePreparedSmilesGraph,
    load_default_connected_nonstereo_molecule_cases,
    validate_rooted_connected_stereo_smiles_support,
)
from tests.helpers.cases import (
    load_connected_atom_stereo_cases,
    load_connected_bond_stereo_cases,
    load_connected_multi_atom_stereo_cases,
)
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles


class CoreDatasetContractsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")

    def test_kernel_prepared_graph_roundtrips_dataset_slice(self) -> None:
        from smiles_next_token import _runtime

        cases = load_default_connected_nonstereo_molecule_cases(limit=25, max_smiles_length=20)
        self.assertEqual(25, len(cases))

        for case in cases:
            with self.subTest(cid=case.cid, smiles=case.smiles):
                prepared = _runtime.prepare_smiles_graph(
                    parse_smiles(case.smiles),
                    flags=_runtime.MolToSmilesFlags(
                        isomeric_smiles=False,
                        rooted_at_atom=0,
                        canonical=False,
                        do_random=True,
                    ),
                )
                kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared)
                self.assertEqual(prepared.to_dict(), kernel_prepared.to_dict())

    def test_kernel_stereo_outputs_canonicalize_on_representative_case_set(self) -> None:
        from smiles_next_token import _runtime

        cases: list[tuple[str, str, str]] = []
        cases.extend(
            (cid, smiles, "atom")
            for cid, smiles in load_connected_atom_stereo_cases(limit=1, max_smiles_length=18)
        )
        cases.extend(
            (cid, smiles, "multi_atom")
            for cid, smiles, _ in load_connected_multi_atom_stereo_cases(limit=1, max_smiles_length=28)
        )
        cases.extend(
            (cid, smiles, "bond")
            for cid, smiles in load_connected_bond_stereo_cases(limit=1, max_smiles_length=18)
        )
        self.assertEqual(3, len(cases))

        total_generated = 0
        for cid, smiles, category in cases:
            prepared = _runtime.prepare_smiles_graph(
                parse_smiles(smiles),
                flags=_runtime.MolToSmilesFlags(
                    isomeric_smiles=True,
                    rooted_at_atom=0,
                    canonical=False,
                    do_random=True,
                ),
            )
            reference_prepared = ReferencePreparedSmilesGraph.from_dict(prepared.to_dict())
            kernel_prepared = CORE_MODULE.PreparedSmilesGraph(prepared)
            generated: set[str] = set()
            for root_idx in range(prepared.atom_count):
                generated.update(kernel_prepared.enumerate_rooted_connected_stereo_support(root_idx))

            with self.subTest(cid=cid, smiles=smiles, category=category):
                self.assertTrue(generated)
                total_generated += len(generated)
                self.assertEqual(
                    [],
                    validate_rooted_connected_stereo_smiles_support(
                        reference_prepared,
                        0,
                        None,
                        generated,
                    ),
                )
                canonicalized = set()
                for output_smiles in generated:
                    parsed = Chem.MolFromSmiles(output_smiles)
                    self.assertIsNotNone(parsed, msg=output_smiles)
                    assert parsed is not None
                    canonicalized.add(reference_prepared.identity_smiles_for(parsed))
                self.assertEqual({reference_prepared.identity_smiles}, canonicalized)

        self.assertGreaterEqual(total_generated, 12)


if __name__ == "__main__":
    unittest.main()
