from __future__ import annotations

import unittest

from rdkit import Chem

from smiles_next_token.reference import (
    CONNECTED_STEREO_SURFACE,
    DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH,
    ReferencePolicy,
    enumerate_rooted_connected_stereo_smiles_support,
    load_default_molecule_cases,
    molecule_is_connected,
    prepare_smiles_graph,
    validate_rooted_connected_stereo_smiles_support,
)

try:
    from smiles_next_token import _core
except ImportError:  # pragma: no cover - exercised only when the extension is absent
    _core = None


def _load_connected_atom_stereo_cases(*, limit: int, max_smiles_length: int) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for case in load_default_molecule_cases(limit=5000, max_smiles_length=max_smiles_length):
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None or not molecule_is_connected(mol):
            continue
        if any(str(bond.GetStereo()) != "STEREONONE" or str(bond.GetBondDir()) != "NONE" for bond in mol.GetBonds()):
            continue
        if not any(str(atom.GetChiralTag()) != "CHI_UNSPECIFIED" for atom in mol.GetAtoms()):
            continue
        selected.append((case.cid, case.smiles))
        if len(selected) >= limit:
            break
    return selected


def _load_connected_multi_atom_stereo_cases(
    *,
    limit: int,
    max_smiles_length: int,
) -> list[tuple[str, str, int]]:
    selected: list[tuple[str, str, int]] = []
    for case in load_default_molecule_cases(limit=50000, max_smiles_length=max_smiles_length):
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None or not molecule_is_connected(mol):
            continue
        if any(str(bond.GetStereo()) != "STEREONONE" or str(bond.GetBondDir()) != "NONE" for bond in mol.GetBonds()):
            continue
        chiral_count = sum(1 for atom in mol.GetAtoms() if str(atom.GetChiralTag()) != "CHI_UNSPECIFIED")
        if chiral_count < 3:
            continue
        selected.append((case.cid, case.smiles, chiral_count))
        if len(selected) >= limit:
            break
    return selected


def _load_connected_bond_stereo_cases(*, limit: int, max_smiles_length: int) -> list[tuple[str, str]]:
    selected: list[tuple[str, str]] = []
    for case in load_default_molecule_cases(limit=5000, max_smiles_length=max_smiles_length):
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None or not molecule_is_connected(mol):
            continue
        if any(atom.GetIsAromatic() for atom in mol.GetAtoms()):
            continue
        if any(bond.IsInRing() for bond in mol.GetBonds()):
            continue
        if any(str(atom.GetChiralTag()) != "CHI_UNSPECIFIED" for atom in mol.GetAtoms()):
            continue
        stereo_bonds = [bond for bond in mol.GetBonds() if str(bond.GetStereo()) != "STEREONONE"]
        if len(stereo_bonds) != 1:
            continue
        stereo_bond = stereo_bonds[0]
        if stereo_bond.GetBeginAtom().GetAtomicNum() != 6 or stereo_bond.GetEndAtom().GetAtomicNum() != 6:
            continue
        begin_idx = stereo_bond.GetBeginAtomIdx()
        end_idx = stereo_bond.GetEndAtomIdx()
        begin_single_substituents = sum(
            1
            for neighbor in mol.GetAtomWithIdx(begin_idx).GetNeighbors()
            if neighbor.GetIdx() != end_idx
            and mol.GetBondBetweenAtoms(begin_idx, neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE
        )
        end_single_substituents = sum(
            1
            for neighbor in mol.GetAtomWithIdx(end_idx).GetNeighbors()
            if neighbor.GetIdx() != begin_idx
            and mol.GetBondBetweenAtoms(end_idx, neighbor.GetIdx()).GetBondType() == Chem.BondType.SINGLE
        )
        if begin_single_substituents > 1 or end_single_substituents > 1:
            continue
        selected.append((case.cid, case.smiles))
        if len(selected) >= limit:
            break
    return selected


class CoreRootedConnectedStereoTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if _core is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        cls.policy = ReferencePolicy.from_path(DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH)

    def test_kernel_matches_python_reference_on_curated_stereo_cases(self) -> None:
        cases = [
            "F[C@H](Cl)Br",
            "F[C@](Cl)(Br)I",
            "C[C@H](O)[C@@H](F)Cl",
            "F/C=C\\Cl",
            "F/C=C/C",
            "C(/C=C/Cl)Cl",
            "C(=C/Cl)\\Cl",
            "CC/C=C\\CCO",
            "C/C=C/C=O",
            "C/C=C/C=C/C(=O)O",
            "C/C=C(\\C)/C(=O)O",
            "C/C(=N\\\\OC(=O)NC)/SC",
            "F/C(Cl)=C/F",
        ]

        for smiles in cases:
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            assert mol is not None

            prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)
            kernel_prepared = _core.PreparedSmilesGraph(prepared)

            for root_idx in range(prepared.atom_count):
                with self.subTest(smiles=smiles, root_idx=root_idx):
                    python_support = enumerate_rooted_connected_stereo_smiles_support(prepared, root_idx)
                    kernel_support = set(
                        kernel_prepared.enumerate_rooted_connected_stereo_support(root_idx)
                    )
                    self.assertEqual(python_support, kernel_support)
                    self.assertEqual(
                        [],
                        validate_rooted_connected_stereo_smiles_support(
                            prepared,
                            root_idx,
                            None,
                            kernel_support,
                        ),
                    )

    def test_kernel_matches_python_reference_on_atom_stereo_dataset_slice(self) -> None:
        cases = _load_connected_atom_stereo_cases(limit=4, max_smiles_length=16)
        self.assertEqual(4, len(cases))

        for cid, smiles in cases:
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            assert mol is not None

            prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)
            kernel_prepared = _core.PreparedSmilesGraph(prepared)
            for root_idx in range(prepared.atom_count):
                with self.subTest(cid=cid, smiles=smiles, root_idx=root_idx):
                    python_support = enumerate_rooted_connected_stereo_smiles_support(prepared, root_idx)
                    kernel_support = set(
                        kernel_prepared.enumerate_rooted_connected_stereo_support(root_idx)
                    )
                    self.assertEqual(python_support, kernel_support)

    def test_kernel_matches_python_reference_on_multi_center_atom_stereo_slice(self) -> None:
        cases = _load_connected_multi_atom_stereo_cases(limit=2, max_smiles_length=28)
        self.assertEqual(2, len(cases))

        for cid, smiles, chiral_count in cases:
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            assert mol is not None

            prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)
            kernel_prepared = _core.PreparedSmilesGraph(prepared)
            for root_idx in range(prepared.atom_count):
                with self.subTest(cid=cid, smiles=smiles, chiral_count=chiral_count, root_idx=root_idx):
                    python_support = enumerate_rooted_connected_stereo_smiles_support(prepared, root_idx)
                    kernel_support = set(
                        kernel_prepared.enumerate_rooted_connected_stereo_support(root_idx)
                    )
                    self.assertEqual(python_support, kernel_support)

    def test_kernel_matches_python_reference_on_bond_stereo_dataset_slice(self) -> None:
        cases = _load_connected_bond_stereo_cases(limit=3, max_smiles_length=18)
        self.assertEqual(3, len(cases))

        for cid, smiles in cases:
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            assert mol is not None

            prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)
            kernel_prepared = _core.PreparedSmilesGraph(prepared)
            for root_idx in range(prepared.atom_count):
                with self.subTest(cid=cid, smiles=smiles, root_idx=root_idx):
                    python_support = enumerate_rooted_connected_stereo_smiles_support(prepared, root_idx)
                    kernel_support = set(
                        kernel_prepared.enumerate_rooted_connected_stereo_support(root_idx)
                    )
                    self.assertEqual(python_support, kernel_support)

    def test_kernel_outputs_canonicalize_on_connected_stereo_case_set(self) -> None:
        cases: list[tuple[str, str, str]] = []
        cases.extend(
            (cid, smiles, "atom")
            for cid, smiles in _load_connected_atom_stereo_cases(limit=4, max_smiles_length=18)
        )
        cases.extend(
            (cid, smiles, "multi_atom")
            for cid, smiles, _ in _load_connected_multi_atom_stereo_cases(limit=2, max_smiles_length=28)
        )
        cases.extend(
            (cid, smiles, "bond")
            for cid, smiles in _load_connected_bond_stereo_cases(limit=3, max_smiles_length=18)
        )
        self.assertEqual(9, len(cases))

        total_generated = 0
        for cid, smiles, category in cases:
            mol = Chem.MolFromSmiles(smiles)
            self.assertIsNotNone(mol)
            assert mol is not None

            prepared = prepare_smiles_graph(mol, self.policy, surface_kind=CONNECTED_STEREO_SURFACE)
            kernel_prepared = _core.PreparedSmilesGraph(prepared)
            generated: set[str] = set()
            for root_idx in range(prepared.atom_count):
                generated.update(kernel_prepared.enumerate_rooted_connected_stereo_support(root_idx))

            with self.subTest(cid=cid, smiles=smiles, category=category):
                self.assertTrue(generated)
                total_generated += len(generated)
                self.assertEqual(
                    [],
                    validate_rooted_connected_stereo_smiles_support(
                        prepared,
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
                    canonicalized.add(prepared.identity_smiles_for(parsed))
                self.assertEqual({prepared.identity_smiles}, canonicalized)

        self.assertGreaterEqual(total_generated, 40)


if __name__ == "__main__":
    unittest.main()
