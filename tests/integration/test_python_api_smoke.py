from __future__ import annotations

import unittest

import grimace
from rdkit import Chem, rdBase
from grimace._reference.prepared_graph import (
    prepare_smiles_graph_from_mol_to_smiles_kwargs,
)
from tests.helpers.kernel import CORE_MODULE
from tests.helpers.mols import parse_smiles


DISCONNECTED_PARITY_CASES_ROOT_ZERO: tuple[str, ...] = (
    "[Na+].[Cl-]",
    "[K+].[Cl-]",
    "[F-].[Na+]",
    "[NH4+].[Cl-]",
    "[O-2].[Zn+2]",
    "CC.O",
    "C#N.O",
    "CC.[Na+]",
    "C=C.[Na+]",
    "[Na+].C#N",
    "[Na+].[O-]C=O",
    "[Cl-].C#N",
    "[NH4+].C#N",
    "C#N.[Na+]",
    "C#N.[NH4+]",
    "c1ccccc1.O",
    "CO.O",
    "CCN.O",
    "CCCl.O",
    "CCBr.O",
    "CC(=O)O.O",
    "CC(=O)[O-].[Na+]",
    "C(=O)([O-])[O-].[Ca+2]",
    "[O-]P(=O)([O-])F.[Na+].[Na+]",
    "[O-]S(=O)(=O)[O-].[Ba+2]",
    "C1CCCCC1.O",
    "CC(C)O.O",
    "CC(C)N.O",
    "CCS.O",
    "CC#N.O",
)


class PythonApiSmokeTests(unittest.TestCase):
    def _sample_rdkit_random_support(
        self,
        mol: Chem.Mol,
        *,
        root_idx: int,
        isomeric_smiles: bool,
        draw_budget: int,
        seed: int = 12345,
    ) -> set[str]:
        rdBase.SeedRandomNumberGenerator(seed)
        return {
            Chem.MolToSmiles(
                Chem.Mol(mol),
                rootedAtAtom=root_idx,
                isomericSmiles=isomeric_smiles,
                canonical=False,
                doRandom=True,
            )
            for _ in range(draw_budget)
        }

    def test_top_level_api_exposes_only_final_runtime_surface(self) -> None:
        self.assertTrue(callable(grimace.MolToSmilesDecoder))
        self.assertTrue(callable(grimace.MolToSmilesEnum))
        self.assertTrue(callable(grimace.MolToSmilesTokenInventory))
        self.assertFalse(hasattr(grimace, "MolToSmilesSupport"))
        self.assertFalse(hasattr(grimace, "ReferencePolicy"))
        self.assertFalse(hasattr(grimace, "enumerate_rooted_connected_nonstereo_smiles_support"))
        self.assertFalse(hasattr(grimace, "enumerate_rooted_connected_stereo_smiles_support"))
        if CORE_MODULE is None:
            with self.assertRaises(ImportError):
                tuple(
                    grimace.MolToSmilesEnum(
                        parse_smiles("CCO"),
                        rootedAtAtom=0,
                        isomericSmiles=False,
                        canonical=False,
                        doRandom=True,
                    )
                )
            return

        from grimace import _runtime

        self.assertEqual(
            _runtime.enumerate_rooted_connected_nonstereo_smiles_support(
                parse_smiles("CCO"),
                0,
            ),
            set(
                grimace.MolToSmilesEnum(
                    parse_smiles("CCO"),
                    rootedAtAtom=0,
                    isomericSmiles=False,
                    canonical=False,
                    doRandom=True,
                )
            ),
        )
        self.assertEqual(
            set(
                grimace.MolToSmilesEnum(
                    parse_smiles("CCO"),
                    rootedAtAtom=0,
                    isomericSmiles=False,
                    canonical=False,
                    doRandom=True,
                )
            ),
            _runtime.mol_to_smiles_support(
                parse_smiles("CCO"),
                rooted_at_atom=0,
                isomeric_smiles=False,
                canonical=False,
                do_random=True,
            ),
        )

    def test_public_api_rejects_unsupported_flag_combination(self) -> None:
        with self.assertRaisesRegex(NotImplementedError, "canonical=False"):
            tuple(
                grimace.MolToSmilesEnum(
                    parse_smiles("CCO"),
                    rootedAtAtom=0,
                )
            )
        with self.assertRaisesRegex(NotImplementedError, "doRandom=True"):
            tuple(
                grimace.MolToSmilesEnum(
                    parse_smiles("CCO"),
                    rootedAtAtom=0,
                    canonical=False,
                )
            )

    def test_public_api_matches_rdkit_on_disconnected_cyanide_salt(self) -> None:
        mol = parse_smiles("[Na+].C#N")

        for isomeric_smiles in (False, True):
            for root_idx in range(mol.GetNumAtoms()):
                with self.subTest(isomeric_smiles=isomeric_smiles, root_idx=root_idx):
                    expected = self._sample_rdkit_random_support(
                        mol,
                        root_idx=root_idx,
                        isomeric_smiles=isomeric_smiles,
                        draw_budget=256,
                    )
                    self.assertEqual(
                        expected,
                        set(
                            grimace.MolToSmilesEnum(
                                mol,
                                rootedAtAtom=root_idx,
                                isomericSmiles=isomeric_smiles,
                                canonical=False,
                                doRandom=True,
                            )
                        ),
                    )

    def test_public_api_matches_rdkit_on_disconnected_case_suite_root_zero(self) -> None:
        self.assertEqual(30, len(DISCONNECTED_PARITY_CASES_ROOT_ZERO))

        for smiles in DISCONNECTED_PARITY_CASES_ROOT_ZERO:
            mol = parse_smiles(smiles)
            self.assertGreater(len(Chem.GetMolFrags(mol)), 1)
            for isomeric_smiles in (False, True):
                with self.subTest(smiles=smiles, isomeric_smiles=isomeric_smiles):
                    expected = self._sample_rdkit_random_support(
                        mol,
                        root_idx=0,
                        isomeric_smiles=isomeric_smiles,
                        draw_budget=256,
                    )
                    self.assertEqual(
                        expected,
                        set(
                            grimace.MolToSmilesEnum(
                                mol,
                                rootedAtAtom=0,
                                isomericSmiles=isomeric_smiles,
                                canonical=False,
                                doRandom=True,
                            )
                        ),
                    )

    def test_internal_runtime_bridge_accepts_reference_prepared_graph(self) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        from grimace import _runtime

        reference_prepared = prepare_smiles_graph_from_mol_to_smiles_kwargs(
            parse_smiles("CCO"),
            surface_kind=_runtime.CONNECTED_NONSTEREO_SURFACE,
            isomeric_smiles=False,
        )
        prepared = _runtime.prepare_smiles_graph(
            reference_prepared,
            flags=_runtime.MolToSmilesFlags(
                isomeric_smiles=False,
                rooted_at_atom=0,
                canonical=False,
                do_random=True,
            ),
        )

        self.assertEqual(reference_prepared.to_dict(), prepared.to_dict())

    def test_top_level_runtime_nonstereo_surface_smoke(self) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        from grimace import _runtime

        mol = parse_smiles("CCO")
        expected = _runtime.enumerate_rooted_connected_nonstereo_smiles_support(
            mol,
            0,
        )

        support_from_enum = set(
            grimace.MolToSmilesEnum(
                mol,
                rootedAtAtom=0,
                isomericSmiles=False,
                canonical=False,
                doRandom=True,
            )
        )
        support = _runtime.mol_to_smiles_support(
            mol,
            rooted_at_atom=0,
            isomeric_smiles=False,
            canonical=False,
            do_random=True,
        )

        self.assertEqual(support_from_enum, support)
        self.assertEqual(expected, support)

    def test_top_level_runtime_stereo_surface_smoke(self) -> None:
        if CORE_MODULE is None:
            raise unittest.SkipTest("private Rust extension is not installed")
        from grimace import _runtime

        mol = parse_smiles("F/C=C\\Cl")
        expected = _runtime.enumerate_rooted_connected_stereo_smiles_support(
            mol,
            0,
        )
        support_from_enum = set(
            grimace.MolToSmilesEnum(
                mol,
                rootedAtAtom=0,
                isomericSmiles=True,
                canonical=False,
                doRandom=True,
            )
        )
        support = _runtime.mol_to_smiles_support(
            mol,
            rooted_at_atom=0,
            isomeric_smiles=True,
            canonical=False,
            do_random=True,
        )

        self.assertEqual(support_from_enum, support)
        self.assertEqual(expected, support)


if __name__ == "__main__":
    unittest.main()
