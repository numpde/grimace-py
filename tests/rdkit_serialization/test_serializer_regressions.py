from __future__ import annotations

import unittest

from rdkit import rdBase

from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import (
    public_enum_support,
    public_token_inventory,
    supported_public_kwargs,
)


PINNED_RDKIT_VERSION = "2026.03.1"


class SerializerRegressionTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if rdBase.rdkitVersion != PINNED_RDKIT_VERSION:
            raise unittest.SkipTest(
                f"serializer regressions are pinned to RDKit {PINNED_RDKIT_VERSION}, "
                f"got {rdBase.rdkitVersion}"
            )

    def test_ignore_atom_map_numbers_rooted_surface(self) -> None:
        # Regression for the bug where ignoreAtomMapNumbers=True still leaked
        # atom-map text through the prepared-graph/reference writer surface.
        mol = parse_smiles("[CH3:7]C")

        rooted_expectations = (
            (0, {"[CH3:7]C"}, ("C", "[CH3:7]"), {"CC"}, ("C",)),
            (1, {"C[CH3:7]"}, ("C", "[CH3:7]"), {"CC"}, ("C",)),
        )
        for (
            rooted_at_atom,
            expected_kept_support,
            expected_kept_inventory,
            expected_ignored_support,
            expected_ignored_inventory,
        ) in rooted_expectations:
            with self.subTest(rooted_at_atom=rooted_at_atom):
                self.assertEqual(
                    expected_kept_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=False,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_kept_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=False,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_ignored_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=True,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_ignored_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=True,
                        ),
                    ),
                )

    def test_rooted_atom_on_disconnected_fragment(self) -> None:
        # Regression for RDKit Github 8328: rootedAtAtom on multi-fragment
        # molecules must only root the fragment that contains the atom while
        # leaving other fragments in their own rooted-random support space.
        mol = parse_smiles("[C:1][C:2].[N:3]([C:4])=[O:5]")

        expectations = (
            (
                3,
                {
                    "[C:1][C:2].[C:4][N:3]=[O:5]",
                    "[C:2][C:1].[C:4][N:3]=[O:5]",
                },
                (".", "=", "[C:1]", "[C:2]", "[C:4]", "[N:3]", "[O:5]"),
            ),
            (
                4,
                {
                    "[C:1][C:2].[O:5]=[N:3][C:4]",
                    "[C:2][C:1].[O:5]=[N:3][C:4]",
                },
                (".", "=", "[C:1]", "[C:2]", "[C:4]", "[N:3]", "[O:5]"),
            ),
        )
        for rooted_at_atom, expected_support, expected_inventory in expectations:
            with self.subTest(rooted_at_atom=rooted_at_atom):
                self.assertEqual(
                    expected_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=False,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=False,
                        ),
                    ),
                )

    def test_atoms_bound_to_metals_keep_explicit_hydrogens(self) -> None:
        # Regression pinned to RDKit's serializer rule that atoms bound to
        # metals retain explicit Hs in the emitted atom tokens.
        cases = (
            (
                "Cl[Pt](F)([NH2])[OH]",
                0,
                {
                    "[Cl][Pt]([F])([NH2])[OH]",
                    "[Cl][Pt]([F])([OH])[NH2]",
                    "[Cl][Pt]([NH2])([F])[OH]",
                    "[Cl][Pt]([NH2])([OH])[F]",
                    "[Cl][Pt]([OH])([F])[NH2]",
                    "[Cl][Pt]([OH])([NH2])[F]",
                },
                ("(", ")", "[Cl]", "[F]", "[NH2]", "[OH]", "[Pt]"),
            ),
            (
                "Cl[Pt](F)(<-[NH3])[OH]",
                0,
                {
                    "[Cl][Pt](<-[NH3])([F])[OH]",
                    "[Cl][Pt](<-[NH3])([OH])[F]",
                    "[Cl][Pt]([F])(<-[NH3])[OH]",
                    "[Cl][Pt]([F])([OH])<-[NH3]",
                    "[Cl][Pt]([OH])(<-[NH3])[F]",
                    "[Cl][Pt]([OH])([F])<-[NH3]",
                },
                ("(", ")", "<-", "[Cl]", "[F]", "[NH3]", "[OH]", "[Pt]"),
            ),
        )
        for smiles, rooted_at_atom, expected_support, expected_inventory in cases:
            mol = parse_smiles(smiles)
            with self.subTest(smiles=smiles, rooted_at_atom=rooted_at_atom):
                self.assertEqual(
                    expected_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                        ),
                    ),
                )

    def test_all_bonds_explicit_uses_aromatic_colons(self) -> None:
        # Regression pinned to RDKit's allBondsExplicit behavior on aromatic
        # systems: aromatic bonds emit ":" instead of flipping to single/double.
        cases = (
            (
                "c1ccccc1",
                0,
                {"c1:c:c:c:c:c:1"},
                ("1", ":", "c"),
            ),
            (
                "c1ccncc1",
                1,
                {"c1:c:c:c:n:c:1", "c1:c:n:c:c:c:1"},
                ("1", ":", "c", "n"),
            ),
        )
        for smiles, rooted_at_atom, expected_support, expected_inventory in cases:
            mol = parse_smiles(smiles)
            with self.subTest(smiles=smiles, rooted_at_atom=rooted_at_atom):
                self.assertEqual(
                    expected_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            allBondsExplicit=True,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            allBondsExplicit=True,
                        ),
                    ),
                )

    def test_explicit_kekule_input_normalizes_to_aromatic_nh(self) -> None:
        # Regression pinned to RDKit's aromatic normalization behavior from
        # explicit non-aromatic inputs like C1=CNC=C1 and
        # [CH]1=[CH][NH][CH]=[CH]1.
        cases = (
            (
                "C1=CNC=C1",
                0,
                {"c1c[nH]cc1", "c1cc[nH]c1"},
                ("1", "[nH]", "c"),
            ),
            (
                "C1=CNC=C1",
                2,
                {"[nH]1cccc1"},
                ("1", "[nH]", "c"),
            ),
            (
                "[CH]1=[CH][NH][CH]=[CH]1",
                0,
                {"c1c[nH]cc1", "c1cc[nH]c1"},
                ("1", "[nH]", "c"),
            ),
            (
                "[CH]1=[CH][NH][CH]=[CH]1",
                2,
                {"[nH]1cccc1"},
                ("1", "[nH]", "c"),
            ),
        )
        for smiles, rooted_at_atom, expected_support, expected_inventory in cases:
            mol = parse_smiles(smiles)
            with self.subTest(smiles=smiles, rooted_at_atom=rooted_at_atom):
                self.assertEqual(
                    expected_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                        ),
                    ),
                )

    def test_rooted_imine_stereo_orientation(self) -> None:
        # Regression pinned to RDKit Java SmilesDetailsTests.testBug1842174:
        # rootedAtAtom changes the emitted orientation surface for F/C=N/Cl.
        mol = parse_smiles("F/C=N/Cl")

        expectations = (
            (
                -1,
                {
                    "C(/F)=N\\Cl",
                    "C(=N\\Cl)/F",
                    "Cl/N=C/F",
                    "F/C=N/Cl",
                    "N(/Cl)=C\\F",
                    "N(=C\\F)/Cl",
                },
                ("(", ")", "/", "=", "C", "Cl", "F", "N", "\\"),
            ),
            (
                1,
                {"C(/F)=N\\Cl", "C(=N\\Cl)/F"},
                ("(", ")", "/", "=", "C", "Cl", "F", "N", "\\"),
            ),
        )
        for rooted_at_atom, expected_support, expected_inventory in expectations:
            with self.subTest(rooted_at_atom=rooted_at_atom):
                self.assertEqual(
                    expected_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=True,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=True,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            allBondsExplicit=True,
                        ),
                    ),
                )

    def test_ignore_atom_map_numbers_on_aromatic_attachment(self) -> None:
        # Regression pinned to RDKit rough_test.testIgnoreAtomMapNumbers:
        # map suppression must change the emitted atom token surface cleanly
        # even when the mapped atom is attached to an aromatic ring.
        mol = parse_smiles("[NH2:1]c1ccccc1")

        expectations = (
            (
                0,
                {"[NH2:1]c1ccccc1"},
                ("1", "[NH2:1]", "c"),
                {"Nc1ccccc1"},
                ("1", "N", "c"),
            ),
            (
                -1,
                {
                    "[NH2:1]c1ccccc1",
                    "c1([NH2:1])ccccc1",
                    "c1(ccccc1)[NH2:1]",
                    "c1c([NH2:1])cccc1",
                    "c1c(cccc1)[NH2:1]",
                    "c1cc([NH2:1])ccc1",
                    "c1cc(ccc1)[NH2:1]",
                    "c1ccc([NH2:1])cc1",
                    "c1ccc(cc1)[NH2:1]",
                    "c1cccc([NH2:1])c1",
                    "c1cccc(c1)[NH2:1]",
                    "c1ccccc1[NH2:1]",
                },
                ("(", ")", "1", "[NH2:1]", "c"),
                {
                    "Nc1ccccc1",
                    "c1(N)ccccc1",
                    "c1(ccccc1)N",
                    "c1c(N)cccc1",
                    "c1c(cccc1)N",
                    "c1cc(N)ccc1",
                    "c1cc(ccc1)N",
                    "c1ccc(N)cc1",
                    "c1ccc(cc1)N",
                    "c1cccc(N)c1",
                    "c1cccc(c1)N",
                    "c1ccccc1N",
                },
                ("(", ")", "1", "N", "c"),
            ),
        )
        for (
            rooted_at_atom,
            expected_kept_support,
            expected_kept_inventory,
            expected_ignored_support,
            expected_ignored_inventory,
        ) in expectations:
            with self.subTest(rooted_at_atom=rooted_at_atom):
                self.assertEqual(
                    expected_kept_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=False,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_kept_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=False,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_ignored_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=True,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_ignored_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=False,
                            ignoreAtomMapNumbers=True,
                        ),
                    ),
                )

    def test_rooted_chiral_center_surface(self) -> None:
        # Regression pinned to RDKit rough_test.testSmilesWriteParams:
        # rooting on a chiral center must preserve the rooted branch order and
        # emit the stereo-marked center tokens expected by the writer.
        mol = parse_smiles("C[C@H](F)Cl")

        self.assertEqual(
            {
                "[C@@H](C)(F)Cl",
                "[C@@H](Cl)(C)F",
                "[C@@H](F)(Cl)C",
                "[C@H](C)(Cl)F",
                "[C@H](Cl)(F)C",
                "[C@H](F)(C)Cl",
            },
            public_enum_support(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=1,
                    isomericSmiles=True,
                ),
            ),
        )
        self.assertEqual(
            ("(", ")", "C", "Cl", "F", "[C@@H]", "[C@H]"),
            public_token_inventory(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=1,
                    isomericSmiles=True,
                ),
            ),
        )
        self.assertEqual(
            {
                "C(C)(Cl)F",
                "C(C)(F)Cl",
                "C(Cl)(C)F",
                "C(Cl)(F)C",
                "C(F)(C)Cl",
                "C(F)(Cl)C",
            },
            public_enum_support(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=1,
                    isomericSmiles=False,
                ),
            ),
        )
        self.assertEqual(
            ("(", ")", "C", "Cl", "F"),
            public_token_inventory(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=1,
                    isomericSmiles=False,
                ),
            ),
        )

    def test_all_bonds_explicit_on_aliphatic_branch(self) -> None:
        # Regression pinned to RDKit rough_test.test75AllBondsExplicit:
        # allBondsExplicit should add explicit "-" tokens on simple aliphatic
        # chains without perturbing rooted branch shape.
        mol = parse_smiles("CCC")
        self.assertEqual(
            {"C(-C)-C"},
            public_enum_support(
                mol,
                **supported_public_kwargs(
                    rootedAtAtom=1,
                    isomericSmiles=False,
                    allBondsExplicit=True,
                ),
            ),
        )

    def test_rooted_tertiary_amine_branch_shape(self) -> None:
        # Regression pinned to RDKit rough_test.test40SmilesRootedAtAtom:
        # rooting on the tertiary amine center should produce the rooted
        # branch shape N(C)(C)C, with the explicit-bond variant preserving
        # that same rooted order.
        source = "RDKit Code/GraphMol/Wrap/rough_test.py:test40SmilesRootedAtAtom"
        mol = parse_smiles("CN(C)C")

        with self.subTest(source=source, rooted_at_atom=1, all_bonds_explicit=False):
            self.assertEqual(
                {"N(C)(C)C"},
                public_enum_support(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=1,
                        isomericSmiles=False,
                    ),
                ),
            )
            self.assertEqual(
                ("(", ")", "C", "N"),
                public_token_inventory(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=1,
                        isomericSmiles=False,
                    ),
                ),
            )

        with self.subTest(source=source, rooted_at_atom=1, all_bonds_explicit=True):
            self.assertEqual(
                {"N(-C)(-C)-C"},
                public_enum_support(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=1,
                        isomericSmiles=False,
                        allBondsExplicit=True,
                    ),
                ),
            )
            self.assertEqual(
                ("(", ")", "-", "C", "N"),
                public_token_inventory(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=1,
                        isomericSmiles=False,
                        allBondsExplicit=True,
                    ),
                ),
            )

    def test_rooted_conjugated_diene_stereo_surface(self) -> None:
        # Regression pinned to the second case in RDKit
        # SmilesDetailsTests.testBug1842174. RDKit's single rooted writer
        # surfaces correspond to one rooted slice of our exact support.
        source = (
            "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
            "SmilesDetailsTests.java:testBug1842174"
        )
        mol = parse_smiles(r"C(\C=C\F)=C(/Cl)Br")

        with self.subTest(source=source, rooted_at_atom=3, isomeric_smiles=True):
            self.assertEqual(
                {"F/C=C/C=C(/Cl)Br", "F/C=C/C=C(\\Br)Cl"},
                public_enum_support(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=3,
                        isomericSmiles=True,
                    ),
                ),
            )
            self.assertEqual(
                ("(", ")", "/", "=", "Br", "C", "Cl", "F", "\\"),
                public_token_inventory(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=3,
                        isomericSmiles=True,
                    ),
                ),
            )

        with self.subTest(source=source, rooted_at_atom=3, isomeric_smiles=False):
            self.assertEqual(
                {"FC=CC=C(Br)Cl", "FC=CC=C(Cl)Br"},
                public_enum_support(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=3,
                        isomericSmiles=False,
                    ),
                ),
            )
            self.assertEqual(
                ("(", ")", "=", "Br", "C", "Cl", "F"),
                public_token_inventory(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=3,
                        isomericSmiles=False,
                    ),
                ),
            )


if __name__ == "__main__":
    unittest.main()
