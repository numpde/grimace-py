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
        source = "RDKit Code/GraphMol/Wrap/rough_test.py:testIgnoreAtomMapNumbers"
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
            with self.subTest(source=source, rooted_at_atom=rooted_at_atom):
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
        source = (
            "RDKit Code/GraphMol/SmilesParse/catch_tests.cpp:"
            "Github 8328 MolToSmiles with rootedAtAtom for multiple fragments"
        )
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
            with self.subTest(source=source, rooted_at_atom=rooted_at_atom):
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
        source = (
            "RDKit Code/GraphMol/SmilesParse/catch_tests.cpp:"
            " atoms bound to metals should always have Hs specified"
        )
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
            with self.subTest(source=source, smiles=smiles, rooted_at_atom=rooted_at_atom):
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
                "RDKit Code/GraphMol/Wrap/rough_test.py:test75AllBondsExplicit",
                "c1ccccc1",
                0,
                {"c1:c:c:c:c:c:1"},
                ("1", ":", "c"),
            ),
            (
                "RDKit Code/GraphMol/SmilesParse/catch_tests.cpp:"
                " aromatic double bonds allBondsExplicit sections",
                "c1ccncc1",
                1,
                {"c1:c:c:c:n:c:1", "c1:c:n:c:c:c:1"},
                ("1", ":", "c", "n"),
            ),
        )
        for source, smiles, rooted_at_atom, expected_support, expected_inventory in cases:
            mol = parse_smiles(smiles)
            with self.subTest(source=source, smiles=smiles, rooted_at_atom=rooted_at_atom):
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
                "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
                "SmilesDetailsTests.java:1121-1126",
                "C1=CNC=C1",
                0,
                {"c1c[nH]cc1", "c1cc[nH]c1"},
                ("1", "[nH]", "c"),
            ),
            (
                "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
                "SmilesDetailsTests.java:1121-1126",
                "C1=CNC=C1",
                2,
                {"[nH]1cccc1"},
                ("1", "[nH]", "c"),
            ),
            (
                "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
                "SmilesDetailsTests.java:1127-1132",
                "[CH]1=[CH][NH][CH]=[CH]1",
                0,
                {"c1c[nH]cc1", "c1cc[nH]c1"},
                ("1", "[nH]", "c"),
            ),
            (
                "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
                "SmilesDetailsTests.java:1127-1132",
                "[CH]1=[CH][NH][CH]=[CH]1",
                2,
                {"[nH]1cccc1"},
                ("1", "[nH]", "c"),
            ),
        )
        for source, smiles, rooted_at_atom, expected_support, expected_inventory in cases:
            mol = parse_smiles(smiles)
            with self.subTest(source=source, smiles=smiles, rooted_at_atom=rooted_at_atom):
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
        source = (
            "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
            "SmilesDetailsTests.java:testBug1842174"
        )
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
            with self.subTest(source=source, rooted_at_atom=rooted_at_atom):
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
        source = "RDKit Code/GraphMol/Wrap/rough_test.py:testIgnoreAtomMapNumbers"
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
            with self.subTest(source=source, rooted_at_atom=rooted_at_atom):
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
        source = "RDKit Code/GraphMol/Wrap/rough_test.py:testSmilesWriteParams"
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

    def test_protonated_ring_normalizes_explicit_vs_implicit_hydrogens(self) -> None:
        # Regression pinned to RDKit SmilesDetailsTests.testBug1670149:
        # explicit and implicit protonation forms should collapse onto the same
        # writer surface for a charged ring nitrogen.
        source = (
            "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
            "SmilesDetailsTests.java:testBug1670149"
        )
        expected_support = {
            "C1CCC[NH2+]1",
            "C1CC[NH2+]C1",
            "C1C[NH2+]CC1",
            "C1[NH2+]CCC1",
            "[NH2+]1CCCC1",
        }
        expected_inventory = ("1", "C", "[NH2+]")

        for smiles in ("C1[NH2+]CCC1", "C1CC[NH2+]C1"):
            mol = parse_smiles(smiles)
            with self.subTest(source=source, smiles=smiles):
                self.assertEqual(
                    expected_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=-1,
                            isomericSmiles=False,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=-1,
                            isomericSmiles=False,
                        ),
                    ),
                )

    def test_cyclohexyl_halide_normalizes_nonstereo_surface(self) -> None:
        # Regression pinned to the first two writer cases in RDKit
        # SmilesDetailsTests.testBug1719046: explicit-valence/chiral input
        # variants must normalize to the same non-stereo cyclohexyl halide
        # support surface.
        source = (
            "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
            "SmilesDetailsTests.java:testBug1719046"
        )
        expected_support = {
            "C1(CCCCC1)Cl",
            "C1(Cl)CCCCC1",
            "C1C(CCCC1)Cl",
            "C1C(Cl)CCCC1",
            "C1CC(CCC1)Cl",
            "C1CC(Cl)CCC1",
            "C1CCC(CC1)Cl",
            "C1CCC(Cl)CC1",
            "C1CCCC(C1)Cl",
            "C1CCCC(Cl)C1",
            "C1CCCCC1Cl",
            "ClC1CCCCC1",
        }
        expected_inventory = ("(", ")", "1", "C", "Cl")

        for smiles in ("Cl[CH]1CCCCC1", "Cl[C@H]1CCCCC1"):
            mol = parse_smiles(smiles)
            with self.subTest(source=source, smiles=smiles):
                self.assertEqual(
                    expected_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=-1,
                            isomericSmiles=False,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=-1,
                            isomericSmiles=False,
                        ),
                    ),
                )

    def test_benzene_and_pyridine_aromatic_normalization(self) -> None:
        # Regression pinned to the aromatic normalization cases immediately
        # before RDKit SmilesDetailsTests.testBug1842174.
        cases = (
            (
                "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
                "SmilesDetailsTests.java:1113-1120",
                "[CH]1=[CH][CH]=[CH][CH]=[CH]1",
                {"c1ccccc1"},
                ("1", "c"),
            ),
            (
                "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
                "SmilesDetailsTests.java:1119-1120",
                "c1ccccn1",
                {"c1ccccn1", "c1cccnc1", "c1ccncc1", "c1cnccc1", "c1ncccc1", "n1ccccc1"},
                ("1", "c", "n"),
            ),
        )
        for source, smiles, expected_support, expected_inventory in cases:
            mol = parse_smiles(smiles)
            with self.subTest(source=source, smiles=smiles):
                self.assertEqual(
                    expected_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=-1,
                            isomericSmiles=False,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=-1,
                            isomericSmiles=False,
                        ),
                    ),
                )

    def test_rooted_chirality_surfaces(self) -> None:
        # Regression pinned to RDKit rough_test.test31ChiralitySmiles:
        # rooted writing on tetrahedral stereocenters and small chiral rings
        # must preserve the rooted branch/ring surface family.
        cases = (
            (
                "RDKit Code/GraphMol/Wrap/rough_test.py:test31ChiralitySmiles",
                "F[C@](Br)(I)Cl",
                1,
                {
                    "[C@@](Br)(Cl)(F)I",
                    "[C@@](Br)(F)(I)Cl",
                    "[C@@](Br)(I)(Cl)F",
                    "[C@@](Cl)(Br)(I)F",
                    "[C@@](Cl)(F)(Br)I",
                    "[C@@](Cl)(I)(F)Br",
                    "[C@@](F)(Br)(Cl)I",
                    "[C@@](F)(Cl)(I)Br",
                    "[C@@](F)(I)(Br)Cl",
                    "[C@@](I)(Br)(F)Cl",
                    "[C@@](I)(Cl)(Br)F",
                    "[C@@](I)(F)(Cl)Br",
                    "[C@](Br)(Cl)(I)F",
                    "[C@](Br)(F)(Cl)I",
                    "[C@](Br)(I)(F)Cl",
                    "[C@](Cl)(Br)(F)I",
                    "[C@](Cl)(F)(I)Br",
                    "[C@](Cl)(I)(Br)F",
                    "[C@](F)(Br)(I)Cl",
                    "[C@](F)(Cl)(Br)I",
                    "[C@](F)(I)(Cl)Br",
                    "[C@](I)(Br)(Cl)F",
                    "[C@](I)(Cl)(F)Br",
                    "[C@](I)(F)(Br)Cl",
                },
                ("(", ")", "Br", "Cl", "F", "I", "[C@@]", "[C@]"),
            ),
            (
                "RDKit Code/GraphMol/Wrap/rough_test.py:test31ChiralitySmiles",
                "CC1C[C@@]1(Cl)F",
                1,
                {
                    "C1(C)C[C@@]1(Cl)F",
                    "C1(C)C[C@]1(F)Cl",
                    "C1(C)[C@@](C1)(F)Cl",
                    "C1(C)[C@@](Cl)(C1)F",
                    "C1(C)[C@@](F)(Cl)C1",
                    "C1(C)[C@](C1)(Cl)F",
                    "C1(C)[C@](Cl)(F)C1",
                    "C1(C)[C@](F)(C1)Cl",
                    "C1(C[C@@]1(Cl)F)C",
                    "C1(C[C@]1(F)Cl)C",
                    "C1([C@@](C1)(F)Cl)C",
                    "C1([C@@](Cl)(C1)F)C",
                    "C1([C@@](F)(Cl)C1)C",
                    "C1([C@](C1)(Cl)F)C",
                    "C1([C@](Cl)(F)C1)C",
                    "C1([C@](F)(C1)Cl)C",
                },
                ("(", ")", "1", "C", "Cl", "F", "[C@@]", "[C@]"),
            ),
            (
                "RDKit Code/GraphMol/Wrap/rough_test.py:test31ChiralitySmiles",
                "CC1C[C@]1(Cl)F",
                1,
                {
                    "C1(C)C[C@@]1(F)Cl",
                    "C1(C)C[C@]1(Cl)F",
                    "C1(C)[C@@](C1)(Cl)F",
                    "C1(C)[C@@](Cl)(F)C1",
                    "C1(C)[C@@](F)(C1)Cl",
                    "C1(C)[C@](C1)(F)Cl",
                    "C1(C)[C@](Cl)(C1)F",
                    "C1(C)[C@](F)(Cl)C1",
                    "C1(C[C@@]1(F)Cl)C",
                    "C1(C[C@]1(Cl)F)C",
                    "C1([C@@](C1)(Cl)F)C",
                    "C1([C@@](Cl)(F)C1)C",
                    "C1([C@@](F)(C1)Cl)C",
                    "C1([C@](C1)(F)Cl)C",
                    "C1([C@](Cl)(C1)F)C",
                    "C1([C@](F)(Cl)C1)C",
                },
                ("(", ")", "1", "C", "Cl", "F", "[C@@]", "[C@]"),
            ),
        )
        for source, smiles, rooted_at_atom, expected_support, expected_inventory in cases:
            mol = parse_smiles(smiles)
            with self.subTest(source=source, smiles=smiles, rooted_at_atom=rooted_at_atom):
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

    def test_rooted_small_center_surfaces(self) -> None:
        # Regression pinned to additional small-center cases in RDKit
        # rough_test.test31ChiralitySmiles: a tetrahedral CH center, a neutral
        # amine, and a charged chiral amine should each preserve their rooted
        # writer surface family.
        cases = (
            (
                "RDKit Code/GraphMol/Wrap/rough_test.py:test31ChiralitySmiles",
                "F[C@H](Cl)Br",
                1,
                True,
                {
                    "[C@@H](Br)(F)Cl",
                    "[C@@H](Cl)(Br)F",
                    "[C@@H](F)(Cl)Br",
                    "[C@H](Br)(Cl)F",
                    "[C@H](Cl)(F)Br",
                    "[C@H](F)(Br)Cl",
                },
                ("(", ")", "Br", "Cl", "F", "[C@@H]", "[C@H]"),
            ),
            (
                "RDKit Code/GraphMol/Wrap/rough_test.py:test31ChiralitySmiles",
                "FN(Cl)Br",
                1,
                False,
                {
                    "N(Br)(Cl)F",
                    "N(Br)(F)Cl",
                    "N(Cl)(Br)F",
                    "N(Cl)(F)Br",
                    "N(F)(Br)Cl",
                    "N(F)(Cl)Br",
                },
                ("(", ")", "Br", "Cl", "F", "N"),
            ),
            (
                "RDKit Code/GraphMol/Wrap/rough_test.py:test31ChiralitySmiles",
                "F[N@H+](Cl)Br",
                1,
                True,
                {
                    "[N@@H+](Br)(F)Cl",
                    "[N@@H+](Cl)(Br)F",
                    "[N@@H+](F)(Cl)Br",
                    "[N@H+](Br)(Cl)F",
                    "[N@H+](Cl)(F)Br",
                    "[N@H+](F)(Br)Cl",
                },
                ("(", ")", "Br", "Cl", "F", "[N@@H+]", "[N@H+]"),
            ),
        )
        for (
            source,
            smiles,
            rooted_at_atom,
            isomeric_smiles,
            expected_support,
            expected_inventory,
        ) in cases:
            mol = parse_smiles(smiles)
            with self.subTest(
                source=source,
                smiles=smiles,
                rooted_at_atom=rooted_at_atom,
                isomeric_smiles=isomeric_smiles,
            ):
                self.assertEqual(
                    expected_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=isomeric_smiles,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=rooted_at_atom,
                            isomericSmiles=isomeric_smiles,
                        ),
                    ),
                )

    def test_rooted_small_ring_variant_surfaces(self) -> None:
        # Regression pinned to the remaining rooted ring cases in
        # RDKit rough_test.test31ChiralitySmiles where the input already uses
        # the serializer's preferred F/Cl ordering on the substituted center.
        source = "RDKit Code/GraphMol/Wrap/rough_test.py:test31ChiralitySmiles"
        cases = (
            (
                "CC1C[C@]1(F)Cl",
                {
                    "C1(C)C[C@@]1(Cl)F",
                    "C1(C)C[C@]1(F)Cl",
                    "C1(C)[C@@](C1)(F)Cl",
                    "C1(C)[C@@](Cl)(C1)F",
                    "C1(C)[C@@](F)(Cl)C1",
                    "C1(C)[C@](C1)(Cl)F",
                    "C1(C)[C@](Cl)(F)C1",
                    "C1(C)[C@](F)(C1)Cl",
                    "C1(C[C@@]1(Cl)F)C",
                    "C1(C[C@]1(F)Cl)C",
                    "C1([C@@](C1)(F)Cl)C",
                    "C1([C@@](Cl)(C1)F)C",
                    "C1([C@@](F)(Cl)C1)C",
                    "C1([C@](C1)(Cl)F)C",
                    "C1([C@](Cl)(F)C1)C",
                    "C1([C@](F)(C1)Cl)C",
                },
            ),
            (
                "CC1C[C@@]1(F)Cl",
                {
                    "C1(C)C[C@@]1(F)Cl",
                    "C1(C)C[C@]1(Cl)F",
                    "C1(C)[C@@](C1)(Cl)F",
                    "C1(C)[C@@](Cl)(F)C1",
                    "C1(C)[C@@](F)(C1)Cl",
                    "C1(C)[C@](C1)(F)Cl",
                    "C1(C)[C@](Cl)(C1)F",
                    "C1(C)[C@](F)(Cl)C1",
                    "C1(C[C@@]1(F)Cl)C",
                    "C1(C[C@]1(Cl)F)C",
                    "C1([C@@](C1)(Cl)F)C",
                    "C1([C@@](Cl)(F)C1)C",
                    "C1([C@@](F)(C1)Cl)C",
                    "C1([C@](C1)(F)Cl)C",
                    "C1([C@](Cl)(C1)F)C",
                    "C1([C@](F)(Cl)C1)C",
                },
            ),
        )
        expected_inventory = ("(", ")", "1", "C", "Cl", "F", "[C@@]", "[C@]")
        for smiles, expected_support in cases:
            mol = parse_smiles(smiles)
            with self.subTest(source=source, smiles=smiles, rooted_at_atom=1):
                self.assertEqual(
                    expected_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=1,
                            isomericSmiles=True,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=1,
                            isomericSmiles=True,
                        ),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
