from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

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

    @staticmethod
    def _parse_molblock(molblock: str) -> Chem.Mol:
        mol = Chem.MolFromMolBlock(molblock)
        if mol is None:
            raise AssertionError("RDKit failed to parse regression mol block")
        return mol

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

    def test_kekule_aromatic_nh_surface(self) -> None:
        # Regression pinned to RDKit catch_tests.cpp Github #2788 basics2:
        # doKekule=True on aromatic [nH] systems should emit a stable kekule
        # family instead of leaving the aromatic form untouched.
        source = (
            "RDKit Code/GraphMol/SmilesParse/catch_tests.cpp:"
            " Github #2788 doKekule=true basics2"
        )
        mol = parse_smiles("c1cc[nH]c1")
        expected_support = {
            "C1=CC=CN1",
            "C1=CNC=C1",
            "C1C=CNC=1",
            "C1NC=CC=1",
            "N1C=CC=C1",
        }
        expected_inventory = ("1", "=", "C", "N")

        with self.subTest(source=source):
            self.assertEqual(
                expected_support,
                public_enum_support(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=-1,
                        isomericSmiles=False,
                        kekuleSmiles=True,
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
                        kekuleSmiles=True,
                    ),
                ),
            )

    def test_rooted_small_chiral_surfaces(self) -> None:
        # Regression pinned to RDKit Java wrapper rooted chirality tests:
        # small chiral alcohol and epoxide surfaces should match the exact
        # rooted and all-roots serializer families.
        cases = (
            (
                "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
                "SmilesDetailsTests.java:455-474 chiral alcohol",
                "[C@@H](F)(O)C",
                (
                    (
                        0,
                        {
                            "[C@@H](C)(F)O",
                            "[C@@H](F)(O)C",
                            "[C@@H](O)(C)F",
                            "[C@H](C)(O)F",
                            "[C@H](F)(C)O",
                            "[C@H](O)(F)C",
                        },
                        ("(", ")", "C", "F", "O", "[C@@H]", "[C@H]"),
                    ),
                    (
                        -1,
                        {
                            "C[C@@H](O)F",
                            "C[C@H](F)O",
                            "F[C@@H](C)O",
                            "F[C@H](O)C",
                            "O[C@@H](F)C",
                            "O[C@H](C)F",
                            "[C@@H](C)(F)O",
                            "[C@@H](F)(O)C",
                            "[C@@H](O)(C)F",
                            "[C@H](C)(O)F",
                            "[C@H](F)(C)O",
                            "[C@H](O)(F)C",
                        },
                        ("(", ")", "C", "F", "O", "[C@@H]", "[C@H]"),
                    ),
                ),
            ),
            (
                "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
                "SmilesDetailsTests.java:468-474 chiral epoxide",
                "[C@@H]1(F)OC1",
                (
                    (
                        0,
                        {
                            "[C@@H]1(CO1)F",
                            "[C@@H]1(F)OC1",
                            "[C@H]1(F)CO1",
                            "[C@H]1(OC1)F",
                        },
                        ("(", ")", "1", "C", "F", "O", "[C@@H]", "[C@H]"),
                    ),
                    (
                        -1,
                        {
                            "C1O[C@H]1F",
                            "C1[C@@H](O1)F",
                            "C1[C@H](F)O1",
                            "F[C@@H]1OC1",
                            "F[C@H]1CO1",
                            "O1C[C@@H]1F",
                            "O1[C@@H](F)C1",
                            "O1[C@H](C1)F",
                            "[C@@H]1(CO1)F",
                            "[C@@H]1(F)OC1",
                            "[C@H]1(F)CO1",
                            "[C@H]1(OC1)F",
                        },
                        ("(", ")", "1", "C", "F", "O", "[C@@H]", "[C@H]"),
                    ),
                ),
            ),
        )
        for source, smiles, expectations in cases:
            mol = parse_smiles(smiles)
            for rooted_at_atom, expected_support, expected_inventory in expectations:
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

    def test_small_input_normalization_surfaces(self) -> None:
        # Regression pinned to RDKit Java wrapper chirality normalization
        # checks: alternate explicit-H and stereo-bond spellings should land
        # on the expected exact serializer family.
        cases = (
            (
                "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
                "SmilesDetailsTests.java:483-487 epoxide explicit-H",
                "C1O[C@@]1([H])F",
                {
                    "C1O[C@H]1F",
                    "C1[C@@H](O1)F",
                    "C1[C@H](F)O1",
                    "F[C@@H]1OC1",
                    "F[C@H]1CO1",
                    "O1C[C@@H]1F",
                    "O1[C@@H](F)C1",
                    "O1[C@H](C1)F",
                    "[C@@H]1(CO1)F",
                    "[C@@H]1(F)OC1",
                    "[C@H]1(F)CO1",
                    "[C@H]1(OC1)F",
                },
                ("(", ")", "1", "C", "F", "O", "[C@@H]", "[C@H]"),
            ),
            (
                "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
                "SmilesDetailsTests.java:492-508 cis alkene normalization",
                r"F\C=C/Br",
                {
                    r"Br\C=C/F",
                    r"C(=C\Br)\F",
                    r"C(=C\F)\Br",
                    r"C(\Br)=C\F",
                    r"C(\F)=C\Br",
                    r"F\C=C/Br",
                },
                ("(", ")", "/", "=", "Br", "C", "F", "\\"),
            ),
            (
                "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
                "SmilesDetailsTests.java:510-526 trans alkene normalization",
                r"F\C=C\Br",
                {
                    r"Br\C=C\F",
                    r"C(=C/Br)\F",
                    r"C(=C/F)\Br",
                    r"C(\Br)=C/F",
                    r"C(\F)=C/Br",
                    r"F\C=C\Br",
                },
                ("(", ")", "/", "=", "Br", "C", "F", "\\"),
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
                            isomericSmiles=True,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=-1,
                            isomericSmiles=True,
                        ),
                    ),
                )

    def test_directional_bond_input_normalization(self) -> None:
        # Regression pinned to RDKit SmilesDetailsTests.testIssue159:
        # equivalent directional-bond input spellings should produce the same
        # exact E/Z serializer support.
        source = (
            "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
            "SmilesDetailsTests.java:testIssue159"
        )
        inputs = ("C/C=C/O", r"C(\C)=C/O")
        expected_support = {
            r"C(/C)=C\O",
            r"C(/O)=C\C",
            r"C(=C\C)/O",
            r"C(=C\O)/C",
            "C/C=C/O",
            "O/C=C/C",
        }
        expected_inventory = ("(", ")", "/", "=", "C", "O", "\\")

        for smiles in inputs:
            mol = parse_smiles(smiles)
            with self.subTest(source=source, smiles=smiles):
                self.assertEqual(
                    expected_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=-1,
                            isomericSmiles=True,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=-1,
                            isomericSmiles=True,
                        ),
                    ),
                )

    def test_nondegenerate_chirality_is_suppressed(self) -> None:
        # Regression pinned to RDKit SmilesDetailsTests.testIssue143:
        # chiral annotations on non-stereogenic centers should disappear from
        # the emitted exact support.
        cases = (
            (
                "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
                "SmilesDetailsTests.java:testIssue143 neopentane",
                "C[C@](C)(C)C",
                {"C(C)(C)(C)C", "CC(C)(C)C"},
                ("(", ")", "C"),
            ),
            (
                "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
                "SmilesDetailsTests.java:testIssue143 aldehyde",
                "CC[C@](C)(C)C=O",
                {
                    "C(=O)C(C)(C)CC",
                    "C(=O)C(C)(CC)C",
                    "C(=O)C(CC)(C)C",
                    "C(C(C)(C)C=O)C",
                    "C(C(C)(C)CC)=O",
                    "C(C(C)(C=O)C)C",
                    "C(C(C)(CC)C)=O",
                    "C(C(C=O)(C)C)C",
                    "C(C(CC)(C)C)=O",
                    "C(C)(C)(C=O)CC",
                    "C(C)(C)(CC)C=O",
                    "C(C)(C=O)(C)CC",
                    "C(C)(C=O)(CC)C",
                    "C(C)(CC)(C)C=O",
                    "C(C)(CC)(C=O)C",
                    "C(C)C(C)(C)C=O",
                    "C(C)C(C)(C=O)C",
                    "C(C)C(C=O)(C)C",
                    "C(C=O)(C)(C)CC",
                    "C(C=O)(C)(CC)C",
                    "C(C=O)(CC)(C)C",
                    "C(CC)(C)(C)C=O",
                    "C(CC)(C)(C=O)C",
                    "C(CC)(C=O)(C)C",
                    "CC(C)(C=O)CC",
                    "CC(C)(CC)C=O",
                    "CC(C=O)(C)CC",
                    "CC(C=O)(CC)C",
                    "CC(CC)(C)C=O",
                    "CC(CC)(C=O)C",
                    "CCC(C)(C)C=O",
                    "CCC(C)(C=O)C",
                    "CCC(C=O)(C)C",
                    "O=CC(C)(C)CC",
                    "O=CC(C)(CC)C",
                    "O=CC(CC)(C)C",
                },
                ("(", ")", "=", "C", "O"),
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
                            isomericSmiles=True,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=-1,
                            isomericSmiles=True,
                        ),
                    ),
                )

    def test_small_sulfur_ring_chirality_surface(self) -> None:
        # Regression pinned to RDKit SmilesDetailsTests.testIssue151:
        # a small sulfur-containing ring should preserve the rooted chiral
        # surface through exact all-roots enumeration.
        source = (
            "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
            "SmilesDetailsTests.java:testIssue151 C1S[C@H]1O"
        )
        mol = parse_smiles("C1S[C@H]1O")
        expected_support = {
            "C1S[C@H]1O",
            "C1[C@@H](S1)O",
            "C1[C@H](O)S1",
            "O[C@@H]1SC1",
            "O[C@H]1CS1",
            "S1C[C@@H]1O",
            "S1[C@@H](O)C1",
            "S1[C@H](C1)O",
            "[C@@H]1(CS1)O",
            "[C@@H]1(O)SC1",
            "[C@H]1(O)CS1",
            "[C@H]1(SC1)O",
        }
        expected_inventory = ("(", ")", "1", "C", "O", "S", "[C@@H]", "[C@H]")

        with self.subTest(source=source):
            self.assertEqual(
                expected_support,
                public_enum_support(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=-1,
                        isomericSmiles=True,
                    ),
                ),
            )
            self.assertEqual(
                expected_inventory,
                public_token_inventory(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=-1,
                        isomericSmiles=True,
                    ),
                ),
            )

    def test_fused_ring_chirality_surfaces(self) -> None:
        # Regression pinned to RDKit SmilesDetailsTests.testIssue153:
        # fused-ring ring-label ordering and chiral orientation must remain
        # coupled in the exact serializer support.
        cases = (
            (
                "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
                "SmilesDetailsTests.java:testIssue153 S",
                "C1(O[C@H]12)S2",
                {
                    "C12O[C@H]1S2",
                    "C12O[C@H]2S1",
                    "C12S[C@@H]1O2",
                    "C12S[C@@H]2O1",
                    "C12[C@@H](O1)S2",
                    "C12[C@@H](O2)S1",
                    "C12[C@H](S1)O2",
                    "C12[C@H](S2)O1",
                    "O1C2S[C@@H]21",
                    "O1C2S[C@H]12",
                    "O1C2[C@@H]1S2",
                    "O1[C@@H]2C1S2",
                    "O1[C@H]2SC12",
                    "O1[C@H]2SC21",
                    "S1C2O[C@@H]12",
                    "S1C2O[C@H]21",
                    "S1C2[C@H]1O2",
                    "S1[C@@H]2OC12",
                    "S1[C@@H]2OC21",
                    "S1[C@H]2C1O2",
                    "[C@@H]12C(O2)S1",
                    "[C@@H]12C(S1)O2",
                    "[C@@H]12OC1S2",
                    "[C@@H]12SC2O1",
                    "[C@H]12C(O1)S2",
                    "[C@H]12C(S2)O1",
                    "[C@H]12OC2S1",
                    "[C@H]12SC1O2",
                },
            ),
            (
                "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
                "SmilesDetailsTests.java:testIssue153 R",
                "C1(O[C@H]21)S2",
                {
                    "C12O[C@@H]1S2",
                    "C12O[C@@H]2S1",
                    "C12S[C@H]1O2",
                    "C12S[C@H]2O1",
                    "C12[C@@H](S1)O2",
                    "C12[C@@H](S2)O1",
                    "C12[C@H](O1)S2",
                    "C12[C@H](O2)S1",
                    "O1C2S[C@@H]12",
                    "O1C2S[C@H]21",
                    "O1C2[C@H]1S2",
                    "O1[C@@H]2SC12",
                    "O1[C@@H]2SC21",
                    "O1[C@H]2C1S2",
                    "S1C2O[C@@H]21",
                    "S1C2O[C@H]12",
                    "S1C2[C@@H]1O2",
                    "S1[C@@H]2C1O2",
                    "S1[C@H]2OC12",
                    "S1[C@H]2OC21",
                    "[C@@H]12C(O1)S2",
                    "[C@@H]12C(S2)O1",
                    "[C@@H]12OC2S1",
                    "[C@@H]12SC1O2",
                    "[C@H]12C(O2)S1",
                    "[C@H]12C(S1)O2",
                    "[C@H]12OC1S2",
                    "[C@H]12SC2O1",
                },
            ),
        )
        expected_inventory = ("(", ")", "1", "2", "C", "O", "S", "[C@@H]", "[C@H]")

        for source, smiles, expected_support in cases:
            mol = parse_smiles(smiles)
            with self.subTest(source=source, smiles=smiles):
                self.assertEqual(
                    expected_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=-1,
                            isomericSmiles=True,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=-1,
                            isomericSmiles=True,
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

    def test_explicit_kekule_benzene_normalizes_to_aromatic(self) -> None:
        # Regression pinned to RDKit SmilesDetailsTests.testBug1719046:
        # explicit kekule benzene input should normalize to aromatic output.
        source = (
            "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
            "SmilesDetailsTests.java:testBug1719046 benzene"
        )
        mol = parse_smiles("[CH]1=[CH][CH]=[CH][CH]=[CH]1")
        expected_support = {"c1ccccc1"}
        expected_inventory = ("1", "c")

        with self.subTest(source=source):
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

    def test_conjugated_diene_stereo_normalization(self) -> None:
        # Regression pinned to RDKit SmilesDetailsTests.testBug1842174:
        # conjugated diene stereo input should normalize to the expected
        # serializer support family under exact all-roots enumeration.
        source = (
            "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
            "SmilesDetailsTests.java:testBug1842174 diene"
        )
        mol = parse_smiles(r"C(\C=C\F)=C(/Cl)Br")
        expected_support = {
            r"Br/C(=C\C=C\F)Cl",
            r"Br/C(Cl)=C\C=C\F",
            r"C(/Br)(=C/C=C/F)Cl",
            r"C(/Br)(Cl)=C/C=C/F",
            r"C(/C=C(/Cl)Br)=C\F",
            r"C(/C=C(\Br)Cl)=C\F",
            r"C(/C=C/F)=C(/Br)Cl",
            r"C(/C=C/F)=C(\Cl)Br",
            r"C(/Cl)(=C\C=C\F)Br",
            r"C(/Cl)(Br)=C\C=C\F",
            r"C(/F)=C\C=C(/Br)Cl",
            r"C(/F)=C\C=C(\Cl)Br",
            r"C(=C(/Br)Cl)/C=C/F",
            r"C(=C(\Cl)Br)/C=C/F",
            r"C(=C/C=C/F)(/Br)Cl",
            r"C(=C\C=C(/Br)Cl)/F",
            r"C(=C\C=C(\Cl)Br)/F",
            r"C(=C\C=C\F)(/Cl)Br",
            r"C(=C\F)/C=C(/Cl)Br",
            r"C(=C\F)/C=C(\Br)Cl",
            r"Cl/C(=C/C=C/F)Br",
            r"Cl/C(Br)=C/C=C/F",
            r"F/C=C/C=C(/Cl)Br",
            r"F/C=C/C=C(\Br)Cl",
        }
        expected_inventory = ("(", ")", "/", "=", "Br", "C", "Cl", "F", "\\")

        with self.subTest(source=source):
            self.assertEqual(
                expected_support,
                public_enum_support(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=-1,
                        isomericSmiles=True,
                    ),
                ),
            )
            self.assertEqual(
                expected_inventory,
                public_token_inventory(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=-1,
                        isomericSmiles=True,
                    ),
                ),
            )

    def test_isotope_surface_respects_isomeric_smiles(self) -> None:
        # Regression pinned to RDKit SmilesDetailsTests.testIsotopes:
        # isotope labels must disappear when isomericSmiles=False and reappear
        # when isomericSmiles=True.
        source = (
            "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
            "SmilesDetailsTests.java:testIsotopes"
        )
        mol = parse_smiles("C[13C](C)(C)C")

        with self.subTest(source=source, isomeric_smiles=False):
            self.assertEqual(
                {"CC(C)(C)C", "C(C)(C)(C)C"},
                public_enum_support(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=-1,
                        isomericSmiles=False,
                    ),
                ),
            )
            self.assertEqual(
                ("(", ")", "C"),
                public_token_inventory(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=-1,
                        isomericSmiles=False,
                    ),
                ),
            )

        with self.subTest(source=source, isomeric_smiles=True):
            self.assertEqual(
                {"C[13C](C)(C)C", "[13C](C)(C)(C)C"},
                public_enum_support(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=-1,
                        isomericSmiles=True,
                    ),
                ),
            )
            self.assertEqual(
                ("(", ")", "C", "[13C]"),
                public_token_inventory(
                    mol,
                    **supported_public_kwargs(
                        rootedAtAtom=-1,
                        isomericSmiles=True,
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
        # support surface, including the multisubstituted ring case.
        source = (
            "RDKit Code/JavaWrappers/gmwrapper/src-test/org/RDKit/"
            "SmilesDetailsTests.java:testBug1719046"
        )
        cases = (
            (
                "Cl[CH]1CCCCC1",
                {
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
                },
                ("(", ")", "1", "C", "Cl"),
            ),
            (
                "Cl[C@H]1CCCCC1",
                {
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
                },
                ("(", ")", "1", "C", "Cl"),
            ),
            (
                "Cl[C@H]1C(Br)CCCC1",
                {
                    "BrC1C(CCCC1)Cl",
                    "BrC1C(Cl)CCCC1",
                    "BrC1CCCCC1Cl",
                    "C1(Br)C(CCCC1)Cl",
                    "C1(Br)C(Cl)CCCC1",
                    "C1(Br)CCCCC1Cl",
                    "C1(C(Br)CCCC1)Cl",
                    "C1(C(CCCC1)Br)Cl",
                    "C1(C(CCCC1)Cl)Br",
                    "C1(C(Cl)CCCC1)Br",
                    "C1(CCCCC1Br)Cl",
                    "C1(CCCCC1Cl)Br",
                    "C1(Cl)C(Br)CCCC1",
                    "C1(Cl)C(CCCC1)Br",
                    "C1(Cl)CCCCC1Br",
                    "C1C(Br)C(CCC1)Cl",
                    "C1C(Br)C(Cl)CCC1",
                    "C1C(C(Br)CCC1)Cl",
                    "C1C(C(CCC1)Br)Cl",
                    "C1C(C(CCC1)Cl)Br",
                    "C1C(C(Cl)CCC1)Br",
                    "C1C(Cl)C(Br)CCC1",
                    "C1C(Cl)C(CCC1)Br",
                    "C1CC(Br)C(CC1)Cl",
                    "C1CC(Br)C(Cl)CC1",
                    "C1CC(C(Br)CC1)Cl",
                    "C1CC(C(CC1)Br)Cl",
                    "C1CC(C(CC1)Cl)Br",
                    "C1CC(C(Cl)CC1)Br",
                    "C1CC(Cl)C(Br)CC1",
                    "C1CC(Cl)C(CC1)Br",
                    "C1CCC(Br)C(C1)Cl",
                    "C1CCC(Br)C(Cl)C1",
                    "C1CCC(C(Br)C1)Cl",
                    "C1CCC(C(C1)Br)Cl",
                    "C1CCC(C(C1)Cl)Br",
                    "C1CCC(C(Cl)C1)Br",
                    "C1CCC(Cl)C(Br)C1",
                    "C1CCC(Cl)C(C1)Br",
                    "C1CCCC(Br)C1Cl",
                    "C1CCCC(C1Br)Cl",
                    "C1CCCC(C1Cl)Br",
                    "C1CCCC(Cl)C1Br",
                    "ClC1C(Br)CCCC1",
                    "ClC1C(CCCC1)Br",
                    "ClC1CCCCC1Br",
                },
                ("(", ")", "1", "Br", "C", "Cl"),
            ),
        )

        for smiles, expected_support, expected_inventory in cases:
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

    def test_dative_bond_direction_surfaces(self) -> None:
        # Regression pinned to RDKit catch_tests github #3774:
        # serializer output must preserve dative-bond direction, and rooting
        # flips which side of the arrow is emitted first.
        source = (
            "RDKit Code/GraphMol/SmilesParse/catch_tests.cpp:"
            " github #3774 MolToSmarts inverts direction of dative bond"
        )
        cases = (
            (
                "[NH3][Cu+]",
                -1,
                {"[Cu+]<-[NH3]", "[NH3]->[Cu+]"},
                ("->", "<-", "[Cu+]", "[NH3]"),
            ),
            (
                "[NH3][Cu+]",
                0,
                {"[NH3]->[Cu+]"},
                ("->", "[Cu+]", "[NH3]"),
            ),
            (
                "[NH3][Cu+]",
                1,
                {"[Cu+]<-[NH3]"},
                ("<-", "[Cu+]", "[NH3]"),
            ),
            (
                "[NH2]<-[Cu+]",
                -1,
                {"[Cu+]->[NH2]", "[NH2]<-[Cu+]"},
                ("->", "<-", "[Cu+]", "[NH2]"),
            ),
            (
                "[NH2]<-[Cu+]",
                0,
                {"[NH2]<-[Cu+]"},
                ("<-", "[Cu+]", "[NH2]"),
            ),
            (
                "[NH2]<-[Cu+]",
                1,
                {"[Cu+]->[NH2]"},
                ("->", "[Cu+]", "[NH2]"),
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

    def test_fragment_and_ring_output_order_surfaces(self) -> None:
        # Regression pinned to RDKit catch_tests smilesBondOutputOrder:
        # exact support must preserve fragment-local writer traversal and ring
        # closure ordering for these small non-canonical examples.
        cases = (
            (
                "RDKit Code/GraphMol/SmilesParse/catch_tests.cpp:"
                " smilesBondOutputOrder basics",
                "OCCN.CCO",
                -1,
                {
                    "C(CN)O.C(C)O",
                    "C(CN)O.C(O)C",
                    "C(CN)O.CCO",
                    "C(CN)O.OCC",
                    "C(CO)N.C(C)O",
                    "C(CO)N.C(O)C",
                    "C(CO)N.CCO",
                    "C(CO)N.OCC",
                    "C(N)CO.C(C)O",
                    "C(N)CO.C(O)C",
                    "C(N)CO.CCO",
                    "C(N)CO.OCC",
                    "C(O)CN.C(C)O",
                    "C(O)CN.C(O)C",
                    "C(O)CN.CCO",
                    "C(O)CN.OCC",
                    "NCCO.C(C)O",
                    "NCCO.C(O)C",
                    "NCCO.CCO",
                    "NCCO.OCC",
                    "OCCN.C(C)O",
                    "OCCN.C(O)C",
                    "OCCN.CCO",
                    "OCCN.OCC",
                },
                ("(", ")", ".", "C", "N", "O"),
            ),
            (
                "RDKit Code/GraphMol/SmilesParse/catch_tests.cpp:"
                " smilesBondOutputOrder basics rooted first fragment",
                "OCCN.CCO",
                0,
                {
                    "OCCN.C(C)O",
                    "OCCN.C(O)C",
                    "OCCN.CCO",
                    "OCCN.OCC",
                },
                ("(", ")", ".", "C", "N", "O"),
            ),
            (
                "RDKit Code/GraphMol/SmilesParse/catch_tests.cpp:"
                " smilesBondOutputOrder basics rooted second fragment",
                "OCCN.CCO",
                4,
                {
                    "C(CN)O.CCO",
                    "C(CO)N.CCO",
                    "C(N)CO.CCO",
                    "C(O)CN.CCO",
                    "NCCO.CCO",
                    "OCCN.CCO",
                },
                ("(", ")", ".", "C", "N", "O"),
            ),
            (
                "RDKit Code/GraphMol/SmilesParse/catch_tests.cpp:"
                " github #5585 incorrect ordering for ring closures",
                "OC1CCCN1",
                -1,
                {
                    "C1(CCCN1)O",
                    "C1(NCCC1)O",
                    "C1(O)CCCN1",
                    "C1(O)NCCC1",
                    "C1C(NCC1)O",
                    "C1C(O)NCC1",
                    "C1CC(NC1)O",
                    "C1CC(O)NC1",
                    "C1CCC(N1)O",
                    "C1CCC(O)N1",
                    "C1CCNC1O",
                    "C1CNC(C1)O",
                    "C1CNC(O)C1",
                    "C1NC(CC1)O",
                    "C1NC(O)CC1",
                    "N1C(CCC1)O",
                    "N1C(O)CCC1",
                    "N1CCCC1O",
                    "OC1CCCN1",
                    "OC1NCCC1",
                },
                ("(", ")", "1", "C", "N", "O"),
            ),
            (
                "RDKit Code/GraphMol/SmilesParse/catch_tests.cpp:"
                " github #5585 incorrect ordering for ring closures rooted",
                "OC1CCCN1",
                0,
                {"OC1CCCN1", "OC1NCCC1"},
                ("1", "C", "N", "O"),
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

    def test_molblock_rooted_stereochemistry_surfaces(self) -> None:
        # Regression pinned to RDKit rough_test._test32MolFilesWithChirality:
        # wedge-bond stereoperception from mol blocks must yield the expected
        # rooted writer surface family, not just the same result from SMILES.
        cases = (
            (
                "RDKit Code/GraphMol/Wrap/rough_test.py:_test32MolFilesWithChirality chiral2",
                """chiral2.cdxml
  ChemDraw10160314052D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0553    0.6188    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.0553   -0.2062    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7697   -0.6188    0.0000 I   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6592   -0.6188    0.0000 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -0.7697   -0.2062    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  2  4  1  6
  2  5  1  0
M  END
""",
                {
                    "[C@@](Br)(Cl)(I)F",
                    "[C@@](Br)(F)(Cl)I",
                    "[C@@](Br)(I)(F)Cl",
                    "[C@@](Cl)(Br)(F)I",
                    "[C@@](Cl)(F)(I)Br",
                    "[C@@](Cl)(I)(Br)F",
                    "[C@@](F)(Br)(I)Cl",
                    "[C@@](F)(Cl)(Br)I",
                    "[C@@](F)(I)(Cl)Br",
                    "[C@@](I)(Br)(Cl)F",
                    "[C@@](I)(Cl)(F)Br",
                    "[C@@](I)(F)(Br)Cl",
                    "[C@](Br)(Cl)(F)I",
                    "[C@](Br)(F)(I)Cl",
                    "[C@](Br)(I)(Cl)F",
                    "[C@](Cl)(Br)(I)F",
                    "[C@](Cl)(F)(Br)I",
                    "[C@](Cl)(I)(F)Br",
                    "[C@](F)(Br)(Cl)I",
                    "[C@](F)(Cl)(I)Br",
                    "[C@](F)(I)(Br)Cl",
                    "[C@](I)(Br)(F)Cl",
                    "[C@](I)(Cl)(Br)F",
                    "[C@](I)(F)(Cl)Br",
                },
                ("(", ")", "Br", "Cl", "F", "I", "[C@@]", "[C@]"),
            ),
            (
                "RDKit Code/GraphMol/Wrap/rough_test.py:_test32MolFilesWithChirality case 10-14-3",
                """Case 10-14-3
  ChemDraw10140308512D

  4  3  0  0  0  0  0  0  0  0999 V2000
   -0.8250   -0.4125    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.4125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.8250   -0.4125    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.4125    0.0000 Br  0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  2  4  1  1
M  END
""",
                {
                    "[C@@H](Br)(Cl)F",
                    "[C@@H](Cl)(F)Br",
                    "[C@@H](F)(Br)Cl",
                    "[C@H](Br)(F)Cl",
                    "[C@H](Cl)(Br)F",
                    "[C@H](F)(Cl)Br",
                },
                ("(", ")", "Br", "Cl", "F", "[C@@H]", "[C@H]"),
            ),
            (
                "RDKit Code/GraphMol/Wrap/rough_test.py:_test32MolFilesWithChirality case 10-14-4",
                """Case 10-14-4
  ChemDraw10140308512D

  4  3  0  0  0  0  0  0  0  0999 V2000
   -0.8250   -0.4125    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.0000   -0.4125    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.8250   -0.4125    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
    0.0000    0.4125    0.0000 Br  0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  1
  2  4  1  0
M  END
""",
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
                "RDKit Code/GraphMol/Wrap/rough_test.py:_test32MolFilesWithChirality chiral4 wedge",
                """chiral4.mol
  ChemDraw10160315172D

  6  6  0  0  0  0  0  0  0  0999 V2000
   -0.4422    0.1402    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4422   -0.6848    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.2723   -0.2723    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8547    0.8547    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.6848    0.4422    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.8547   -0.8547    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  1  1  0
  1  4  1  0
  3  5  1  1
  3  6  1  0
M  END
""",
                {
                    "C1C(C)[C@@]1(Cl)F",
                    "C1C(C)[C@]1(F)Cl",
                    "C1C([C@@]1(Cl)F)C",
                    "C1C([C@]1(F)Cl)C",
                    "C1[C@@](C1C)(F)Cl",
                    "C1[C@@](Cl)(C1C)F",
                    "C1[C@@](F)(Cl)C1C",
                    "C1[C@](C1C)(Cl)F",
                    "C1[C@](Cl)(F)C1C",
                    "C1[C@](F)(C1C)Cl",
                },
                ("(", ")", "1", "C", "Cl", "F", "[C@@]", "[C@]"),
            ),
            (
                "RDKit Code/GraphMol/Wrap/rough_test.py:_test32MolFilesWithChirality chiral4 dash",
                """chiral4.mol
  ChemDraw10160315172D

  6  6  0  0  0  0  0  0  0  0999 V2000
   -0.4422    0.1402    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.4422   -0.6848    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.2723   -0.2723    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.8547    0.8547    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.6848    0.4422    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.8547   -0.8547    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  3  1  1  0
  1  4  1  0
  3  5  1  6
  3  6  1  0
M  END
""",
                {
                    "C1C(C)[C@@]1(F)Cl",
                    "C1C(C)[C@]1(Cl)F",
                    "C1C([C@@]1(F)Cl)C",
                    "C1C([C@]1(Cl)F)C",
                    "C1[C@@](C1C)(Cl)F",
                    "C1[C@@](Cl)(F)C1C",
                    "C1[C@@](F)(C1C)Cl",
                    "C1[C@](C1C)(F)Cl",
                    "C1[C@](Cl)(C1C)F",
                    "C1[C@](F)(Cl)C1C",
                },
                ("(", ")", "1", "C", "Cl", "F", "[C@@]", "[C@]"),
            ),
        )
        for source, molblock, expected_support, expected_inventory in cases:
            mol = self._parse_molblock(molblock)
            with self.subTest(source=source, rooted_at_atom=1):
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

    def test_molblock_nitrogen_stereochemistry_surfaces(self) -> None:
        # Regression pinned to RDKit rough_test._test32MolFilesWithChirality:
        # molblock wedge perception should distinguish neutral amine
        # non-stereochemistry from ammonium stereochemistry.
        cases = (
            (
                "RDKit Code/GraphMol/Wrap/rough_test.py:_test32MolFilesWithChirality chiral4 neutral N",
                """chiral4.mol
  ChemDraw10160314362D

  4  3  0  0  0  0  0  0  0  0999 V2000
    0.4125    0.6188    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.4125   -0.2062    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3020   -0.6188    0.0000 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -0.4125   -0.2062    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  1
  2  4  1  0
M  END
""",
                (
                    (
                        1,
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
                        -1,
                        {
                            "BrN(Cl)F",
                            "BrN(F)Cl",
                            "ClN(Br)F",
                            "ClN(F)Br",
                            "FN(Br)Cl",
                            "FN(Cl)Br",
                            "N(Br)(Cl)F",
                            "N(Br)(F)Cl",
                            "N(Cl)(Br)F",
                            "N(Cl)(F)Br",
                            "N(F)(Br)Cl",
                            "N(F)(Cl)Br",
                        },
                        ("(", ")", "Br", "Cl", "F", "N"),
                    ),
                ),
            ),
            (
                "RDKit Code/GraphMol/Wrap/rough_test.py:_test32MolFilesWithChirality chiral5 ammonium",
                """chiral5.mol
  ChemDraw10160314362D

  4  3  0  0  0  0  0  0  0  0999 V2000
    0.4125    0.6188    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.4125   -0.2062    0.0000 N   0  0  0  0  0  0  0  0  0  0  0  0
   -0.3020   -0.6188    0.0000 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -0.4125   -0.2062    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  1
  2  4  1  0
M  CHG  1   2   1
M  END
""",
                (
                    (
                        1,
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
                    (
                        -1,
                        {
                            "Br[N@@H+](Cl)F",
                            "Br[N@H+](F)Cl",
                            "Cl[N@@H+](F)Br",
                            "Cl[N@H+](Br)F",
                            "F[N@@H+](Br)Cl",
                            "F[N@H+](Cl)Br",
                            "[N@@H+](Br)(F)Cl",
                            "[N@@H+](Cl)(Br)F",
                            "[N@@H+](F)(Cl)Br",
                            "[N@H+](Br)(Cl)F",
                            "[N@H+](Cl)(F)Br",
                            "[N@H+](F)(Br)Cl",
                        },
                        ("(", ")", "Br", "Cl", "F", "[N@@H+]", "[N@H+]"),
                    ),
                ),
            ),
        )
        for source, molblock, expectations in cases:
            mol = self._parse_molblock(molblock)
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

    def test_molblock_atom_order_invariance_for_chiral_center(self) -> None:
        # Regression pinned to repeated chiral1 molblock variants in
        # rough_test._test32MolFilesWithChirality: atom-order differences in
        # the mol block should not change the exact stereochemical support.
        source = (
            "RDKit Code/GraphMol/Wrap/rough_test.py:"
            "_test32MolFilesWithChirality chiral1 variants"
        )
        molblocks = (
            """chiral1.mol
  ChemDraw10160313232D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0553    0.6188    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0
    0.0553   -0.2062    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
    0.7697   -0.6188    0.0000 I   0  0  0  0  0  0  0  0  0  0  0  0
   -0.6592   -0.6188    0.0000 Br  0  0  0  0  0  0  0  0  0  0  0  0
   -0.7697   -0.2062    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  2  3  1  0
  2  4  1  1
  2  5  1  0
M  END
""",
            """chiral1.mol
  ChemDraw10160313232D

  5  4  0  0  0  0  0  0  0  0999 V2000
    0.0553   -0.2062    0.0000 C   0  0  0  0  0  0  0  0  0  0  0  0
   -0.7697   -0.2062    0.0000 Cl  0  0  0  0  0  0  0  0  0  0  0  0
   -0.6592   -0.6188    0.0000 Br  0  0  0  0  0  0  0  0  0  0  0  0
    0.7697   -0.6188    0.0000 I   0  0  0  0  0  0  0  0  0  0  0  0
    0.0553    0.6188    0.0000 F   0  0  0  0  0  0  0  0  0  0  0  0
  1  2  1  0
  1  3  1  1
  1  4  1  0
  1  5  1  0
M  END
""",
        )
        expected_support = {
            "Br[C@@](Cl)(F)I",
            "Br[C@@](F)(I)Cl",
            "Br[C@@](I)(Cl)F",
            "Br[C@](Cl)(I)F",
            "Br[C@](F)(Cl)I",
            "Br[C@](I)(F)Cl",
            "Cl[C@@](Br)(I)F",
            "Cl[C@@](F)(Br)I",
            "Cl[C@@](I)(F)Br",
            "Cl[C@](Br)(F)I",
            "Cl[C@](F)(I)Br",
            "Cl[C@](I)(Br)F",
            "F[C@@](Br)(Cl)I",
            "F[C@@](Cl)(I)Br",
            "F[C@@](I)(Br)Cl",
            "F[C@](Br)(I)Cl",
            "F[C@](Cl)(Br)I",
            "F[C@](I)(Cl)Br",
            "I[C@@](Br)(F)Cl",
            "I[C@@](Cl)(Br)F",
            "I[C@@](F)(Cl)Br",
            "I[C@](Br)(Cl)F",
            "I[C@](Cl)(F)Br",
            "I[C@](F)(Br)Cl",
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
        }
        expected_inventory = ("(", ")", "Br", "Cl", "F", "I", "[C@@]", "[C@]")

        for variant_idx, molblock in enumerate(molblocks, start=1):
            mol = self._parse_molblock(molblock)
            with self.subTest(source=source, variant=variant_idx):
                self.assertEqual(
                    expected_support,
                    public_enum_support(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=-1,
                            isomericSmiles=True,
                        ),
                    ),
                )
                self.assertEqual(
                    expected_inventory,
                    public_token_inventory(
                        mol,
                        **supported_public_kwargs(
                            rootedAtAtom=-1,
                            isomericSmiles=True,
                        ),
                    ),
                )


if __name__ == "__main__":
    unittest.main()
