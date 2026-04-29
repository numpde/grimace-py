from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import (
    public_enum_support,
    public_token_inventory,
    supported_public_kwargs,
)
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
