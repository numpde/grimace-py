from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RootedRandomCase:
    smiles: str
    rooted_outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ExactWriterCase:
    smiles: str
    expected: str
    isomeric_smiles: bool
    rooted_at_atom: int | None = None
    rdkit_canonical: bool = True
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False


ROOTED_RANDOM_CASES: tuple[RootedRandomCase, ...] = (
    RootedRandomCase(
        smiles="COc1ccnc(CC)c1C",
        rooted_outputs=(
            "COc1ccnc(CC)c1C",
            "O(C)c1ccnc(CC)c1C",
            "c1(OC)ccnc(CC)c1C",
            "c1c(OC)c(C)c(CC)nc1",
            "c1cc(OC)c(C)c(CC)n1",
            "n1ccc(OC)c(C)c1CC",
            "c1(CC)nccc(OC)c1C",
            "C(c1nccc(OC)c1C)C",
            "CCc1nccc(OC)c1C",
            "c1(C)c(OC)ccnc1CC",
            "Cc1c(OC)ccnc1CC",
        ),
    ),
)
DISCONNECTED_ROOT_ZERO_CASES: tuple[str, ...] = (
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


CANONICAL_CHIRALITY_CASES: tuple[ExactWriterCase, ...] = (
    ExactWriterCase("F[C@](Br)(I)Cl", "F[C@](Cl)(Br)I", True),
    ExactWriterCase("CC1C[C@@]1(Cl)F", "CC1C[C@]1(F)Cl", True),
    ExactWriterCase("CC1C[C@]1(Cl)F", "CC1C[C@@]1(F)Cl", True),
    ExactWriterCase("F[C@H](Cl)Br", "F[C@H](Cl)Br", True),
    ExactWriterCase("FN(Cl)Br", "FN(Cl)Br", True),
    ExactWriterCase("F[N@H+](Cl)Br", "F[N@H+](Cl)Br", True),
)


WRITER_FLAG_CASES: tuple[ExactWriterCase, ...] = (
    # Code/GraphMol/Wrap/rough_test.py:test75AllBondsExplicit()
    ExactWriterCase(
        smiles="CCC",
        expected="C-C-C",
        isomeric_smiles=False,
        all_bonds_explicit=True,
    ),
    ExactWriterCase(
        smiles="c1ccccc1",
        expected="c1:c:c:c:c:c:1",
        isomeric_smiles=False,
        all_bonds_explicit=True,
    ),
    # Code/GraphMol/Wrap/rough_test.py:testIgnoreAtomMapNumbers()
    ExactWriterCase(
        smiles="[NH2:1]c1ccccc1",
        expected="Nc1ccccc1",
        isomeric_smiles=False,
        rdkit_canonical=False,
        ignore_atom_map_numbers=True,
    ),
    ExactWriterCase(
        smiles="[NH2:1]c1ccccc1",
        expected="c1ccc([NH2:1])cc1",
        isomeric_smiles=False,
        ignore_atom_map_numbers=False,
    ),
    # Code/GraphMol/Wrap/rough_test.py:testIssue266()
    ExactWriterCase(
        smiles="c1ccccc1",
        expected="C1=CC=CC=C1",
        isomeric_smiles=False,
        kekule_smiles=True,
    ),
    ExactWriterCase(
        smiles="c1ccccc1c1ccccc1",
        expected="C1=CC=C(C2=CC=CC=C2)C=C1",
        isomeric_smiles=False,
        kekule_smiles=True,
    ),
    # Code/GraphMol/JavaWrappers/.../SmilesDetailsTests.java:testRootedAt()
    ExactWriterCase(
        smiles="CN(C)C",
        expected="CN(C)C",
        isomeric_smiles=False,
        rooted_at_atom=None,
    ),
    ExactWriterCase(
        smiles="CN(C)C",
        expected="N(C)(C)C",
        isomeric_smiles=False,
        rooted_at_atom=1,
    ),
    # Code/GraphMol/JavaWrappers/.../SmilesDetailsTests.java:testBug1719046()
    ExactWriterCase(
        smiles="Cl[C@H]1C(Br)CCCC1",
        expected="ClC1CCCCC1Br",
        isomeric_smiles=False,
        rooted_at_atom=None,
    ),
    ExactWriterCase(
        smiles="c1ccccn1",
        expected="c1ccncc1",
        isomeric_smiles=False,
        rooted_at_atom=None,
    ),
    ExactWriterCase(
        smiles="C1=CNC=C1",
        expected="c1cc[nH]c1",
        isomeric_smiles=False,
        rooted_at_atom=None,
    ),
    # Code/GraphMol/JavaWrappers/.../SmilesDetailsTests.java:testBug1842174()
    ExactWriterCase(
        smiles="F/C=N/Cl",
        expected="F/C=N/Cl",
        isomeric_smiles=True,
        rooted_at_atom=None,
    ),
    ExactWriterCase(
        smiles="F/C=N/Cl",
        expected="C(/F)=N\\Cl",
        isomeric_smiles=True,
        rooted_at_atom=1,
    ),
    # Code/GraphMol/SmilesParse/test.cpp:testGithub1219()
    ExactWriterCase(
        smiles="C[C@H](F)Cl",
        expected="[CH3][C@H]([F])[Cl]",
        isomeric_smiles=True,
        rooted_at_atom=None,
        all_hs_explicit=True,
    ),
)


DATASET_REGRESSION_CASES: tuple[ExactWriterCase, ...] = (
    ExactWriterCase(
        smiles="CC\\1=C(C2=C(/C1=C\\C3=CC=C(C=C3)S(=O)C)C=CC(=C2)F)CC(=O)O",
        expected="CC1=C(CC(=O)O)c2c(ccc(F)c2)/C1=C\\c1ccc(S(=O)C)cc1",
        isomeric_smiles=True,
        rooted_at_atom=0,
        rdkit_canonical=False,
    ),
    ExactWriterCase(
        smiles=(
            "C1=CC=C2/C/3=N/C4=C5C(=C([N-]4)/N=C/6\\[N-]/C(=N\\C7=C8C(=C([N-]7)"
            "/N=C(/C2=C1)\\[N-]3)C=CC=C8)/C9=CC=CC=C69)C=CC=C5.[Cu]"
        ),
        expected=(
            "c1ccc2/c3[n-]/c(c2c1)=N\\c1c2c(c([n-]1)/N=c1\\[n-]/c(c4c1cccc4)"
            "=N\\c1c4c(c([n-]1)\\N=3)cccc4)cccc2.[Cu]"
        ),
        isomeric_smiles=True,
        rooted_at_atom=0,
        rdkit_canonical=False,
    ),
    ExactWriterCase(
        smiles="CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C(=O)O)/C)/C",
        expected="CC(/C=C/C=C(/C=C/C1=C(C)CCCC1(C)C)C)=C\\C(=O)O",
        isomeric_smiles=True,
        rooted_at_atom=20,
        rdkit_canonical=False,
    ),
    ExactWriterCase(
        smiles="CC1=C(C(CCC1)(C)C)/C=C/C(=C/C=C/C(=C/C(=O)O)/C)/C",
        expected="CC1=C(/C=C/C(=C/C=C/C(=C/C(=O)O)C)C)C(C)(C)CCC1",
        isomeric_smiles=True,
        rooted_at_atom=None,
        rdkit_canonical=False,
    ),
)
