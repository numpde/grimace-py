from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class RootedRandomCase:
    smiles: str
    rooted_outputs: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CanonicalCase:
    smiles: str
    expected: str


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


CANONICAL_CHIRALITY_CASES: tuple[CanonicalCase, ...] = (
    CanonicalCase("F[C@](Br)(I)Cl", "F[C@](Cl)(Br)I"),
    CanonicalCase("CC1C[C@@]1(Cl)F", "CC1C[C@]1(F)Cl"),
    CanonicalCase("CC1C[C@]1(Cl)F", "CC1C[C@@]1(F)Cl"),
    CanonicalCase("F[C@H](Cl)Br", "F[C@H](Cl)Br"),
    CanonicalCase("FN(Cl)Br", "FN(Cl)Br"),
    CanonicalCase("F[N@H+](Cl)Br", "F[N@H+](Cl)Br"),
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

