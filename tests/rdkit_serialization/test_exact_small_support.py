from __future__ import annotations

from dataclasses import dataclass
import unittest

from rdkit import Chem, rdBase

from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import (
    make_decoder,
    make_determinized_decoder,
    public_token_inventory,
    reachable_outputs_from_decoder,
    supported_public_kwargs,
)
from tests.rdkit_serialization._support import (
    assert_grimace_support_equals,
    sample_rdkit_random_support,
)


@dataclass(frozen=True, slots=True)
class PinnedExactSmallSupportCase:
    smiles: str
    rooted_at_atom: int
    isomeric_smiles: bool
    expected: tuple[str, ...]
    expected_inventory: tuple[str, ...]
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False


PINNED_EXACT_SMALL_SUPPORT_CASES_BY_RDKIT_VERSION: dict[str, tuple[PinnedExactSmallSupportCase, ...]] = {
    "2026.03.1": (
        PinnedExactSmallSupportCase(
            smiles="CCO",
            rooted_at_atom=1,
            isomeric_smiles=False,
            expected=("C(C)O", "C(O)C"),
            expected_inventory=("(", ")", "C", "O"),
        ),
        PinnedExactSmallSupportCase(
            smiles="c1ccncc1",
            rooted_at_atom=1,
            isomeric_smiles=False,
            expected=("c1cccnc1", "c1cnccc1"),
            expected_inventory=("1", "c", "n"),
        ),
        PinnedExactSmallSupportCase(
            smiles="c1ccncc1",
            rooted_at_atom=0,
            isomeric_smiles=False,
            expected=("C1=CC=NC=C1", "C1C=CN=CC=1"),
            expected_inventory=("1", "=", "C", "N"),
            kekule_smiles=True,
        ),
        PinnedExactSmallSupportCase(
            smiles="CC#N",
            rooted_at_atom=1,
            isomeric_smiles=False,
            expected=("C(#N)C", "C(C)#N"),
            expected_inventory=("#", "(", ")", "C", "N"),
        ),
        PinnedExactSmallSupportCase(
            smiles="CC#N",
            rooted_at_atom=1,
            isomeric_smiles=False,
            expected=("C(#N)-C", "C(-C)#N"),
            expected_inventory=("#", "(", ")", "-", "C", "N"),
            all_bonds_explicit=True,
        ),
        PinnedExactSmallSupportCase(
            smiles="F/C=C\\Cl",
            rooted_at_atom=1,
            isomeric_smiles=True,
            expected=("C(/F)=C/Cl", "C(=C/Cl)/F"),
            expected_inventory=("(", ")", "/", "=", "C", "Cl", "F"),
        ),
        PinnedExactSmallSupportCase(
            smiles="F/C=C\\Cl",
            rooted_at_atom=1,
            isomeric_smiles=False,
            expected=("C(/F)=C/Cl", "C(=C/Cl)/F"),
            expected_inventory=("(", ")", "/", "=", "C", "Cl", "F"),
            all_bonds_explicit=True,
        ),
        PinnedExactSmallSupportCase(
            smiles="F[C@H](Cl)Br",
            rooted_at_atom=1,
            isomeric_smiles=True,
            expected=(
                "[C@@H](Br)(F)Cl",
                "[C@@H](Cl)(Br)F",
                "[C@@H](F)(Cl)Br",
                "[C@H](Br)(Cl)F",
                "[C@H](Cl)(F)Br",
                "[C@H](F)(Br)Cl",
            ),
            expected_inventory=("(", ")", "Br", "Cl", "F", "[C@@H]", "[C@H]"),
        ),
        PinnedExactSmallSupportCase(
            smiles="[NH3][Cu]",
            rooted_at_atom=1,
            isomeric_smiles=False,
            expected=("[Cu]<-[NH3]",),
            expected_inventory=("<-", "[Cu]", "[NH3]"),
        ),
        PinnedExactSmallSupportCase(
            smiles="[CH3:7]C",
            rooted_at_atom=0,
            isomeric_smiles=False,
            expected=("CC",),
            expected_inventory=("C",),
            ignore_atom_map_numbers=True,
        ),
        PinnedExactSmallSupportCase(
            smiles="C",
            rooted_at_atom=0,
            isomeric_smiles=False,
            expected=("[CH4]",),
            expected_inventory=("[CH4]",),
            all_hs_explicit=True,
        ),
    ),
}


class RdkitExactSmallSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        if rdBase.rdkitVersion not in PINNED_EXACT_SMALL_SUPPORT_CASES_BY_RDKIT_VERSION:
            raise unittest.SkipTest(
                f"no pinned exact small-support corpus for RDKit {rdBase.rdkitVersion}"
            )
        cls.cases = PINNED_EXACT_SMALL_SUPPORT_CASES_BY_RDKIT_VERSION[rdBase.rdkitVersion]

    @staticmethod
    def _public_kwargs(case: PinnedExactSmallSupportCase) -> dict[str, object]:
        return supported_public_kwargs(
            rootedAtAtom=case.rooted_at_atom,
            isomericSmiles=case.isomeric_smiles,
            kekuleSmiles=case.kekule_smiles,
            allBondsExplicit=case.all_bonds_explicit,
            allHsExplicit=case.all_hs_explicit,
            ignoreAtomMapNumbers=case.ignore_atom_map_numbers,
        )

    def test_pinned_fixture_matches_repeated_rdkit_sampling(self) -> None:
        # These cases are intentionally tiny rooted supports that saturated
        # exactly in repeated rooted-random RDKit draws for this specific
        # pinned RDKit build.
        for case in self.cases:
            mol = parse_smiles(case.smiles)
            expected = set(case.expected)
            with self.subTest(
                smiles=case.smiles,
                rooted_at_atom=case.rooted_at_atom,
                isomeric_smiles=case.isomeric_smiles,
            ):
                for seed in (12345, 54321):
                    self.assertEqual(
                        expected,
                        sample_rdkit_random_support(
                            mol,
                            root_idx=case.rooted_at_atom,
                            isomeric_smiles=case.isomeric_smiles,
                            draw_budget=10_000,
                            kekule_smiles=case.kekule_smiles,
                            all_bonds_explicit=case.all_bonds_explicit,
                            all_hs_explicit=case.all_hs_explicit,
                            ignore_atom_map_numbers=case.ignore_atom_map_numbers,
                            seed=seed,
                        ),
                    )

    def test_grimace_exact_support_matches_pinned_fixture(self) -> None:
        for case in self.cases:
            mol = Chem.MolFromSmiles(case.smiles)
            self.assertIsNotNone(mol)
            with self.subTest(
                smiles=case.smiles,
                rooted_at_atom=case.rooted_at_atom,
                isomeric_smiles=case.isomeric_smiles,
            ):
                assert_grimace_support_equals(
                    self,
                    mol=mol,
                    expected=set(case.expected),
                    rooted_at_atom=case.rooted_at_atom,
                    isomeric_smiles=case.isomeric_smiles,
                    kekule_smiles=case.kekule_smiles,
                    all_bonds_explicit=case.all_bonds_explicit,
                    all_hs_explicit=case.all_hs_explicit,
                    ignore_atom_map_numbers=case.ignore_atom_map_numbers,
                )

    def test_public_token_inventory_matches_pinned_fixture(self) -> None:
        for case in self.cases:
            mol = parse_smiles(case.smiles)
            with self.subTest(
                smiles=case.smiles,
                rooted_at_atom=case.rooted_at_atom,
                isomeric_smiles=case.isomeric_smiles,
            ):
                self.assertEqual(
                    case.expected_inventory,
                    public_token_inventory(mol, **self._public_kwargs(case)),
                )

    def test_decoder_reachable_outputs_match_pinned_fixture(self) -> None:
        for case in self.cases:
            mol = parse_smiles(case.smiles)
            decoder = make_decoder(mol, **self._public_kwargs(case))
            with self.subTest(
                smiles=case.smiles,
                rooted_at_atom=case.rooted_at_atom,
                isomeric_smiles=case.isomeric_smiles,
            ):
                self.assertEqual(
                    set(case.expected),
                    reachable_outputs_from_decoder(decoder),
                )

    def test_determinized_decoder_reachable_outputs_match_pinned_fixture(self) -> None:
        for case in self.cases:
            mol = parse_smiles(case.smiles)
            decoder = make_determinized_decoder(mol, **self._public_kwargs(case))
            with self.subTest(
                smiles=case.smiles,
                rooted_at_atom=case.rooted_at_atom,
                isomeric_smiles=case.isomeric_smiles,
            ):
                self.assertEqual(
                    set(case.expected),
                    reachable_outputs_from_decoder(decoder),
                )


if __name__ == "__main__":
    unittest.main()
