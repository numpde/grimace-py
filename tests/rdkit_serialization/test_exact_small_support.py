from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

from tests.helpers.rdkit_exact_small_support import (
    PinnedExactSmallSupportCase,
    load_pinned_exact_small_support_cases,
)
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


class RdkitExactSmallSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.cases = load_pinned_exact_small_support_cases(rdBase.rdkitVersion)
        except FileNotFoundError:
            raise unittest.SkipTest(
                f"no pinned exact small-support corpus for RDKit {rdBase.rdkitVersion}"
            )

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
                case_id=case.case_id,
                source=case.source,
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
                case_id=case.case_id,
                source=case.source,
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
                case_id=case.case_id,
                source=case.source,
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
                case_id=case.case_id,
                source=case.source,
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
                case_id=case.case_id,
                source=case.source,
            ):
                self.assertEqual(
                    set(case.expected),
                    reachable_outputs_from_decoder(decoder),
                )


if __name__ == "__main__":
    unittest.main()
