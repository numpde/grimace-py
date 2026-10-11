from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

from tests.helpers.fixture_paths import checked_in_fixture_path
from tests.helpers.pinned_rdkit_fixtures import PinnedFixtureCase
from tests.helpers.pinned_rdkit_fixtures import load_pinned_rdkit_fixture_cases
from tests.helpers.pinned_rdkit_fixtures import normalized_unique_sorted_strings
from tests.south_star2.test_preparation_semantics import _accepted_strings


PINNED_RDKIT_VERSION = "2026.03.1"
FIXTURE_ROOT = checked_in_fixture_path("rdkit_south_star2_exact_support")
RENUMBERED_CASE_IDS = {
    "ss2_disconnected_cc_o_ethane_component_first",
    "ss2_disconnected_cc_o_oxygen_component_first",
}


def _input_molecule(case: PinnedFixtureCase) -> Chem.Mol:
    raw = case.raw
    mol = Chem.MolFromSmiles(raw["input_smiles"])
    if mol is None:
        raise AssertionError(f"RDKit could not parse fixture input {raw['input_smiles']!r}")
    if case.case_id not in RENUMBERED_CASE_IDS:
        if "atom_order" in raw:
            raise ValueError(
                f"fixture case {case.case_id!r} must not define atom_order"
            )
        return mol

    if "atom_order" not in raw:
        raise ValueError(f"fixture case {case.case_id!r} requires atom_order")
    atom_order = raw["atom_order"]
    if (
        type(atom_order) is not list
        or not all(type(atom) is int for atom in atom_order)
        or len(atom_order) != mol.GetNumAtoms()
        or sorted(atom_order) != list(range(mol.GetNumAtoms()))
    ):
        raise ValueError(
            f"fixture case {case.case_id!r} atom_order must be a complete permutation; "
            f"got {atom_order!r}"
        )
    return Chem.RenumberAtoms(mol, atom_order)


@unittest.skipUnless(
    rdBase.rdkitVersion == PINNED_RDKIT_VERSION,
    f"South Star 2 exact support fixture is pinned to RDKit {PINNED_RDKIT_VERSION}",
)
class SouthStar2RdkitExactSupportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_cases = load_pinned_rdkit_fixture_cases(
            fixture_root=FIXTURE_ROOT,
            rdkit_version=PINNED_RDKIT_VERSION,
            fixture_label="South Star 2 exact support",
        )

    def test_native_support_equals_source_bounded_rdkit_support(self) -> None:
        self.assertEqual(
            {
                "ss2_c_o_linear_ethanol",
                "ss2_c_o_isopropanol",
                "ss2_c_cycle_triangle",
                "ss2_disconnected_carbon_oxygen",
                "ss2_disconnected_cc_o_ethane_component_first",
                "ss2_disconnected_cc_o_oxygen_component_first",
            },
            {case.case_id for case in self.fixture_cases},
        )
        for case in self.fixture_cases:
            with self.subTest(case=case.case_id):
                expected = normalized_unique_sorted_strings(
                    case.raw["expected_support"],
                    field_name="expected_support",
                    fixture_path=case.fixture_path,
                    case_id=case.case_id,
                )
                self.assertEqual(case.raw["upper_bound"], len(expected))
                witness_seeds = case.raw["witness_seeds"]
                self.assertEqual(len(expected), len(witness_seeds))

                for expected_text, seed in zip(expected, witness_seeds, strict=True):
                    with self.subTest(seed=seed, expected=expected_text):
                        self.assertIs(type(seed), int)
                        self.assertGreater(seed, 0)
                        mol = _input_molecule(case)
                        self.assertEqual(
                            [expected_text],
                            Chem.MolToRandomSmilesVect(
                                mol,
                                1,
                                seed,
                                True,
                                False,
                                False,
                                False,
                            ),
                        )

                self.assertEqual(set(expected), _accepted_strings(_input_molecule(case)))


if __name__ == "__main__":
    unittest.main()
