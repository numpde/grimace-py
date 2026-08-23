"""Pinned parser evidence for dative text at both ring endpoints."""

from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

from tests.helpers.pinned_rdkit_fixtures import load_pinned_rdkit_fixture_cases
from tests.helpers.pinned_rdkit_fixtures import pinned_rdkit_fixture_root
from tests.helpers.pinned_rdkit_fixtures import required_int
from tests.helpers.pinned_rdkit_fixtures import required_int_tuple
from tests.helpers.pinned_rdkit_fixtures import required_string


class RingDativeParserPlacementTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.cases = load_pinned_rdkit_fixture_cases(
            fixture_root=pinned_rdkit_fixture_root(
                "rdkit_south_star_ring_dative_audit"
            ),
            rdkit_version=rdBase.rdkitVersion,
            fixture_label="South Star ring dative parser audit",
        )

    def test_both_endpoint_dative_placements_retain_direction(self) -> None:
        for case in self.cases:
            with self.subTest(case=case.case_id):
                smiles = required_string(
                    case.raw,
                    field_name="smiles",
                    fixture_path=case.fixture_path,
                    case_id=case.case_id,
                )
                atom_pair = required_int_tuple(
                    case.raw["bond_atoms"],
                    field_name="bond_atoms",
                    fixture_path=case.fixture_path,
                    case_id=case.case_id,
                )
                self.assertEqual(len(atom_pair), 2)
                molecule = Chem.MolFromSmiles(smiles)
                self.assertIsNotNone(molecule)
                bond = molecule.GetBondBetweenAtoms(*atom_pair)
                self.assertIsNotNone(bond)
                self.assertEqual(bond.GetBondType(), Chem.BondType.DATIVE)
                self.assertEqual(
                    bond.GetBeginAtomIdx(),
                    required_int(
                        case.raw,
                        field_name="expected_begin_atom",
                        fixture_path=case.fixture_path,
                        case_id=case.case_id,
                    ),
                )
                self.assertEqual(
                    bond.GetEndAtomIdx(),
                    required_int(
                        case.raw,
                        field_name="expected_end_atom",
                        fixture_path=case.fixture_path,
                        case_id=case.case_id,
                    ),
                )


if __name__ == "__main__":
    unittest.main()
