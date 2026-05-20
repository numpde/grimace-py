from __future__ import annotations

import unittest

from grimace._south_star.support_gates import south_star_support_gate_report
from grimace._south_star.tetrahedral import (
    IMPLICIT_HYDROGEN_LIGAND,
    extract_tetrahedral_center_facts,
    preserving_tetrahedral_token,
    tetrahedral_token_preserves_orientation,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles


class SouthStarTetrahedralFactTests(unittest.TestCase):
    def test_extracts_implicit_hydrogen_center_fact(self) -> None:
        facts = extract_tetrahedral_center_facts(parse_smiles("C[C@H](F)Cl"))

        self.assertEqual(1, len(facts))
        fact = facts[0]
        self.assertEqual(1, fact.center_atom_idx)
        self.assertEqual("CHI_TETRAHEDRAL_CCW", fact.chiral_tag)
        self.assertEqual("@", fact.source_token)
        self.assertEqual((0, 2, 3), fact.explicit_neighbor_atom_indices)
        self.assertEqual(1, fact.implicit_hydrogen_count)
        self.assertEqual(
            ("atom:0", "atom:2", "atom:3", IMPLICIT_HYDROGEN_LIGAND),
            fact.source_ligand_order,
        )

    def test_extracts_quaternary_center_fact(self) -> None:
        facts = extract_tetrahedral_center_facts(parse_smiles("C[C@](F)(Cl)Br"))

        self.assertEqual(1, len(facts))
        fact = facts[0]
        self.assertEqual("@", fact.source_token)
        self.assertEqual((0, 2, 3, 4), fact.explicit_neighbor_atom_indices)
        self.assertEqual(0, fact.implicit_hydrogen_count)
        self.assertEqual(
            ("atom:0", "atom:2", "atom:3", "atom:4"),
            fact.source_ligand_order,
        )

    def test_preserving_token_flips_on_odd_ligand_permutation(self) -> None:
        source_order = ("a", "b", "c", "d")

        cases = (
            (source_order, "@"),
            (("b", "a", "c", "d"), "@@"),
            (("b", "c", "a", "d"), "@"),
            (("d", "c", "b", "a"), "@"),
        )
        for emitted_order, expected_token in cases:
            with self.subTest(emitted_order=emitted_order):
                self.assertEqual(
                    expected_token,
                    preserving_tetrahedral_token(
                        source_token="@",
                        source_ligand_order=source_order,
                        emitted_ligand_order=emitted_order,
                    ),
                )

    def test_candidate_token_preserves_orientation_by_parity(self) -> None:
        source_order = ("a", "b", "c", "d")
        emitted_order = ("b", "a", "c", "d")

        self.assertTrue(
            tetrahedral_token_preserves_orientation(
                candidate_token="@@",
                source_token="@",
                source_ligand_order=source_order,
                emitted_ligand_order=emitted_order,
            )
        )
        self.assertFalse(
            tetrahedral_token_preserves_orientation(
                candidate_token="@",
                source_token="@",
                source_ligand_order=source_order,
                emitted_ligand_order=emitted_order,
            )
        )

    def test_atom_stereo_remains_fail_fast_for_enumeration(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C[C@H](F)Cl"))

        self.assertIn("atom_stereo", report.categories)
        with self.assertRaisesRegex(NotImplementedError, "atom_stereo"):
            report.fail_if_unsupported()


if __name__ == "__main__":
    unittest.main()
