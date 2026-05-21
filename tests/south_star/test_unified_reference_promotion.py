from __future__ import annotations

import unittest

from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from tests.helpers.south_star_domain_manifest import (
    SOUTH_STAR_SINGLE_ATOM_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
)
from tests.helpers.south_star_exact_support import load_south_star_expanded_support_cases
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_unified_reference import (
    is_single_atom_atom_text_domain,
    single_atom_atom_text_support_from_facts,
)


SINGLE_ATOM_ATOM_TEXT_CASE_IDS = frozenset(
    {
        "radical_atom_text_hydrogen",
        "radical_atom_text_methyl",
        "radical_atom_text_oxygen",
        "charged_atom_text_chloride",
        "charged_atom_text_ammonium",
    }
)


class SouthStarUnifiedReferencePromotionTests(unittest.TestCase):
    def test_single_atom_atom_text_cases_have_facts_derived_singleton_support(
        self,
    ) -> None:
        cases = {
            case.case_id: case
            for case in load_south_star_expanded_support_cases()
            if case.case_id in SINGLE_ATOM_ATOM_TEXT_CASE_IDS
        }
        self.assertEqual(SINGLE_ATOM_ATOM_TEXT_CASE_IDS, frozenset(cases))

        for case in cases.values():
            with self.subTest(case_id=case.case_id):
                facts = SouthStarMoleculeFacts.from_mol(parse_smiles(case.source_smiles))
                self.assertTrue(is_single_atom_atom_text_domain(facts))

                support = single_atom_atom_text_support_from_facts(facts)
                self.assertEqual(case.expected_support, support.support)

                result = mol_to_smiles_enum_s_graph_native(
                    case.source_smiles,
                    case_id=case.case_id,
                )
                self.assertEqual(support.support, result.outputs)
                diagnostics = result.generation_diagnostics
                self.assertIsNotNone(diagnostics)
                if diagnostics is None:
                    continue
                self.assertEqual(1, diagnostics.fragment_count)
                self.assertEqual(1, diagnostics.traversal_skeleton_count)
                self.assertEqual(0, diagnostics.marker_slot_count)
                self.assertEqual(1, diagnostics.local_assignment_count)
                self.assertEqual(1, diagnostics.solved_assignment_count)
                self.assertEqual(1, diagnostics.raw_output_count)
                self.assertEqual(1, diagnostics.output_count)
                self.assertEqual(
                    SOUTH_STAR_SINGLE_ATOM_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
                    case.support_authority,
                )

    def test_single_atom_atom_text_domain_rejects_wider_atom_text_cases(self) -> None:
        cases = {
            case.case_id: case for case in load_south_star_expanded_support_cases()
        }
        for case_id in (
            "explicit_bracket_hydrogen_h2",
            "charged_atom_text_methylammonium",
        ):
            case = cases[case_id]
            with self.subTest(case_id=case_id):
                facts = SouthStarMoleculeFacts.from_mol(parse_smiles(case.source_smiles))
                self.assertFalse(is_single_atom_atom_text_domain(facts))
                with self.assertRaisesRegex(NotImplementedError, "single-atom"):
                    single_atom_atom_text_support_from_facts(facts)
                self.assertNotEqual(
                    SOUTH_STAR_SINGLE_ATOM_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
                    case.support_authority,
                )


if __name__ == "__main__":
    unittest.main()
