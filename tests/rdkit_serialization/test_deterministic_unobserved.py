from __future__ import annotations

import unittest

from rdkit import Chem, rdBase

from tests.helpers.rdkit_writer_membership import (
    load_pinned_deterministic_unobserved_cases,
)
from tests.rdkit_serialization._support import (
    mol_from_pinned_source,
    rdkit_exact_writer_output,
    sample_rdkit_random_support,
)


def _draw_budget_for_diagnostic(mol: Chem.Mol) -> int:
    # This is bounded diagnostic evidence, not proof of non-membership.
    if mol.GetNumAtoms() > 35:
        return 2_000
    return 20_000


class RDKitDeterministicUnobservedTests(unittest.TestCase):
    """Pinned deterministic RDKit outputs not seen in bounded random sampling."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.cases = load_pinned_deterministic_unobserved_cases(rdBase.rdkitVersion)
        except FileNotFoundError:
            raise unittest.SkipTest(
                "no pinned deterministic-unobserved corpus for "
                f"RDKit {rdBase.rdkitVersion}"
            )

    def test_deterministic_outputs_are_not_bounded_random_observations(self) -> None:
        for case in self.cases:
            with self.subTest(
                case_id=case.case_id,
                rooted_at_atom=case.rooted_at_atom,
                isomeric_smiles=case.isomeric_smiles,
            ):
                mol = mol_from_pinned_source(case)
                rdkit_out = rdkit_exact_writer_output(case)
                self.assertEqual(case.expected, rdkit_out)

                sampled = sample_rdkit_random_support(
                    mol,
                    root_idx=case.rooted_at_atom,
                    isomeric_smiles=case.isomeric_smiles,
                    draw_budget=_draw_budget_for_diagnostic(mol),
                    kekule_smiles=case.kekule_smiles,
                    all_bonds_explicit=case.all_bonds_explicit,
                    all_hs_explicit=case.all_hs_explicit,
                    ignore_atom_map_numbers=case.ignore_atom_map_numbers,
                )
                self.assertNotIn(rdkit_out, sampled)


if __name__ == "__main__":
    unittest.main()
