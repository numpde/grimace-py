"""Tests for shared stereo compatibility under atom/bond mappings."""

from __future__ import annotations

import ast
from dataclasses import replace
from pathlib import Path
import unittest

from grimace._south_star1.facts import StereoFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.stereo_mapping import map_ligand_occurrence
from grimace._south_star1.stereo_mapping import (
    specified_stereo_compatible_under_mapping,
)

from tests.south_star1.helpers import tetrahedral_facts


class StereoMappingTest(unittest.TestCase):
    def test_maps_ligand_occurrence_by_semantic_identity_not_ordinal(self) -> None:
        facts = tetrahedral_facts()
        left = replace(facts.ligand_occurrences[0], id=OccurrenceId(99), ordinal=17)
        right = replace(facts.ligand_occurrences[0], id=OccurrenceId(12), ordinal=3)

        mapped = map_ligand_occurrence(
            left,
            atom_map=_identity_atom_map(facts),
            bond_map=_identity_bond_map(facts),
            target_occurrences=(right,),
        )

        self.assertEqual(mapped, right)

    def test_tetra_reference_order_parity_uses_shared_relation(self) -> None:
        facts = tetrahedral_facts()
        site = facts.stereo.tetrahedral[0]
        changed_site = replace(
            site,
            target=TetraValue.MINUS,
            reference_order=(
                site.reference_order[1],
                site.reference_order[0],
                site.reference_order[2],
                site.reference_order[3],
            ),
        )
        changed = replace(facts, stereo=StereoFacts(tetrahedral=(changed_site,)))

        self.assertTrue(
            specified_stereo_compatible_under_mapping(
                facts,
                changed,
                atom_map=_identity_atom_map(facts),
                bond_map=_identity_bond_map(facts),
            )
        )

    def test_module_has_no_rdkit_import(self) -> None:
        path = Path("python/grimace/_south_star1/stereo_mapping.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))

        self.assertFalse(_imports_rdkit(tree))


def _identity_atom_map(facts):
    return {atom.id: atom.id for atom in facts.atoms}


def _identity_bond_map(facts):
    return {bond.id: bond.id for bond in facts.bonds}


def _imports_rdkit(tree: ast.AST) -> bool:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            if any(alias.name.startswith("rdkit") for alias in node.names):
                return True
        if isinstance(node, ast.ImportFrom):
            if node.module is not None and node.module.startswith("rdkit"):
                return True
    return False


if __name__ == "__main__":
    unittest.main()
