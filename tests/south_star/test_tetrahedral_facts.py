from __future__ import annotations

import unittest

from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.enum_s import mol_to_smiles_enum_s_tree_traversals_for_case
from grimace._south_star.support_gates import south_star_support_gate_report
from grimace._south_star.tetrahedral import (
    IMPLICIT_HYDROGEN_LIGAND,
    RING_TETRAHEDRAL_REQUIRED_FACT_AND_EVENT_FIELDS,
    emitted_tetrahedral_ligand_order_from_observation,
    extract_ring_tetrahedral_interaction_obligations,
    extract_tetrahedral_center_facts,
    preserving_tetrahedral_token,
    tetrahedral_token_preserves_orientation,
    tetrahedral_traversal_observation_from_connected_graph_plan,
)
from tests.helpers.south_star_semantic_oracle import parse_smiles
from tests.helpers.south_star_semantics import SouthStarAnnotationPolicyExpectation
from tests.helpers.south_star_semantics import SouthStarSemanticCase


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

    def test_tetrahedral_atom_stereo_is_inside_current_gate_scope(self) -> None:
        report = south_star_support_gate_report(parse_smiles("C[C@H](F)Cl"))

        self.assertTrue(report.supported, report.unsupported_features)

    def test_ring_tetrahedral_obligation_names_required_event_fields(self) -> None:
        obligations = extract_ring_tetrahedral_interaction_obligations(
            parse_smiles("F[C@H]1CCCC(C)C1")
        )

        self.assertEqual(1, len(obligations))
        obligation = obligations[0]
        self.assertEqual(1, obligation.center_atom_idx)
        self.assertTrue(obligation.center_in_ring)
        self.assertEqual("@@", obligation.source_token)
        self.assertEqual(
            ("atom:0", "atom:2", "atom:7", IMPLICIT_HYDROGEN_LIGAND),
            obligation.source_ligand_order,
        )
        self.assertEqual((2, 7), obligation.ring_ligand_atom_indices)
        self.assertEqual((0,), obligation.acyclic_ligand_atom_indices)
        self.assertEqual(1, obligation.implicit_hydrogen_count)
        self.assertEqual(
            RING_TETRAHEDRAL_REQUIRED_FACT_AND_EVENT_FIELDS,
            obligation.required_fact_and_event_fields,
        )

    def test_ring_adjacent_tetrahedral_obligation_names_ring_ligand(self) -> None:
        obligations = extract_ring_tetrahedral_interaction_obligations(
            parse_smiles("F[C@H](Cl)C1CCCCC1")
        )

        self.assertEqual(1, len(obligations))
        obligation = obligations[0]
        self.assertFalse(obligation.center_in_ring)
        self.assertEqual((3,), obligation.ring_ligand_atom_indices)
        self.assertEqual((0, 2), obligation.acyclic_ligand_atom_indices)
        self.assertEqual(
            RING_TETRAHEDRAL_REQUIRED_FACT_AND_EVENT_FIELDS,
            obligation.required_fact_and_event_fields,
        )

    def test_tetrahedral_traversal_observation_uses_plan_root_closure_ligands(
        self,
    ) -> None:
        traversal = _traversal_by_render("FC1CCCCC1", "C1(F)CCCCC1")
        plan = traversal.connected_graph_plan
        self.assertIsNotNone(plan)
        assert plan is not None

        observation = tetrahedral_traversal_observation_from_connected_graph_plan(
            plan,
            center_atom_idx=1,
            implicit_hydrogen_count=1,
        )

        self.assertIsNone(observation.parent_atom_idx)
        self.assertEqual((0, 2), observation.child_atom_indices)
        self.assertEqual((6,), observation.ring_closure_ligand_atom_indices)
        self.assertEqual(("1",), observation.ring_closure_labels)
        self.assertEqual(
            (
                IMPLICIT_HYDROGEN_LIGAND,
                "atom:6",
                "atom:0",
                "atom:2",
            ),
            emitted_tetrahedral_ligand_order_from_observation(observation),
        )

    def test_tetrahedral_traversal_observation_uses_plan_parent_and_closure(
        self,
    ) -> None:
        traversal = _traversal_by_render("FC1CCCCC1", "FC1CCCCC1")
        plan = traversal.connected_graph_plan
        self.assertIsNotNone(plan)
        assert plan is not None

        observation = tetrahedral_traversal_observation_from_connected_graph_plan(
            plan,
            center_atom_idx=1,
            implicit_hydrogen_count=1,
        )

        self.assertEqual(0, observation.parent_atom_idx)
        self.assertEqual((2,), observation.child_atom_indices)
        self.assertEqual((6,), observation.ring_closure_ligand_atom_indices)
        self.assertEqual(("1",), observation.ring_closure_labels)
        self.assertEqual(
            (
                "atom:0",
                "atom:6",
                "atom:2",
                IMPLICIT_HYDROGEN_LIGAND,
            ),
            emitted_tetrahedral_ligand_order_from_observation(observation),
        )

    def test_ring_tetrahedral_support_still_fails_before_gate_widening(self) -> None:
        with self.assertRaisesRegex(
            NotImplementedError,
            "ring_tetrahedral_interaction",
        ):
            mol_to_smiles_enum_s_graph_native("F[C@H]1CCCC(C)C1")


def _traversal_by_render(source_smiles: str, rendered: str):
    for traversal in mol_to_smiles_enum_s_tree_traversals_for_case(
        SouthStarSemanticCase(
            case_id="fluorocyclohexane",
            semantic_feature="monocycle traversal observation",
            source_smiles=source_smiles,
            eligible_carrier_edges=(),
            maximal_eligible_carrier=SouthStarAnnotationPolicyExpectation(
                required_marker_edge_count=0,
            ),
            rdkit_writer_membership_status="not_checked",
            rdkit_writer_membership_notes=(
                "Synthetic traversal-observation provenance case."
            ),
            positive_semantic_smiles=(),
            negative_semantic_smiles=(),
        )
    ):
        if traversal.render() == rendered:
            return traversal
    raise AssertionError(f"missing traversal rendering {rendered!r}")


if __name__ == "__main__":
    unittest.main()
