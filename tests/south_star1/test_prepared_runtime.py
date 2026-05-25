"""Tests for the South Star prepared runtime boundary."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.online_decoder_api import online_decode_token_texts_for_policy
from grimace._south_star1.online_serialization_stream import collect_online_serializations
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.support_enumeration import enumerate_stereo_support
from tests.south_star1.helpers import cyclopropane_facts
from tests.south_star1.helpers import tetrahedral_facts


class PreparedRuntimeTest(unittest.TestCase):
    def test_prepared_mol_reuses_facts_policy_semantics(self) -> None:
        facts = tetrahedral_facts()
        policy = ordinary_policy_for_facts(facts)
        semantics = OrdinarySmilesSemantics()

        prepared = prepare_south_star_mol_from_facts(
            facts,
            writer_surface=SouthStarWriterSurface(),
            policy=policy,
            semantics=semantics,
        )

        self.assertIs(prepared.facts, facts)
        self.assertIs(prepared.policy, policy)
        self.assertIs(prepared.semantics, semantics)

    def test_prepared_writer_surface_flags_are_baked(self) -> None:
        surface = SouthStarWriterSurface(ignore_atom_map_numbers=True)

        prepared = prepare_south_star_mol_from_facts(
            cyclopropane_facts(),
            writer_surface=surface,
        )

        self.assertEqual(prepared.writer_surface, surface)
        self.assertTrue(prepared.writer_surface.ignore_atom_map_numbers)

    def test_conflicting_writer_surface_flags_raise(self) -> None:
        with self.assertRaises(SouthStarError):
            prepare_south_star_mol_from_facts(
                tetrahedral_facts(),
                writer_surface=SouthStarWriterSurface(isomeric_smiles=False),
            )

    def test_runtime_root_options_do_not_change_prepared_identity(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        with self.assertRaises(SouthStarError):
            collect_online_serializations(
                prepared=prepared,
                runtime_options=SouthStarRuntimeOptions(rooted_at_atom=0),
            )

        self.assertEqual(prepared.atom_count, len(prepared.facts.atoms))
        self.assertEqual(prepared.component_count, len(prepared.facts.components))

    def test_prepared_online_stream_matches_facts_level_stream(self) -> None:
        facts = tetrahedral_facts()
        prepared = prepare_south_star_mol_from_facts(
            facts,
            writer_surface=SouthStarWriterSurface(),
        )

        prepared_result = collect_online_serializations(prepared=prepared)
        facts_result = collect_online_serializations(
            facts=facts,
            policy=prepared.policy,
            semantics=prepared.semantics,
        )

        self.assertEqual(set(prepared_result.strings), set(facts_result.strings))
        self.assertEqual(
            prepared_result.witness_completion_count,
            facts_result.witness_completion_count,
        )

    def test_prepared_offline_support_matches_facts_level_support(self) -> None:
        facts = cyclopropane_facts()
        prepared = prepare_south_star_mol_from_facts(
            facts,
            writer_surface=SouthStarWriterSurface(),
        )

        prepared_support = enumerate_stereo_support(
            facts=prepared.facts,
            policy=prepared.policy,
            semantics=prepared.semantics,
        )
        facts_support = enumerate_stereo_support(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )

        self.assertEqual(set(prepared_support.strings), set(facts_support.strings))
        self.assertEqual(prepared_support.distinct_count, facts_support.distinct_count)
        self.assertEqual(prepared_support.witness_count, facts_support.witness_count)

    def test_prepared_templates_are_not_recomputed_per_decoder_query(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )

        with patch(
            "grimace._south_star1.online_search_vm.build_stereo_templates",
            side_effect=AssertionError("query recomputed stereo templates"),
        ), patch(
            "grimace._south_star1.online_stereo_witness.build_stereo_templates",
            side_effect=AssertionError("query recomputed stereo templates"),
        ):
            result = collect_online_serializations(prepared=prepared)

        self.assertEqual(result.support_count, len(result.strings))

    def test_prepared_token_inventory_superset_contains_exact_online_inventory(self) -> None:
        prepared = prepare_south_star_mol_from_facts(
            tetrahedral_facts(),
            writer_surface=SouthStarWriterSurface(),
        )
        exact_inventory = online_decode_token_texts_for_policy(
            facts=prepared.facts,
            policy=prepared.policy,
        )

        self.assertGreater(len(prepared.token_inventory_superset), 0)
        self.assertLessEqual(set(exact_inventory), set(prepared.token_inventory_superset))


if __name__ == "__main__":
    unittest.main()
