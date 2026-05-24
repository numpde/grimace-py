"""Tests for stable South Star proof-term constructors."""

from __future__ import annotations

import unittest

from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.proof_terms import prefix_key
from grimace._south_star1.proof_terms import render_duplicate_node_id
from grimace._south_star1.proof_terms import skeleton_key
from grimace._south_star1.proof_terms import stereo_solution_key
from grimace._south_star1.proof_terms import witness_node_id
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.slots import allocate_traversal_slots
from grimace._south_star1.stereo_csp import build_stereo_csp
from grimace._south_star1.stereo_csp import solve_stereo_csp
from grimace._south_star1.stereo_csp import stereo_solution_canonical_key
from grimace._south_star1.stereo_witness import enumerate_presentation_prefixes
from grimace._south_star1.support_enumeration import (
    enumerate_traced_certified_stereo_support,
)
from tests.south_star1.helpers import tetrahedral_facts


class ProofTermsTest(unittest.TestCase):
    def test_skeleton_key_is_stable_under_json_roundtrip_relevant_values(self) -> None:
        facts = tetrahedral_facts()
        policy = ordinary_policy_for_facts(facts)
        skeleton = enumerate_traversal_skeletons(
            facts,
            build_graph_index(facts),
            policy,
        )[0]

        key = skeleton_key(skeleton)
        roundtripped = _tuple_from_jsonable(_jsonable(key))

        self.assertEqual(roundtripped, key)

    def test_prefix_key_is_stable(self) -> None:
        facts = tetrahedral_facts()
        policy = ordinary_policy_for_facts(facts)
        skeleton = enumerate_traversal_skeletons(
            facts,
            build_graph_index(facts),
            policy,
        )[0]
        slots = allocate_traversal_slots(facts, skeleton)
        prefix = next(
            enumerate_presentation_prefixes(
                facts=facts,
                slots=slots,
                policy=policy,
            )
        )

        key = prefix_key(prefix)

        self.assertEqual(_tuple_from_jsonable(_jsonable(key)), key)

    def test_stereo_solution_key_matches_csp_canonical_key(self) -> None:
        facts = tetrahedral_facts()
        policy = ordinary_policy_for_facts(facts)
        skeleton = enumerate_traversal_skeletons(
            facts,
            build_graph_index(facts),
            policy,
        )[0]
        slots = allocate_traversal_slots(facts, skeleton)
        prefix = next(
            enumerate_presentation_prefixes(
                facts=facts,
                slots=slots,
                policy=policy,
            )
        )
        csp = build_stereo_csp(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            prefix=prefix,
            policy=policy,
            semantics=OrdinarySmilesSemantics(),
        )
        solution = next(iter(solve_stereo_csp(csp)))

        self.assertEqual(
            stereo_solution_key(solution),
            stereo_solution_canonical_key(solution),
        )

    def test_generator_and_checker_use_same_witness_node_id(self) -> None:
        facts = tetrahedral_facts()
        result = enumerate_traced_certified_stereo_support(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )
        accepted = result.trace.accepted[0]

        self.assertEqual(accepted.node, witness_node_id(accepted.witness_id))

    def test_render_duplicate_node_id_is_disjoint_from_witness_node_id(self) -> None:
        witness_id = "witness:abc"

        self.assertNotEqual(
            render_duplicate_node_id(witness_id),
            witness_node_id(witness_id),
        )


def _jsonable(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _tuple_from_jsonable(value: object) -> tuple[object, ...]:
    if isinstance(value, list):
        return tuple(
            _tuple_from_jsonable(item) if isinstance(item, list) else item
            for item in value
        )
    raise TypeError(value)


if __name__ == "__main__":
    unittest.main()
