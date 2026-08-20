from __future__ import annotations

import unittest

from tests.south_star1.writer_test_fixtures import (
    chain_facts,
    directional_non_single_ring_carrier_facts,
    directional_ring_carrier_facts,
    duplicate_single_atom_policy,
    shared_directional_ring_carrier_facts,
    terminal_tetra_center_facts,
    terminal_tetra_center_policy,
)


class WriterTestFixturesTest(unittest.TestCase):
    def test_fact_fixtures_are_independent_and_preparable(self) -> None:
        from tests.south_star1.writer_test_context import prepare_writer_facts

        for factory, policy in (
            (directional_ring_carrier_facts, None),
            (directional_non_single_ring_carrier_facts, None),
            (shared_directional_ring_carrier_facts, None),
            (terminal_tetra_center_facts, terminal_tetra_center_policy()),
        ):
            first = factory()
            second = factory()
            self.assertEqual(first, second)
            self.assertIsNot(first, second)
            prepare_writer_facts(first, policy=policy)

    def test_policy_fixtures_are_repeatedly_equal(self) -> None:
        self.assertEqual(terminal_tetra_center_policy(), terminal_tetra_center_policy())
        self.assertEqual(duplicate_single_atom_policy(), duplicate_single_atom_policy())

    def test_chain_fixture_is_valid(self) -> None:
        self.assertEqual(len(chain_facts(("C", "C", "O")).bonds), 2)


if __name__ == "__main__":
    unittest.main()
