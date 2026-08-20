from __future__ import annotations

import unittest

from tests.south_star1.default_writer_capability_ledger import ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
from tests.south_star1.qualification_plan import (
    CONTINUATION_AUTHORITY_DIAGNOSTIC_LAYERS,
    CONTINUATION_AUTHORITY_PRODUCT_LAYERS,
    CONTINUATION_PROOF_QUALIFIED_CASES,
    FAST_ACCEPTED_CASES,
    MATERIALIZED_ARTIFACT_QUALIFIED_CASES,
    PUBLIC_PROOF_SHARD_COUNT,
    SLOW_COUPLED_CASES,
    SLOW_COUPLED_CASE_NAMES,
    SLOW_QUALIFICATION_LAYERS,
    SLOW_QUALIFICATION_SHARDS,
    validate_qualification_plan,
)


class QualificationPlanTest(unittest.TestCase):
    def test_validator_accepts_authoritative_plan(self) -> None:
        validate_qualification_plan()

    def test_authority_partition_is_complete(self) -> None:
        accepted = {case.name for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES}
        materialized = {case.name for case in MATERIALIZED_ARTIFACT_QUALIFIED_CASES}
        continuation = {case.name for case in CONTINUATION_PROOF_QUALIFIED_CASES}
        self.assertTrue(materialized.isdisjoint(continuation))
        self.assertEqual(materialized | continuation, accepted)

    def test_shards_are_disjoint_and_ledger_ordered(self) -> None:
        shard_sets = [set(shard.case_names) for shard in SLOW_QUALIFICATION_SHARDS.values()]
        for index, left in enumerate(shard_sets):
            for right in shard_sets[index + 1:]:
                self.assertTrue(left.isdisjoint(right))
        self.assertEqual(set().union(*shard_sets), set(SLOW_COUPLED_CASE_NAMES))
        self.assertEqual(tuple(case.name for case in SLOW_COUPLED_CASES), tuple(case.name for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES if case.name in SLOW_COUPLED_CASE_NAMES))

    def test_product_and_diagnostic_layers_are_disjoint(self) -> None:
        self.assertTrue(set(CONTINUATION_AUTHORITY_PRODUCT_LAYERS).isdisjoint(CONTINUATION_AUTHORITY_DIAGNOSTIC_LAYERS))
        self.assertEqual(sum(name.startswith("public-proofs-") for name in CONTINUATION_AUTHORITY_PRODUCT_LAYERS), PUBLIC_PROOF_SHARD_COUNT)
        self.assertNotIn("continuation-proof-complete", SLOW_QUALIFICATION_LAYERS)


if __name__ == "__main__":
    unittest.main()
