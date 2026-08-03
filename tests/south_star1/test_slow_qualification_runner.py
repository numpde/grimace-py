from __future__ import annotations

import unittest

from tests import run_south_star1_slow as runner
from tests.south_star1.qualification_plan import (
    CONTINUATION_AUTHORITY_DIAGNOSTIC_LAYERS,
    CONTINUATION_AUTHORITY_PRODUCT_LAYERS,
    FAST_ACCEPTED_CASES,
    SLOW_COUPLED_CASES,
    SLOW_QUALIFICATION_LAYERS,
    SLOW_QUALIFICATION_SHARDS,
    bind_slow_qualification_shard,
    reset_slow_qualification_shard,
    selected_slow_qualification_cases,
    validate_qualification_plan,
)


def _test_ids(suite: unittest.TestSuite) -> tuple[str, ...]:
    ids: list[str] = []
    for test in suite:
        if isinstance(test, unittest.TestSuite):
            ids.extend(_test_ids(test))
        else:
            ids.append(test.id())
    return tuple(ids)


class SlowQualificationRunnerTest(unittest.TestCase):
    def test_plan_validates(self) -> None:
        validate_qualification_plan()

    def test_every_registered_test_id_resolves(self) -> None:
        loader = unittest.defaultTestLoader
        for definition in SLOW_QUALIFICATION_LAYERS.values():
            for test_id in definition.test_ids:
                with self.subTest(test_id=test_id):
                    suite = loader.loadTestsFromName(test_id)
                    self.assertNotIn(".loadTestsFromName", _test_ids(suite))

    def test_layer_ids_are_unique_and_roles_are_disjoint(self) -> None:
        definitions = tuple(SLOW_QUALIFICATION_LAYERS.values())
        ids = [test_id for definition in definitions for test_id in definition.test_ids]
        self.assertEqual(len(ids), len(set(ids)))
        product = set(CONTINUATION_AUTHORITY_PRODUCT_LAYERS)
        diagnostic = set(CONTINUATION_AUTHORITY_DIAGNOSTIC_LAYERS)
        self.assertTrue(product.isdisjoint(diagnostic))
        self.assertTrue(all(SLOW_QUALIFICATION_LAYERS[name].kind == "product" for name in product))
        self.assertTrue(all(SLOW_QUALIFICATION_LAYERS[name].kind == "diagnostic" for name in diagnostic))

    def test_continuation_product_order_and_proof_indices(self) -> None:
        self.assertEqual(
            CONTINUATION_AUTHORITY_PRODUCT_LAYERS,
            (
                "public-build", "public-certify", "public-runtime",
                "public-recertification", "public-proofs-0", "public-proofs-1",
                "public-proofs-2", "public-proofs-3", "support-reparse",
                "continuation", "stereo-audit",
            ),
        )
        proof_layers = tuple(name for name in CONTINUATION_AUTHORITY_PRODUCT_LAYERS if name.startswith("public-proofs-"))
        self.assertEqual(proof_layers, tuple(f"public-proofs-{i}" for i in range(4)))
        self.assertNotIn("continuation-proof-complete", SLOW_QUALIFICATION_LAYERS)

    def test_shards_resolve_to_declared_ledger_cases(self) -> None:
        accepted = {case.name for case in SLOW_COUPLED_CASES}
        for shard in SLOW_QUALIFICATION_SHARDS.values():
            with self.subTest(shard=shard.name):
                self.assertTrue(set(shard.case_names) <= accepted)
                token = bind_slow_qualification_shard(shard.name)
                try:
                    self.assertEqual(
                        tuple(case.name for case in selected_slow_qualification_cases()),
                        shard.case_names,
                    )
                finally:
                    reset_slow_qualification_shard(token)

    def test_selected_layer_loads_only_its_declared_test(self) -> None:
        for shard_name, layer_name in (("remote-a", "public-build"), ("remote-b", "public-proofs-2")):
            with self.subTest(shard=shard_name, layer=layer_name):
                suite, token = runner.load_selected_layer(
                    unittest.defaultTestLoader, shard_name, layer_name
                )
                try:
                    self.assertEqual(
                        _test_ids(suite),
                        SLOW_QUALIFICATION_LAYERS[layer_name].test_ids,
                    )
                finally:
                    reset_slow_qualification_shard(token)

    def test_invalid_selection_fails_before_loading_tests(self) -> None:
        for shard, layer in ((None, "public-build"), ("unknown", "public-build"), ("remote-a", None), ("remote-a", "unknown")):
            with self.subTest(shard=shard, layer=layer):
                with self.assertRaises(ValueError):
                    runner.validate_selection(shard, layer)

    def test_fast_cases_never_select_a_slow_case(self) -> None:
        self.assertTrue({case.name for case in FAST_ACCEPTED_CASES}.isdisjoint(
            case.name for case in SLOW_COUPLED_CASES
        ))


if __name__ == "__main__":
    unittest.main()
