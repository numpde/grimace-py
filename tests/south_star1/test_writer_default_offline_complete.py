"""Default ordinary writer offline-complete support contract tests."""

from __future__ import annotations

import unittest

from grimace._south_star1.writer_support_artifact_offline_verifier import (
    classify_residual_stereo_obligations_offline,
)
from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.default_writer_capability_ledger import (
    BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.test_writer_default_parity_corpus import _accepted_case_result
from tests.south_star1.test_writer_default_parity_corpus import _artifact
from tests.south_star1.test_writer_default_parity_corpus import _facts
from tests.south_star1.test_writer_default_parity_corpus import _prepare_default


class WriterDefaultOfflineCompleteTest(unittest.TestCase):
    def test_accepted_default_cases_are_offline_complete(self) -> None:
        for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES:
            with self.subTest(case=case.name):
                result = _accepted_case_result(case)

                self.assertTrue(case.expected_structural_artifact)
                self.assertTrue(case.expected_live_artifact_verifier)
                self.assertTrue(case.expected_facts_bound_verifier)
                self.assertTrue(case.expected_offline_replay_complete)
                self.assertTrue(result["structural_accepted"])
                self.assertTrue(result["live_accepted"])
                self.assertTrue(result["facts_bound_accepted"])
                self.assertTrue(result["facts_bound_offline_complete"])
                self.assertEqual(result["facts_bound_unchecked_object_kinds"], ())
                self.assertEqual(
                    result["facts_bound_unchecked_obligation_families"],
                    (),
                )
                self.assertLessEqual(
                    set(case.expected_offline_relation_families),
                    set(result["facts_bound_relation_families"]),
                )

    def test_blocked_default_cases_have_no_artifact_contract(self) -> None:
        for case in BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES:
            with self.subTest(case=case.name):
                self.assertEqual(case.expected, "blocked")
                self.assertIsNotNone(case.blocker_phase)
                self.assertIsNotNone(case.blocker_kind)
                self.assertFalse(case.expected_structural_artifact)
                self.assertFalse(case.expected_live_artifact_verifier)
                self.assertFalse(case.expected_facts_bound_verifier)
                self.assertFalse(case.expected_offline_replay_complete)
                self.assertEqual(case.expected_offline_relation_families, ())

    def test_incomplete_obligation_family_is_not_reported_complete(self) -> None:
        case = next(
            item
            for item in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
            if item.name == "ethanol"
        )
        artifact = _artifact(_prepare_default(_facts(case)))
        objects = {item["object_id"]: item for item in artifact["objects"]}
        branch = next(
            item for item in artifact["objects"] if item["kind"] == "branch_support"
        )
        manifest = branch["payload"]["obligation_manifests"]["stereo_lifecycle"][0]
        manifest["is_discharged"] = False
        manifest["is_noop"] = False
        manifest["is_empty"] = False
        manifest["terminal_clean"] = False

        classification = classify_residual_stereo_obligations_offline(
            artifact=artifact,
            objects=objects,
        )

        self.assertTrue(classification.accepted, classification.reason)
        self.assertIn("stereo_lifecycle", classification.unchecked_families)


if __name__ == "__main__":
    unittest.main()
