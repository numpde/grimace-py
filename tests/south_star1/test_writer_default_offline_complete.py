"""Default ordinary writer offline-complete support contract tests."""

from __future__ import annotations

import unittest
import os

from grimace._south_star1.writer_support_artifact_offline_verifier import (
    classify_residual_stereo_obligations_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_writer_support_artifact_offline_replay,
)
from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.default_writer_capability_ledger import (
    BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.qualification_plan import FAST_ACCEPTED_CASES
from tests.south_star1.qualification_plan import SLOW_COUPLED_CASES
from tests.south_star1.qualification_plan import (
    selected_slow_qualification_cases,
)
from tests.south_star1.qualification_support import accepted_case_result
from tests.south_star1.qualification_support import support_artifact_for_prepared
from tests.south_star1.qualification_support import facts_for_case
from tests.south_star1.qualification_support import prepare_default_case
from tests.south_star1.qualification_assertions import assert_materialized_case_matches_ledger
from tests.south_star1.qualification_assertions import assert_offline_case_matches_ledger
from tests.south_star1.qualification_plan import case_by_name


class WriterDefaultOfflineCompleteTest(unittest.TestCase):
    def _assert_cases_are_offline_complete(self, cases) -> None:
        for case in cases:
            with self.subTest(case=case.name):
                result = accepted_case_result(case)
                assert_materialized_case_matches_ledger(self, case=case, result=result)

    def test_fast_cases_are_offline_complete(self) -> None:
        self._assert_cases_are_offline_complete(FAST_ACCEPTED_CASES)

    @unittest.skipUnless(
        os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1",
        "set SOUTH_STAR1_RUN_SLOW=1 to run coupled cases",
    )
    def test_slow_coupled_cases_are_offline_complete(self) -> None:
        self._assert_cases_are_offline_complete(selected_slow_qualification_cases())

    @unittest.skipUnless(
        os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1",
        "set SOUTH_STAR1_RUN_SLOW=1 to run coupled cases",
    )
    def test_zero_h_tetrahedral_is_offline_complete(self) -> None:
        cases = selected_slow_qualification_cases()
        self._assert_cases_are_offline_complete(
            tuple(case for case in cases if case.name == "zero_h_tetrahedral")
        )

    @unittest.skipUnless(
        os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1",
        "set SOUTH_STAR1_RUN_SLOW=1 to run coupled cases",
    )
    def test_adjacent_specified_tetrahedral_is_offline_complete(self) -> None:
        cases = selected_slow_qualification_cases()
        self._assert_cases_are_offline_complete(
            tuple(
                case
                for case in cases
                if case.name == "adjacent_specified_tetrahedral"
            )
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
                self.assertFalse(case.expected_live_frontier_agreement_complete)
                self.assertFalse(case.expected_live_count_agreement_complete)
                self.assertFalse(case.expected_snapshot_resume_agreement_complete)
                self.assertEqual(case.expected_offline_object_kinds, ())
                self.assertEqual(case.expected_offline_unchecked_object_kinds, ())
                self.assertEqual(case.expected_offline_relation_families, ())
                self.assertEqual(
                    case.expected_offline_unchecked_obligation_families,
                    (),
                )

    def test_descriptive_lifecycle_flags_do_not_remove_replay_credit(self) -> None:
        case = case_by_name("ethanol")
        artifact = support_artifact_for_prepared(prepare_default_case(facts_for_case(case)))
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
            facts=facts_for_case(case),
            artifact=artifact,
            objects=objects,
        )

        self.assertTrue(classification.accepted, classification.reason)
        self.assertNotIn("stereo_lifecycle", classification.unchecked_families)

        replay = verify_writer_support_artifact_offline_replay(
            facts=facts_for_case(case),
            artifact=artifact,
        )

        self.assertTrue(replay.accepted, replay.reason)
        self.assertTrue(replay.offline_replay_complete)
        assert_offline_case_matches_ledger(self, case=case, verification=replay)
        self.assertNotIn(
            "stereo_lifecycle",
            replay.unchecked_obligation_families,
        )


if __name__ == "__main__":
    unittest.main()
