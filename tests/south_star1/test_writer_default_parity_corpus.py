"""Default ordinary writer parity corpus over table-backed artifacts."""

from __future__ import annotations

import os
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.fact_isomorphism import facts_are_isomorphic
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_envelope_work import (
    default_writer_envelope_work_budget,
)
from grimace._south_star1.writer_frontier import (
    _snapshot_advance_writer_frontier_product,
)
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_frontier_count_envelope import (
    verify_writer_frontier_count_envelope,
)
from grimace._south_star1.writer_frontier_count_envelope import (
    writer_frontier_count_envelope_for_snapshot,
)
from grimace._south_star1.writer_runtime import count_writer_runtime_completions
from grimace._south_star1.writer_runtime import count_writer_runtime_support
from grimace._south_star1.writer_runtime import initial_writer_runtime_state
from grimace._south_star1.writer_runtime import writer_runtime_choices
from grimace._south_star1.writer_runtime import writer_runtime_state_from_snapshot
from grimace._south_star1.writer_snapshot import advance_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot import resume_writer_frontier_choices_from_snapshot
from grimace._south_star1.writer_snapshot import (
    _writer_snapshot_advance_outcome_by_emitted_text,
)
from grimace._south_star1.writer_support import (
    enumerate_prepared_writer_shaped_support,
)
from grimace._south_star1.writer_support_artifact_checker import (
    verify_writer_support_artifact_consistency,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    verify_writer_support_artifact_envelope,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_snapshot,
)
from grimace._south_star1.writer_support_artifact_fact_verifier import (
    verify_writer_support_artifact_for_facts,
)
from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.default_writer_capability_ledger import (
    BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.default_writer_capability_ledger import DefaultWriterCapabilityCase
from tests.south_star1.qualification_plan import FAST_ACCEPTED_CASES
from tests.south_star1.qualification_plan import SLOW_COUPLED_CASES
from tests.south_star1.qualification_plan import SLOW_COUPLED_CASE_NAMES
from tests.south_star1.qualification_plan import (
    selected_slow_qualification_cases,
)
from tests.south_star1.qualification_support import accepted_case_result as _accepted_case_result
from tests.south_star1.qualification_support import case_facts as _facts
from tests.south_star1.qualification_support import prepare_default_case as _prepare_default
from tests.south_star1.qualification_support import runtime_options_for_case as _case_runtime_options
from tests.south_star1.qualification_support import support_image_for_case as _support_image
from tests.south_star1.qualification_support import artifact_for_prepared as _artifact
from tests.south_star1.qualification_support import _initial_snapshot
from tests.south_star1.qualification_support import _writer_options

ACCEPTED_CASES = ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
BLOCKED_CASES = BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES
RUN_SLOW_ENV = "SOUTH_STAR1_RUN_SLOW"
_SPECIAL_CASES = tuple(case.name for case in SLOW_COUPLED_CASES)
_ZERO_H_AND_ADJACENT = SLOW_COUPLED_CASE_NAMES[:2]
_REMOTE_COUPLED_A = SLOW_COUPLED_CASE_NAMES[2:3]
_REMOTE_COUPLED_B = SLOW_COUPLED_CASE_NAMES[3:]


class WriterDefaultParityCorpusTest(unittest.TestCase):
    def test_accepted_default_shards_are_complete_and_deterministic(self) -> None:
        accepted_names = tuple(case.name for case in ACCEPTED_CASES)
        self.assertEqual(
            tuple(case.name for case in FAST_ACCEPTED_CASES),
            tuple(name for name in accepted_names if name not in SLOW_COUPLED_CASE_NAMES),
        )
        self.assertEqual(
            tuple(case.name for case in SLOW_COUPLED_CASES),
            tuple(name for name in accepted_names if name in SLOW_COUPLED_CASE_NAMES),
        )

    def _run_cases(self):
        yield from FAST_ACCEPTED_CASES

    def test_accepted_default_corpus_verifies_support_artifacts(self) -> None:
        for case in self._run_cases():
            with self.subTest(case=case.name):
                result = _accepted_case_result(case)
                self._assert_accepted_case_result(case, result)

    def test_accepted_default_corpus_reparses_to_isomorphic_facts(self) -> None:
        for case in self._run_cases():
            with self.subTest(case=case.name):
                facts = _facts(case)
                image = _support_image(case)
                for text in image.strings:
                    with self.subTest(case=case.name, text=text):
                        reparsed = ordinary_molecule_facts_from_smiles(
                            text,
                            case.extraction_options,
                        )
                        self.assertTrue(
                            facts_are_isomorphic(facts, reparsed).isomorphic,
                            text,
                        )

    def test_slow_coupled_corpus_verifies_support_artifacts(self) -> None:
        if os.environ.get(RUN_SLOW_ENV) != "1":
            self.skipTest(f"set {RUN_SLOW_ENV}=1 to run coupled cases")
        for case in selected_slow_qualification_cases():
            self._assert_slow_support_case(case)

    def test_zero_h_tetrahedral_support_artifact(self) -> None:
        self._assert_slow_support_case_named("zero_h_tetrahedral")

    def test_adjacent_specified_tetrahedral_support_artifact(self) -> None:
        self._assert_slow_support_case_named("adjacent_specified_tetrahedral")

    def _assert_slow_support_case_named(self, name: str) -> None:
        if os.environ.get(RUN_SLOW_ENV) != "1":
            self.skipTest(f"set {RUN_SLOW_ENV}=1 to run coupled cases")
        cases = tuple(case for case in selected_slow_qualification_cases() if case.name == name)
        self.assertEqual(tuple(case.name for case in cases), (name,))
        self._assert_slow_support_case(cases[0])

    def _assert_slow_support_case(self, case) -> None:
        with self.subTest(case=case.name):
            result = _accepted_case_result(case)
            self._assert_accepted_case_result(case, result)

    def test_slow_coupled_corpus_reparses_to_isomorphic_facts(self) -> None:
        if os.environ.get(RUN_SLOW_ENV) != "1":
            self.skipTest(f"set {RUN_SLOW_ENV}=1 to run coupled cases")
        for case in selected_slow_qualification_cases():
            with self.subTest(case=case.name):
                facts = _facts(case)
                for text in _support_image(case).strings:
                    with self.subTest(text=text):
                        reparsed = ordinary_molecule_facts_from_smiles(
                            text,
                            case.extraction_options,
                        )
                        self.assertTrue(
                            facts_are_isomorphic(facts, reparsed).isomorphic,
                            text,
                        )

    def _assert_accepted_case_result(
        self,
        case: DefaultWriterCapabilityCase,
        result: dict[str, object],
    ) -> None:
        self.assertEqual(
            result["support_count"],
            result["artifact_support_count"],
        )
        self.assertEqual(
            result["completion_count"],
            result["artifact_witness_count"],
        )
        self.assertEqual(
            result["support_count"],
            case.expected_support_count,
        )
        self.assertEqual(
            result["completion_count"],
            case.expected_completion_count,
        )
        self.assertEqual(
            result["support_count"],
            result["materialized_support_count"],
        )
        self.assertEqual(
            result["completion_count"],
            result["materialized_witness_count"],
        )
        self.assertEqual(
            result["support_count"],
            result["artifact_metrics"]["support_string_count"],
        )
        self.assertGreater(result["artifact_metrics"]["object_count"], 0)
        self.assertEqual(
            result["artifact_metrics"]["reachable_object_count"],
            result["artifact_metrics"]["object_count"],
        )
        self.assertEqual(
            result["artifact_metrics"]["unreferenced_object_count"],
            0,
        )
        self.assertIsNotNone(result["artifact_metrics"]["count_dag_node_count"])
        self.assertLessEqual(
            result["artifact_metrics"]["largest_object_digest_payload_bytes"],
            default_writer_envelope_work_budget(None).max_digest_term_bytes,
        )
        self.assertEqual(
            result["structural_accepted"],
            case.expected_structural_artifact,
        )
        self.assertEqual(
            result["live_accepted"],
            case.expected_live_artifact_verifier,
        )
        self.assertEqual(
            result["facts_bound_accepted"],
            case.expected_facts_bound_verifier,
        )
        self.assertEqual(
            result["facts_bound_offline_complete"],
            case.expected_offline_replay_complete,
        )
        self.assertEqual(
            result["live_frontier_agreement_complete"],
            case.expected_live_frontier_agreement_complete,
        )
        self.assertEqual(
            result["live_count_agreement_complete"],
            case.expected_live_count_agreement_complete,
        )
        self.assertEqual(
            result["snapshot_resume_agreement_complete"],
            case.expected_snapshot_resume_agreement_complete,
        )
        self.assertEqual(
            result["facts_bound_object_kinds"],
            case.expected_offline_object_kinds,
        )
        self.assertEqual(
            result["facts_bound_unchecked_object_kinds"],
            case.expected_offline_unchecked_object_kinds,
        )
        self.assertEqual(
            result["facts_bound_unchecked_obligation_families"],
            case.expected_offline_unchecked_obligation_families,
        )
        self.assertLessEqual(
            set(case.expected_offline_relation_families),
            set(result["facts_bound_relation_families"]),
        )

    def test_blocked_default_corpus_has_typed_blockers(self) -> None:
        for case in BLOCKED_CASES:
            with self.subTest(case=case.name):
                blocked = _blocked_case_result(case)

                if case.blocker_error_kind is not None:
                    self.assertEqual(blocked["stage"], "prepare")
                    self.assertIs(blocked["error_kind"], case.blocker_error_kind)
                    self.assertIsNotNone(case.blocker_message_contains)
                    blocker_message_contains = case.blocker_message_contains
                    self.assertIn(
                        blocker_message_contains,
                        blocked["message"],
                    )
                    continue

                self.assertEqual(blocked["stage"], "frontier")
                self.assertEqual(
                    {item.kind for item in blocked["blockers"]},
                    {case.blocker_kind},
                )
                self.assertEqual(
                    {item.operation for item in blocked["blockers"]},
                    {case.blocker_operation},
                )

    def test_default_cyclopropene_artifact_policy_identity_is_default_joint(
        self,
    ) -> None:
        case = next(item for item in ACCEPTED_CASES if item.name == "cyclopropene_double_closure")
        facts = _facts(case)
        prepared = _prepare_default(facts)
        artifact = _artifact(prepared)

        default_verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=_writer_options(),
            artifact=artifact,
        )


        self.assertTrue(default_verification.accepted, default_verification.reason)
        with self.assertRaisesRegex(SouthStarError, "non-single ring closures"):
            ordinary_policy_for_facts(
                facts,
                OrdinaryPolicyOptions(non_single_ring_closures="unsupported"),
            )
def _blocked_case_result(case: DefaultWriterCapabilityCase) -> dict[str, object]:
    try:
        facts = _facts(case)
        prepared = _prepare_default(facts)
    except SouthStarError as error:
        return {
            "stage": "prepare",
            "error_kind": error.kind,
            "message": str(error),
        }

    blockers = _reachable_stereo_policy_blockers(prepared)
    return {
        "stage": "frontier",
        "blockers": blockers,
    }


def _reachable_stereo_policy_blockers(prepared):
    pending = [
        initial_writer_runtime_state(
            prepared=prepared,
            runtime_options=_writer_options(),
        ).snapshot
    ]
    seen = set()
    blockers = []
    while pending:
        snapshot = pending.pop(0)
        if snapshot.cursor in seen:
            continue
        seen.add(snapshot.cursor)
        product = _snapshot_advance_writer_frontier_product(
            prepared,
            snapshot.cursor,
        )
        if product.blocked:
            blockers.extend(
                item.blocker
                for item in (
                    product
                    .blocked_frontier_certificate
                    .stereo_policy_blocker_certificates
                )
            )
            continue
        projection = product.projection_certificate
        if projection.terminal_projection_certificate is not None:
            continue
        for text_projection in projection.text_choice_projection_certificates:
            outcome = _writer_snapshot_advance_outcome_by_emitted_text(
                snapshot,
                prepared=prepared,
                emitted_text=text_projection.emitted_text,
            )
            if outcome.advanced_snapshot is not None:
                pending.append(outcome.advanced_snapshot)
    return tuple(blockers)


if __name__ == "__main__":
    unittest.main()

