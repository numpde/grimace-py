"""Default ordinary writer parity corpus over table-backed artifacts."""

from __future__ import annotations

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
from tests.south_star1.default_writer_capability_ledger import (
    DefaultWriterCapabilityCase,
)

ACCEPTED_CASES = ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
BLOCKED_CASES = BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES


class WriterDefaultParityCorpusTest(unittest.TestCase):
    def test_accepted_default_corpus_verifies_support_artifacts(self) -> None:
        for case in ACCEPTED_CASES:
            with self.subTest(case=case.name):
                result = _accepted_case_result(case)

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

    def test_accepted_default_corpus_reparses_to_isomorphic_facts(self) -> None:
        for case in ACCEPTED_CASES:
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


def _facts(case: DefaultWriterCapabilityCase):
    return ordinary_molecule_facts_from_smiles(
        case.smiles,
        case.extraction_options,
    )


def _prepare_default(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _writer_options(rooted_at_atom: int = 0) -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def _initial_snapshot(prepared, rooted_at_atom: int = 0):
    options = _writer_options(rooted_at_atom)
    return capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=initial_writer_frontier_cursor(prepared, options),
    )


def _support_image(case: DefaultWriterCapabilityCase):
    return enumerate_prepared_writer_shaped_support(
        prepared=_prepare_default(_facts(case)),
        runtime_options=_writer_options(case.rooted_at_atom),
    )


def _artifact(prepared, rooted_at_atom: int = 0):
    return writer_support_artifact_envelope_for_snapshot(
        prepared=prepared,
        snapshot=_initial_snapshot(prepared, rooted_at_atom),
    )


def _accepted_case_result(case: DefaultWriterCapabilityCase) -> dict[str, object]:
    facts = _facts(case)
    prepared = _prepare_default(facts)
    options = _writer_options(case.rooted_at_atom)
    state = initial_writer_runtime_state(
        prepared=prepared,
        runtime_options=options,
    )
    image = enumerate_prepared_writer_shaped_support(
        prepared=prepared,
        runtime_options=options,
    )
    snapshot = _initial_snapshot(prepared, case.rooted_at_atom)
    count_envelope = writer_frontier_count_envelope_for_snapshot(
        prepared=prepared,
        snapshot=snapshot,
    )
    count_verification = verify_writer_frontier_count_envelope(
        prepared=prepared,
        envelope=count_envelope,
    )
    artifact = writer_support_artifact_envelope_for_snapshot(
        prepared=prepared,
        snapshot=snapshot,
    )
    structural = verify_writer_support_artifact_consistency(artifact)
    live = verify_writer_support_artifact_envelope(
        prepared=prepared,
        envelope=artifact,
    )
    fact_bound = verify_writer_support_artifact_for_facts(
        facts=facts,
        runtime_options=options,
        artifact=artifact,
    )

    assert count_verification.accepted, count_verification.reason
    assert structural.accepted, structural.reason
    assert live.accepted, live.reason
    assert fact_bound.accepted, fact_bound.reason

    snapshot_resume = _snapshot_resume_agreement(prepared, snapshot)
    live_count_agreement_complete = (
        count_writer_runtime_support(
            prepared=prepared,
            state=state,
        )
        == image.distinct_count
        == structural.support_count
        == artifact["metrics"]["support_string_count"]
        and count_writer_runtime_completions(
            prepared=prepared,
            state=state,
        )
        == image.witness_count
        == structural.witness_count
    )

    return {
        "support_count": count_writer_runtime_support(
            prepared=prepared,
            state=state,
        ),
        "completion_count": count_writer_runtime_completions(
            prepared=prepared,
            state=state,
        ),
        "materialized_support_count": image.distinct_count,
        "materialized_witness_count": image.witness_count,
        "artifact_support_count": structural.support_count,
        "artifact_witness_count": structural.witness_count,
        "artifact_metrics": artifact["metrics"],
        "structural_accepted": structural.accepted,
        "live_accepted": live.accepted,
        "facts_bound_accepted": fact_bound.accepted,
        "facts_bound_offline_complete": fact_bound.offline_replay_complete,
        "live_frontier_agreement_complete": (
            snapshot_resume["frontier_traversal_complete"]
            and count_verification.accepted
            and live.accepted
        ),
        "live_count_agreement_complete": live_count_agreement_complete,
        "snapshot_resume_agreement_complete": (
            snapshot_resume["frontier_traversal_complete"]
            and snapshot_resume["strings"] == set(image.strings)
        ),
        "facts_bound_object_kinds": fact_bound.offline_checked_object_kinds,
        "facts_bound_unchecked_object_kinds": (
            fact_bound.offline_unchecked_object_kinds
        ),
        "facts_bound_unchecked_obligation_families": (
            fact_bound.offline_unchecked_obligation_families
        ),
        "facts_bound_relation_families": fact_bound.offline_checked_relation_families,
    }


def _snapshot_resume_agreement(prepared, snapshot) -> dict[str, object]:
    pending = [(snapshot, "")]
    seen = set()
    strings = set()
    frontier_traversal_complete = True
    while pending:
        current, emitted = pending.pop(0)
        # A shared cursor can still represent distinct emitted prefixes.
        seen_key = (current.cursor, emitted)
        if seen_key in seen:
            continue
        seen.add(seen_key)
        resumed_state = writer_runtime_state_from_snapshot(
            current,
            prepared=prepared,
        )
        resumed_choices = resume_writer_frontier_choices_from_snapshot(
            current,
            prepared=prepared,
        )
        runtime_choices = writer_runtime_choices(
            prepared=prepared,
            state=resumed_state,
        )
        if resumed_choices != runtime_choices:
            frontier_traversal_complete = False
            continue
        product = _snapshot_advance_writer_frontier_product(
            prepared,
            current.cursor,
        )
        if product.blocked:
            frontier_traversal_complete = False
            continue
        projection = product.projection_certificate
        if projection.terminal_projection_certificate is not None:
            strings.add(emitted)
            continue
        for choice in resumed_choices.choices:
            advanced = advance_writer_frontier_snapshot(
                current,
                prepared=prepared,
                emitted_text=choice.emitted_text,
            )
            pending.append(
                (
                    advanced,
                    emitted + choice.emitted_text,
                )
            )
    return {
        "frontier_traversal_complete": frontier_traversal_complete,
        "strings": strings,
    }


def _blocked_case_result(case: DefaultWriterCapabilityCase) -> dict[str, object]:
    facts = _facts(case)
    try:
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
