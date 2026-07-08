"""Default ordinary writer parity corpus over table-backed artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.fact_isomorphism import facts_are_isomorphic
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
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
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
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


_GRAPH_EXTRACTION = RdkitOrdinaryExtractionOptions(include_potential_sites=False)
_POTENTIAL_STEREO_EXTRACTION = RdkitOrdinaryExtractionOptions(
    include_potential_sites=True,
)


@dataclass(frozen=True, slots=True)
class DefaultParityCase:
    name: str
    smiles: str
    extraction_options: RdkitOrdinaryExtractionOptions
    expected: str
    expected_error_kind: SouthStarErrorKind | None = None
    expected_blocker_kind: str | None = None
    expected_blocker_operation: str | None = None


ACCEPTED_CASES = (
    DefaultParityCase("ethanol", "CCO", _GRAPH_EXTRACTION, "accepted"),
    DefaultParityCase("branched_alcohol", "CC(C)O", _GRAPH_EXTRACTION, "accepted"),
    DefaultParityCase("cyclopropane", "C1CC1", _GRAPH_EXTRACTION, "accepted"),
    DefaultParityCase("cyclobutane", "C1CCC1", _GRAPH_EXTRACTION, "accepted"),
    DefaultParityCase(
        "cyclopropene_double_closure",
        "C1=CC1",
        _GRAPH_EXTRACTION,
        "accepted",
    ),
    DefaultParityCase(
        "cyclopropyne_triple_closure",
        "C1#CC1",
        _GRAPH_EXTRACTION,
        "accepted",
    ),
    DefaultParityCase(
        "branched_cyclobutane",
        "C1CC(C)C1",
        _GRAPH_EXTRACTION,
        "accepted",
    ),
)

BLOCKED_CASES = (
    DefaultParityCase(
        "ammonium_charge",
        "[NH4+]",
        _GRAPH_EXTRACTION,
        "blocked",
        expected_error_kind=SouthStarErrorKind.UNSUPPORTED_ATOM,
    ),
    DefaultParityCase(
        "isotopic_methane",
        "[13CH4]",
        _GRAPH_EXTRACTION,
        "blocked",
        expected_error_kind=SouthStarErrorKind.UNSUPPORTED_ATOM,
    ),
    DefaultParityCase(
        "cyclopropene_potential_directional_boundary",
        "C1=CC1",
        _POTENTIAL_STEREO_EXTRACTION,
        "blocked",
        expected_blocker_kind="unsupported_directional_non_neighbor_ligand",
        expected_blocker_operation="directional carrier-mark restriction",
    ),
)


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

                if case.expected_error_kind is not None:
                    self.assertEqual(blocked["stage"], "prepare")
                    self.assertIs(blocked["error_kind"], case.expected_error_kind)
                    continue

                self.assertEqual(blocked["stage"], "frontier")
                self.assertEqual(
                    {item.kind for item in blocked["blockers"]},
                    {case.expected_blocker_kind},
                )
                self.assertEqual(
                    {item.operation for item in blocked["blockers"]},
                    {case.expected_blocker_operation},
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


def _facts(case: DefaultParityCase):
    return ordinary_molecule_facts_from_smiles(
        case.smiles,
        case.extraction_options,
    )


def _prepare_default(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _writer_options() -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=0,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def _initial_snapshot(prepared):
    options = _writer_options()
    return capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=initial_writer_frontier_cursor(prepared, options),
    )


def _support_image(case: DefaultParityCase):
    return enumerate_prepared_writer_shaped_support(
        prepared=_prepare_default(_facts(case)),
        runtime_options=_writer_options(),
    )


def _artifact(prepared):
    return writer_support_artifact_envelope_for_snapshot(
        prepared=prepared,
        snapshot=_initial_snapshot(prepared),
    )


def _accepted_case_result(case: DefaultParityCase) -> dict[str, object]:
    facts = _facts(case)
    prepared = _prepare_default(facts)
    state = initial_writer_runtime_state(
        prepared=prepared,
        runtime_options=_writer_options(),
    )
    image = enumerate_prepared_writer_shaped_support(
        prepared=prepared,
        runtime_options=_writer_options(),
    )
    snapshot = _initial_snapshot(prepared)
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
        runtime_options=_writer_options(),
        artifact=artifact,
    )

    assert count_verification.accepted, count_verification.reason
    assert structural.accepted, structural.reason
    assert live.accepted, live.reason
    assert fact_bound.accepted, fact_bound.reason

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
    }


def _blocked_case_result(case: DefaultParityCase) -> dict[str, object]:
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
