"""Continuation/Rust tiers for every accepted default writer case."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from grimace import MolToSmilesContinuationDecoder
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_branch_transition_artifact import verify_writer_branch_transition_artifact_envelope
from grimace._south_star1.writer_branch_transition_artifact_checker import verify_writer_branch_transition_artifact_consistency
from grimace._south_star1.writer_branch_transition_artifact_fact_verifier import verify_writer_branch_transition_artifact_for_facts
from grimace._south_star1.writer_continuation_asset import branch_transition_artifact_from_continuation_asset
from grimace._south_star1.writer_continuation_asset import open_writer_continuation_core
from grimace._south_star1.writer_continuation_asset import terminalization_artifact_from_continuation_asset
from grimace._south_star1.writer_continuation_asset import verify_writer_continuation_asset_consistency
from grimace._south_star1.writer_continuation_asset import verify_writer_continuation_asset_live
from grimace._south_star1.writer_continuation_asset import write_writer_continuation_asset
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_terminalization_artifact import verify_writer_terminalization_artifact_envelope
from grimace._south_star1.writer_terminalization_artifact_checker import verify_writer_terminalization_artifact_consistency
from grimace._south_star1.writer_terminalization_artifact_fact_verifier import verify_writer_terminalization_artifact_for_facts
from tests.south_star1.default_writer_capability_ledger import ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
from tests.south_star1.test_writer_default_parity_corpus import _facts
from tests.south_star1.test_writer_default_parity_corpus import _support_image
from tests.south_star1.test_writer_default_parity_corpus import _writer_options


class WriterDefaultContinuationCorpusTest(unittest.TestCase):
    def test_every_accepted_case_crosses_all_continuation_tiers(self) -> None:
        for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES:
            with self.subTest(case=case.name), TemporaryDirectory() as directory:
                options = _writer_options(case.rooted_at_atom)
                facts = _facts(case)
                prepared = prepare_south_star_mol_from_facts(
                    facts,
                    writer_surface=SouthStarWriterSurface(),
                )
                snapshot = capture_writer_frontier_snapshot(
                    prepared=prepared,
                    runtime_options=options,
                    cursor=initial_writer_frontier_cursor(prepared, options),
                )
                path = Path(directory) / "asset"
                with (
                    patch(
                        "grimace._south_star1.writer_frontier_count_envelope.writer_frontier_count_envelope_for_snapshot",
                        side_effect=AssertionError("count envelope invoked"),
                    ),
                    patch(
                        "grimace._south_star1.writer_count_dag_envelope.writer_count_certificate_dag_envelope_for_product",
                        side_effect=AssertionError("count DAG invoked"),
                    ),
                    patch(
                        "grimace._south_star1.writer_snapshot._iter_writer_snapshot_certified_support_strings",
                        side_effect=AssertionError("support materialization invoked"),
                    ),
                ):
                    write_writer_continuation_asset(
                        path=path,
                        prepared=prepared,
                        snapshot=snapshot,
                    )
                asset = open_writer_continuation_core(path)
                structural = verify_writer_continuation_asset_consistency(path)
                live = verify_writer_continuation_asset_live(
                    prepared=prepared,
                    asset=asset,
                    full=True,
                )
                self.assertEqual(structural.accepted, case.expected_continuation_asset_complete)
                self.assertTrue(live.accepted, live.reason)

                decoder = MolToSmilesContinuationDecoder.from_asset(path)
                support = _decoder_support(decoder)
                expected = tuple(sorted(_support_image(case).strings))
                self.assertEqual(support, expected)
                self.assertEqual(decoder.support_count, case.expected_support_count)
                self.assertEqual(decoder.completion_count, case.expected_completion_count)
                self.assertEqual(
                    sum(item.numerator for item in decoder.exact_probabilities()),
                    decoder.completion_count,
                )
                if case.expected_support_digest is not None:
                    self.assertEqual(_support_digest(support), case.expected_support_digest)
                advanced = decoder.next_choices[0].next_state
                resumed = MolToSmilesContinuationDecoder.from_snapshot(
                    path,
                    advanced.snapshot(),
                )
                self.assertEqual(resumed.cache_key(), advanced.cache_key())

                proof_decoder = MolToSmilesContinuationDecoder.from_asset(
                    path,
                    proof_capable=True,
                    prepared=prepared,
                )
                self.assertIsNotNone(proof_decoder._state.proof_cursor)
                _verify_all_local_proofs(
                    test=self,
                    facts=facts,
                    prepared=prepared,
                    options=options,
                    asset=asset,
                )


def _verify_all_local_proofs(*, test, facts, prepared, options, asset) -> None:
    for edge in asset.records("edge_records"):
        for digest in edge.branch_certificate_digests:
            artifact = branch_transition_artifact_from_continuation_asset(
                prepared=prepared,
                asset=asset,
                source_raw_cursor_digest=edge.source_raw_cursor_digest,
                emitted_text=edge.emitted_text,
                branch_certificate_digest=digest,
            )
            structural = verify_writer_branch_transition_artifact_consistency(artifact)
            live = verify_writer_branch_transition_artifact_envelope(
                prepared=prepared,
                artifact=artifact,
            )
            offline = verify_writer_branch_transition_artifact_for_facts(
                facts=facts,
                runtime_options=options,
                artifact=artifact,
            )
            test.assertTrue(structural.accepted, structural.reason)
            test.assertTrue(live.accepted, live.reason)
            test.assertTrue(offline.accepted, offline.reason)
            test.assertEqual(offline.unchecked_obligation_families, ())
    for terminal in asset.records("terminal_records"):
        for digest in terminal.terminal_support_identity_digests:
            artifact = terminalization_artifact_from_continuation_asset(
                prepared=prepared,
                asset=asset,
                source_raw_cursor_digest=terminal.source_raw_cursor_digest,
                terminal_support_identity_digest=digest,
            )
            structural = verify_writer_terminalization_artifact_consistency(artifact)
            live = verify_writer_terminalization_artifact_envelope(
                prepared=prepared,
                artifact=artifact,
            )
            offline = verify_writer_terminalization_artifact_for_facts(
                facts=facts,
                runtime_options=options,
                artifact=artifact,
            )
            test.assertTrue(structural.accepted, structural.reason)
            test.assertTrue(live.accepted, live.reason)
            test.assertTrue(offline.accepted, offline.reason)
            test.assertEqual(offline.unchecked_obligation_families, ())


def _decoder_support(decoder) -> tuple[str, ...]:
    pending = [decoder]
    values = []
    while pending:
        state = pending.pop()
        if state.is_terminal:
            values.append(state.prefix)
        pending.extend(choice.next_state for choice in state.next_choices)
    return tuple(sorted(values))


def _support_digest(strings: tuple[str, ...]) -> str:
    return hashlib.sha256(
        json.dumps(
            strings,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
    ).hexdigest()


if __name__ == "__main__":
    unittest.main()
