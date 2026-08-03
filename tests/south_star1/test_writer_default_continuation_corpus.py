"""Continuation/Rust tiers for every accepted default writer case."""

from __future__ import annotations

import os
import hashlib
import json
import time
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from grimace import MolToSmilesContinuationDecoder
from rdkit import Chem

from grimace._south_star1.fact_isomorphism import facts_are_isomorphic
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_rdkit
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.writer_continuation_asset import open_writer_continuation_core
from grimace._south_star1.writer_continuation_asset import verify_writer_continuation_asset_consistency
from grimace._south_star1.writer_continuation_asset import verify_writer_continuation_asset_for_prepared
from grimace._south_star1.writer_continuation_asset import write_writer_continuation_asset
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from tests.south_star1.default_writer_capability_ledger import ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
from tests.south_star1.qualification_plan import FAST_ACCEPTED_CASES
from tests.south_star1.qualification_plan import SLOW_COUPLED_CASES
from tests.south_star1.qualification_plan import (
    selected_slow_qualification_cases,
)
from tests.south_star1.qualification_plan import SLOW_COUPLED_CASE_NAMES
from tests.south_star1.slow_qualification_assets import require_slow_qualification_asset
from tests.south_star1.qualification_support import case_facts as _facts
from tests.south_star1.qualification_support import decoder_support_strings as _decoder_support
from tests.south_star1.qualification_support import support_image_for_case as _support_image
from tests.south_star1.qualification_support import support_strings_digest as _support_digest
from tests.south_star1.qualification_support import _writer_options


class WriterDefaultContinuationCorpusTest(unittest.TestCase):
    def test_accepted_default_shards_are_complete_and_deterministic(self) -> None:
        accepted_names = tuple(case.name for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES)
        self.assertEqual(tuple(case.name for case in FAST_ACCEPTED_CASES), tuple(name for name in accepted_names if name not in SLOW_COUPLED_CASE_NAMES))
        self.assertEqual(tuple(case.name for case in SLOW_COUPLED_CASES), tuple(name for name in accepted_names if name in SLOW_COUPLED_CASE_NAMES))

    def _run_cases(self):
        yield from FAST_ACCEPTED_CASES

    def _cross_all_continuation_tiers(self, cases) -> None:
        for case in cases:
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
                live = verify_writer_continuation_asset_for_prepared(
                    prepared=prepared,
                    asset=asset,
                )
                self.assertEqual(structural.accepted, case.expected_continuation_asset_complete)
                self.assertTrue(live.accepted, live.reason)
                self.assertTrue(live.structurally_verified)
                self.assertTrue(live.live_replay_complete)
                self.assertEqual(live.branch_locator_count, live.branch_proof_count)
                self.assertEqual(live.terminal_locator_count, live.terminal_proof_count)
                self.assertEqual(live.unchecked_obligation_families, ())

                if case.name in {"remote_coupled_tetrahedral_a", "remote_coupled_tetrahedral_b"}:
                    self.assertEqual(live.raw_cursor_count, 3075)
                    self.assertEqual(live.edge_locator_count, 3074)
                    self.assertEqual(live.branch_locator_count, 3848)
                    self.assertEqual(live.terminal_locator_count, 216)
                    self.assertEqual(live.terminal_record_count, 216)

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

    def test_every_fast_accepted_case_crosses_all_continuation_tiers(self) -> None:
        cases = tuple(self._run_cases())
        self.assertTrue(cases)
        self._cross_all_continuation_tiers(cases)

    def test_slow_coupled_cases_cross_all_continuation_tiers(self) -> None:
        if os.environ.get("SOUTH_STAR1_RUN_SLOW") != "1":
            self.skipTest("set SOUTH_STAR1_RUN_SLOW=1 to run coupled cases")
        self._cross_cached_continuation_tiers(selected_slow_qualification_cases())

    def _cross_cached_continuation_tiers(self, cases) -> None:
        for case in cases:
            with self.subTest(case=case.name):
                cache_started = time.monotonic()
                cached = require_slow_qualification_asset(case)
                cache_validation_seconds = time.monotonic() - cache_started
                prepared_started = time.monotonic()
                options = _writer_options(case.rooted_at_atom)
                facts = _facts(case)
                prepared = prepare_south_star_mol_from_facts(
                    facts,
                    writer_surface=SouthStarWriterSurface(),
                )
                capture_writer_frontier_snapshot(
                    prepared=prepared,
                    runtime_options=options,
                    cursor=initial_writer_frontier_cursor(prepared, options),
                )
                prepared_reconstruction_seconds = time.monotonic() - prepared_started
                asset = open_writer_continuation_core(cached.asset_path)
                live_started = time.monotonic()
                structural = verify_writer_continuation_asset_consistency(asset.path)
                live = verify_writer_continuation_asset_for_prepared(
                    prepared=prepared,
                    asset=asset,
                )
                live_replay_seconds = time.monotonic() - live_started
                self.assertEqual(structural.accepted, case.expected_continuation_asset_complete)
                self.assertTrue(live.accepted, live.reason)
                self.assertTrue(live.structurally_verified)
                self.assertTrue(live.live_replay_complete)
                self.assertEqual(live.branch_locator_count, live.branch_proof_count)
                self.assertEqual(live.terminal_locator_count, live.terminal_proof_count)
                self.assertEqual(live.unchecked_obligation_families, ())
                if case.name in {"remote_coupled_tetrahedral_a", "remote_coupled_tetrahedral_b"}:
                    self.assertEqual(live.raw_cursor_count, 3075)
                    self.assertEqual(live.edge_locator_count, 3074)
                    self.assertEqual(live.branch_locator_count, 3848)
                    self.assertEqual(live.terminal_locator_count, 216)
                    self.assertEqual(live.terminal_record_count, 216)

                rust_started = time.monotonic()
                with patch(
                    "grimace.BuildMolToSmilesContinuationAsset",
                    side_effect=AssertionError("public asset build invoked"),
                ), patch(
                    "grimace._south_star1.writer_continuation_asset.write_writer_continuation_asset",
                    side_effect=AssertionError("asset writer invoked"),
                ):
                    decoder = MolToSmilesContinuationDecoder.from_asset(
                        cached.asset_path,
                        expected_manifest_digest=cached.manifest_digest,
                    )
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
                        cached.asset_path,
                        advanced.snapshot(),
                    )
                    self.assertEqual(resumed.cache_key(), advanced.cache_key())
                    proof_decoder = MolToSmilesContinuationDecoder.from_asset(
                        cached.asset_path,
                        proof_capable=True,
                        prepared=prepared,
                    )
                    self.assertIsNotNone(proof_decoder._state.proof_cursor)
                rust_decoder_seconds = time.monotonic() - rust_started
                print(f"cache_validation_seconds={cache_validation_seconds:.3f}", flush=True)
                print(f"prepared_reconstruction_seconds={prepared_reconstruction_seconds:.3f}", flush=True)
                print(f"live_replay_seconds={live_replay_seconds:.3f}", flush=True)
                print(f"rust_decoder_seconds={rust_decoder_seconds:.3f}", flush=True)

    def test_renumbered_rdkit_stereo_keeps_certified_rust_language(self) -> None:
        cases = (
            ("[C@H](F)(Cl)Br", 0, None),
            ("C[C@H](F)Cl", 0, None),
            ("[C@H](F)(Cl)C[C@@H](Br)I", 0, None),
            # Fixed-order composition permits renumbering within a component;
            # fragment reordering is a distinct product operation.
            ("[C@H](F)(Cl)Br.O", 0, (3, 2, 1, 0, 4)),
            ("F/C=C/Cl", 0, None),
            ("F/C=C\\Cl", 0, None),
            ("CCO", 0, None),
        )
        for source, root, requested_order in cases:
            with self.subTest(source=source), TemporaryDirectory() as directory:
                mol = Chem.MolFromSmiles(source)
                order = requested_order or tuple(
                    reversed(range(mol.GetNumAtoms()))
                )
                renumbered = Chem.RenumberAtoms(mol, list(order))
                renumbered_root = order.index(root)
                original_facts = ordinary_molecule_facts_from_rdkit(mol)
                renumbered_facts = ordinary_molecule_facts_from_rdkit(renumbered)
                comparison = facts_are_isomorphic(
                    original_facts,
                    renumbered_facts,
                )
                self.assertTrue(comparison.isomorphic, comparison.reason)

                original = _certified_rdkit_support(
                    facts=original_facts,
                    root=root,
                    path=Path(directory) / "original",
                )
                changed = _certified_rdkit_support(
                    facts=renumbered_facts,
                    root=renumbered_root,
                    path=Path(directory) / "renumbered",
                )
                self.assertEqual(changed, original)
                for text in original[0]:
                    reparsed = ordinary_molecule_facts_from_smiles(text)
                    reparsed_comparison = facts_are_isomorphic(
                        original_facts,
                        reparsed,
                    )
                    self.assertTrue(
                        reparsed_comparison.isomorphic,
                        (text, reparsed_comparison.reason),
                    )


def _certified_rdkit_support(*, facts, root: int, path: Path):
    options = _writer_options(root)
    prepared = prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )
    snapshot = capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=initial_writer_frontier_cursor(prepared, options),
    )
    write_writer_continuation_asset(
        path=path,
        prepared=prepared,
        snapshot=snapshot,
    )
    asset = open_writer_continuation_core(path)
    structural = verify_writer_continuation_asset_consistency(path)
    live = verify_writer_continuation_asset_for_prepared(
        prepared=prepared,
        asset=asset,
    )
    if not structural.accepted or not live.accepted:
        raise AssertionError((structural.reason, live.reason))
    decoder = MolToSmilesContinuationDecoder.from_asset(path)
    return (
        _decoder_support(decoder),
        decoder.support_count,
        decoder.completion_count,
    )


if __name__ == "__main__":
    unittest.main()
