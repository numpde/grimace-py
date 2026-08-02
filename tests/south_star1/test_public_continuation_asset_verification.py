"""Public whole-asset continuation recertification regressions."""

from __future__ import annotations

from pathlib import Path
import os
import shutil
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import grimace
from rdkit import Chem

from grimace._south_star1 import writer_count_dag_envelope
from grimace._south_star1 import writer_continuation_asset
from grimace._south_star1 import writer_frontier_count_envelope
from grimace._south_star1 import writer_snapshot
from grimace._south_star1 import writer_support
from grimace._south_star1 import writer_support_artifact_envelope

from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.default_writer_qualification_shards import FAST_ACCEPTED_CASES
from tests.south_star1.default_writer_qualification_shards import SLOW_COUPLED_CASES
from tests.south_star1.default_writer_qualification_shards import (
    selected_slow_qualification_cases,
)
from tests.south_star1.slow_qualification_assets import (
    require_slow_qualification_asset,
)
from tests.south_star1.test_continuation_qualification_contract import (
    assert_continuation_recertification_matches_case,
)
import time


class PublicContinuationAssetVerificationTest(unittest.TestCase):
    def _assert_cases_recertify_copied_assets(self, cases) -> None:
        for case in cases:
            with self.subTest(case=case.name), TemporaryDirectory() as directory:
                source = Path(directory) / "source"
                copied = Path(directory) / "copied"
                mol = Chem.MolFromSmiles(case.smiles)
                digest = grimace.BuildMolToSmilesContinuationAsset(
                    mol,
                    source,
                    rootedAtAtom=case.rooted_at_atom,
                )
                shutil.copytree(source, copied)
                report = grimace.VerifyMolToSmilesContinuationAsset(
                    mol,
                    copied,
                    expected_manifest_digest=digest,
                )
                self.assertTrue(report.accepted)
                self.assertTrue(report.live_replay_complete)
                self.assertEqual(report.unchecked_obligation_families, ())
                self.assertEqual(
                    report.branch_locator_count,
                    report.branch_proof_count,
                )
                self.assertEqual(
                    report.terminal_locator_count,
                    report.terminal_proof_count,
                )

    def test_fast_cases_recertify_copied_assets(self) -> None:
        self._assert_cases_recertify_copied_assets(FAST_ACCEPTED_CASES)

    @unittest.skipUnless(
        os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1",
        "set SOUTH_STAR1_RUN_SLOW=1 to run coupled cases",
    )
    def test_slow_coupled_cases_recertify_copied_assets(self) -> None:
        for case in selected_slow_qualification_cases():
            with self.subTest(case=case.name), TemporaryDirectory() as directory:
                cache_started = time.monotonic()
                cached = require_slow_qualification_asset(case)
                cache_validation_seconds = time.monotonic() - cache_started
                copied = Path(directory) / "copied"
                copy_started = time.monotonic()
                shutil.copytree(cached.asset_path, copied)
                copy_seconds = time.monotonic() - copy_started
                before = _bundle_bytes(copied)
                recert_started = time.monotonic()
                with (
                    patch.object(
                        grimace,
                        "BuildMolToSmilesContinuationAsset",
                        side_effect=AssertionError("public asset build invoked"),
                    ),
                    patch.object(
                        writer_count_dag_envelope,
                        "writer_count_certificate_dag_envelope_for_product",
                        side_effect=AssertionError("count DAG invoked"),
                    ),
                    patch.object(
                        writer_frontier_count_envelope,
                        "writer_frontier_count_envelope_for_snapshot",
                        side_effect=AssertionError("count envelope invoked"),
                    ),
                    patch.object(
                        writer_support_artifact_envelope,
                        "writer_support_artifact_envelope_for_snapshot",
                        side_effect=AssertionError("rich support artifact invoked"),
                    ),
                    patch.object(
                        writer_support_artifact_envelope,
                        "_writer_support_artifact_envelope_for_snapshot_with_count_envelope",
                        side_effect=AssertionError("cached rich support artifact invoked"),
                    ),
                    patch.object(
                        writer_snapshot,
                        "_iter_writer_snapshot_certified_support_strings",
                        side_effect=AssertionError("support strings materialized"),
                    ),
                    patch.object(
                        writer_support,
                        "enumerate_prepared_writer_shaped_support",
                        side_effect=AssertionError("legacy support enumeration invoked"),
                    ),
                    patch.object(
                        writer_continuation_asset,
                        "write_writer_continuation_asset",
                        side_effect=AssertionError("asset writer invoked"),
                    ),
                ):
                    report = grimace.VerifyMolToSmilesContinuationAsset(
                        Chem.MolFromSmiles(case.smiles),
                        copied,
                        expected_manifest_digest=cached.manifest_digest,
                    )
                public_recertification_seconds = time.monotonic() - recert_started
                assert_continuation_recertification_matches_case(
                    self, case=case, report=report
                )
                self.assertEqual(_bundle_bytes(copied), before)
                print(f"cache_validation_seconds={cache_validation_seconds:.3f}", flush=True)
                print(f"copy_seconds={copy_seconds:.3f}", flush=True)
                print(
                    f"public_recertification_seconds={public_recertification_seconds:.3f}",
                    flush=True,
                )
                for name in (
                    "raw_cursor_count",
                    "edge_locator_count",
                    "branch_locator_count",
                    "branch_proof_count",
                    "terminal_record_count",
                    "terminal_locator_count",
                    "terminal_proof_count",
                    "semantically_replayed_operations",
                    "checked_relation_families",
                    "checked_obligation_families",
                    "unchecked_obligation_families",
                ):
                    print(f"{name}={getattr(report, name)}", flush=True)

    def test_copied_asset_is_recertified_without_mutation(self) -> None:
        with TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            copied = Path(directory) / "copied"
            mol = Chem.MolFromSmiles("CCO")
            digest = grimace.BuildMolToSmilesContinuationAsset(
                mol, source, rootedAtAtom=0
            )
            shutil.copytree(source, copied)
            before = _bundle_bytes(copied)
            report = grimace.VerifyMolToSmilesContinuationAsset(
                mol,
                copied,
                expected_manifest_digest=digest,
            )
            repeated = grimace.VerifyMolToSmilesContinuationAsset(
                mol,
                copied,
                expected_manifest_digest=digest,
            )
            self.assertIsInstance(
                report, grimace.MolToSmilesContinuationAssetVerification
            )
            self.assertTrue(report.accepted)
            self.assertEqual(report, repeated)
            self.assertEqual(report.manifest_digest, digest)
            self.assertTrue(report.live_replay_complete)
            self.assertEqual(report.unchecked_obligation_families, ())
            self.assertEqual(report.semantically_replayed_operations, ())
            self.assertEqual(report.checked_relation_families, ())
            self.assertEqual(
                report.checked_obligation_families,
                (
                    "graph_obligation_work",
                    "stereo_lifecycle",
                    "terminal_graph_obligation_work",
                    "terminal_stereo_lifecycle",
                ),
            )
            self.assertEqual(report.branch_locator_count, report.branch_proof_count)
            self.assertEqual(
                report.terminal_locator_count,
                report.terminal_proof_count,
            )
            self.assertEqual(_bundle_bytes(copied), before)

    def test_identity_and_manifest_mismatches_are_typed_before_replay(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            mol = Chem.MolFromSmiles("CCO")
            digest = grimace.BuildMolToSmilesContinuationAsset(
                mol, path, rootedAtAtom=0
            )
            with self.assertRaises(grimace.SouthStarError) as raised:
                grimace.VerifyMolToSmilesContinuationAsset(
                    mol,
                    path,
                    expected_manifest_digest="0" * 64,
                )
            self.assertIs(
                raised.exception.kind,
                grimace.SouthStarErrorKind.SEMANTIC_MISMATCH,
            )
            renumbered = Chem.RenumberAtoms(mol, [2, 1, 0])
            with self.assertRaises(grimace.SouthStarError) as raised:
                grimace.VerifyMolToSmilesContinuationAsset(
                    renumbered,
                    path,
                    expected_manifest_digest=digest,
                )
            self.assertIs(
                raised.exception.kind,
                grimace.SouthStarErrorKind.SEMANTIC_MISMATCH,
            )

            rooted_path = Path(directory) / "rooted"
            rooted_digest = grimace.BuildMolToSmilesContinuationAsset(
                mol, rooted_path, rootedAtAtom=1
            )
            with self.assertRaises(grimace.SouthStarError) as raised:
                grimace.VerifyMolToSmilesContinuationAsset(
                    mol,
                    rooted_path,
                    expected_manifest_digest=digest,
                )
            self.assertIs(
                raised.exception.kind,
                grimace.SouthStarErrorKind.SEMANTIC_MISMATCH,
            )
            report = grimace.VerifyMolToSmilesContinuationAsset(
                mol,
                rooted_path,
                expected_manifest_digest=rooted_digest,
            )
            self.assertEqual(report.manifest_digest, rooted_digest)

    def test_verifier_does_not_use_decoder_or_asset_writer_paths(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            mol = Chem.MolFromSmiles("CCO")
            digest = grimace.BuildMolToSmilesContinuationAsset(mol, path)
            with (
                patch(
                    "grimace._south_star1.writer_continuation_rust.MolToSmilesContinuationDecoder",
                    side_effect=AssertionError("decoder invoked"),
                ),
                patch(
                    "grimace._south_star1.writer_continuation_asset.write_writer_continuation_asset",
                    side_effect=AssertionError("asset writer invoked"),
                ),
                patch(
                    "grimace._south_star1.writer_support_artifact_envelope.writer_support_artifact_envelope_for_snapshot",
                    side_effect=AssertionError("rich support invoked"),
                ),
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
                    side_effect=AssertionError("support materializer invoked"),
                ),
            ):
                report = grimace.VerifyMolToSmilesContinuationAsset(
                    mol, path, expected_manifest_digest=digest
                )
            self.assertTrue(report.live_replay_complete)

    def test_facts_rejection_prevents_a_public_report(self) -> None:
        with TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            mol = Chem.MolFromSmiles("CCO")
            digest = grimace.BuildMolToSmilesContinuationAsset(mol, path)
            with patch(
                "grimace._south_star1.public_continuation_asset.verify_writer_continuation_asset_for_prepared",
                return_value=type(
                    "Rejected",
                    (),
                    {
                        "accepted": False,
                        "reason": "forced_asset_facts_rejection",
                    },
                )(),
            ):
                with self.assertRaisesRegex(
                    grimace.SouthStarError,
                    "forced_asset_facts_rejection",
                ):
                    grimace.VerifyMolToSmilesContinuationAsset(
                        mol, path, expected_manifest_digest=digest
                    )
            self.assertTrue(path.exists())


def _bundle_bytes(path: Path) -> tuple[tuple[str, bytes], ...]:
    return tuple(
        (str(item.relative_to(path)), item.read_bytes())
        for item in sorted(path.rglob("*"))
        if item.is_file()
    )


if __name__ == "__main__":
    unittest.main()
