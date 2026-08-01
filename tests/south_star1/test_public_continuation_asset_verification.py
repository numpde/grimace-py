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

from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.default_writer_qualification_shards import FAST_ACCEPTED_CASES
from tests.south_star1.default_writer_qualification_shards import SLOW_COUPLED_CASES
from tests.south_star1.default_writer_qualification_shards import (
    selected_slow_qualification_cases,
)


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
        self._assert_cases_recertify_copied_assets(selected_slow_qualification_cases())

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
