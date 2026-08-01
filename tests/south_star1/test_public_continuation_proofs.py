"""Public molecule-bound continuation proof sessions."""

from __future__ import annotations

from dataclasses import replace
import json
import os
from pathlib import Path
import shutil
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import grimace
from rdkit import Chem

from grimace._south_star1.writer_terminalization_artifact import (
    _writer_terminalization_artifact_and_live_verification_for_selected_support,
)
from grimace._south_star1.writer_snapshot_prefix_envelope import (
    _terminal_support_identity_envelope_from_certificate,
)
from grimace._south_star1.writer_envelope_work import (
    default_writer_envelope_work_budget,
)

from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.default_writer_qualification_shards import FAST_ACCEPTED_CASES
from tests.south_star1.default_writer_qualification_shards import SLOW_COUPLED_CASES


class PublicContinuationProofTest(unittest.TestCase):
    def _assert_cases_expose_and_verify_every_local_proof(self, cases) -> None:
        for case in cases:
            with self.subTest(case=case.name), TemporaryDirectory() as directory:
                mol = Chem.MolFromSmiles(case.smiles)
                path = Path(directory) / "asset"
                digest = grimace.BuildMolToSmilesContinuationAsset(
                    mol,
                    path,
                    rootedAtAtom=case.rooted_at_atom,
                )
                decoder = grimace.MolToSmilesContinuationDecoder.from_asset(
                    path,
                    expected_manifest_digest=digest,
                    proof_capable=True,
                    mol=mol,
                )
                branch_count, terminal_count = _verify_all_public_proofs(decoder)
                self.assertGreater(branch_count, 0)
                self.assertGreater(terminal_count, 0)

    def test_fast_cases_expose_and_verify_every_local_proof(self) -> None:
        self._assert_cases_expose_and_verify_every_local_proof(FAST_ACCEPTED_CASES)

    @unittest.skipUnless(
        os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1",
        "set SOUTH_STAR1_RUN_SLOW=1 to run coupled cases",
    )
    def test_slow_coupled_cases_expose_and_verify_every_local_proof(self) -> None:
        self._assert_cases_expose_and_verify_every_local_proof(SLOW_COUPLED_CASES)

    def test_copy_and_snapshot_resume_share_the_molecule_bound_session(self) -> None:
        with TemporaryDirectory() as directory:
            mol = Chem.MolFromSmiles("F/C=C/Cl")
            path = Path(directory) / "asset"
            digest = grimace.BuildMolToSmilesContinuationAsset(
                mol,
                path,
                rootedAtAtom=0,
            )
            decoder = grimace.MolToSmilesContinuationDecoder.from_asset(
                path,
                expected_manifest_digest=digest,
                proof_capable=True,
                mol=mol,
            )
            copied = decoder.copy()
            self.assertEqual(copied.branch_proof_locators, decoder.branch_proof_locators)
            advanced = decoder.next_choices[0].next_state
            resumed = grimace.MolToSmilesContinuationDecoder.from_snapshot(
                path,
                advanced.snapshot(),
                proof_capable=True,
                mol=mol,
            )
            self.assertEqual(resumed.cache_key(), advanced.cache_key())
            self.assertEqual(
                resumed.branch_proof_locators,
                advanced.branch_proof_locators,
            )

    def test_original_and_renumbered_stereo_bind_only_their_own_assets(self) -> None:
        with TemporaryDirectory() as directory:
            original = Chem.MolFromSmiles("[C@H](F)(Cl)Br")
            order = tuple(reversed(range(original.GetNumAtoms())))
            renumbered = Chem.RenumberAtoms(original, list(order))
            original_bytes = original.ToBinary()
            renumbered_bytes = renumbered.ToBinary()
            original_path = Path(directory) / "original"
            renumbered_path = Path(directory) / "renumbered"
            original_digest = grimace.BuildMolToSmilesContinuationAsset(
                original,
                original_path,
                rootedAtAtom=0,
            )
            renumbered_digest = grimace.BuildMolToSmilesContinuationAsset(
                renumbered,
                renumbered_path,
                rootedAtAtom=order.index(0),
            )
            grimace.MolToSmilesContinuationDecoder.from_asset(
                original_path,
                expected_manifest_digest=original_digest,
                proof_capable=True,
                mol=original,
            )
            grimace.MolToSmilesContinuationDecoder.from_asset(
                renumbered_path,
                expected_manifest_digest=renumbered_digest,
                proof_capable=True,
                mol=renumbered,
            )
            for path, digest, wrong in (
                (original_path, original_digest, renumbered),
                (renumbered_path, renumbered_digest, original),
            ):
                with self.assertRaises(grimace.SouthStarError):
                    grimace.MolToSmilesContinuationDecoder.from_asset(
                        path,
                        expected_manifest_digest=digest,
                        proof_capable=True,
                        mol=wrong,
                    )
            self.assertEqual(original.ToBinary(), original_bytes)
            self.assertEqual(renumbered.ToBinary(), renumbered_bytes)

    def test_locators_are_bound_to_the_exact_asset_and_state(self) -> None:
        with TemporaryDirectory() as directory:
            mol = Chem.MolFromSmiles("CCO")
            first_path = Path(directory) / "first"
            second_path = Path(directory) / "second"
            first_digest = grimace.BuildMolToSmilesContinuationAsset(
                mol, first_path, rootedAtAtom=0
            )
            grimace.BuildMolToSmilesContinuationAsset(mol, second_path)
            first = grimace.MolToSmilesContinuationDecoder.from_asset(
                first_path,
                expected_manifest_digest=first_digest,
                proof_capable=True,
                mol=mol,
            )
            second = grimace.MolToSmilesContinuationDecoder.from_asset(
                second_path,
                proof_capable=True,
                mol=mol,
            )
            branch = first.branch_proof_locators[0]
            self.assertIsInstance(branch, grimace.MolToSmilesBranchProofLocator)
            with self.assertRaises(grimace.SouthStarError):
                second.branch_artifact(branch)
            successor = first.next_choices[0].next_state
            with self.assertRaises(grimace.SouthStarError):
                successor.branch_artifact(branch)
            for forged in (
                replace(branch, emitted_text=branch.emitted_text + "x"),
                replace(branch, branch_certificate_digest="0" * 64),
            ):
                with self.assertRaises(grimace.SouthStarError):
                    first.branch_artifact(forged)

            terminal_state = _first_terminal_state(first)
            terminal = terminal_state.terminal_proof_locators[0]
            self.assertIsInstance(
                terminal,
                grimace.MolToSmilesTerminalProofLocator,
            )
            with self.assertRaises(grimace.SouthStarError):
                terminal_state.terminalization_artifact(
                    replace(terminal, terminal_support_identity_digest="0" * 64)
                )

    def test_proof_snapshot_requires_the_exact_molecule_binding(self) -> None:
        with TemporaryDirectory() as directory:
            mol = Chem.MolFromSmiles("CCO")
            other = Chem.MolFromSmiles("OCC")
            path = Path(directory) / "asset"
            other_root_path = Path(directory) / "other-root"
            digest = grimace.BuildMolToSmilesContinuationAsset(
                mol, path, rootedAtAtom=0
            )
            grimace.BuildMolToSmilesContinuationAsset(
                mol, other_root_path, rootedAtAtom=1
            )
            decoder = grimace.MolToSmilesContinuationDecoder.from_asset(
                path,
                expected_manifest_digest=digest,
                proof_capable=True,
                mol=mol,
            )
            snapshot = decoder.next_choices[0].next_state.snapshot()
            with self.assertRaises(grimace.SouthStarError):
                grimace.MolToSmilesContinuationDecoder.from_snapshot(
                    path,
                    snapshot,
                    proof_capable=True,
                )
            with self.assertRaises(grimace.SouthStarError):
                grimace.MolToSmilesContinuationDecoder.from_snapshot(
                    path,
                    snapshot,
                    proof_capable=True,
                    mol=other,
                )
            with self.assertRaises(grimace.SouthStarError):
                grimace.MolToSmilesContinuationDecoder.from_snapshot(
                    other_root_path,
                    snapshot,
                    proof_capable=True,
                    mol=mol,
                )

    def test_wrong_manifest_and_dual_bindings_reject_typed(self) -> None:
        with TemporaryDirectory() as directory:
            mol = Chem.MolFromSmiles("CCO")
            path = Path(directory) / "asset"
            grimace.BuildMolToSmilesContinuationAsset(mol, path)
            with self.assertRaises(grimace.SouthStarError):
                grimace.MolToSmilesContinuationDecoder.from_asset(
                    path,
                    expected_manifest_digest="0" * 64,
                    proof_capable=True,
                    mol=mol,
                )
            with self.assertRaises(grimace.SouthStarError):
                grimace.MolToSmilesContinuationDecoder.from_asset(path, mol=mol)
            with self.assertRaises(grimace.SouthStarError):
                grimace.MolToSmilesContinuationDecoder.from_asset(
                    path,
                    proof_capable=True,
                    mol=mol,
                    prepared=object(),
                )

    def test_tampered_prepared_identity_prevents_proof_open(self) -> None:
        with TemporaryDirectory() as directory:
            mol = Chem.MolFromSmiles("CCO")
            original = Path(directory) / "original"
            forged = Path(directory) / "forged"
            grimace.BuildMolToSmilesContinuationAsset(mol, original)
            shutil.copytree(original, forged)
            manifest_path = forged / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["prepared_identity"]["digest"] = "0" * 64
            manifest_path.write_text(
                json.dumps(manifest, sort_keys=True, separators=(",", ":"))
            )
            with self.assertRaises(grimace.SouthStarError):
                grimace.MolToSmilesContinuationDecoder.from_asset(
                    forged,
                    proof_capable=True,
                    mol=mol,
                )

    def test_facts_rejection_prevents_public_artifact_access(self) -> None:
        with TemporaryDirectory() as directory:
            mol = Chem.MolFromSmiles("CCO")
            before = mol.ToBinary()
            path = Path(directory) / "asset"
            grimace.BuildMolToSmilesContinuationAsset(mol, path, rootedAtAtom=0)
            decoder = grimace.MolToSmilesContinuationDecoder.from_asset(
                path,
                proof_capable=True,
                mol=mol,
            )
            locator = decoder.branch_proof_locators[0]
            rejected = SimpleNamespace(
                accepted=False,
                reason="forced_public_branch_facts_rejection",
                unchecked_obligation_families=(),
            )
            with patch(
                "grimace._south_star1.writer_continuation_asset._verify_writer_branch_transition_artifact_for_facts_with_context",
                return_value=rejected,
            ):
                with self.assertRaises(grimace.SouthStarError):
                    decoder.branch_artifact(locator)
            terminal_state = _first_terminal_state(decoder)
            with patch(
                "grimace._south_star1.writer_continuation_asset._verify_writer_terminalization_artifact_for_facts_with_context",
                return_value=rejected,
            ):
                with self.assertRaises(grimace.SouthStarError):
                    terminal_state.terminalization_artifact(
                        terminal_state.terminal_proof_locators[0]
                    )
            self.assertEqual(mol.ToBinary(), before)

    def test_cc_dot_cc_terminal_proofs_are_individually_bound(self) -> None:
        with TemporaryDirectory() as directory:
            mol = Chem.MolFromSmiles("CC.CC")
            path = Path(directory) / "asset"
            digest = grimace.BuildMolToSmilesContinuationAsset(
                mol,
                path,
                rootedAtAtom=0,
            )
            decoder = grimace.MolToSmilesContinuationDecoder.from_asset(
                path,
                expected_manifest_digest=digest,
                proof_capable=True,
                mol=mol,
            )
            self.assertEqual(decoder.support_count, 1)
            self.assertEqual(decoder.completion_count, 2)
            state = _first_terminal_state(decoder)
            locators = state.terminal_proof_locators
            self.assertEqual(len(locators), 2)
            proof_state = state._state
            batch = proof_state.proof_session.batch(
                locators[0].source_raw_cursor_digest
            )

            artifacts = tuple(
                state.terminalization_artifact(locator) for locator in locators
            )
            self.assertEqual(
                tuple(
                    _terminal_identity_digest(artifact)
                    for artifact in artifacts
                ),
                tuple(
                    _terminal_support_identity_envelope_from_certificate(
                        batch.index.terminal_support_by_digest[
                            locator.terminal_support_identity_digest
                        ].checked_terminal_certificate,
                        budget=default_writer_envelope_work_budget(None),
                    )
                    for locator in locators
                ),
            )
            self.assertNotEqual(artifacts[0], artifacts[1])

            selected = batch.index.terminal_support_by_digest[
                locators[1].terminal_support_identity_digest
            ]
            reused, live = (
                _writer_terminalization_artifact_and_live_verification_for_selected_support(
                    prepared=proof_state.proof_session.prepared,
                    artifact=artifacts[0],
                    snapshot=batch.snapshot,
                    selected=selected,
                )
            )
            self.assertIsNone(reused)
            self.assertFalse(live.accepted)

            self.assertEqual(len(set(locators)), 2)

    def test_core_only_open_reads_no_proof_inputs(self) -> None:
        with TemporaryDirectory() as directory:
            mol = Chem.MolFromSmiles("CCO")
            path = Path(directory) / "asset"
            grimace.BuildMolToSmilesContinuationAsset(mol, path)
            with (
                patch(
                    "grimace._south_star1.writer_continuation_rust.prepare_public_continuation_molecule",
                    side_effect=AssertionError("molecule preparation invoked"),
                ),
                patch(
                    "grimace._south_star1.writer_continuation_asset.WriterContinuationAsset.records",
                    side_effect=AssertionError("provenance read"),
                ),
                patch(
                    "grimace._south_star1.writer_continuation_asset._source_snapshot_from_asset",
                    side_effect=AssertionError("source snapshot read"),
                ),
            ):
                decoder = grimace.MolToSmilesContinuationDecoder.from_asset(path)
                self.assertGreater(decoder.support_count, 0)
                self.assertTrue(decoder.next_choices)

    def test_locator_discovery_and_token_advance_do_not_replay_live_frontiers(self) -> None:
        with TemporaryDirectory() as directory:
            mol = Chem.MolFromSmiles("CCO")
            path = Path(directory) / "asset"
            grimace.BuildMolToSmilesContinuationAsset(mol, path, rootedAtAtom=0)
            decoder = grimace.MolToSmilesContinuationDecoder.from_asset(
                path,
                proof_capable=True,
                mol=mol,
            )
            with patch(
                "grimace._south_star1.writer_continuation_asset._frontier_batch",
                side_effect=AssertionError("live frontier replayed"),
            ):
                self.assertTrue(decoder.branch_proof_locators)
                self.assertEqual(decoder.terminal_proof_locators, ())
                advanced = decoder.next_choices[0].next_state
                self.assertTrue(advanced.branch_proof_locators)

    def test_proof_retrieval_uses_no_legacy_materialization_path(self) -> None:
        with TemporaryDirectory() as directory:
            mol = Chem.MolFromSmiles("CCO")
            path = Path(directory) / "asset"
            grimace.BuildMolToSmilesContinuationAsset(mol, path, rootedAtAtom=0)
            with (
                patch(
                    "grimace._runtime.mol_to_smiles_enum",
                    side_effect=AssertionError("legacy decoder invoked"),
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
                    side_effect=AssertionError("support materialization invoked"),
                ),
            ):
                decoder = grimace.MolToSmilesContinuationDecoder.from_asset(
                    path,
                    proof_capable=True,
                    mol=mol,
                )
                decoder.branch_artifact(decoder.branch_proof_locators[0])
                terminal = _first_terminal_state(decoder)
                terminal.terminalization_artifact(
                    terminal.terminal_proof_locators[0]
                )

    def test_multiple_locators_share_one_live_frontier_batch(self) -> None:
        from grimace._south_star1 import writer_continuation_rust

        with TemporaryDirectory() as directory:
            mol = Chem.MolFromSmiles("[C@H](F)(Cl)Br")
            path = Path(directory) / "asset"
            grimace.BuildMolToSmilesContinuationAsset(mol, path, rootedAtAtom=0)
            decoder = grimace.MolToSmilesContinuationDecoder.from_asset(
                path,
                proof_capable=True,
                mol=mol,
            )
            self.assertEqual(len(decoder.branch_proof_locators), 2)
            with patch.object(
                writer_continuation_rust,
                "_continuation_asset_proof_batch",
                wraps=writer_continuation_rust._continuation_asset_proof_batch,
            ) as build_batch:
                for locator in decoder.branch_proof_locators:
                    decoder.branch_artifact(locator)
            self.assertEqual(build_batch.call_count, 1)

    def test_one_facts_context_is_shared_across_the_proof_session(self) -> None:
        from grimace._south_star1 import writer_continuation_rust

        with TemporaryDirectory() as directory:
            mol = Chem.MolFromSmiles("CCO")
            path = Path(directory) / "asset"
            grimace.BuildMolToSmilesContinuationAsset(mol, path, rootedAtAtom=0)
            with patch.object(
                writer_continuation_rust,
                "_writer_facts_replay_context",
                wraps=writer_continuation_rust._writer_facts_replay_context,
            ) as build_context:
                decoder = grimace.MolToSmilesContinuationDecoder.from_asset(
                    path,
                    proof_capable=True,
                    mol=mol,
                )
                decoder.branch_artifact(decoder.branch_proof_locators[0])
                terminal = _first_terminal_state(decoder)
                terminal.terminalization_artifact(
                    terminal.terminal_proof_locators[0]
                )
            self.assertEqual(build_context.call_count, 1)


def _verify_all_public_proofs(decoder) -> tuple[int, int]:
    pending = [decoder]
    visited = set()
    branch_count = 0
    terminal_count = 0
    while pending:
        state = pending.pop()
        if state.cache_key() in visited:
            continue
        visited.add(state.cache_key())
        for locator in state.branch_proof_locators:
            artifact = state.branch_artifact(locator)
            if artifact["schema_name"] != "writer_branch_transition_artifact":
                raise AssertionError("unexpected public branch artifact")
            branch_count += 1
        for locator in state.terminal_proof_locators:
            artifact = state.terminalization_artifact(locator)
            if artifact["schema_name"] != "writer_terminalization_artifact":
                raise AssertionError("unexpected public terminal artifact")
            terminal_count += 1
        pending.extend(choice.next_state for choice in state.next_choices)
    return branch_count, terminal_count


def _first_terminal_state(decoder):
    pending = [decoder]
    while pending:
        state = pending.pop()
        if state.terminal_proof_locators:
            return state
        pending.extend(choice.next_state for choice in state.next_choices)
    raise AssertionError("no terminal proof locator")


def _terminal_identity_digest(artifact) -> dict[str, object]:
    objects = {item["object_id"]: item for item in artifact["objects"]}
    payload = objects[artifact["roots"]["terminal_support_ref"]]["payload"]
    return {
        key: payload[key]
        for key in (
            "source_state_digest",
            "finalized_state_digest",
            "parent_weight",
            "terminal_ordinal",
            "terminal_support_key_digest",
            "terminal_execution_capabilities_digest",
            "terminal_residual_work_evidence_digest",
            "terminal_stereo_lifecycle_evidence_digest",
            "graph_obligation_work_evidence_digest",
            "terminal_certificate_digests",
            "digest",
        )
    }


if __name__ == "__main__":
    unittest.main()
