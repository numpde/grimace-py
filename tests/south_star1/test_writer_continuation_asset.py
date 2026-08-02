"""Replay-addressed continuation asset regressions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from grimace import MolToSmilesContinuationDecoder
import grimace._south_star1.writer_branch_transition_artifact as branch_artifact_module
import grimace._south_star1.writer_continuation_asset as continuation_asset_module
import grimace._south_star1.writer_terminalization_artifact as terminal_artifact_module
from grimace._south_star1.errors import SouthStarError

from grimace._south_star1.writer_continuation_asset import (
    _certify_writer_continuation_asset_candidate,
)
from grimace._south_star1.writer_continuation_asset import (
    _materialize_writer_continuation_asset_candidate,
)
from grimace._south_star1.writer_continuation_asset import (
    advance_writer_continuation_proof,
)
from grimace._south_star1.writer_branch_transition_artifact_fact_verifier import (
    WriterBranchTransitionArtifactFactVerification,
)
from grimace._south_star1.writer_branch_transition_artifact_fact_verifier import (
    verify_writer_branch_transition_artifact_for_facts,
)
from grimace._south_star1.writer_continuation_asset import (
    branch_transition_artifact_from_continuation_asset,
)
from grimace._south_star1.writer_continuation_asset import (
    open_writer_continuation_core,
)
from grimace._south_star1.writer_continuation_asset import (
    terminalization_artifact_from_continuation_asset,
)
from grimace._south_star1.writer_continuation_asset import (
    verify_writer_continuation_asset_consistency,
)
from grimace._south_star1.writer_continuation_asset import (
    verify_writer_continuation_asset_live,
)
from grimace._south_star1.writer_continuation_asset import (
    verify_writer_continuation_asset_for_prepared,
)
from grimace._south_star1.writer_continuation_asset import (
    verify_writer_continuation_cursor_envelope,
)
from grimace._south_star1.writer_continuation_asset import (
    write_writer_continuation_asset,
)
from grimace._south_star1.writer_continuation_asset import (
    writer_continuation_cursor_envelope,
)
from grimace._south_star1.writer_continuation_automaton import (
    writer_continuation_choices,
)
from grimace._south_star1.writer_continuation_automaton import (
    writer_continuation_completion_count,
)
from grimace._south_star1.writer_continuation_automaton import (
    writer_continuation_support_count,
)
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_terminalization_artifact_fact_verifier import (
    WriterTerminalizationArtifactFactsVerification,
)
from grimace._south_star1.writer_terminalization_artifact_fact_verifier import (
    verify_writer_terminalization_artifact_for_facts,
)
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.test_writer_branch_transition_artifact import (
    _shared_ring_branch_sources,
)
from tests.south_star1.test_writer_stereo_residual import (
    _directional_non_single_ring_carrier_facts,
)
from tests.south_star1.test_writer_stereo_residual import (
    _directional_ring_carrier_facts,
)
from tests.south_star1.test_writer_stereo_residual import (
    _shared_directional_ring_carrier_facts,
)
from tests.south_star1.test_writer_stereo_residual import (
    terminal_tetra_center_facts,
)
from tests.south_star1.test_writer_stereo_residual import (
    terminal_tetra_center_policy,
)
from tests.south_star1.test_writer_terminalization_artifact import _terminal_source
from tests.south_star1.test_writer_support_artifact_fact_verifier import (
    _initial_snapshot,
)
from tests.south_star1.test_writer_support_artifact_fact_verifier import _prepare
from tests.south_star1.test_writer_support_artifact_fact_verifier import (
    _writer_options,
)


class _CallCounter:
    def __init__(self, target):
        self._target = target
        self.count = 0

    def __call__(self, *args, **kwargs):
        self.count += 1
        return self._target(*args, **kwargs)


class WriterContinuationAssetTest(unittest.TestCase):
    def test_staged_candidate_matches_public_composition_and_is_not_published(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared, _writer_options())
        with TemporaryDirectory() as directory:
            candidate = Path(directory) / "candidate"
            destination = Path(directory) / "asset"
            manifest = _materialize_writer_continuation_asset_candidate(
                path=candidate, prepared=prepared, snapshot=snapshot
            )
            self.assertTrue(open_writer_continuation_core(candidate).manifest_digest)
            self.assertFalse(destination.exists())
            certified = _certify_writer_continuation_asset_candidate(
                path=candidate,
                prepared=prepared,
                expected_manifest_digest=manifest["digest"],
            )
            self.assertTrue(certified.accepted)
            public_manifest = write_writer_continuation_asset(
                path=destination, prepared=prepared, snapshot=snapshot
            )
            self.assertEqual(manifest, public_manifest)
            self.assertEqual(_bundle_bytes(candidate), _bundle_bytes(destination))

    def test_public_composition_orders_materialize_certify_publish_and_cleans_failures(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared, _writer_options())
        with TemporaryDirectory() as directory:
            destination = Path(directory) / "asset"
            events = []
            original_materialize = continuation_asset_module._materialize_writer_continuation_asset_candidate
            original_certify = continuation_asset_module._certify_writer_continuation_asset_candidate
            original_replace = continuation_asset_module.os.replace

            def materialize(*args, **kwargs):
                events.append("materialize")
                return original_materialize(*args, **kwargs)

            def certify(*args, **kwargs):
                events.append("certify")
                return original_certify(*args, **kwargs)

            def publish(*args, **kwargs):
                if len(args) >= 2 and Path(args[1]) == destination:
                    events.append("publish")
                return original_replace(*args, **kwargs)

            with (
                patch.object(continuation_asset_module, "_materialize_writer_continuation_asset_candidate", materialize),
                patch.object(continuation_asset_module, "_certify_writer_continuation_asset_candidate", certify),
                patch.object(continuation_asset_module.os, "replace", publish),
            ):
                write_writer_continuation_asset(path=destination, prepared=prepared, snapshot=snapshot)
            self.assertEqual(events, ["materialize", "certify", "publish"])

            for phase in ("materialize", "certify", "publish"):
                failed = Path(directory) / f"{phase}-failure"
                target = {
                    "materialize": "grimace._south_star1.writer_continuation_asset._materialize_writer_continuation_asset_candidate",
                    "certify": "grimace._south_star1.writer_continuation_asset._certify_writer_continuation_asset_candidate",
                    "publish": "grimace._south_star1.writer_continuation_asset.os.replace",
                }[phase]
                with patch(target, side_effect=AssertionError(phase)):
                    with self.assertRaises(AssertionError):
                        write_writer_continuation_asset(
                            path=failed, prepared=prepared, snapshot=snapshot
                        )
                self.assertFalse(failed.exists())

    def test_asset_is_deterministic_core_first_and_lazily_provable(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared, _writer_options())
        with TemporaryDirectory() as directory:
            first = Path(directory) / "first"
            second = Path(directory) / "second"
            first_manifest = write_writer_continuation_asset(
                path=first,
                prepared=prepared,
                snapshot=snapshot,
            )
            second_manifest = write_writer_continuation_asset(
                path=second,
                prepared=prepared,
                snapshot=snapshot,
            )
            self.assertEqual(first_manifest, second_manifest)
            self.assertEqual(_bundle_bytes(first), _bundle_bytes(second))
            for descriptor_field in (
                "raw_cursor_chunks",
                "primitive_chunks",
                "edge_chunks",
                "terminal_chunks",
            ):
                for descriptor in first_manifest[descriptor_field]:
                    payload = (
                        first
                        / "chunks"
                        / f"{descriptor['digest']}.json"
                    ).read_text()
                    self.assertNotIn("WriterFrontierCursor", payload)
                    self.assertNotIn("WriterStateKey", payload)
            self.assertTrue(
                verify_writer_continuation_asset_consistency(first).accepted
            )

            original_read = __import__(
                "grimace._south_star1.writer_continuation_asset",
                fromlist=["_read_chunk"],
            )._read_chunk

            def core_chunks_only(path, descriptor):
                if descriptor["kind"] not in {
                    "source_snapshot",
                    "automaton_core",
                }:
                    raise AssertionError("provenance chunk read")
                return original_read(path, descriptor)

            with patch(
                "grimace._south_star1.writer_continuation_asset._read_chunk",
                side_effect=core_chunks_only,
            ):
                asset = open_writer_continuation_core(first)
                choices = writer_continuation_choices(asset.core)
                self.assertTrue(choices)

            asset = open_writer_continuation_core(first)
            semantic = verify_writer_continuation_asset_for_prepared(
                prepared=prepared,
                asset=asset,
            )
            self.assertTrue(semantic.accepted, semantic.reason)
            self.assertTrue(semantic.structurally_verified)
            self.assertTrue(semantic.live_replay_complete)
            self.assertEqual(
                semantic.branch_locator_count,
                semantic.branch_proof_count,
            )
            self.assertEqual(
                semantic.terminal_locator_count,
                semantic.terminal_proof_count,
            )
            self.assertEqual(semantic.unchecked_obligation_families, ())
            self.assertEqual(semantic.semantically_replayed_operations, ())
            self.assertEqual(semantic.checked_relation_families, ())
            self.assertEqual(
                semantic.checked_obligation_families,
                (
                    "graph_obligation_work",
                    "stereo_lifecycle",
                    "terminal_graph_obligation_work",
                    "terminal_stereo_lifecycle",
                ),
            )
            mismatched = verify_writer_continuation_asset_for_prepared(
                prepared=_prepare(_directional_non_single_ring_carrier_facts()),
                asset=open_writer_continuation_core(first),
            )
            self.assertFalse(mismatched.accepted)
            self.assertIn("prepared_identity_mismatch", mismatched.reason)
            root = asset.root_proof_cursor
            cursor_envelope = writer_continuation_cursor_envelope(
                asset=asset,
                cursor=root,
                raw_cursor_digest=root.raw_cursor_digest,
            )
            self.assertTrue(
                verify_writer_continuation_cursor_envelope(
                    asset=asset,
                    envelope=cursor_envelope,
                ).accepted
            )
            detached = dict(cursor_envelope)
            detached["asset_manifest_digest"] = "0" * 64
            unsigned = dict(detached)
            unsigned.pop("digest")
            detached["digest"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
            self.assertFalse(
                verify_writer_continuation_cursor_envelope(
                    asset=asset,
                    envelope=detached,
                ).accepted
            )
            choice = writer_continuation_choices(asset.core)[0]
            advanced = advance_writer_continuation_proof(
                asset, root, choice.emitted_text
            )
            self.assertNotEqual(advanced.raw_cursor_digest, root.raw_cursor_digest)
            edge = next(
                item
                for item in asset.records("edge_records")
                if item.source_raw_cursor_digest == root.raw_cursor_digest
                and item.emitted_text == choice.emitted_text
            )
            branch = branch_transition_artifact_from_continuation_asset(
                prepared=prepared,
                asset=asset,
                source_raw_cursor_digest=root.raw_cursor_digest,
                emitted_text=choice.emitted_text,
                branch_certificate_digest=edge.branch_certificate_digests[0],
            )
            self.assertEqual(branch["schema_name"], "writer_branch_transition_artifact")
            with self.assertRaises(SouthStarError):
                branch_transition_artifact_from_continuation_asset(
                    prepared=prepared,
                    asset=asset,
                    source_raw_cursor_digest=root.raw_cursor_digest,
                    emitted_text=choice.emitted_text,
                    branch_certificate_digest="0" * 64,
                )
            terminal = asset.records("terminal_records")[0]
            terminal_artifact = terminalization_artifact_from_continuation_asset(
                prepared=prepared,
                asset=asset,
                source_raw_cursor_digest=terminal.source_raw_cursor_digest,
                terminal_support_identity_digest=(
                    terminal.terminal_support_identity_digests[0]
                ),
            )
            self.assertEqual(
                terminal_artifact["schema_name"],
                "writer_terminalization_artifact",
            )
            with self.assertRaises(SouthStarError):
                terminalization_artifact_from_continuation_asset(
                    prepared=prepared,
                    asset=asset,
                    source_raw_cursor_digest=terminal.source_raw_cursor_digest,
                    terminal_support_identity_digest="0" * 64,
                )
            with patch(
                "grimace._south_star1.writer_continuation_asset._frontier_batch",
                wraps=continuation_asset_module._frontier_batch,
            ) as frontier_batch:
                live = verify_writer_continuation_asset_live(
                    prepared=prepared,
                    asset=asset,
                    full=True,
                )
            self.assertTrue(live.accepted, live.reason)
            self.assertEqual(
                frontier_batch.call_count,
                len(asset.records("raw_cursor_records")),
            )

    def test_structural_verifier_rejects_coherent_core_and_provenance_forgeries(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared, _writer_options())
        cases = (
            ("core_count", _forge_core_count),
            ("predecessor", _forge_predecessor),
            ("duplicate_record", _forge_duplicate_raw_record),
            ("raw_node", _forge_raw_node),
            ("primitive_representative", _forge_primitive_representative),
            ("edge_successor_node", _forge_edge_successor_node),
            ("terminal_node", _forge_terminal_node),
            ("chunk_key_range", _forge_chunk_key_range),
        )
        for name, forge in cases:
            with self.subTest(name=name), TemporaryDirectory() as directory:
                path = Path(directory) / "asset"
                write_writer_continuation_asset(
                    path=path,
                    prepared=prepared,
                    snapshot=snapshot,
                )
                forge(path)
                checked = verify_writer_continuation_asset_consistency(path)
                self.assertFalse(checked.accepted)
                self.assertIn("continuation", checked.reason)
                if name == "core_count":
                    with self.assertRaises(SouthStarError):
                        open_writer_continuation_core(path)

    def test_live_verifier_rejects_coherent_projection_substitution(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared, _writer_options())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            write_writer_continuation_asset(
                path=path,
                prepared=prepared,
                snapshot=snapshot,
            )
            _forge_unused_projection_digest(path)
            structural = verify_writer_continuation_asset_consistency(path)
            self.assertTrue(structural.accepted, structural.reason)
            asset = open_writer_continuation_core(path)
            live = verify_writer_continuation_asset_live(
                prepared=prepared,
                asset=asset,
                full=True,
            )
            self.assertFalse(live.accepted)
            self.assertIn("live_projection_mismatch", live.reason)

    def test_semantic_certifier_rejects_coherent_locator_substitution(self) -> None:
        cases = (
            (
                cco_facts(),
                _writer_options(),
                _forge_branch_digest_transplant,
            ),
            (
                directional_facts(),
                _writer_options(rooted_at_atom=2),
                _forge_terminal_identity_transplant,
            ),
        )
        for facts, options, forge in cases:
            with (
                self.subTest(forge=forge.__name__),
                TemporaryDirectory() as directory,
            ):
                prepared = _prepare(facts)
                snapshot = _initial_snapshot(prepared, options)
                path = Path(directory) / "asset"
                write_writer_continuation_asset(
                    path=path,
                    prepared=prepared,
                    snapshot=snapshot,
                )
                forge(path)
                structural = verify_writer_continuation_asset_consistency(path)
                self.assertTrue(structural.accepted, structural.reason)
                semantic = verify_writer_continuation_asset_for_prepared(
                    prepared=prepared,
                    asset=open_writer_continuation_core(path),
                )
                self.assertFalse(semantic.accepted)
                self.assertTrue(semantic.structurally_verified)
                self.assertIn("continuation_asset", semantic.reason)

    def test_publication_requires_branch_and_terminal_facts_proofs(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared, _writer_options())
        cases = (
            (
                "_verify_writer_branch_transition_artifact_for_facts_with_context",
                WriterBranchTransitionArtifactFactVerification(
                    accepted=False,
                    reason="forced_facts_rejection",
                ),
            ),
            (
                "_verify_writer_terminalization_artifact_for_facts_with_context",
                WriterTerminalizationArtifactFactsVerification(
                    accepted=False,
                    reason="forced_facts_rejection",
                ),
            ),
        )
        for verifier_name, rejection in cases:
            with (
                self.subTest(verifier=verifier_name),
                TemporaryDirectory() as directory,
            ):
                path = Path(directory) / "asset"
                with patch.object(
                    continuation_asset_module,
                    verifier_name,
                    return_value=rejection,
                ):
                    with self.assertRaises(SouthStarError) as raised:
                        write_writer_continuation_asset(
                            path=path,
                            prepared=prepared,
                            snapshot=snapshot,
                        )
                self.assertIn("forced_facts_rejection", str(raised.exception))
                self.assertFalse(path.exists())
                self.assertEqual(
                    tuple(Path(directory).glob(f".{path.name}.*")),
                    (),
                )

    def test_certification_uses_no_rich_support_or_count_path(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared, _writer_options())
        blocked = AssertionError("legacy materialization path invoked")
        with (
            TemporaryDirectory() as directory,
            patch(
                "grimace._south_star1.writer_support_artifact_envelope."
                "writer_support_artifact_envelope_for_snapshot",
                side_effect=blocked,
            ),
            patch(
                "grimace._south_star1.writer_frontier_count_envelope."
                "writer_frontier_count_envelope_for_snapshot",
                side_effect=blocked,
            ),
            patch(
                "grimace._south_star1.writer_count_dag_envelope."
                "writer_count_certificate_dag_envelope_for_product",
                side_effect=blocked,
            ),
            patch(
                "grimace._south_star1.writer_snapshot."
                "_iter_writer_snapshot_certified_support_strings",
                side_effect=blocked,
            ),
        ):
            path = Path(directory) / "asset"
            write_writer_continuation_asset(
                path=path,
                prepared=prepared,
                snapshot=snapshot,
            )
            self.assertTrue(path.is_dir())

    def test_non_single_certification_streams_each_locator_once(self) -> None:
        prepared = _prepare(_directional_non_single_ring_carrier_facts())
        snapshot = _initial_snapshot(
            prepared,
            _writer_options(rooted_at_atom=0),
        )
        counted_names = (
            "_frontier_batch",
            "_source_snapshot_from_asset",
            "_writer_facts_replay_context",
            "verify_writer_branch_transition_artifact_consistency",
            "verify_writer_terminalization_artifact_consistency",
            "_writer_branch_transition_artifact_and_live_verification_for_selected_support",
            "_writer_terminalization_artifact_and_live_verification_for_selected_support",
            "_verify_writer_branch_transition_artifact_for_facts_with_context",
            "_verify_writer_terminalization_artifact_for_facts_with_context",
        )
        counters = {
            name: _CallCounter(getattr(continuation_asset_module, name))
            for name in counted_names
        }
        patches = {
            name: patch.object(continuation_asset_module, name, new=counter)
            for name, counter in counters.items()
        }
        canonical_counters = {
            "canonical_branch_artifact": _CallCounter(
                branch_artifact_module._writer_branch_transition_artifact_for_prelocated_support
            ),
            "canonical_terminal_artifact": _CallCounter(
                terminal_artifact_module._writer_terminalization_artifact_for_prelocated_support
            ),
        }
        canonical_patches = {
            "canonical_branch_artifact": patch.object(
                branch_artifact_module,
                "_writer_branch_transition_artifact_for_prelocated_support",
                new=canonical_counters["canonical_branch_artifact"],
            ),
            "canonical_terminal_artifact": patch.object(
                terminal_artifact_module,
                "_writer_terminalization_artifact_for_prelocated_support",
                new=canonical_counters["canonical_terminal_artifact"],
            ),
        }
        with TemporaryDirectory() as directory:
            for item in (*canonical_patches.values(), *patches.values()):
                item.start()
            try:
                write_writer_continuation_asset(
                    path=Path(directory) / "asset",
                    prepared=prepared,
                    snapshot=snapshot,
                )
            finally:
                for item in (*patches.values(), *canonical_patches.values()):
                    item.stop()
        self.assertEqual(counters["_frontier_batch"].count, 456)
        self.assertEqual(counters["_source_snapshot_from_asset"].count, 1)
        self.assertEqual(counters["_writer_facts_replay_context"].count, 1)
        self.assertEqual(
            canonical_counters["canonical_branch_artifact"].count, 491
        )
        self.assertEqual(
            canonical_counters["canonical_terminal_artifact"].count, 72
        )
        for name in (
            "verify_writer_branch_transition_artifact_consistency",
            "_writer_branch_transition_artifact_and_live_verification_for_selected_support",
            "_verify_writer_branch_transition_artifact_for_facts_with_context",
        ):
            self.assertEqual(counters[name].count, 491)
        for name in (
            "verify_writer_terminalization_artifact_consistency",
            "_writer_terminalization_artifact_and_live_verification_for_selected_support",
            "_verify_writer_terminalization_artifact_for_facts_with_context",
        ):
            self.assertEqual(counters[name].count, 72)

    def test_missing_and_extra_chunks_reject(self) -> None:
        prepared = _prepare(cco_facts())
        snapshot = _initial_snapshot(prepared, _writer_options())
        with TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            manifest = write_writer_continuation_asset(
                path=path,
                prepared=prepared,
                snapshot=snapshot,
            )
            descriptor = manifest["edge_chunks"][0]
            (path / "chunks" / f"{descriptor['digest']}.json").unlink()
            self.assertFalse(
                verify_writer_continuation_asset_consistency(path).accepted
            )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            write_writer_continuation_asset(
                path=path,
                prepared=prepared,
                snapshot=snapshot,
            )
            manifest_path = path / "manifest.json"
            manifest = json.loads(manifest_path.read_text())
            manifest["schema_version"] = 0
            unsigned = dict(manifest)
            unsigned.pop("digest")
            manifest["digest"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
            manifest_path.write_bytes(_canonical(manifest))
            checked = verify_writer_continuation_asset_consistency(path)
            self.assertFalse(checked.accepted)
            self.assertIn("unknown_schema_version", checked.reason)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            write_writer_continuation_asset(
                path=path,
                prepared=prepared,
                snapshot=snapshot,
            )
            (path / "chunks" / "0.json").write_text("{}")
            self.assertFalse(
                verify_writer_continuation_asset_consistency(path).accepted
            )

    def test_terminalization_matrix_reconstructs_lazily(self) -> None:
        cases = (
            (cco_facts(), _writer_options(), None),
            (
                terminal_tetra_center_facts(),
                _writer_options(rooted_at_atom=0),
                terminal_tetra_center_policy(),
            ),
            (
                _directional_ring_carrier_facts(),
                _writer_options(rooted_at_atom=0),
                None,
            ),
            (
                _shared_directional_ring_carrier_facts(),
                _writer_options(rooted_at_atom=1),
                None,
            ),
            (
                _directional_non_single_ring_carrier_facts(),
                _writer_options(rooted_at_atom=0),
                None,
            ),
        )
        for facts, options, policy in cases:
            with self.subTest(facts=facts), TemporaryDirectory() as directory:
                prepared, snapshot, support = _terminal_source(
                    facts, options, policy
                )
                path = Path(directory) / "asset"
                write_writer_continuation_asset(
                    path=path,
                    prepared=prepared,
                    snapshot=snapshot,
                )
                asset = open_writer_continuation_core(path)
                terminal = asset.records("terminal_records")[0]
                decoder = _rust_decoder_at_raw_cursor(
                    path=path,
                    asset=asset,
                    prepared=prepared,
                    raw_cursor_digest=terminal.source_raw_cursor_digest,
                )
                artifact = decoder.terminalization_artifact(
                    terminal.terminal_support_identity_digests[0]
                )
                self.assertEqual(
                    artifact["schema_name"],
                    "writer_terminalization_artifact",
                )
                checked = verify_writer_terminalization_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=artifact,
                    policy=policy,
                )
                self.assertTrue(checked.accepted, checked.reason)

    @unittest.skipUnless(
        os.environ.get("SOUTH_STAR1_RUN_SLOW") == "1",
        "full shared-ring continuation asset is slow-gated",
    )
    def test_full_shared_root_metrics_and_six_lazy_branch_proofs(self) -> None:
        facts = _shared_directional_ring_carrier_facts()
        prepared = _prepare(facts)
        snapshot = _initial_snapshot(
            prepared,
            _writer_options(rooted_at_atom=1),
        )
        with TemporaryDirectory() as directory:
            path = Path(directory) / "asset"
            patches = (
                patch(
                    "grimace._south_star1.writer_frontier_count_envelope."
                    "writer_frontier_count_envelope_for_snapshot",
                    side_effect=AssertionError("count envelope path"),
                ),
                patch(
                    "grimace._south_star1.writer_count_dag_envelope."
                    "writer_count_certificate_dag_envelope_for_product",
                    side_effect=AssertionError("count DAG path"),
                ),
                patch(
                    "grimace._south_star1.writer_support_string_envelope."
                    "_iter_writer_snapshot_certified_support_strings",
                    side_effect=AssertionError("support materialization path"),
                ),
            )
            expected_calls = {
                "_frontier_batch": 19_595,
                "_source_snapshot_from_asset": 1,
                "_writer_facts_replay_context": 1,
                "verify_writer_branch_transition_artifact_consistency": 19_847,
                "verify_writer_terminalization_artifact_consistency": 3_744,
                "_writer_branch_transition_artifact_and_live_verification_for_selected_support": 19_847,
                "_writer_terminalization_artifact_and_live_verification_for_selected_support": 3_744,
                "_verify_writer_branch_transition_artifact_for_facts_with_context": 19_847,
                "_verify_writer_terminalization_artifact_for_facts_with_context": 3_744,
            }
            counters = {
                name: _CallCounter(getattr(continuation_asset_module, name))
                for name in expected_calls
            }
            with (
                patches[0],
                patches[1],
                patches[2],
                *(
                    patch.object(
                        continuation_asset_module,
                        name,
                        new=counter,
                    )
                    for name, counter in counters.items()
                ),
            ):
                manifest = write_writer_continuation_asset(
                    path=path,
                    prepared=prepared,
                    snapshot=snapshot,
                )
            for name, expected in expected_calls.items():
                self.assertEqual(counters[name].count, expected, name)
            metrics = manifest["deterministic_metrics"]
            self.assertEqual(metrics["semantic_node_count"], 2_101)
            self.assertEqual(metrics["semantic_edge_count"], 2_843)
            self.assertLess(metrics["core_chunk_bytes"], 25_000_000)
            self.assertLessEqual(
                metrics["compact_provenance_bytes"], 64_000_000
            )
            self.assertLessEqual(
                max(
                    item["canonical_bytes"]
                    for item in (
                        manifest["source_snapshot_chunk"],
                        manifest["core_chunk"],
                        *manifest["raw_cursor_chunks"],
                        *manifest["primitive_chunks"],
                        *manifest["edge_chunks"],
                        *manifest["terminal_chunks"],
                    )
                ),
                4_000_000,
            )
            asset = open_writer_continuation_core(path)
            self.assertEqual(writer_continuation_support_count(asset.core), 3_744)
            self.assertEqual(
                writer_continuation_completion_count(asset.core), 3_744
            )
            rust_decoder = MolToSmilesContinuationDecoder.from_asset(
                path,
                proof_capable=True,
                prepared=prepared,
            )
            self.assertEqual(rust_decoder.support_count, 3_744)
            self.assertEqual(rust_decoder.completion_count, 3_744)
            self.assertLessEqual(rust_decoder.rust_resident_bytes, 16_000_000)
            self.assertEqual(
                _rust_decoder_strings(rust_decoder),
                _core_strings(asset.core),
            )
            sources = _shared_ring_branch_sources()
            for phase, mark in sources:
                _facts, _options, _prepared, source, support = sources[(phase, mark)]
                decoder = _rust_decoder_at_raw_cursor(
                    path=path,
                    asset=asset,
                    prepared=prepared,
                    raw_cursor_digest=_identity_digest(source.cursor),
                )
                artifact = decoder.branch_artifact(
                    _identity_digest(support.checked_branch_certificate)
                )
                self.assertEqual(
                    artifact["schema_name"],
                    "writer_branch_transition_artifact",
                )
                checked = verify_writer_branch_transition_artifact_for_facts(
                    facts=facts,
                    runtime_options=_options,
                    artifact=artifact,
                )
                self.assertTrue(checked.accepted, checked.reason)
                self.assertEqual(checked.unchecked_obligation_families, ())


def _rust_decoder_at_raw_cursor(
    *, path, asset, prepared, raw_cursor_digest
):
    emitted_texts = []
    current = asset.raw_cursor_record(raw_cursor_digest)
    while current.predecessor_edge_id is not None:
        edge = asset.edge_record_by_id(current.predecessor_edge_id)
        emitted_texts.append(edge.emitted_text)
        current = asset.raw_cursor_record(edge.source_raw_cursor_digest)
    decoder = MolToSmilesContinuationDecoder.from_asset(
        path,
        proof_capable=True,
        prepared=prepared,
    )
    for text in reversed(emitted_texts):
        decoder = decoder.advance(text)
    if decoder._state.proof_cursor.raw_cursor_digest != raw_cursor_digest:
        raise AssertionError("Rust proof cursor replay ended at the wrong cursor")
    return decoder


def _rust_decoder_strings(decoder):
    values = []
    pending = [decoder]
    while pending:
        current = pending.pop()
        if current.is_terminal:
            values.append(current.prefix)
        pending.extend(choice.next_state for choice in current.next_choices)
    return tuple(sorted(values))


def _core_strings(core):
    memo = {}

    def visit(node_id):
        known = memo.get(node_id)
        if known is not None:
            return known
        node = core.nodes[node_id]
        values = [""] if node.terminal_available else []
        for choice in node.choices:
            values.extend(
                choice.emitted_text + suffix
                for suffix in visit(choice.successor_node_id)
            )
        result = tuple(sorted(values))
        memo[node_id] = result
        return result

    return visit(core.root.node_id)


def _bundle_bytes(path):
    return {
        item.relative_to(path).as_posix(): item.read_bytes()
        for item in sorted(path.rglob("*"))
        if item.is_file()
    }


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _refresh_chunk(path, *, descriptor_field, chunk_index, mutate):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    descriptors = manifest[descriptor_field]
    descriptor = descriptors[chunk_index]
    old_path = path / "chunks" / f"{descriptor['digest']}.json"
    chunk = json.loads(old_path.read_text())
    mutate(chunk)
    payload = _canonical(chunk)
    digest = hashlib.sha256(payload).hexdigest()
    keys = [_record_key(chunk["kind"], item) for item in chunk["items"]]
    updated = {
        **descriptor,
        "digest": digest,
        "canonical_bytes": len(payload),
        "item_count": len(chunk["items"]),
        "first_key": keys[0],
        "last_key": keys[-1],
    }
    new_path = path / "chunks" / f"{digest}.json"
    new_path.write_bytes(payload)
    old_path.unlink()
    descriptors[chunk_index] = updated
    _refresh_size_metrics(manifest)
    unsigned = dict(manifest)
    unsigned.pop("digest")
    manifest["digest"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    manifest_path.write_bytes(_canonical(manifest))


def _refresh_size_metrics(manifest):
    metrics = manifest["deterministic_metrics"]
    mapping = {
        "raw_cursor_record_bytes": "raw_cursor_chunks",
        "primitive_record_bytes": "primitive_chunks",
        "edge_record_bytes": "edge_chunks",
        "terminal_record_bytes": "terminal_chunks",
    }
    for metric, field in mapping.items():
        metrics[metric] = sum(item["canonical_bytes"] for item in manifest[field])
    metrics["compact_provenance_bytes"] = sum(
        metrics[item] for item in mapping
    )
    descriptors = (
        manifest["source_snapshot_chunk"],
        manifest["core_chunk"],
        *manifest["raw_cursor_chunks"],
        *manifest["primitive_chunks"],
        *manifest["edge_chunks"],
        *manifest["terminal_chunks"],
    )
    metrics["peak_serialization_buffer_bytes"] = max(
        item["canonical_bytes"] for item in descriptors
    )


def _forge_core_count(path):
    descriptor = json.loads((path / "manifest.json").read_text())["core_chunk"]
    old = path / "chunks" / f"{descriptor['digest']}.json"
    chunk = json.loads(old.read_text())
    chunk["items"][0]["nodes"][0]["completion_count"] += 1
    payload = _canonical(chunk)
    digest = hashlib.sha256(payload).hexdigest()
    new = path / "chunks" / f"{digest}.json"
    new.write_bytes(payload)
    old.unlink()
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["core_chunk"] = {
        **descriptor,
        "digest": digest,
        "canonical_bytes": len(payload),
    }
    manifest["deterministic_metrics"]["core_chunk_bytes"] = len(payload)
    manifest["deterministic_metrics"]["peak_serialization_buffer_bytes"] = max(
        item["canonical_bytes"] for item in (
            manifest["source_snapshot_chunk"],
            manifest["core_chunk"],
            *manifest["raw_cursor_chunks"],
            *manifest["primitive_chunks"],
            *manifest["edge_chunks"],
            *manifest["terminal_chunks"],
        )
    )
    unsigned = dict(manifest)
    unsigned.pop("digest")
    manifest["digest"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    manifest_path.write_bytes(_canonical(manifest))


def _forge_predecessor(path):
    def mutate(chunk):
        item = next(
            item for item in chunk["items"] if item["predecessor_edge_id"] is not None
        )
        item["predecessor_edge_id"] = None

    _refresh_chunk(
        path,
        descriptor_field="raw_cursor_chunks",
        chunk_index=0,
        mutate=mutate,
    )


def _forge_duplicate_raw_record(path):
    def mutate(chunk):
        chunk["items"].append(dict(chunk["items"][-1]))

    _refresh_chunk(
        path,
        descriptor_field="raw_cursor_chunks",
        chunk_index=0,
        mutate=mutate,
    )


def _forge_raw_node(path):
    _refresh_chunk(
        path,
        descriptor_field="raw_cursor_chunks",
        chunk_index=0,
        mutate=lambda chunk: chunk["items"][0].__setitem__(
            "compiled_node_id", 999
        ),
    )


def _forge_primitive_representative(path):
    _refresh_chunk(
        path,
        descriptor_field="primitive_chunks",
        chunk_index=0,
        mutate=lambda chunk: chunk["items"][0].__setitem__(
            "representative_raw_cursor_digest", "0" * 64
        ),
    )


def _forge_edge_successor_node(path):
    _refresh_chunk(
        path,
        descriptor_field="edge_chunks",
        chunk_index=0,
        mutate=lambda chunk: chunk["items"][0].__setitem__(
            "successor_node_id", 999
        ),
    )


def _forge_terminal_node(path):
    _refresh_chunk(
        path,
        descriptor_field="terminal_chunks",
        chunk_index=0,
        mutate=lambda chunk: chunk["items"][0].__setitem__(
            "source_node_id", 999
        ),
    )


def _forge_chunk_key_range(path):
    manifest_path = path / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["edge_chunks"][0]["first_key"] = ["0" * 64, "", "0" * 64]
    unsigned = dict(manifest)
    unsigned.pop("digest")
    manifest["digest"] = hashlib.sha256(_canonical(unsigned)).hexdigest()
    manifest_path.write_bytes(_canonical(manifest))


def _forge_unused_projection_digest(path):
    changed = {}

    def mutate(chunk):
        item = chunk["items"][0]
        changed["old"] = item["edge_id"]
        item["text_projection_digest"] = "0" * 64
        identity = (
            item["source_raw_cursor_digest"],
            item["emitted_text"],
            item["text_projection_digest"],
            item["branch_certificate_digests"],
            item["successor_raw_cursor_digest"],
        )
        item["edge_id"] = hashlib.sha256(
            _canonical(_dataclass_term(identity))
        ).hexdigest()
        changed["new"] = item["edge_id"]

    _refresh_chunk(
        path,
        descriptor_field="edge_chunks",
        chunk_index=0,
        mutate=mutate,
    )
    manifest = json.loads((path / "manifest.json").read_text())
    for index in range(len(manifest["raw_cursor_chunks"])):
        descriptor = manifest["raw_cursor_chunks"][index]
        chunk = json.loads(
            (path / "chunks" / f"{descriptor['digest']}.json").read_text()
        )
        if not any(
            item["predecessor_edge_id"] == changed["old"]
            for item in chunk["items"]
        ):
            continue

        def replace_predecessor(value):
            for item in value["items"]:
                if item["predecessor_edge_id"] == changed["old"]:
                    item["predecessor_edge_id"] = changed["new"]

        _refresh_chunk(
            path,
            descriptor_field="raw_cursor_chunks",
            chunk_index=index,
            mutate=replace_predecessor,
        )


def _forge_branch_digest_transplant(path):
    manifest = json.loads((path / "manifest.json").read_text())
    branch_digests = []
    for descriptor in manifest["edge_chunks"]:
        chunk = json.loads(
            (path / "chunks" / f"{descriptor['digest']}.json").read_text()
        )
        branch_digests.extend(
            item["branch_certificate_digests"][0]
            for item in chunk["items"]
            if item["branch_certificate_digests"]
        )
    if len(set(branch_digests)) < 2:
        raise AssertionError("fixture needs two distinct branch identities")
    replacement = next(
        item for item in branch_digests if item != branch_digests[0]
    )
    changed = {}

    def mutate(chunk):
        item = next(
            item
            for item in chunk["items"]
            if item["branch_certificate_digests"]
        )
        changed["old"] = item["edge_id"]
        item["branch_certificate_digests"] = [replacement]
        identity = (
            item["source_raw_cursor_digest"],
            item["emitted_text"],
            item["text_projection_digest"],
            item["branch_certificate_digests"],
            item["successor_raw_cursor_digest"],
        )
        item["edge_id"] = hashlib.sha256(
            _canonical(_dataclass_term(identity))
        ).hexdigest()
        changed["new"] = item["edge_id"]

    _refresh_chunk(
        path,
        descriptor_field="edge_chunks",
        chunk_index=0,
        mutate=mutate,
    )
    manifest = json.loads((path / "manifest.json").read_text())
    for index, descriptor in enumerate(manifest["raw_cursor_chunks"]):
        chunk = json.loads(
            (path / "chunks" / f"{descriptor['digest']}.json").read_text()
        )
        if not any(
            item["predecessor_edge_id"] == changed["old"]
            for item in chunk["items"]
        ):
            continue

        def replace_predecessor(value):
            for item in value["items"]:
                if item["predecessor_edge_id"] == changed["old"]:
                    item["predecessor_edge_id"] = changed["new"]

        _refresh_chunk(
            path,
            descriptor_field="raw_cursor_chunks",
            chunk_index=index,
            mutate=replace_predecessor,
        )


def _forge_terminal_identity_transplant(path):
    manifest = json.loads((path / "manifest.json").read_text())
    identities = []
    for descriptor in manifest["terminal_chunks"]:
        chunk = json.loads(
            (path / "chunks" / f"{descriptor['digest']}.json").read_text()
        )
        identities.extend(
            digest
            for item in chunk["items"]
            for digest in item["terminal_support_identity_digests"]
        )
    if len(set(identities)) < 2:
        raise AssertionError("fixture needs two distinct terminal identities")
    replacement = next(item for item in identities if item != identities[0])
    _refresh_chunk(
        path,
        descriptor_field="terminal_chunks",
        chunk_index=0,
        mutate=lambda chunk: chunk["items"][0].__setitem__(
            "terminal_support_identity_digests", [replacement]
        ),
    )


def _dataclass_term(value):
    if isinstance(value, tuple):
        return [_dataclass_term(item) for item in value]
    if isinstance(value, list):
        return [_dataclass_term(item) for item in value]
    return value


def _record_key(kind, item):
    if kind == "raw_cursor_records":
        return item["raw_cursor_digest"]
    if kind == "primitive_records":
        return item["primitive_cursor_digest"]
    if kind == "edge_records":
        return [
            item["source_raw_cursor_digest"],
            item["emitted_text"],
            item["text_projection_digest"],
        ]
    return item["source_raw_cursor_digest"]


if __name__ == "__main__":
    unittest.main()
