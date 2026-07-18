"""Replay-addressed continuation asset regressions."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import grimace._south_star1.writer_continuation_asset as continuation_asset_module
from grimace._south_star1.errors import SouthStarError

from grimace._south_star1.writer_continuation_asset import (
    advance_writer_continuation_proof,
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
    verify_writer_terminalization_artifact_for_facts,
)
from tests.south_star1.helpers import cco_facts
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


class WriterContinuationAssetTest(unittest.TestCase):
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
                artifact = terminalization_artifact_from_continuation_asset(
                    prepared=prepared,
                    asset=asset,
                    source_raw_cursor_digest=terminal.source_raw_cursor_digest,
                    terminal_support_identity_digest=(
                        terminal.terminal_support_identity_digests[0]
                    ),
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
            with patches[0], patches[1], patches[2]:
                manifest = write_writer_continuation_asset(
                    path=path,
                    prepared=prepared,
                    snapshot=snapshot,
                )
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
            sources = _shared_ring_branch_sources()
            for phase, mark in sources:
                _facts, _options, _prepared, source, support = sources[(phase, mark)]
                artifact = branch_transition_artifact_from_continuation_asset(
                    prepared=prepared,
                    asset=asset,
                    source_raw_cursor_digest=_identity_digest(source.cursor),
                    emitted_text=support.emitted_text,
                    branch_certificate_digest=_identity_digest(
                        support.checked_branch_certificate
                    ),
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
            live = verify_writer_continuation_asset_live(
                prepared=prepared,
                asset=asset,
                full=True,
            )
            self.assertTrue(live.accepted, live.reason)


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
