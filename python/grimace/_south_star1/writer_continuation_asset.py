"""Durable replay-addressed assets for compiled writer continuations."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_branch_transition_artifact import (
    verify_writer_branch_transition_artifact_envelope,
)
from .writer_branch_transition_artifact import (
    writer_branch_transition_artifact_for_support,
)
from .writer_branch_transition_artifact_checker import (
    verify_writer_branch_transition_artifact_consistency,
)
from .writer_branch_transition_artifact_fact_verifier import (
    verify_writer_branch_transition_artifact_for_facts,
)
from .writer_continuation_automaton import _canonical_predecessor_tree
from .writer_continuation_automaton import _frontier_batch
from .writer_continuation_automaton import _normalize_cursor
from .writer_continuation_automaton import _verify_internal_consistency
from .writer_continuation_automaton import _verify_core_consistency
from .writer_continuation_automaton import advance_writer_continuation
from .writer_continuation_automaton import compile_writer_continuation_automaton
from .writer_continuation_automaton import WriterContinuationAutomaton
from .writer_continuation_automaton import WriterContinuationChoice
from .writer_continuation_automaton import WriterContinuationCore
from .writer_continuation_automaton import WriterContinuationCursor
from .writer_continuation_automaton import WriterContinuationEdgeRecord
from .writer_continuation_automaton import WriterContinuationMetrics
from .writer_continuation_automaton import WriterContinuationNode
from .writer_continuation_automaton import WriterContinuationPrimitiveRecord
from .writer_continuation_automaton import WriterContinuationProvenance
from .writer_continuation_automaton import WriterContinuationRawCursorRecord
from .writer_continuation_automaton import WriterContinuationTerminalRecord
from .writer_envelope_terms import _identity_envelope
from .writer_envelope_terms import _identity_digest
from .writer_envelope_terms import _runtime_options_from_terms
from .writer_envelope_terms import _snapshot_identity_envelope
from .writer_snapshot import capture_writer_frontier_snapshot
from .writer_snapshot import WriterDecoderBoundary
from .writer_snapshot_closed_terms import writer_frontier_cursor_from_closed_terms
from .writer_prepared_identity import writer_prepared_identity
from .writer_terminalization_artifact import (
    verify_writer_terminalization_artifact_envelope,
)
from .writer_terminalization_artifact import (
    writer_terminalization_artifact_for_support,
)
from .writer_terminalization_artifact_checker import (
    verify_writer_terminalization_artifact_consistency,
)
from .writer_terminalization_artifact_fact_verifier import (
    verify_writer_terminalization_artifact_for_facts,
)


SCHEMA_NAME = "writer_continuation_asset"
SCHEMA_VERSION = 1
CHUNK_SCHEMA_NAME = "writer_continuation_chunk"
CHUNK_SCHEMA_VERSION = 1

_MAX_CHUNK_BYTES = 4_000_000
_MAX_MANIFEST_BYTES = 1_000_000
_MAX_COMPACT_PROVENANCE_BYTES = 64_000_000
_CHUNK_KINDS = (
    "source_snapshot",
    "automaton_core",
    "raw_cursor_records",
    "primitive_records",
    "edge_records",
    "terminal_records",
)
_RECORD_KINDS = _CHUNK_KINDS[2:]
_MANIFEST_FIELDS = frozenset(
    (
        "schema_name",
        "schema_version",
        "prepared_identity",
        "root_raw_cursor_digest",
        "source_snapshot_chunk",
        "core_chunk",
        "raw_cursor_chunks",
        "primitive_chunks",
        "edge_chunks",
        "terminal_chunks",
        "deterministic_metrics",
        "digest",
    )
)
_DESCRIPTOR_FIELDS = frozenset(
    ("kind", "digest", "canonical_bytes", "item_count", "first_key", "last_key")
)
_RAW_RECORD_FIELDS = frozenset(
    (
        "raw_cursor_digest",
        "primitive_cursor_digest",
        "normalization_scale",
        "compiled_node_id",
        "token_depth",
        "predecessor_edge_id",
    )
)
_PRIMITIVE_RECORD_FIELDS = frozenset(
    (
        "primitive_cursor_digest",
        "compiled_node_id",
        "representative_raw_cursor_digest",
    )
)
_EDGE_RECORD_FIELDS = frozenset(
    (
        "edge_id",
        "source_raw_cursor_digest",
        "source_node_id",
        "emitted_text",
        "text_projection_digest",
        "branch_certificate_digests",
        "successor_raw_cursor_digest",
        "successor_node_id",
        "successor_scale",
    )
)
_TERMINAL_RECORD_FIELDS = frozenset(
    (
        "source_raw_cursor_digest",
        "source_node_id",
        "terminal_support_identity_digests",
        "finalized_cursor_digest",
    )
)
_ASSET_METRIC_FIELDS = frozenset(
    (
        "raw_cursor_count",
        "primitive_cursor_count",
        "semantic_node_count",
        "semantic_edge_count",
        "terminal_node_count",
        "maximum_depth",
        "maximum_out_degree",
        "largest_equivalence_class_membership",
        "weight_normalization_merge_count",
        "semantic_minimization_merge_count",
        "canonical_core_bytes",
        "provenance_index_bytes",
        "peak_active_depth",
        "peak_primitive_memo_size",
        "peak_signature_memo_size",
        "source_snapshot_bytes",
        "core_chunk_bytes",
        "raw_cursor_record_bytes",
        "primitive_record_bytes",
        "edge_record_bytes",
        "terminal_record_bytes",
        "compact_provenance_bytes",
        "largest_record_bytes",
        "chunk_count",
        "peak_serialization_buffer_bytes",
    )
)


@dataclass(frozen=True, slots=True)
class WriterContinuationProofCursor:
    node_id: int
    completion_scale: int
    raw_cursor_digest: str


@dataclass(frozen=True, slots=True)
class WriterContinuationAssetVerification:
    accepted: bool
    reason: str | None = None


@dataclass(frozen=True, slots=True)
class WriterContinuationAssetSemanticVerification:
    accepted: bool
    structurally_verified: bool = False
    live_replay_complete: bool = False
    raw_cursor_count: int = 0
    edge_locator_count: int = 0
    branch_locator_count: int = 0
    branch_proof_count: int = 0
    terminal_record_count: int = 0
    terminal_locator_count: int = 0
    terminal_proof_count: int = 0
    unchecked_obligation_families: tuple[str, ...] = ()
    reason: str | None = None


class WriterContinuationAsset:
    """A core-first asset session whose provenance indexes load on demand."""

    def __init__(self, *, path: Path, manifest, core) -> None:
        self.path = path
        self.manifest = manifest
        self.core = core
        self._records: dict[str, tuple[object, ...]] = {}
        self._indexes = {}
        self._cursor_cache = {}
        self._source_snapshot_cache = None

    @property
    def manifest_digest(self) -> str:
        return self.manifest["digest"]

    @property
    def root_proof_cursor(self) -> WriterContinuationProofCursor:
        return WriterContinuationProofCursor(
            node_id=self.core.root.node_id,
            completion_scale=self.core.root.completion_scale,
            raw_cursor_digest=self.manifest["root_raw_cursor_digest"],
        )

    def records(self, kind: str):
        known = self._records.get(kind)
        if known is not None:
            return known
        descriptors = self.manifest[_descriptor_list_field(kind)]
        values = []
        for descriptor in descriptors:
            chunk = _read_chunk(self.path, descriptor)
            values.extend(_record_from_mapping(kind, item) for item in chunk["items"])
        result = tuple(values)
        self._records[kind] = result
        return result

    def raw_cursor_record(self, raw_cursor_digest):
        index = self._indexes.get("raw_by_digest")
        if index is None:
            index = {
                item.raw_cursor_digest: item
                for item in self.records("raw_cursor_records")
            }
            self._indexes["raw_by_digest"] = index
        return index.get(raw_cursor_digest)

    def edge_record(self, source_raw_cursor_digest, emitted_text):
        index = self._indexes.get("edge_by_source_text")
        if index is None:
            index = {
                (item.source_raw_cursor_digest, item.emitted_text): item
                for item in self.records("edge_records")
            }
            self._indexes["edge_by_source_text"] = index
        return index.get((source_raw_cursor_digest, emitted_text))

    def edge_record_by_id(self, edge_id):
        index = self._indexes.get("edge_by_id")
        if index is None:
            index = {item.edge_id: item for item in self.records("edge_records")}
            self._indexes["edge_by_id"] = index
        return index.get(edge_id)

    def edges_from(self, source_raw_cursor_digest):
        index = self._indexes.get("edges_by_source")
        if index is None:
            grouped = {}
            for item in self.records("edge_records"):
                grouped.setdefault(item.source_raw_cursor_digest, []).append(item)
            index = {key: tuple(values) for key, values in grouped.items()}
            self._indexes["edges_by_source"] = index
        return index.get(source_raw_cursor_digest, ())

    def terminal_record(self, source_raw_cursor_digest):
        index = self._indexes.get("terminal_by_source")
        if index is None:
            index = {
                item.source_raw_cursor_digest: item
                for item in self.records("terminal_records")
            }
            self._indexes["terminal_by_source"] = index
        return index.get(source_raw_cursor_digest)


def write_writer_continuation_asset(
    *, path, prepared, snapshot, automaton=None
) -> Mapping[str, object]:
    destination = Path(path)
    if destination.exists():
        _violation("continuation_asset_destination_exists")
    automaton = (
        compile_writer_continuation_automaton(prepared=prepared, snapshot=snapshot)
        if automaton is None
        else automaton
    )
    _verify_internal_consistency(
        automaton,
        signature_digest_function=_identity_digest,
    )
    parent = destination.parent
    parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{destination.name}.", dir=parent))
    try:
        (temporary / "chunks").mkdir()
        source = _snapshot_identity_envelope(
            snapshot,
            operation="continuation_asset.source_snapshot",
        )
        source_descriptor, source_peak = _write_singleton_chunk(
            temporary,
            kind="source_snapshot",
            item=source,
            key="source_snapshot",
        )
        core_item = _core_mapping(automaton)
        core_descriptor, core_peak = _write_singleton_chunk(
            temporary,
            kind="automaton_core",
            item=core_item,
            key="automaton_core",
        )
        raw_descriptors, raw_peak, raw_bytes, raw_largest = _write_record_chunks(
            temporary,
            kind="raw_cursor_records",
            records=(
                _raw_record_mapping(item)
                for item in automaton.provenance.raw_cursors
            ),
            key_function=lambda item: item["raw_cursor_digest"],
        )
        (
            primitive_descriptors,
            primitive_peak,
            primitive_bytes,
            primitive_largest,
        ) = _write_record_chunks(
            temporary,
            kind="primitive_records",
            records=(
                _primitive_record_mapping(item)
                for item in automaton.provenance.primitives
            ),
            key_function=lambda item: item["primitive_cursor_digest"],
        )
        edge_descriptors, edge_peak, edge_bytes, edge_largest = _write_record_chunks(
            temporary,
            kind="edge_records",
            records=(
                _edge_record_mapping(item) for item in automaton.provenance.edges
            ),
            key_function=lambda item: (
                item["source_raw_cursor_digest"],
                item["emitted_text"],
                item["text_projection_digest"],
            ),
        )
        (
            terminal_descriptors,
            terminal_peak,
            terminal_bytes,
            terminal_largest,
        ) = _write_record_chunks(
            temporary,
            kind="terminal_records",
            records=(
                _terminal_record_mapping(item)
                for item in automaton.provenance.terminals
            ),
            key_function=lambda item: item["source_raw_cursor_digest"],
        )
        compact_bytes = raw_bytes + primitive_bytes + edge_bytes + terminal_bytes
        if compact_bytes > _MAX_COMPACT_PROVENANCE_BYTES:
            _violation("continuation_asset_compact_provenance_too_large")
        descriptors = (
            raw_descriptors
            + primitive_descriptors
            + edge_descriptors
            + terminal_descriptors
        )
        metrics = {
            **_deterministic_metrics(automaton),
            "source_snapshot_bytes": source_descriptor["canonical_bytes"],
            "core_chunk_bytes": core_descriptor["canonical_bytes"],
            "raw_cursor_record_bytes": raw_bytes,
            "primitive_record_bytes": primitive_bytes,
            "edge_record_bytes": edge_bytes,
            "terminal_record_bytes": terminal_bytes,
            "compact_provenance_bytes": compact_bytes,
            "largest_record_bytes": max(
                raw_largest,
                primitive_largest,
                edge_largest,
                terminal_largest,
            ),
            "chunk_count": 2 + len(descriptors),
            "peak_serialization_buffer_bytes": max(
                source_peak,
                core_peak,
                raw_peak,
                primitive_peak,
                edge_peak,
                terminal_peak,
            ),
        }
        manifest = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "prepared_identity": _identity_envelope(snapshot.prepared_identity),
            "root_raw_cursor_digest": automaton.provenance.root_raw_cursor_digest,
            "source_snapshot_chunk": source_descriptor,
            "core_chunk": core_descriptor,
            "raw_cursor_chunks": list(raw_descriptors),
            "primitive_chunks": list(primitive_descriptors),
            "edge_chunks": list(edge_descriptors),
            "terminal_chunks": list(terminal_descriptors),
            "deterministic_metrics": metrics,
        }
        manifest["digest"] = _digest_mapping(manifest)
        payload = _canonical_bytes(manifest)
        if len(payload) > _MAX_MANIFEST_BYTES:
            _violation("continuation_asset_manifest_too_large")
        (temporary / "manifest.json").write_bytes(payload)
        checked = verify_writer_continuation_asset_consistency(temporary)
        if not checked.accepted:
            _violation(checked.reason or "continuation_asset_structural_rejection")
        asset = open_writer_continuation_core(temporary)
        semantic = verify_writer_continuation_asset_for_prepared(
            prepared=prepared,
            asset=asset,
        )
        if not semantic.accepted:
            _violation(
                semantic.reason or "continuation_asset_semantic_certification_rejection"
            )
        os.replace(temporary, destination)
        return manifest
    except BaseException:
        if temporary.exists():
            shutil.rmtree(temporary)
        raise


def open_writer_continuation_core(path) -> WriterContinuationAsset:
    directory = Path(path)
    manifest = _read_manifest(directory)
    core_chunk = _read_chunk(directory, manifest["core_chunk"])
    if len(core_chunk["items"]) != 1:
        _violation("continuation_asset_core_item_count_mismatch")
    core = _core_from_mapping(
        core_chunk["items"][0],
        metrics=manifest["deterministic_metrics"],
    )
    _verify_core_consistency(
        core,
        signature_digest_function=_identity_digest,
    )
    return WriterContinuationAsset(path=directory, manifest=manifest, core=core)


def advance_writer_continuation_proof(
    asset: WriterContinuationAsset,
    cursor: WriterContinuationProofCursor,
    emitted_text: str,
) -> WriterContinuationProofCursor:
    edge = asset.edge_record(cursor.raw_cursor_digest, emitted_text)
    if edge is None:
        _violation("continuation_asset_proof_edge_not_unique")
    if edge.source_node_id != cursor.node_id:
        _violation("continuation_asset_proof_source_node_mismatch")
    advanced = advance_writer_continuation(
        asset.core,
        WriterContinuationCursor(cursor.node_id, cursor.completion_scale),
        emitted_text,
    )
    if (
        advanced.node_id != edge.successor_node_id
        or advanced.completion_scale != edge.successor_scale
    ):
        _violation("continuation_asset_proof_advance_mismatch")
    return WriterContinuationProofCursor(
        node_id=advanced.node_id,
        completion_scale=advanced.completion_scale,
        raw_cursor_digest=edge.successor_raw_cursor_digest,
    )


def writer_continuation_cursor_envelope(*, asset, cursor, raw_cursor_digest=None):
    payload = {
        "asset_manifest_digest": asset.manifest_digest,
        "node_id": cursor.node_id,
        "completion_scale": cursor.completion_scale,
        "raw_cursor_digest": raw_cursor_digest,
    }
    payload["digest"] = _digest_mapping(payload)
    return payload


def verify_writer_continuation_cursor_envelope(*, asset, envelope):
    try:
        if set(envelope) != {
            "asset_manifest_digest",
            "node_id",
            "completion_scale",
            "raw_cursor_digest",
            "digest",
        }:
            _violation("continuation_asset_cursor_shape_mismatch")
        unsigned = dict(envelope)
        digest = unsigned.pop("digest")
        if digest != _digest_mapping(unsigned):
            _violation("continuation_asset_cursor_digest_mismatch")
        if envelope["asset_manifest_digest"] != asset.manifest_digest:
            _violation("continuation_asset_cursor_asset_mismatch")
        cursor = WriterContinuationCursor(
            node_id=envelope["node_id"],
            completion_scale=envelope["completion_scale"],
        )
        if not 0 <= cursor.node_id < len(asset.core.nodes):
            _violation("continuation_asset_cursor_node_mismatch")
        raw_digest = envelope["raw_cursor_digest"]
        if raw_digest is None:
            return WriterContinuationAssetVerification(accepted=True)
        record = asset.raw_cursor_record(raw_digest)
        if (
            record is None
            or record.compiled_node_id != cursor.node_id
            or record.normalization_scale != cursor.completion_scale
        ):
            _violation("continuation_asset_proof_cursor_mismatch")
        return WriterContinuationAssetVerification(accepted=True)
    except SouthStarError as exc:
        return WriterContinuationAssetVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "continuation_asset_cursor_error",
        )
    except (KeyError, TypeError, ValueError) as exc:
        return WriterContinuationAssetVerification(
            accepted=False,
            reason=f"malformed_continuation_asset_cursor:{type(exc).__name__}",
        )


def reconstruct_writer_cursor_from_asset(*, prepared, asset, raw_cursor_digest):
    known = asset._cursor_cache.get(raw_cursor_digest)
    if known is not None:
        return known
    requested = asset.raw_cursor_record(raw_cursor_digest)
    if requested is None:
        _violation("continuation_asset_raw_cursor_not_found")
    path = []
    current = requested
    seen = set()
    while current.predecessor_edge_id is not None:
        if current.raw_cursor_digest in seen:
            _violation("continuation_asset_predecessor_cycle")
        seen.add(current.raw_cursor_digest)
        edge = asset.edge_record_by_id(current.predecessor_edge_id)
        if edge is None:
            _violation("continuation_asset_predecessor_edge_missing")
        path.append(edge)
        current = asset.raw_cursor_record(edge.source_raw_cursor_digest)
        if current is None:
            _violation("continuation_asset_predecessor_source_missing")
    if current.raw_cursor_digest != asset.manifest["root_raw_cursor_digest"]:
        _violation("continuation_asset_predecessor_root_mismatch")
    snapshot = _source_snapshot_from_asset(prepared=prepared, asset=asset)
    cursor = snapshot.cursor
    root_digest = _identity_digest(cursor)
    if root_digest != asset.manifest["root_raw_cursor_digest"]:
        _violation("continuation_asset_root_cursor_mismatch")
    asset._cursor_cache[root_digest] = cursor
    for edge in reversed(path):
        cached = asset._cursor_cache.get(edge.successor_raw_cursor_digest)
        if cached is not None:
            cursor = cached
            continue
        if _identity_digest(cursor) != edge.source_raw_cursor_digest:
            _violation("continuation_asset_replay_source_mismatch")
        batch = _frontier_batch(prepared, cursor)
        projections = tuple(
            item
            for item in batch.text_choice_projection_certificates
            if item.emitted_text == edge.emitted_text
        )
        if len(projections) != 1:
            _violation("continuation_asset_replay_projection_not_unique")
        projection = projections[0]
        if (
            _identity_digest(projection) != edge.text_projection_digest
            or tuple(
                _identity_digest(item) for item in projection.branch_certificates
            )
            != edge.branch_certificate_digests
            or _identity_digest(projection.successor_cursor)
            != edge.successor_raw_cursor_digest
        ):
            _violation("continuation_asset_replay_edge_mismatch")
        record = asset.raw_cursor_record(edge.successor_raw_cursor_digest)
        if record is None:
            _violation("continuation_asset_replay_successor_record_missing")
        primitive, scale = _normalize_cursor(projection.successor_cursor)
        if (
            _identity_digest(primitive) != record.primitive_cursor_digest
            or scale != record.normalization_scale
            or record.compiled_node_id != edge.successor_node_id
            or scale != edge.successor_scale
        ):
            _violation("continuation_asset_replay_cursor_class_mismatch")
        cursor = projection.successor_cursor
        asset._cursor_cache[edge.successor_raw_cursor_digest] = cursor
    if _identity_digest(cursor) != raw_cursor_digest:
        _violation("continuation_asset_replay_target_mismatch")
    return cursor


def branch_transition_artifact_from_continuation_asset(
    *, prepared, asset, source_raw_cursor_digest, emitted_text, branch_certificate_digest
):
    locator = asset.edge_record(source_raw_cursor_digest, emitted_text)
    if (
        locator is None
        or branch_certificate_digest
        not in locator.branch_certificate_digests
    ):
        _violation("continuation_asset_branch_locator_mismatch")
    cursor = reconstruct_writer_cursor_from_asset(
        prepared=prepared,
        asset=asset,
        raw_cursor_digest=source_raw_cursor_digest,
    )
    batch = _frontier_batch(prepared, cursor)
    matches = tuple(
        support
        for support in batch.supports
        if support.emitted_text == emitted_text
        and _identity_digest(support.checked_branch_certificate)
        == branch_certificate_digest
    )
    if len(matches) != 1:
        _violation("continuation_asset_branch_identity_not_unique")
    snapshot = _snapshot_for_raw_cursor(
        prepared=prepared,
        asset=asset,
        cursor=cursor,
        raw_cursor_digest=source_raw_cursor_digest,
    )
    artifact = writer_branch_transition_artifact_for_support(
        prepared=prepared,
        snapshot=snapshot,
        support=matches[0],
    )
    checked = verify_writer_branch_transition_artifact_consistency(artifact)
    if not checked.accepted:
        _violation(checked.reason or "continuation_asset_branch_artifact_rejected")
    return artifact


def terminalization_artifact_from_continuation_asset(
    *, prepared, asset, source_raw_cursor_digest, terminal_support_identity_digest
):
    locator = asset.terminal_record(source_raw_cursor_digest)
    if (
        locator is None
        or terminal_support_identity_digest
        not in locator.terminal_support_identity_digests
    ):
        _violation("continuation_asset_terminal_locator_mismatch")
    cursor = reconstruct_writer_cursor_from_asset(
        prepared=prepared,
        asset=asset,
        raw_cursor_digest=source_raw_cursor_digest,
    )
    batch = _frontier_batch(prepared, cursor)
    matches = tuple(
        support
        for support in batch.terminal_supports
        if _identity_digest(support.checked_terminal_certificate)
        == terminal_support_identity_digest
    )
    if len(matches) != 1:
        _violation("continuation_asset_terminal_identity_not_unique")
    snapshot = _snapshot_for_raw_cursor(
        prepared=prepared,
        asset=asset,
        cursor=cursor,
        raw_cursor_digest=source_raw_cursor_digest,
    )
    artifact = writer_terminalization_artifact_for_support(
        prepared=prepared,
        snapshot=snapshot,
        support=matches[0],
    )
    checked = verify_writer_terminalization_artifact_consistency(artifact)
    if not checked.accepted:
        _violation(checked.reason or "continuation_asset_terminal_artifact_rejected")
    return artifact


def verify_writer_continuation_asset_consistency(path):
    try:
        directory = Path(path)
        manifest = _read_manifest(directory)
        descriptors = _all_descriptors(manifest)
        expected_files = {f"{item['digest']}.json" for item in descriptors}
        actual_files = {item.name for item in (directory / "chunks").iterdir()}
        if expected_files != actual_files:
            _violation("continuation_asset_chunk_set_mismatch")
        records = {}
        for kind in _RECORD_KINDS:
            values = []
            previous_key = None
            for descriptor in manifest[_descriptor_list_field(kind)]:
                chunk = _read_chunk(directory, descriptor)
                for item in chunk["items"]:
                    key = _record_key(kind, item)
                    if previous_key is not None and key <= previous_key:
                        _violation("continuation_asset_record_order_mismatch")
                    previous_key = key
                    values.append(_record_from_mapping(kind, item))
            records[kind] = tuple(values)
        core_chunk = _read_chunk(directory, manifest["core_chunk"])
        source_chunk = _read_chunk(directory, manifest["source_snapshot_chunk"])
        if len(core_chunk["items"]) != 1 or len(source_chunk["items"]) != 1:
            _violation("continuation_asset_singleton_chunk_mismatch")
        core = _core_from_mapping(
            core_chunk["items"][0], metrics=manifest["deterministic_metrics"]
        )
        provenance = WriterContinuationProvenance(
            source_snapshot_digest=source_chunk["items"][0]["digest"],
            root_raw_cursor_digest=manifest["root_raw_cursor_digest"],
            raw_cursors=records["raw_cursor_records"],
            primitives=records["primitive_records"],
            edges=records["edge_records"],
            terminals=records["terminal_records"],
        )
        automaton = WriterContinuationAutomaton(
            root=core.root,
            nodes=core.nodes,
            provenance=provenance,
            metrics=core.metrics,
        )
        _verify_internal_consistency(
            automaton, signature_digest_function=_identity_digest
        )
        source_terms = source_chunk["items"][0]
        if (
            set(manifest["prepared_identity"]) != {"terms", "digest"}
            or manifest["prepared_identity"]["digest"]
            != hashlib.sha256(
                _canonical_bytes(manifest["prepared_identity"]["terms"])
            ).hexdigest()
            or manifest["prepared_identity"]["terms"]
            != source_terms["prepared_identity_terms"]
            or manifest["prepared_identity"]["digest"]
            != source_terms["prepared_identity_digest"]
            or source_terms["cursor"]["digest"]
            != hashlib.sha256(
                _canonical_bytes(source_terms["cursor"]["terms"])
            ).hexdigest()
            or manifest["root_raw_cursor_digest"]
            != source_terms["cursor"]["digest"]
        ):
            _violation("continuation_asset_source_identity_mismatch")
        _verify_predecessor_canonicality(provenance)
        _verify_manifest_metrics(manifest=manifest, automaton=automaton)
        return WriterContinuationAssetVerification(accepted=True)
    except SouthStarError as exc:
        return WriterContinuationAssetVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "continuation_asset_error",
        )
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as exc:
        return WriterContinuationAssetVerification(
            accepted=False,
            reason=f"malformed_continuation_asset:{type(exc).__name__}:{exc}",
        )


def verify_writer_continuation_asset_live(*, prepared, asset, full):
    try:
        structural = verify_writer_continuation_asset_consistency(asset.path)
        if not structural.accepted:
            return structural
        raw_records = asset.records("raw_cursor_records")
        root = next(
            item
            for item in raw_records
            if item.raw_cursor_digest
            == asset.manifest["root_raw_cursor_digest"]
        )
        selected = raw_records if full else (root,)
        if full:
            source = _source_snapshot_from_asset(prepared=prepared, asset=asset)
            asset._cursor_cache[root.raw_cursor_digest] = source.cursor
        for record in sorted(selected, key=lambda item: item.token_depth):
            if full:
                cursor = asset._cursor_cache.get(record.raw_cursor_digest)
                if cursor is None:
                    _violation("continuation_asset_live_predecessor_not_replayed")
            else:
                cursor = reconstruct_writer_cursor_from_asset(
                    prepared=prepared,
                    asset=asset,
                    raw_cursor_digest=record.raw_cursor_digest,
                )
            primitive, scale = _normalize_cursor(cursor)
            if (
                _identity_digest(primitive) != record.primitive_cursor_digest
                or scale != record.normalization_scale
            ):
                _violation("continuation_asset_live_cursor_mismatch")
            batch = _frontier_batch(prepared, cursor)
            edge_records = asset.edges_from(record.raw_cursor_digest)
            if tuple(item.emitted_text for item in edge_records) != tuple(
                item.emitted_text
                for item in batch.text_choice_projection_certificates
            ):
                _violation("continuation_asset_live_edge_coverage_mismatch")
            node = asset.core.nodes[record.compiled_node_id]
            projections = batch.text_choice_projection_certificates
            if tuple(
                record.normalization_scale * choice.immediate_multiplicity
                for choice in node.choices
            ) != tuple(item.immediate_multiplicity for item in projections):
                _violation("continuation_asset_live_multiplicity_mismatch")
            for edge, projection in zip(edge_records, projections):
                if (
                    edge.text_projection_digest != _identity_digest(projection)
                    or edge.branch_certificate_digests
                    != tuple(
                        _identity_digest(item)
                        for item in projection.branch_certificates
                    )
                    or edge.successor_raw_cursor_digest
                    != _identity_digest(projection.successor_cursor)
                ):
                    _violation("continuation_asset_live_projection_mismatch")
                known_successor = asset._cursor_cache.get(
                    edge.successor_raw_cursor_digest
                )
                if (
                    known_successor is not None
                    and known_successor != projection.successor_cursor
                ):
                    _violation("continuation_asset_live_successor_alias_mismatch")
                asset._cursor_cache[edge.successor_raw_cursor_digest] = (
                    projection.successor_cursor
                )
            terminal_record = asset.terminal_record(record.raw_cursor_digest)
            if (terminal_record is not None) != bool(batch.terminal_supports):
                _violation("continuation_asset_live_terminal_coverage_mismatch")
            terminal = batch.choices.terminal
            expected_terminal_multiplicity = (
                0 if terminal is None else terminal.multiplicity
            )
            if (
                record.normalization_scale * node.terminal_multiplicity
                != expected_terminal_multiplicity
            ):
                _violation("continuation_asset_live_terminal_multiplicity_mismatch")
            if terminal_record is not None:
                if (
                    terminal_record.terminal_support_identity_digests
                    != tuple(
                        _identity_digest(item.checked_terminal_certificate)
                        for item in batch.terminal_supports
                    )
                    or terminal_record.finalized_cursor_digest
                    != _identity_digest(terminal.finalized_cursor)
                ):
                    _violation("continuation_asset_live_terminal_mismatch")
        return WriterContinuationAssetVerification(accepted=True)
    except SouthStarError as exc:
        return WriterContinuationAssetVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "continuation_asset_live_error",
        )
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as exc:
        return WriterContinuationAssetVerification(
            accepted=False,
            reason=f"malformed_live_continuation_asset:{type(exc).__name__}:{exc}",
        )


def verify_writer_continuation_asset_for_prepared(
    *, prepared, asset
) -> WriterContinuationAssetSemanticVerification:
    """Certify every live branch and terminal locator in one continuation asset."""
    structurally_verified = False
    live_replay_complete = False
    raw_cursor_count = 0
    edge_locator_count = 0
    branch_locator_count = 0
    branch_proof_count = 0
    terminal_record_count = 0
    terminal_locator_count = 0
    terminal_proof_count = 0
    unchecked_families: set[str] = set()
    try:
        structural = verify_writer_continuation_asset_consistency(asset.path)
        if not structural.accepted:
            _violation(
                structural.reason or "continuation_asset_structural_rejection"
            )
        structurally_verified = True

        source = _source_snapshot_from_asset(prepared=prepared, asset=asset)
        if (
            _identity_digest(source.cursor)
            != asset.manifest["root_raw_cursor_digest"]
        ):
            _violation("continuation_asset_semantic_root_cursor_mismatch")

        live = verify_writer_continuation_asset_live(
            prepared=prepared,
            asset=asset,
            full=True,
        )
        if not live.accepted:
            _violation(live.reason or "continuation_asset_live_replay_rejection")
        live_replay_complete = True

        raw_records = asset.records("raw_cursor_records")
        edge_records = asset.records("edge_records")
        terminal_records = asset.records("terminal_records")
        raw_cursor_count = len(raw_records)
        edge_locator_count = len(edge_records)
        terminal_record_count = len(terminal_records)

        branch_locators = tuple(
            (
                edge.source_raw_cursor_digest,
                edge.emitted_text,
                digest,
            )
            for edge in edge_records
            for digest in edge.branch_certificate_digests
        )
        terminal_locators = tuple(
            (terminal.source_raw_cursor_digest, digest)
            for terminal in terminal_records
            for digest in terminal.terminal_support_identity_digests
        )
        branch_locator_count = len(branch_locators)
        terminal_locator_count = len(terminal_locators)
        if len(set(branch_locators)) != branch_locator_count:
            _violation("continuation_asset_duplicate_branch_locator")
        if len(set(terminal_locators)) != terminal_locator_count:
            _violation("continuation_asset_duplicate_terminal_locator")

        proved_branches = set()
        for source_digest, emitted_text, certificate_digest in branch_locators:
            artifact = branch_transition_artifact_from_continuation_asset(
                prepared=prepared,
                asset=asset,
                source_raw_cursor_digest=source_digest,
                emitted_text=emitted_text,
                branch_certificate_digest=certificate_digest,
            )
            structural_branch = (
                verify_writer_branch_transition_artifact_consistency(artifact)
            )
            if not structural_branch.accepted:
                _violation(
                    structural_branch.reason
                    or "continuation_asset_branch_structural_rejection"
                )
            live_branch = verify_writer_branch_transition_artifact_envelope(
                prepared=prepared,
                artifact=artifact,
            )
            if not live_branch.accepted:
                _violation(
                    live_branch.reason or "continuation_asset_branch_live_rejection"
                )
            facts_branch = verify_writer_branch_transition_artifact_for_facts(
                facts=prepared.facts,
                runtime_options=source.runtime_options,
                artifact=artifact,
                policy=prepared.policy,
            )
            unchecked_families.update(facts_branch.unchecked_obligation_families)
            if not facts_branch.accepted:
                _violation(
                    facts_branch.reason or "continuation_asset_branch_facts_rejection"
                )
            if facts_branch.unchecked_obligation_families:
                _violation("continuation_asset_branch_obligations_unchecked")
            locator = (source_digest, emitted_text, certificate_digest)
            if locator in proved_branches:
                _violation("continuation_asset_duplicate_branch_proof_credit")
            proved_branches.add(locator)
            branch_proof_count = len(proved_branches)

        proved_terminals = set()
        for source_digest, support_digest in terminal_locators:
            artifact = terminalization_artifact_from_continuation_asset(
                prepared=prepared,
                asset=asset,
                source_raw_cursor_digest=source_digest,
                terminal_support_identity_digest=support_digest,
            )
            structural_terminal = (
                verify_writer_terminalization_artifact_consistency(artifact)
            )
            if not structural_terminal.accepted:
                _violation(
                    structural_terminal.reason
                    or "continuation_asset_terminal_structural_rejection"
                )
            live_terminal = verify_writer_terminalization_artifact_envelope(
                prepared=prepared,
                artifact=artifact,
            )
            if not live_terminal.accepted:
                _violation(
                    live_terminal.reason
                    or "continuation_asset_terminal_live_rejection"
                )
            facts_terminal = verify_writer_terminalization_artifact_for_facts(
                facts=prepared.facts,
                runtime_options=source.runtime_options,
                artifact=artifact,
                policy=prepared.policy,
            )
            unchecked_families.update(
                facts_terminal.unchecked_obligation_families
            )
            if not facts_terminal.accepted:
                _violation(
                    facts_terminal.reason
                    or "continuation_asset_terminal_facts_rejection"
                )
            if facts_terminal.unchecked_obligation_families:
                _violation("continuation_asset_terminal_obligations_unchecked")
            locator = (source_digest, support_digest)
            if locator in proved_terminals:
                _violation("continuation_asset_duplicate_terminal_proof_credit")
            proved_terminals.add(locator)
            terminal_proof_count = len(proved_terminals)

        if proved_branches != set(branch_locators):
            _violation("continuation_asset_branch_proof_coverage_mismatch")
        if proved_terminals != set(terminal_locators):
            _violation("continuation_asset_terminal_proof_coverage_mismatch")
        return WriterContinuationAssetSemanticVerification(
            accepted=True,
            structurally_verified=True,
            live_replay_complete=True,
            raw_cursor_count=raw_cursor_count,
            edge_locator_count=edge_locator_count,
            branch_locator_count=branch_locator_count,
            branch_proof_count=branch_proof_count,
            terminal_record_count=terminal_record_count,
            terminal_locator_count=terminal_locator_count,
            terminal_proof_count=terminal_proof_count,
        )
    except SouthStarError as exc:
        return WriterContinuationAssetSemanticVerification(
            accepted=False,
            structurally_verified=structurally_verified,
            live_replay_complete=live_replay_complete,
            raw_cursor_count=raw_cursor_count,
            edge_locator_count=edge_locator_count,
            branch_locator_count=branch_locator_count,
            branch_proof_count=branch_proof_count,
            terminal_record_count=terminal_record_count,
            terminal_locator_count=terminal_locator_count,
            terminal_proof_count=terminal_proof_count,
            unchecked_obligation_families=tuple(sorted(unchecked_families)),
            reason=exc.args[-1] if exc.args else "continuation_asset_semantic_error",
        )
    except (AssertionError, KeyError, OSError, TypeError, ValueError) as exc:
        return WriterContinuationAssetSemanticVerification(
            accepted=False,
            structurally_verified=structurally_verified,
            live_replay_complete=live_replay_complete,
            raw_cursor_count=raw_cursor_count,
            edge_locator_count=edge_locator_count,
            branch_locator_count=branch_locator_count,
            branch_proof_count=branch_proof_count,
            terminal_record_count=terminal_record_count,
            terminal_locator_count=terminal_locator_count,
            terminal_proof_count=terminal_proof_count,
            unchecked_obligation_families=tuple(sorted(unchecked_families)),
            reason=(
                "malformed_semantic_continuation_asset:"
                f"{type(exc).__name__}:{exc}"
            ),
        )


def _write_singleton_chunk(directory, *, kind, item, key):
    return _write_chunk(directory, kind=kind, items=(item,), keys=(key,))


def _write_record_chunks(directory, *, kind, records, key_function):
    descriptors = []
    current = []
    current_keys = []
    current_payload_bytes = 0
    empty_chunk_bytes = len(_chunk_bytes(kind=kind, items=()))
    total_bytes = 0
    peak = 0
    largest = 0
    previous_key = None
    for record in records:
        key = key_function(record)
        if previous_key is not None and key <= previous_key:
            _violation("continuation_asset_writer_record_order_mismatch")
        previous_key = key
        record_bytes = _canonical_bytes(record)
        largest = max(largest, len(record_bytes))
        candidate_size = (
            empty_chunk_bytes
            + current_payload_bytes
            + len(record_bytes)
            + len(current)
        )
        if current and candidate_size > _MAX_CHUNK_BYTES:
            descriptor, chunk_peak = _write_chunk(
                directory,
                kind=kind,
                items=tuple(current),
                keys=tuple(current_keys),
            )
            descriptors.append(descriptor)
            total_bytes += descriptor["canonical_bytes"]
            peak = max(peak, chunk_peak)
            current = [record]
            current_keys = [key]
            current_payload_bytes = len(record_bytes)
        else:
            current.append(record)
            current_keys.append(key)
            current_payload_bytes += len(record_bytes)
    if current:
        descriptor, chunk_peak = _write_chunk(
            directory,
            kind=kind,
            items=tuple(current),
            keys=tuple(current_keys),
        )
        descriptors.append(descriptor)
        total_bytes += descriptor["canonical_bytes"]
        peak = max(peak, chunk_peak)
    return tuple(descriptors), peak, total_bytes, largest


def _write_chunk(directory, *, kind, items, keys):
    payload = _chunk_bytes(kind=kind, items=items)
    if len(payload) > _MAX_CHUNK_BYTES:
        _violation("continuation_asset_chunk_too_large")
    digest = hashlib.sha256(payload).hexdigest()
    descriptor = {
        "kind": kind,
        "digest": digest,
        "canonical_bytes": len(payload),
        "item_count": len(items),
        "first_key": _json_key(keys[0]),
        "last_key": _json_key(keys[-1]),
    }
    temporary = directory / "chunks" / f".{digest}.tmp"
    final = directory / "chunks" / f"{digest}.json"
    temporary.write_bytes(payload)
    os.replace(temporary, final)
    return descriptor, len(payload)


def _chunk_bytes(*, kind, items):
    return _canonical_bytes(
        {
            "schema_name": CHUNK_SCHEMA_NAME,
            "schema_version": CHUNK_SCHEMA_VERSION,
            "kind": kind,
            "items": list(items),
        }
    )


def _read_manifest(directory):
    payload = (directory / "manifest.json").read_bytes()
    manifest = json.loads(payload)
    if payload != _canonical_bytes(manifest):
        _violation("continuation_asset_manifest_not_canonical")
    if set(manifest) != _MANIFEST_FIELDS:
        _violation("continuation_asset_manifest_shape_mismatch")
    if manifest["schema_name"] != SCHEMA_NAME:
        _violation("continuation_asset_unknown_schema_name")
    if manifest["schema_version"] != SCHEMA_VERSION:
        _violation("continuation_asset_unknown_schema_version")
    expected = dict(manifest)
    digest = expected.pop("digest")
    if digest != _digest_mapping(expected):
        _violation("continuation_asset_manifest_digest_mismatch")
    if len(payload) > _MAX_MANIFEST_BYTES:
        _violation("continuation_asset_manifest_too_large")
    return manifest


def _read_chunk(directory, descriptor):
    if set(descriptor) != _DESCRIPTOR_FIELDS:
        _violation("continuation_asset_descriptor_shape_mismatch")
    payload = (directory / "chunks" / f"{descriptor['digest']}.json").read_bytes()
    if len(payload) != descriptor["canonical_bytes"]:
        _violation("continuation_asset_chunk_size_mismatch")
    if len(payload) > _MAX_CHUNK_BYTES:
        _violation("continuation_asset_chunk_too_large")
    if hashlib.sha256(payload).hexdigest() != descriptor["digest"]:
        _violation("continuation_asset_chunk_digest_mismatch")
    chunk = json.loads(payload)
    if payload != _canonical_bytes(chunk):
        _violation("continuation_asset_chunk_not_canonical")
    if set(chunk) != {"schema_name", "schema_version", "kind", "items"}:
        _violation("continuation_asset_chunk_shape_mismatch")
    if (
        chunk["schema_name"] != CHUNK_SCHEMA_NAME
        or chunk["schema_version"] != CHUNK_SCHEMA_VERSION
        or chunk["kind"] != descriptor["kind"]
        or chunk["kind"] not in _CHUNK_KINDS
    ):
        _violation("continuation_asset_chunk_identity_mismatch")
    if len(chunk["items"]) != descriptor["item_count"]:
        _violation("continuation_asset_chunk_item_count_mismatch")
    keys = tuple(_record_key(chunk["kind"], item) for item in chunk["items"])
    if not keys or (
        _json_key(keys[0]) != descriptor["first_key"]
        or _json_key(keys[-1]) != descriptor["last_key"]
    ):
        _violation("continuation_asset_chunk_key_range_mismatch")
    return chunk


def _source_snapshot_from_asset(*, prepared, asset):
    if asset._source_snapshot_cache is not None:
        cached_prepared, cached_snapshot = asset._source_snapshot_cache
        if cached_prepared is not prepared:
            _violation("continuation_asset_session_prepared_mismatch")
        return cached_snapshot
    terms, options = _source_terms_for_prepared(prepared=prepared, asset=asset)
    cursor = writer_frontier_cursor_from_closed_terms(terms["cursor"]["terms"])
    depth = terms["decoder_boundary"]["consumed_token_count"]
    snapshot = capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=cursor,
        decoder_boundary=WriterDecoderBoundary(consumed_token_count=depth),
    )
    if (
        _identity_envelope(snapshot.prepared_identity)
        != asset.manifest["prepared_identity"]
        or _snapshot_identity_envelope(snapshot) != terms
    ):
        _violation("continuation_asset_source_snapshot_mismatch")
    asset._source_snapshot_cache = (prepared, snapshot)
    return snapshot


def _source_terms_for_prepared(*, prepared, asset):
    descriptor = asset.manifest["source_snapshot_chunk"]
    chunk = _read_chunk(asset.path, descriptor)
    terms = chunk["items"][0]
    options = _runtime_options_from_terms(terms["runtime_options"])
    expected_identity = _identity_envelope(
        writer_prepared_identity(prepared, options)
    )
    if expected_identity != asset.manifest["prepared_identity"]:
        _violation("continuation_asset_prepared_identity_mismatch")
    return terms, options


def _snapshot_for_raw_cursor(*, prepared, asset, cursor, raw_cursor_digest):
    source = _source_snapshot_from_asset(prepared=prepared, asset=asset)
    record = asset.raw_cursor_record(raw_cursor_digest)
    if record is None:
        _violation("continuation_asset_snapshot_cursor_record_missing")
    return capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=source.runtime_options,
        cursor=cursor,
        decoder_boundary=WriterDecoderBoundary(
            consumed_token_count=(
                source.decoder_boundary.consumed_token_count + record.token_depth
            )
        ),
    )


def _verify_predecessor_canonicality(provenance):
    reachable, depths, predecessors = _canonical_predecessor_tree(
        root_raw_cursor_digest=provenance.root_raw_cursor_digest,
        edges=provenance.edges,
    )
    records = {item.raw_cursor_digest: item for item in provenance.raw_cursors}
    if set(records) != set(reachable):
        _violation("continuation_asset_predecessor_coverage_mismatch")
    for digest, record in records.items():
        if (
            record.token_depth != depths[digest]
            or record.predecessor_edge_id != predecessors.get(digest)
        ):
            _violation("continuation_asset_predecessor_canonicality_mismatch")


def _verify_manifest_metrics(*, manifest, automaton):
    metrics = manifest["deterministic_metrics"]
    if set(metrics) != _ASSET_METRIC_FIELDS:
        _violation("continuation_asset_metric_shape_mismatch")
    expected = _deterministic_metrics(automaton)
    for key, value in expected.items():
        if metrics.get(key) != value:
            _violation("continuation_asset_metric_mismatch")
    compact = sum(
        metrics[key]
        for key in (
            "raw_cursor_record_bytes",
            "primitive_record_bytes",
            "edge_record_bytes",
            "terminal_record_bytes",
        )
    )
    if compact != metrics["compact_provenance_bytes"]:
        _violation("continuation_asset_compact_metric_mismatch")
    descriptor_bytes = {
        "raw_cursor_record_bytes": sum(
            item["canonical_bytes"] for item in manifest["raw_cursor_chunks"]
        ),
        "primitive_record_bytes": sum(
            item["canonical_bytes"] for item in manifest["primitive_chunks"]
        ),
        "edge_record_bytes": sum(
            item["canonical_bytes"] for item in manifest["edge_chunks"]
        ),
        "terminal_record_bytes": sum(
            item["canonical_bytes"] for item in manifest["terminal_chunks"]
        ),
    }
    if any(metrics[key] != value for key, value in descriptor_bytes.items()):
        _violation("continuation_asset_chunk_byte_metric_mismatch")
    descriptors = _all_descriptors(manifest)
    if (
        metrics["source_snapshot_bytes"]
        != manifest["source_snapshot_chunk"]["canonical_bytes"]
        or metrics["core_chunk_bytes"]
        != manifest["core_chunk"]["canonical_bytes"]
        or metrics["chunk_count"] != len(descriptors)
        or metrics["peak_serialization_buffer_bytes"]
        != max(item["canonical_bytes"] for item in descriptors)
    ):
        _violation("continuation_asset_deterministic_size_metric_mismatch")
    record_sizes = (
        *(
            len(_canonical_bytes(_raw_record_mapping(item)))
            for item in automaton.provenance.raw_cursors
        ),
        *(
            len(_canonical_bytes(_primitive_record_mapping(item)))
            for item in automaton.provenance.primitives
        ),
        *(
            len(_canonical_bytes(_edge_record_mapping(item)))
            for item in automaton.provenance.edges
        ),
        *(
            len(_canonical_bytes(_terminal_record_mapping(item)))
            for item in automaton.provenance.terminals
        ),
    )
    if metrics["largest_record_bytes"] != max(record_sizes, default=0):
        _violation("continuation_asset_largest_record_metric_mismatch")


def _core_mapping(automaton):
    return {
        "root": {
            "node_id": automaton.root.node_id,
            "completion_scale": automaton.root.completion_scale,
        },
        "nodes": [_node_mapping(item) for item in automaton.nodes],
    }


def _core_from_mapping(item, *, metrics):
    if set(item) != {"root", "nodes"}:
        _violation("continuation_asset_core_shape_mismatch")
    root = item["root"]
    if set(root) != {"node_id", "completion_scale"}:
        _violation("continuation_asset_root_shape_mismatch")
    nodes = tuple(_node_from_mapping(node) for node in item["nodes"])
    runtime_metrics = WriterContinuationMetrics(
        raw_cursor_count=metrics["raw_cursor_count"],
        primitive_cursor_count=metrics["primitive_cursor_count"],
        semantic_node_count=metrics["semantic_node_count"],
        semantic_edge_count=metrics["semantic_edge_count"],
        terminal_node_count=metrics["terminal_node_count"],
        maximum_depth=metrics["maximum_depth"],
        maximum_out_degree=metrics["maximum_out_degree"],
        largest_equivalence_class_membership=metrics[
            "largest_equivalence_class_membership"
        ],
        weight_normalization_merge_count=metrics[
            "weight_normalization_merge_count"
        ],
        semantic_minimization_merge_count=metrics[
            "semantic_minimization_merge_count"
        ],
        canonical_core_bytes=metrics["canonical_core_bytes"],
        provenance_index_bytes=metrics["provenance_index_bytes"],
        compile_time_ns=0,
        peak_active_depth=metrics["peak_active_depth"],
        peak_primitive_memo_size=metrics["peak_primitive_memo_size"],
        peak_signature_memo_size=metrics["peak_signature_memo_size"],
    )
    return WriterContinuationCore(
        root=WriterContinuationCursor(
            node_id=root["node_id"],
            completion_scale=root["completion_scale"],
        ),
        nodes=nodes,
        metrics=runtime_metrics,
    )


def _node_mapping(node):
    return {
        "node_id": node.node_id,
        "signature_digest": node.signature_digest,
        "terminal_available": node.terminal_available,
        "terminal_multiplicity": node.terminal_multiplicity,
        "terminal_completion_count": node.terminal_completion_count,
        "choices": [
            {
                "emitted_text": choice.emitted_text,
                "immediate_multiplicity": choice.immediate_multiplicity,
                "successor_node_id": choice.successor_node_id,
                "successor_scale": choice.successor_scale,
                "support_count": choice.support_count,
                "completion_count": choice.completion_count,
            }
            for choice in node.choices
        ],
        "support_count": node.support_count,
        "completion_count": node.completion_count,
    }


def _node_from_mapping(item):
    fields = {
        "node_id",
        "signature_digest",
        "terminal_available",
        "terminal_multiplicity",
        "terminal_completion_count",
        "choices",
        "support_count",
        "completion_count",
    }
    if set(item) != fields:
        _violation("continuation_asset_node_shape_mismatch")
    choices = []
    for choice in item["choices"]:
        if set(choice) != {
            "emitted_text",
            "immediate_multiplicity",
            "successor_node_id",
            "successor_scale",
            "support_count",
            "completion_count",
        }:
            _violation("continuation_asset_choice_shape_mismatch")
        choices.append(WriterContinuationChoice(**choice))
    return WriterContinuationNode(
        node_id=item["node_id"],
        signature_digest=item["signature_digest"],
        terminal_available=item["terminal_available"],
        terminal_multiplicity=item["terminal_multiplicity"],
        terminal_completion_count=item["terminal_completion_count"],
        choices=tuple(choices),
        support_count=item["support_count"],
        completion_count=item["completion_count"],
    )


def _raw_record_mapping(item):
    return {
        "raw_cursor_digest": item.raw_cursor_digest,
        "primitive_cursor_digest": item.primitive_cursor_digest,
        "normalization_scale": item.normalization_scale,
        "compiled_node_id": item.compiled_node_id,
        "token_depth": item.token_depth,
        "predecessor_edge_id": item.predecessor_edge_id,
    }


def _primitive_record_mapping(item):
    return {
        "primitive_cursor_digest": item.primitive_cursor_digest,
        "compiled_node_id": item.compiled_node_id,
        "representative_raw_cursor_digest": item.representative_raw_cursor_digest,
    }


def _edge_record_mapping(item):
    return {
        "edge_id": item.edge_id,
        "source_raw_cursor_digest": item.source_raw_cursor_digest,
        "source_node_id": item.source_node_id,
        "emitted_text": item.emitted_text,
        "text_projection_digest": item.text_projection_digest,
        "branch_certificate_digests": list(item.branch_certificate_digests),
        "successor_raw_cursor_digest": item.successor_raw_cursor_digest,
        "successor_node_id": item.successor_node_id,
        "successor_scale": item.successor_scale,
    }


def _terminal_record_mapping(item):
    return {
        "source_raw_cursor_digest": item.source_raw_cursor_digest,
        "source_node_id": item.source_node_id,
        "terminal_support_identity_digests": list(
            item.terminal_support_identity_digests
        ),
        "finalized_cursor_digest": item.finalized_cursor_digest,
    }


def _record_from_mapping(kind, item):
    expected = {
        "raw_cursor_records": _RAW_RECORD_FIELDS,
        "primitive_records": _PRIMITIVE_RECORD_FIELDS,
        "edge_records": _EDGE_RECORD_FIELDS,
        "terminal_records": _TERMINAL_RECORD_FIELDS,
    }[kind]
    if set(item) != expected:
        _violation("continuation_asset_record_shape_mismatch")
    if kind == "raw_cursor_records":
        return WriterContinuationRawCursorRecord(**item)
    if kind == "primitive_records":
        return WriterContinuationPrimitiveRecord(**item)
    if kind == "edge_records":
        return WriterContinuationEdgeRecord(
            **{**item, "branch_certificate_digests": tuple(item["branch_certificate_digests"])}
        )
    return WriterContinuationTerminalRecord(
        **{
            **item,
            "terminal_support_identity_digests": tuple(
                item["terminal_support_identity_digests"]
            ),
        }
    )


def _record_key(kind, item):
    if kind == "source_snapshot":
        return "source_snapshot"
    if kind == "automaton_core":
        return "automaton_core"
    if kind == "raw_cursor_records":
        return item["raw_cursor_digest"]
    if kind == "primitive_records":
        return item["primitive_cursor_digest"]
    if kind == "edge_records":
        return (
            item["source_raw_cursor_digest"],
            item["emitted_text"],
            item["text_projection_digest"],
        )
    if kind == "terminal_records":
        return item["source_raw_cursor_digest"]
    _violation("continuation_asset_unknown_chunk_kind")


def _deterministic_metrics(automaton):
    return {
        field.name: getattr(automaton.metrics, field.name)
        for field in fields(automaton.metrics)
        if field.name != "compile_time_ns"
    }


def _all_descriptors(manifest):
    return (
        manifest["source_snapshot_chunk"],
        manifest["core_chunk"],
        *manifest["raw_cursor_chunks"],
        *manifest["primitive_chunks"],
        *manifest["edge_chunks"],
        *manifest["terminal_chunks"],
    )


def _descriptor_list_field(kind):
    return {
        "raw_cursor_records": "raw_cursor_chunks",
        "primitive_records": "primitive_chunks",
        "edge_records": "edge_chunks",
        "terminal_records": "terminal_chunks",
    }[kind]


def _json_key(key):
    return list(key) if isinstance(key, tuple) else key


def _canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _digest_mapping(value):
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def _violation(kind):
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer continuation asset violation: {kind}",
    )


__all__ = (
    "WriterContinuationAsset",
    "WriterContinuationAssetSemanticVerification",
    "WriterContinuationAssetVerification",
    "WriterContinuationProofCursor",
    "advance_writer_continuation_proof",
    "branch_transition_artifact_from_continuation_asset",
    "open_writer_continuation_core",
    "reconstruct_writer_cursor_from_asset",
    "terminalization_artifact_from_continuation_asset",
    "verify_writer_continuation_asset_consistency",
    "verify_writer_continuation_asset_for_prepared",
    "verify_writer_continuation_asset_live",
    "verify_writer_continuation_cursor_envelope",
    "write_writer_continuation_asset",
    "writer_continuation_cursor_envelope",
)
