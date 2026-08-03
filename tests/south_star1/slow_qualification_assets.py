"""Domain operations over the shared slow-qualification cache state machine."""

from __future__ import annotations

from dataclasses import dataclass
import time
from pathlib import Path
from unittest.mock import patch

import grimace

from grimace._south_star1.public_continuation_asset import prepare_public_continuation_molecule
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions, SouthStarWriterSurface
from grimace._south_star1.writer_continuation_asset import (
    _certify_writer_continuation_asset_candidate,
    _materialize_writer_continuation_asset_candidate,
    open_writer_continuation_core,
    verify_writer_continuation_asset_consistency,
)
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_frontier import _checked_writer_frontier_product
from grimace._south_star1.writer_frontier_count_envelope import (
    _envelope_from_product,
    _verify_writer_frontier_count_envelope_against_product,
    verify_writer_frontier_count_envelope,
)
from grimace._south_star1.writer_support_artifact_checker import verify_writer_support_artifact_consistency
from grimace._south_star1.writer_support_artifact_envelope import _artifact_from_image
from grimace._south_star1.writer_support_image_envelope import _support_image_certificate_for_source
from grimace._south_star1.writer_snapshot import capture_initial_writer_frontier_snapshot
import grimace._south_star1.writer_continuation_asset as asset_module
from tests.south_star1.default_writer_capability_ledger import DefaultWriterCapabilityCase
from tests.south_star1.qualification_cache import (
    CachedQualificationEntry,
    QualificationCacheKind,
    QualificationCachePayloadKind,
    QualificationCacheState,
    atomic_write_text,
    canonical_json_sha256,
    canonical_json_text,
    cleanup_incomplete_qualification_cache,
    hidden_staging_path,
    inspect_qualification_cache,
    promote_directory_qualification_cache,
    publish_directory_qualification_cache,
    publish_json_qualification_cache,
    qualification_cache_context,
    qualification_cache_metadata,
    qualification_cache_paths,
    read_json_mapping,
)
from tests.south_star1.qualification_support import support_strings_digest


@dataclass(frozen=True, slots=True)
class CachedQualificationCandidate:
    entry: CachedQualificationEntry
    manifest_digest: str


@dataclass(frozen=True, slots=True)
class CachedQualificationAsset:
    entry: CachedQualificationEntry
    manifest_digest: str


@dataclass(frozen=True, slots=True)
class CachedQualificationCountEnvelope:
    entry: CachedQualificationEntry
    envelope: dict[str, object]
    dag_digest: str


@dataclass(frozen=True, slots=True)
class CachedQualificationSupportArtifact:
    entry: CachedQualificationEntry
    artifact: dict[str, object]
    artifact_digest: str
    support_count: int
    completion_count: int
    support_digest: str
    object_count: int
    canonical_bytes: int


def _paths(case: DefaultWriterCapabilityCase, kind: QualificationCacheKind):
    context = qualification_cache_context(case)
    return context, qualification_cache_paths(context, kind)


def _complete(paths, label: str) -> None:
    if inspect_qualification_cache(paths) is not QualificationCacheState.COMPLETE:
        raise AssertionError(f"slow {label} cache is absent or incomplete: {paths.context.case.name}")


def _entry(paths, metadata, *, cache_reused: bool) -> CachedQualificationEntry:
    return CachedQualificationEntry(
        context=paths.context,
        paths=paths,
        metadata=metadata,
        cache_reused=cache_reused,
    )


def _candidate_details(context, digest: str) -> dict[str, object]:
    return {
        "candidate_manifest_digest": digest,
        "stage": "structurally_verified_candidate",
    }


def _asset_details(context, digest: str) -> dict[str, object]:
    return {"asset_manifest_digest": digest}


def _count_details(envelope: dict[str, object]) -> dict[str, object]:
    dag = envelope["count_dag"]
    return {
        "prepared_identity_digest": envelope["prepared_identity"]["digest"],
        "support_count": envelope["support_count"],
        "completion_count": envelope["completion_count"],
        "count_dag_digest": dag["digest"],
        "count_dag_node_count": dag["metrics"]["node_count"],
        "count_envelope_sha256": canonical_json_sha256(envelope),
    }


def _artifact_details(artifact: dict[str, object], *, artifact_digest: str, canonical_bytes: int) -> dict[str, object]:
    objects = {item["object_id"]: item for item in artifact["objects"]}
    root = objects[artifact["roots"]["support_image_root"]]["payload"]
    return {
        "support_count": root["distinct_count"],
        "completion_count": root["witness_count"],
        "support_digest": support_strings_digest(tuple(root["support_strings"])),
        "artifact_sha256": artifact_digest,
        "object_count": len(artifact["objects"]),
        "canonical_bytes": canonical_bytes,
    }


def _read_metadata(paths):
    return read_json_mapping(paths.metadata_path)


def _read_count_entry(case: DefaultWriterCapabilityCase) -> CachedQualificationCountEnvelope:
    context, paths = _paths(case, QualificationCacheKind.COUNT_ENVELOPE)
    _complete(paths, "count envelope")
    envelope = read_json_mapping(paths.payload_path)
    metadata = _read_metadata(paths)
    expected = qualification_cache_metadata(context, paths.definition.kind, details=_count_details(envelope))
    if metadata != expected:
        raise AssertionError(f"slow count envelope metadata mismatch: {case.name}")
    return CachedQualificationCountEnvelope(
        entry=_entry(paths, metadata, cache_reused=True),
        envelope=envelope,
        dag_digest=envelope["count_dag"]["digest"],
    )


def build_slow_count_envelope(case: DefaultWriterCapabilityCase) -> CachedQualificationCountEnvelope:
    context, paths = _paths(case, QualificationCacheKind.COUNT_ENVELOPE)
    state = inspect_qualification_cache(paths)
    lookup_started = time.monotonic()
    print(f"cache_lookup_seconds={time.monotonic() - lookup_started:.3f}", flush=True)
    if state is QualificationCacheState.COMPLETE:
        cached = require_slow_count_envelope(case)
        print("cache_reused=true", flush=True)
        return cached
    if state is not QualificationCacheState.ABSENT:
        cleanup_incomplete_qualification_cache(paths)
    prepared, snapshot = _prepared_and_snapshot(case)
    started = time.monotonic()
    product = _checked_writer_frontier_product(
        prepared,
        snapshot.cursor,
        include_counts=True,
        include_frontier_certificate=True,
        include_count_certificate=True,
    )
    budget = WriterEnvelopeWorkBudget()
    envelope = _envelope_from_product(
        prepared=prepared,
        source_kind="snapshot",
        source_snapshot=snapshot,
        prefix_read_envelope=None,
        frontier_snapshot=snapshot,
        product=product,
        budget=budget,
    )
    if envelope["count_dag"]["metrics"]["node_count"] > budget.max_count_nodes:
        raise AssertionError("count DAG exceeds configured node ceiling")
    print(f"count_envelope_build_seconds={time.monotonic() - started:.3f}", flush=True)
    started = time.monotonic()
    metadata_details = _count_details(envelope)
    canonical_json_text(envelope)
    qualification_cache_metadata(context, paths.definition.kind, details=metadata_details)
    print(f"canonical_serialization_seconds={time.monotonic() - started:.3f}", flush=True)
    started = time.monotonic()
    metadata = publish_json_qualification_cache(
        paths, payload=envelope, metadata_details=metadata_details
    )
    print(f"cache_publish_seconds={time.monotonic() - started:.3f}", flush=True)
    print("cache_reused=false", flush=True)
    return CachedQualificationCountEnvelope(
        entry=_entry(paths, metadata, cache_reused=False),
        envelope=envelope,
        dag_digest=envelope["count_dag"]["digest"],
    )


def require_slow_count_envelope(case: DefaultWriterCapabilityCase) -> CachedQualificationCountEnvelope:
    cached = _read_count_entry(case)
    prepared, _snapshot = _prepared_and_snapshot(case)
    verification = verify_writer_frontier_count_envelope(
        prepared=prepared,
        envelope=cached.envelope,
        budget=WriterEnvelopeWorkBudget(),
    )
    if not verification.accepted:
        raise AssertionError(verification.reason)
    return cached


def build_slow_support_artifact(case: DefaultWriterCapabilityCase) -> CachedQualificationSupportArtifact:
    context, paths = _paths(case, QualificationCacheKind.SUPPORT_ARTIFACT)
    state = inspect_qualification_cache(paths)
    if state is QualificationCacheState.COMPLETE:
        cached = require_slow_support_artifact(case)
        return CachedQualificationSupportArtifact(
            entry=CachedQualificationEntry(
                context=cached.entry.context,
                paths=cached.entry.paths,
                metadata=cached.entry.metadata,
                cache_reused=True,
            ),
            artifact=cached.artifact,
            artifact_digest=cached.artifact_digest,
            support_count=cached.support_count,
            completion_count=cached.completion_count,
            support_digest=cached.support_digest,
            object_count=cached.object_count,
            canonical_bytes=cached.canonical_bytes,
        )
    if state is not QualificationCacheState.ABSENT:
        cleanup_incomplete_qualification_cache(paths)
    budget = WriterEnvelopeWorkBudget()
    started = time.monotonic()
    count_cached = _read_count_entry(case)
    print(f"count_cache_load_seconds={time.monotonic() - started:.3f}", flush=True)
    started = time.monotonic()
    prepared, snapshot = _prepared_and_snapshot(case)
    print(f"prepared_snapshot_seconds={time.monotonic() - started:.3f}", flush=True)
    started = time.monotonic()
    product = _checked_writer_frontier_product(
        prepared,
        snapshot.cursor,
        include_counts=True,
        include_frontier_certificate=True,
        include_count_certificate=True,
    )
    print(f"counted_product_seconds={time.monotonic() - started:.3f}", flush=True)
    started = time.monotonic()
    _verify_writer_frontier_count_envelope_against_product(
        prepared=prepared,
        frontier_snapshot=snapshot,
        product=product,
        envelope=count_cached.envelope,
        budget=budget,
    )
    print(f"count_binding_verification_seconds={time.monotonic() - started:.3f}", flush=True)
    started = time.monotonic()
    image = _support_image_certificate_for_source(
        prepared=prepared, snapshot=snapshot, product=product
    )
    print(f"support_image_seconds={time.monotonic() - started:.3f}", flush=True)
    started = time.monotonic()
    artifact = _artifact_from_image(
        prepared=prepared,
        source_kind="snapshot",
        source_snapshot=snapshot,
        prefix_read_envelope=None,
        count_envelope=count_cached.envelope,
        product=product,
        image=image,
        budget=budget,
    )
    print(f"artifact_assembly_seconds={time.monotonic() - started:.3f}", flush=True)
    structural = verify_writer_support_artifact_consistency(artifact)
    if not structural.accepted:
        raise AssertionError(structural.reason)
    started = time.monotonic()
    artifact_text = canonical_json_text(artifact) + "\n"
    artifact_digest = artifact["digest"]
    metadata_details = _artifact_details(
        artifact,
        artifact_digest=artifact_digest,
        canonical_bytes=len(artifact_text.encode("utf-8")),
    )
    print(f"canonical_serialization_seconds={time.monotonic() - started:.3f}", flush=True)
    started = time.monotonic()
    metadata = publish_json_qualification_cache(
        paths, payload=artifact, metadata_details=metadata_details
    )
    print(f"cache_publish_seconds={time.monotonic() - started:.3f}", flush=True)
    objects = artifact["objects"]
    root = {item["object_id"]: item for item in objects}[artifact["roots"]["support_image_root"]]["payload"]
    result = CachedQualificationSupportArtifact(
        entry=_entry(paths, metadata, cache_reused=False),
        artifact=artifact,
        artifact_digest=artifact_digest,
        support_count=root["distinct_count"],
        completion_count=root["witness_count"],
        support_digest=metadata["support_digest"],
        object_count=len(objects),
        canonical_bytes=len(artifact_text.encode("utf-8")),
    )
    print(f"artifact_object_count={result.object_count}", flush=True)
    print(f"artifact_canonical_bytes={result.canonical_bytes}", flush=True)
    print("cache_reused=false", flush=True)
    return result


def require_slow_support_artifact(case: DefaultWriterCapabilityCase) -> CachedQualificationSupportArtifact:
    context, paths = _paths(case, QualificationCacheKind.SUPPORT_ARTIFACT)
    _complete(paths, "support artifact")
    artifact = read_json_mapping(paths.payload_path)
    metadata = _read_metadata(paths)
    objects = {item["object_id"]: item for item in artifact["objects"]}
    root = objects[artifact["roots"]["support_image_root"]]["payload"]
    artifact_digest = artifact["digest"]
    expected = qualification_cache_metadata(
        context,
        paths.definition.kind,
        details=_artifact_details(
            artifact,
            artifact_digest=artifact_digest,
            canonical_bytes=len((canonical_json_text(artifact) + "\n").encode("utf-8")),
        ),
    )
    if metadata != expected:
        raise AssertionError(f"slow support artifact metadata mismatch: {case.name}")
    structural = verify_writer_support_artifact_consistency(artifact)
    if not structural.accepted:
        raise AssertionError(structural.reason)
    return CachedQualificationSupportArtifact(
        entry=_entry(paths, metadata, cache_reused=True),
        artifact=artifact,
        artifact_digest=artifact_digest,
        support_count=root["distinct_count"],
        completion_count=root["witness_count"],
        support_digest=metadata["support_digest"],
        object_count=len(artifact["objects"]),
        canonical_bytes=len((canonical_json_text(artifact) + "\n").encode("utf-8")),
    )


def build_slow_qualification_candidate(case: DefaultWriterCapabilityCase) -> CachedQualificationCandidate:
    context, paths = _paths(case, QualificationCacheKind.CONTINUATION_CANDIDATE)
    state = inspect_qualification_cache(paths)
    if state is QualificationCacheState.COMPLETE:
        return require_slow_qualification_candidate(case)
    if state is not QualificationCacheState.ABSENT:
        cleanup_incomplete_qualification_cache(paths)
    context.case_dir.mkdir(parents=True, exist_ok=True)
    prepared, snapshot = _prepared_and_snapshot(case)
    staged = hidden_staging_path(paths, "stage")
    manifest = _materialize_candidate_timed(staged, prepared, snapshot)
    metadata = publish_directory_qualification_cache(
        paths,
        staged_payload_path=staged,
        metadata_details=_candidate_details(context, manifest["digest"]),
    )
    return CachedQualificationCandidate(
        entry=_entry(paths, metadata, cache_reused=False),
        manifest_digest=manifest["digest"],
    )


def require_slow_qualification_candidate(case: DefaultWriterCapabilityCase) -> CachedQualificationCandidate:
    context, paths = _paths(case, QualificationCacheKind.CONTINUATION_CANDIDATE)
    _complete(paths, "qualification candidate")
    metadata = _read_metadata(paths)
    digest = metadata.get("candidate_manifest_digest")
    expected = qualification_cache_metadata(
        context, paths.definition.kind, details=_candidate_details(context, digest)
    )
    if metadata != expected:
        raise AssertionError(f"slow qualification candidate metadata mismatch: {case.name}")
    structural = verify_writer_continuation_asset_consistency(paths.payload_path)
    if not structural.accepted:
        raise AssertionError(structural.reason)
    opened = open_writer_continuation_core(paths.payload_path)
    if opened.manifest_digest != digest:
        raise AssertionError(f"slow qualification candidate manifest mismatch: {case.name}")
    return CachedQualificationCandidate(
        entry=_entry(paths, metadata, cache_reused=True), manifest_digest=digest
    )


def certify_slow_qualification_candidate(case: DefaultWriterCapabilityCase) -> CachedQualificationAsset:
    candidate = require_slow_qualification_candidate(case)
    print("candidate_cache_validation_seconds=0.000", flush=True)
    prepared, _snapshot = _prepared_and_snapshot(case)
    print("candidate_semantic_verification_started", flush=True)
    started = time.monotonic()
    semantic = _certify_writer_continuation_asset_candidate(
        path=candidate.entry.paths.payload_path,
        prepared=prepared,
        expected_manifest_digest=candidate.manifest_digest,
    )
    if not semantic.accepted:
        raise AssertionError(semantic.reason)
    print(f"candidate_semantic_verification_seconds={time.monotonic() - started:.3f}", flush=True)
    context, destination = _paths(case, QualificationCacheKind.CONTINUATION_ASSET)
    if inspect_qualification_cache(destination) is not QualificationCacheState.ABSENT:
        raise AssertionError(f"slow qualification final asset already exists: {case.name}")
    started = time.monotonic()
    metadata = promote_directory_qualification_cache(
        source_paths=candidate.entry.paths,
        destination_paths=destination,
        destination_metadata_details=_asset_details(context, candidate.manifest_digest),
    )
    print(f"candidate_publish_seconds={time.monotonic() - started:.3f}", flush=True)
    return CachedQualificationAsset(
        entry=_entry(destination, metadata, cache_reused=False),
        manifest_digest=candidate.manifest_digest,
    )


def require_slow_qualification_asset(case: DefaultWriterCapabilityCase) -> CachedQualificationAsset:
    context, paths = _paths(case, QualificationCacheKind.CONTINUATION_ASSET)
    _complete(paths, "qualification asset")
    metadata = _read_metadata(paths)
    digest = metadata.get("asset_manifest_digest")
    expected = qualification_cache_metadata(
        context, paths.definition.kind, details=_asset_details(context, digest)
    )
    if metadata != expected:
        raise AssertionError(f"slow qualification metadata mismatch: {case.name}")
    structural = verify_writer_continuation_asset_consistency(paths.payload_path)
    if not structural.accepted:
        raise AssertionError(structural.reason)
    opened = open_writer_continuation_core(paths.payload_path)
    if opened.manifest_digest != digest:
        raise AssertionError(f"slow qualification manifest mismatch: {case.name}")
    return CachedQualificationAsset(
        entry=_entry(paths, metadata, cache_reused=True), manifest_digest=digest
    )


def _prepared_and_snapshot(case: DefaultWriterCapabilityCase):
    from grimace._south_star1.policy import SerializationLanguageMode

    options = SouthStarRuntimeOptions(
        rooted_at_atom=case.rooted_at_atom,
        canonical=False,
        do_random=True,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )
    prepared = prepare_public_continuation_molecule(
        _mol(case), writer_surface=SouthStarWriterSurface(), runtime_options=options
    )
    snapshot = capture_initial_writer_frontier_snapshot(
        prepared=prepared, runtime_options=options
    )
    return prepared, snapshot


def _materialize_candidate_timed(candidate, prepared, snapshot):
    started = time.monotonic()
    original_compile = asset_module.compile_writer_continuation_automaton
    original_verify_internal = asset_module._verify_internal_consistency
    original_structural = asset_module.verify_writer_continuation_asset_consistency
    serialization_started = None
    structural_started = None

    def compile_wrapper(*args, **kwargs):
        print("automaton_compile_started", flush=True)
        check_started = time.monotonic()
        result = original_compile(*args, **kwargs)
        print(f"automaton_compile_seconds={time.monotonic() - check_started:.3f}", flush=True)
        return result

    def verify_internal_wrapper(*args, **kwargs):
        print("automaton_internal_verification_started", flush=True)
        check_started = time.monotonic()
        result = original_verify_internal(*args, **kwargs)
        print(f"automaton_internal_verification_seconds={time.monotonic() - check_started:.3f}", flush=True)
        return result

    def singleton_wrapper(*args, **kwargs):
        nonlocal serialization_started
        if serialization_started is None:
            serialization_started = time.monotonic()
            print("candidate_serialization_started", flush=True)
        return original_singleton(*args, **kwargs)

    def structural_wrapper(path, *args, **kwargs):
        nonlocal structural_started
        if structural_started is None:
            print(f"candidate_serialization_seconds={time.monotonic() - serialization_started:.3f}", flush=True)
            print("candidate_structural_verification_started", flush=True)
            structural_started = time.monotonic()
        result = original_structural(path, *args, **kwargs)
        print(f"candidate_structural_verification_seconds={time.monotonic() - structural_started:.3f}", flush=True)
        return result

    original_singleton = asset_module._write_singleton_chunk
    with (
        patch.object(asset_module, "compile_writer_continuation_automaton", compile_wrapper),
        patch.object(asset_module, "_verify_internal_consistency", verify_internal_wrapper),
        patch.object(asset_module, "_write_singleton_chunk", singleton_wrapper),
        patch.object(asset_module, "verify_writer_continuation_asset_consistency", structural_wrapper),
    ):
        manifest = _materialize_writer_continuation_asset_candidate(
            path=candidate, prepared=prepared, snapshot=snapshot
        )
    print(f"candidate_total_seconds={time.monotonic() - started:.3f}", flush=True)
    return manifest


def _mol(case):
    from rdkit import Chem

    return Chem.MolFromSmiles(case.smiles)


__all__ = (
    "CachedQualificationAsset",
    "CachedQualificationCandidate",
    "CachedQualificationCountEnvelope",
    "CachedQualificationSupportArtifact",
    "build_slow_count_envelope",
    "build_slow_support_artifact",
    "build_slow_qualification_candidate",
    "certify_slow_qualification_candidate",
    "require_slow_count_envelope",
    "require_slow_support_artifact",
    "require_slow_qualification_candidate",
    "require_slow_qualification_asset",
)
