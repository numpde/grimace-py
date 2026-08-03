"""Revision-bound candidate and asset caches for slow qualification."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import os
from pathlib import Path
import shutil
import subprocess
import time
from unittest.mock import patch

import grimace

from grimace._south_star1.public_continuation_asset import (
    prepare_public_continuation_molecule,
)
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.writer_continuation_asset import (
    _certify_writer_continuation_asset_candidate,
)
from grimace._south_star1.writer_frontier import _checked_writer_frontier_product
from grimace._south_star1.writer_frontier_count_envelope import (
    _envelope_from_product,
    _verify_writer_frontier_count_envelope_against_product,
    verify_writer_frontier_count_envelope,
)
from grimace._south_star1.writer_support_artifact_checker import (
    verify_writer_support_artifact_consistency,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    _artifact_from_image,
)
from grimace._south_star1.writer_support_image_envelope import (
    _support_image_certificate_for_source,
)
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_continuation_asset import (
    _materialize_writer_continuation_asset_candidate,
)
from grimace._south_star1.writer_continuation_asset import open_writer_continuation_core
from grimace._south_star1.writer_continuation_asset import (
    verify_writer_continuation_asset_consistency,
)
import grimace._south_star1.writer_continuation_asset as asset_module
from grimace._south_star1.writer_snapshot import capture_initial_writer_frontier_snapshot
from tests.south_star1.default_writer_capability_ledger import (
    DefaultWriterCapabilityCase,
)
from tests.south_star1.qualification_support import support_strings_digest


_ASSET_SCHEMA = "south_star1_slow_qualification_asset"
_CANDIDATE_SCHEMA = "south_star1_slow_qualification_candidate"
_COUNT_SCHEMA = "south_star1_slow_count_envelope"
_ARTIFACT_SCHEMA = "south_star1_slow_support_artifact"
_METADATA_VERSION = 1
_ASSET_ROOT_ENV = "SOUTH_STAR1_SLOW_ASSET_ROOT"


@dataclass(frozen=True, slots=True)
class CachedQualificationCandidate:
    case: DefaultWriterCapabilityCase
    candidate_path: Path
    metadata_path: Path
    manifest_digest: str


@dataclass(frozen=True, slots=True)
class CachedQualificationAsset:
    case: DefaultWriterCapabilityCase
    asset_path: Path
    metadata_path: Path
    manifest_digest: str
    cache_reused: bool = False

@dataclass(frozen=True, slots=True)
class CachedQualificationCountEnvelope:
    case: DefaultWriterCapabilityCase
    envelope_path: Path
    metadata_path: Path
    dag_digest: str


@dataclass(frozen=True, slots=True)
class LoadedQualificationCountEnvelope:
    cached: CachedQualificationCountEnvelope
    envelope: dict[str, object]
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class CachedQualificationSupportArtifact:
    case: DefaultWriterCapabilityCase
    artifact_path: Path
    metadata_path: Path
    artifact_digest: str
    support_count: int
    completion_count: int
    support_digest: str
    object_count: int
    canonical_bytes: int
    cache_reused: bool

def build_slow_count_envelope(case):
    case_dir = _case_dir(case)
    case_dir.mkdir(parents=True, exist_ok=True)
    envelope_path = case_dir / "count-envelope.json"
    metadata_path = case_dir / "count-envelope-metadata.json"
    lookup_started = time.monotonic()
    both = envelope_path.exists() and metadata_path.exists()
    either = envelope_path.exists() or metadata_path.exists()
    print(f"cache_lookup_seconds={time.monotonic() - lookup_started:.3f}", flush=True)
    if both:
        cached = require_slow_count_envelope(case)
        print("cache_reused=true", flush=True)
        return cached
    if either:
        print("incomplete_count_cache=true", flush=True)
        envelope_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)
    prepared, snapshot = _prepared_and_snapshot(case)
    build_started = time.monotonic()
    product = _checked_writer_frontier_product(
        prepared, snapshot.cursor, include_counts=True,
        include_frontier_certificate=True, include_count_certificate=True,
    )
    envelope = _envelope_from_product(
        prepared=prepared, source_kind="snapshot", source_snapshot=snapshot,
        prefix_read_envelope=None, frontier_snapshot=snapshot, product=product,
        budget=WriterEnvelopeWorkBudget(),
    )
    metrics = envelope["count_dag"]["metrics"]
    if metrics["node_count"] > 20_000:
        raise AssertionError("count DAG exceeds requalified node ceiling")
    print(f"count_envelope_build_seconds={time.monotonic() - build_started:.3f}", flush=True)
    serialization_started = time.monotonic()
    envelope_json = json.dumps(envelope, sort_keys=True, separators=(",", ":")) + "\n"
    metadata_json = json.dumps(_count_metadata(case, envelope), sort_keys=True, separators=(",", ":")) + "\n"
    print(f"canonical_serialization_seconds={time.monotonic() - serialization_started:.3f}", flush=True)
    publish_started = time.monotonic()
    _atomic_text(envelope_path, envelope_json)
    _atomic_text(metadata_path, metadata_json)
    print(f"cache_publish_seconds={time.monotonic() - publish_started:.3f}", flush=True)
    print("cache_reused=false", flush=True)
    return CachedQualificationCountEnvelope(case, envelope_path, metadata_path, envelope["count_dag"]["digest"])

def store_slow_count_envelope(case, envelope):
    case_dir = _case_dir(case)
    case_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(case_dir / "count-envelope.json", envelope)
    _atomic_json(case_dir / "count-envelope-metadata.json", _count_metadata(case, envelope))
    return require_slow_count_envelope(case)

def require_slow_count_envelope(case):
    case_dir = _case_dir(case)
    envelope_path = case_dir / "count-envelope.json"
    metadata_path = case_dir / "count-envelope-metadata.json"
    if not envelope_path.is_file() or not metadata_path.is_file():
        raise AssertionError(f"slow count envelope is absent: {case.name}")
    loaded = load_slow_count_envelope(case)
    envelope = loaded.envelope
    prepared, snapshot = _prepared_and_snapshot(case)
    verification = verify_writer_frontier_count_envelope(
        prepared=prepared, envelope=envelope, budget=WriterEnvelopeWorkBudget()
    )
    if not verification.accepted:
        raise AssertionError(verification.reason)
    return CachedQualificationCountEnvelope(
        case, envelope_path, metadata_path, envelope["count_dag"]["digest"]
    )


def load_slow_count_envelope(case):
    case_dir = _case_dir(case)
    envelope_path = case_dir / "count-envelope.json"
    metadata_path = case_dir / "count-envelope-metadata.json"
    if not envelope_path.is_file() or not metadata_path.is_file():
        raise AssertionError(f"slow count envelope is absent: {case.name}")
    envelope = json.loads(envelope_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if not isinstance(envelope, dict) or not isinstance(metadata, dict):
        raise AssertionError(f"slow count envelope is not a JSON mapping: {case.name}")
    if metadata != _count_metadata(case, envelope):
        raise AssertionError(f"slow count envelope metadata mismatch: {case.name}")
    return LoadedQualificationCountEnvelope(
        cached=CachedQualificationCountEnvelope(
            case, envelope_path, metadata_path, envelope["count_dag"]["digest"]
        ),
        envelope=envelope,
        metadata=metadata,
    )


def build_slow_support_artifact(case):
    case_dir = _case_dir(case)
    artifact_path = case_dir / "support-artifact.json"
    metadata_path = case_dir / "support-artifact-metadata.json"
    both = artifact_path.is_file() and metadata_path.is_file()
    either = artifact_path.exists() or metadata_path.exists()
    if both:
        cached = require_slow_support_artifact(case)
        return CachedQualificationSupportArtifact(
            case=case,
            artifact_path=cached.artifact_path,
            metadata_path=cached.metadata_path,
            artifact_digest=cached.artifact_digest,
            support_count=cached.support_count,
            completion_count=cached.completion_count,
            support_digest=cached.support_digest,
            object_count=cached.object_count,
            canonical_bytes=cached.canonical_bytes,
            cache_reused=True,
        )
    if either:
        artifact_path.unlink(missing_ok=True)
        metadata_path.unlink(missing_ok=True)

    budget = WriterEnvelopeWorkBudget()
    started = time.monotonic()
    loaded = load_slow_count_envelope(case)
    print(f"count_cache_load_seconds={time.monotonic() - started:.3f}", flush=True)

    started = time.monotonic()
    prepared, snapshot = _prepared_and_snapshot(case)
    print(f"prepared_snapshot_seconds={time.monotonic() - started:.3f}", flush=True)

    started = time.monotonic()
    product = _checked_writer_frontier_product(
        prepared, snapshot.cursor, include_counts=True,
        include_frontier_certificate=True, include_count_certificate=True,
    )
    print(f"counted_product_seconds={time.monotonic() - started:.3f}", flush=True)

    started = time.monotonic()
    _verify_writer_frontier_count_envelope_against_product(
        prepared=prepared,
        frontier_snapshot=snapshot,
        product=product,
        envelope=loaded.envelope,
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
        count_envelope=loaded.envelope,
        product=product,
        image=image,
        budget=budget,
    )
    print(f"artifact_assembly_seconds={time.monotonic() - started:.3f}", flush=True)
    structural = verify_writer_support_artifact_consistency(artifact)
    if not structural.accepted:
        raise AssertionError(structural.reason)
    canonical_started = time.monotonic()
    artifact_text = _canonical_json_text(artifact)
    artifact_digest = sha256(artifact_text.encode()).hexdigest()
    metadata = _artifact_metadata(
        case, artifact, artifact_digest=artifact_digest,
        canonical_bytes=len(artifact_text.encode()),
    )
    metadata_text = _canonical_json_text(metadata)
    print(f"canonical_serialization_seconds={time.monotonic() - canonical_started:.3f}", flush=True)
    publish_started = time.monotonic()
    _atomic_text(artifact_path, artifact_text + "\n")
    _atomic_text(metadata_path, metadata_text + "\n")
    print(f"cache_publish_seconds={time.monotonic() - publish_started:.3f}", flush=True)
    objects = artifact["objects"]
    root = {item["object_id"]: item for item in objects}[artifact["roots"]["support_image_root"]]["payload"]
    result = CachedQualificationSupportArtifact(
        case=case, artifact_path=artifact_path, metadata_path=metadata_path,
        artifact_digest=artifact["digest"], support_count=root["distinct_count"],
        completion_count=root["witness_count"], support_digest=metadata["support_digest"],
        object_count=len(objects), canonical_bytes=len(artifact_text.encode()),
        cache_reused=False,
    )
    print(f"artifact_object_count={result.object_count}", flush=True)
    print(f"artifact_canonical_bytes={result.canonical_bytes}", flush=True)
    print("cache_reused=false", flush=True)
    return result

def require_slow_support_artifact(case):
    case_dir = _case_dir(case)
    artifact_path = case_dir / "support-artifact.json"
    metadata_path = case_dir / "support-artifact-metadata.json"
    if not artifact_path.is_file() or not metadata_path.is_file():
        raise AssertionError(f"slow support artifact is absent: {case.name}")
    artifact = json.loads(artifact_path.read_text())
    metadata = json.loads(metadata_path.read_text())
    if metadata != _artifact_metadata(case, artifact):
        raise AssertionError(f"slow support artifact metadata mismatch: {case.name}")
    structural = verify_writer_support_artifact_consistency(artifact)
    if not structural.accepted:
        raise AssertionError(structural.reason)
    objects = artifact["objects"]
    root = {item["object_id"]: item for item in objects}[artifact["roots"]["support_image_root"]]["payload"]
    canonical_bytes = len(_canonical_json_text(artifact).encode())
    return CachedQualificationSupportArtifact(
        case=case, artifact_path=artifact_path, metadata_path=metadata_path,
        artifact_digest=artifact["digest"], support_count=root["distinct_count"],
        completion_count=root["witness_count"], support_digest=_support_digest(root),
        object_count=len(objects), canonical_bytes=canonical_bytes,
        cache_reused=True,
    )

def _count_metadata(case, envelope):
    digest = envelope["count_dag"]["digest"]
    return {
        "schema": _COUNT_SCHEMA, "schema_version": _METADATA_VERSION,
        "git_head": _git_head(), "case_name": case.name,
        "source_smiles": case.smiles, "rooted_atom": case.rooted_at_atom,
        "prepared_identity_digest": envelope["prepared_identity"]["digest"],
        "support_count": envelope["support_count"],
        "completion_count": envelope["completion_count"],
        "count_dag_digest": digest,
        "count_dag_node_count": envelope["count_dag"]["metrics"]["node_count"],
        "count_envelope_sha256": _json_sha256(envelope),
    }

def _artifact_metadata(case, artifact, *, artifact_digest=None, canonical_bytes=None):
    objects = {item["object_id"]: item for item in artifact["objects"]}
    root = objects[artifact["roots"]["support_image_root"]]["payload"]
    return {
        "schema": _ARTIFACT_SCHEMA, "schema_version": _METADATA_VERSION,
        "git_head": _git_head(), "case_name": case.name,
        "source_smiles": case.smiles, "rooted_atom": case.rooted_at_atom,
        "support_count": root["distinct_count"],
        "completion_count": root["witness_count"],
        "support_digest": support_strings_digest(tuple(root["support_strings"])),
        "artifact_sha256": artifact_digest if artifact_digest is not None else _json_sha256(artifact),
        "object_count": len(artifact["objects"]),
        "canonical_bytes": canonical_bytes if canonical_bytes is not None else len(_canonical_json_text(artifact).encode()),
    }


def _support_digest(root):
    return support_strings_digest(tuple(root["support_strings"]))


def _canonical_json_text(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))

def _json_sha256(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()

def _atomic_json(path, value):
    _atomic_text(path, json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n")

def _atomic_text(path, text):
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def build_slow_qualification_candidate(case: DefaultWriterCapabilityCase):
    case_dir = _case_dir(case)
    _clean_interrupted_state(case_dir)
    candidate = case_dir / "candidate"
    metadata_path = case_dir / "candidate-metadata.json"
    if candidate.exists() or metadata_path.exists():
        try:
            return require_slow_qualification_candidate(case)
        except Exception:
            _clean_interrupted_state(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    prepared, snapshot = _prepared_and_snapshot(case)
    manifest = _materialize_candidate_timed(candidate, prepared, snapshot)
    metadata_path.write_text(
        json.dumps(_candidate_metadata(case, manifest["digest"]), sort_keys=True,
                   separators=(",", ":")) + "\n"
    )
    return CachedQualificationCandidate(case, candidate, metadata_path, manifest["digest"])


def build_slow_qualification_asset(case: DefaultWriterCapabilityCase):
    """Compatibility helper for the older temporary-asset unit tests."""
    from grimace._south_star1 import public_continuation_asset as public_asset

    case_dir = _case_dir(case)
    _clean_interrupted_state(case_dir)
    case_dir.mkdir(parents=True, exist_ok=True)
    asset_path = case_dir / "asset"
    digest = grimace.BuildMolToSmilesContinuationAsset(
        _mol(case), asset_path, rootedAtAtom=case.rooted_at_atom
    )
    structural = verify_writer_continuation_asset_consistency(asset_path)
    if not structural.accepted:
        raise AssertionError(structural.reason)
    metadata_path = case_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(_asset_metadata(case, digest), sort_keys=True, separators=(",", ":")) + "\n"
    )
    return CachedQualificationAsset(case, asset_path, metadata_path, digest)


def require_slow_qualification_candidate(case: DefaultWriterCapabilityCase):
    case_dir = _case_dir(case)
    candidate = case_dir / "candidate"
    metadata_path = case_dir / "candidate-metadata.json"
    if not candidate.is_dir() or not metadata_path.is_file():
        raise AssertionError(f"slow qualification candidate is absent: {case.name}")
    metadata = json.loads(metadata_path.read_text())
    expected = _candidate_metadata(case, metadata.get("candidate_manifest_digest"))
    if metadata != expected:
        raise AssertionError(f"slow qualification candidate metadata mismatch: {case.name}")
    structural = verify_writer_continuation_asset_consistency(candidate)
    if not structural.accepted:
        raise AssertionError(structural.reason)
    opened = open_writer_continuation_core(candidate)
    if opened.manifest_digest != metadata["candidate_manifest_digest"]:
        raise AssertionError(f"slow qualification candidate manifest mismatch: {case.name}")
    return CachedQualificationCandidate(case, candidate, metadata_path, opened.manifest_digest)


def certify_slow_qualification_candidate(case: DefaultWriterCapabilityCase):
    cached = require_slow_qualification_candidate(case)
    validation_started = time.monotonic()
    require_slow_qualification_candidate(case)
    print(f"candidate_cache_validation_seconds={time.monotonic() - validation_started:.3f}", flush=True)
    prepared, _snapshot = _prepared_and_snapshot(case)
    print("candidate_semantic_verification_started", flush=True)
    semantic_started = time.monotonic()
    semantic = _certify_writer_continuation_asset_candidate(
        path=cached.candidate_path,
        prepared=prepared,
        expected_manifest_digest=cached.manifest_digest,
    )
    if not semantic.accepted:
        raise AssertionError(semantic.reason)
    print(f"candidate_semantic_verification_seconds={time.monotonic() - semantic_started:.3f}", flush=True)
    case_dir = cached.candidate_path.parent
    asset_path = case_dir / "asset"
    metadata_path = case_dir / "metadata.json"
    if asset_path.exists() or metadata_path.exists():
        raise AssertionError(f"slow qualification final asset already exists: {case.name}")
    publish_started = time.monotonic()
    os.replace(cached.candidate_path, asset_path)
    final_metadata = _asset_metadata(case, cached.manifest_digest)
    metadata_path.write_text(
        json.dumps(final_metadata, sort_keys=True, separators=(",", ":")) + "\n"
    )
    cached.metadata_path.unlink()
    print(f"candidate_publish_seconds={time.monotonic() - publish_started:.3f}", flush=True)
    return CachedQualificationAsset(case, asset_path, metadata_path, cached.manifest_digest)


def _materialize_candidate_timed(candidate, prepared, snapshot):
    started = time.monotonic()
    compile_started = None
    serialization_started = None
    structural_started = None
    original_compile = asset_module.compile_writer_continuation_automaton
    original_verify_internal = asset_module._verify_internal_consistency
    original_structural = asset_module.verify_writer_continuation_asset_consistency

    def compile_wrapper(*args, **kwargs):
        nonlocal compile_started
        compile_started = time.monotonic()
        print("automaton_compile_started", flush=True)
        result = original_compile(*args, **kwargs)
        print(f"automaton_compile_seconds={time.monotonic() - compile_started:.3f}", flush=True)
        return result

    def verify_internal_wrapper(*args, **kwargs):
        print("automaton_internal_verification_started", flush=True)
        check_started = time.monotonic()
        result = original_verify_internal(*args, **kwargs)
        print(f"automaton_internal_verification_seconds={time.monotonic() - check_started:.3f}", flush=True)
        return result

    def structural_wrapper(path, *args, **kwargs):
        nonlocal serialization_started, structural_started
        if serialization_started is not None and structural_started is None:
            print(f"candidate_serialization_seconds={time.monotonic() - serialization_started:.3f}", flush=True)
            print("candidate_structural_verification_started", flush=True)
            structural_started = time.monotonic()
        result = original_structural(path, *args, **kwargs)
        if structural_started is not None:
            print(f"candidate_structural_verification_seconds={time.monotonic() - structural_started:.3f}", flush=True)
        return result

    original_singleton = asset_module._write_singleton_chunk

    def singleton_wrapper(*args, **kwargs):
        nonlocal serialization_started
        if serialization_started is None:
            serialization_started = time.monotonic()
            print("candidate_serialization_started", flush=True)
        return original_singleton(*args, **kwargs)

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


def require_slow_qualification_asset(case: DefaultWriterCapabilityCase):
    case_dir = _case_dir(case)
    asset_path = case_dir / "asset"
    metadata_path = case_dir / "metadata.json"
    if not asset_path.is_dir() or not metadata_path.is_file():
        raise AssertionError(f"slow qualification asset is absent: {case.name}")
    metadata = json.loads(metadata_path.read_text())
    expected = _asset_metadata(case, metadata.get("asset_manifest_digest"))
    if metadata != expected:
        raise AssertionError(f"slow qualification metadata mismatch: {case.name}")
    structural = verify_writer_continuation_asset_consistency(asset_path)
    if not structural.accepted:
        raise AssertionError(structural.reason)
    opened = open_writer_continuation_core(asset_path)
    if opened.manifest_digest != metadata["asset_manifest_digest"]:
        raise AssertionError(f"slow qualification manifest mismatch: {case.name}")
    return CachedQualificationAsset(case, asset_path, metadata_path, opened.manifest_digest)


def _prepared_and_snapshot(case):
    from grimace._south_star1.policy import SerializationLanguageMode

    options = SouthStarRuntimeOptions(
        rooted_at_atom=case.rooted_at_atom,
        canonical=False,
        do_random=True,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )
    prepared = prepare_public_continuation_molecule(
        _mol(case),
        writer_surface=SouthStarWriterSurface(),
        runtime_options=options,
    )
    snapshot = capture_initial_writer_frontier_snapshot(
        prepared=prepared, runtime_options=options
    )
    return prepared, snapshot


def _clean_interrupted_state(case_dir: Path) -> None:
    if not case_dir.exists():
        return
    candidate = case_dir / "candidate"
    candidate_metadata = case_dir / "candidate-metadata.json"
    asset = case_dir / "asset"
    metadata = case_dir / "metadata.json"
    if candidate.exists() and not candidate_metadata.exists() or candidate_metadata.exists() and not candidate.exists():
        _report_stale_candidate(candidate)
        shutil.rmtree(candidate, ignore_errors=True)
        candidate_metadata.unlink(missing_ok=True)
    if asset.exists() and not metadata.exists() or metadata.exists() and not asset.exists():
        shutil.rmtree(asset, ignore_errors=True)
        metadata.unlink(missing_ok=True)
    for stale in case_dir.glob(".asset.*"):
        _report_stale_candidate(stale)
        shutil.rmtree(stale, ignore_errors=True)
    for stale in case_dir.glob(".candidate.*"):
        _report_stale_candidate(stale)
        shutil.rmtree(stale, ignore_errors=True)


def _report_stale_candidate(path: Path) -> None:
    files = [item for item in path.rglob("*") if item.is_file()] if path.is_dir() else []
    print(f"stale_candidate_manifest_present={'true' if (path / 'manifest.json').is_file() else 'false'}")
    print(f"stale_candidate_chunk_count={sum(1 for item in files if item.parent.name == 'chunks')}")
    print(f"stale_candidate_total_bytes={sum(item.stat().st_size for item in files)}")


def _case_dir(case):
    root = os.environ.get(_ASSET_ROOT_ENV)
    if not root:
        raise RuntimeError(f"{_ASSET_ROOT_ENV} is required for slow asset qualification")
    return Path(root) / case.name


def _metadata_common(case, manifest_digest):
    if not isinstance(manifest_digest, str):
        raise AssertionError(f"missing asset manifest digest: {case.name}")
    return {
        "schema_version": _METADATA_VERSION,
        "git_head": _git_head(),
        "case_name": case.name,
        "source_smiles": case.smiles,
        "rooted_atom": case.rooted_at_atom,
        "expected_support_count": case.expected_support_count,
        "expected_completion_count": case.expected_completion_count,
        "expected_support_digest": case.expected_support_digest,
    }


def _candidate_metadata(case, digest):
    return {"schema": _CANDIDATE_SCHEMA, **_metadata_common(case, digest),
            "candidate_manifest_digest": digest, "stage": "structurally_verified_candidate"}


def _asset_metadata(case, digest):
    return {"schema": _ASSET_SCHEMA, **_metadata_common(case, digest),
            "asset_manifest_digest": digest}


def _git_head():
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=Path(__file__).resolve().parents[2], text=True
    ).strip()


def _mol(case):
    from rdkit import Chem

    return Chem.MolFromSmiles(case.smiles)


__all__ = (
    "CachedQualificationAsset",
    "CachedQualificationCandidate",
    "CachedQualificationCountEnvelope",
    "CachedQualificationSupportArtifact",
    "build_slow_qualification_asset",
    "build_slow_qualification_candidate",
    "build_slow_count_envelope",
    "build_slow_support_artifact",
    "store_slow_count_envelope",
    "certify_slow_qualification_candidate",
    "require_slow_qualification_asset",
    "require_slow_qualification_candidate",
    "require_slow_count_envelope",
    "require_slow_support_artifact",
)
