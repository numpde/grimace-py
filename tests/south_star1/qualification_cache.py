"""Authoritative identity and storage vocabulary for slow qualification caches."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import hashlib
import os
from pathlib import Path
import json
import subprocess
import uuid
from typing import Mapping

from tests.south_star1.default_writer_capability_ledger import DefaultWriterCapabilityCase


QUALIFICATION_CACHE_SCHEMA_VERSION = 1
SLOW_QUALIFICATION_ASSET_ROOT_ENV = "SOUTH_STAR1_SLOW_ASSET_ROOT"


class QualificationCacheKind(Enum):
    CONTINUATION_CANDIDATE = "continuation_candidate"
    CONTINUATION_ASSET = "continuation_asset"
    COUNT_ENVELOPE = "count_envelope"
    SUPPORT_ARTIFACT = "support_artifact"


class QualificationCachePayloadKind(Enum):
    DIRECTORY = "directory"
    JSON = "json"


@dataclass(frozen=True, slots=True)
class QualificationCacheEntryDefinition:
    kind: QualificationCacheKind
    payload_kind: QualificationCachePayloadKind
    payload_name: str
    metadata_name: str
    schema_name: str


QUALIFICATION_CACHE_ENTRY_DEFINITIONS = (
    QualificationCacheEntryDefinition(
        QualificationCacheKind.CONTINUATION_CANDIDATE,
        QualificationCachePayloadKind.DIRECTORY,
        "candidate",
        "candidate-metadata.json",
        "south_star1_slow_qualification_candidate",
    ),
    QualificationCacheEntryDefinition(
        QualificationCacheKind.CONTINUATION_ASSET,
        QualificationCachePayloadKind.DIRECTORY,
        "asset",
        "metadata.json",
        "south_star1_slow_qualification_asset",
    ),
    QualificationCacheEntryDefinition(
        QualificationCacheKind.COUNT_ENVELOPE,
        QualificationCachePayloadKind.JSON,
        "count-envelope.json",
        "count-envelope-metadata.json",
        "south_star1_slow_count_envelope",
    ),
    QualificationCacheEntryDefinition(
        QualificationCacheKind.SUPPORT_ARTIFACT,
        QualificationCachePayloadKind.JSON,
        "support-artifact.json",
        "support-artifact-metadata.json",
        "south_star1_slow_support_artifact",
    ),
)


@dataclass(frozen=True, slots=True)
class QualificationCacheContext:
    case: DefaultWriterCapabilityCase
    root: Path
    git_head: str
    case_dir: Path


@dataclass(frozen=True, slots=True)
class QualificationCachePaths:
    context: QualificationCacheContext
    definition: QualificationCacheEntryDefinition
    payload_path: Path
    metadata_path: Path


class QualificationCacheState(Enum):
    ABSENT = "absent"
    PAYLOAD_ONLY = "payload_only"
    METADATA_ONLY = "metadata_only"
    COMPLETE = "complete"


@dataclass(frozen=True, slots=True)
class CachedQualificationEntry:
    context: QualificationCacheContext
    paths: QualificationCachePaths
    metadata: Mapping[str, object]
    cache_reused: bool


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    ).strip()


def qualification_cache_context(case: DefaultWriterCapabilityCase) -> QualificationCacheContext:
    root_text = os.environ.get(SLOW_QUALIFICATION_ASSET_ROOT_ENV)
    if not root_text:
        raise RuntimeError(
            f"{SLOW_QUALIFICATION_ASSET_ROOT_ENV} is required for slow asset qualification"
        )
    root = Path(root_text)
    git_head = _git_head()
    return QualificationCacheContext(
        case=case,
        root=root,
        git_head=git_head,
        case_dir=root / case.name,
    )


def _definition_by_kind(kind: QualificationCacheKind) -> QualificationCacheEntryDefinition:
    for definition in QUALIFICATION_CACHE_ENTRY_DEFINITIONS:
        if definition.kind is kind:
            return definition
    raise ValueError(f"unknown qualification cache kind: {kind!r}")


def qualification_cache_paths(
    context: QualificationCacheContext,
    kind: QualificationCacheKind,
) -> QualificationCachePaths:
    definition = _definition_by_kind(kind)
    return QualificationCachePaths(
        context=context,
        definition=definition,
        payload_path=context.case_dir / definition.payload_name,
        metadata_path=context.case_dir / definition.metadata_name,
    )


def qualification_cache_common_metadata(
    context: QualificationCacheContext,
) -> dict[str, object]:
    case = context.case
    return {
        "schema_version": QUALIFICATION_CACHE_SCHEMA_VERSION,
        "git_head": context.git_head,
        "case_name": case.name,
        "source_smiles": case.smiles,
        "rooted_atom": case.rooted_at_atom,
        "expected_support_count": case.expected_support_count,
        "expected_completion_count": case.expected_completion_count,
        "expected_support_digest": case.expected_support_digest,
    }


def qualification_cache_metadata(
    context: QualificationCacheContext,
    kind: QualificationCacheKind,
    *,
    details: Mapping[str, object],
) -> dict[str, object]:
    definition = _definition_by_kind(kind)
    return {
        "schema": definition.schema_name,
        **qualification_cache_common_metadata(context),
        **dict(details),
    }


def validate_qualification_cache_registry() -> None:
    definitions = QUALIFICATION_CACHE_ENTRY_DEFINITIONS
    if len({item.kind for item in definitions}) != len(definitions):
        raise ValueError("duplicate qualification cache kinds")
    payload_names = [item.payload_name for item in definitions]
    metadata_names = [item.metadata_name for item in definitions]
    if len(set(payload_names)) != len(payload_names):
        raise ValueError("duplicate qualification cache payload names")
    if len(set(metadata_names)) != len(metadata_names):
        raise ValueError("duplicate qualification cache metadata names")
    if any(not item.schema_name for item in definitions):
        raise ValueError("empty qualification cache schema name")
    if any(not isinstance(item.payload_kind, QualificationCachePayloadKind) for item in definitions):
        raise ValueError("unknown qualification cache payload kind")
    if set(payload_names) & set(metadata_names):
        raise ValueError("qualification cache payload and metadata names overlap")


def inspect_qualification_cache(paths: QualificationCachePaths) -> QualificationCacheState:
    payload_exists = paths.payload_path.exists()
    metadata_exists = paths.metadata_path.is_file()
    if payload_exists and metadata_exists:
        return QualificationCacheState.COMPLETE
    if payload_exists:
        return QualificationCacheState.PAYLOAD_ONLY
    if metadata_exists:
        return QualificationCacheState.METADATA_ONLY
    return QualificationCacheState.ABSENT


def hidden_staging_path(paths: QualificationCachePaths, role: str) -> Path:
    return paths.context.case_dir / (
        f".{paths.definition.payload_name}.{role}.{uuid.uuid4().hex}"
    )


def cleanup_incomplete_qualification_cache(paths: QualificationCachePaths) -> None:
    state = inspect_qualification_cache(paths)
    if state is QualificationCacheState.COMPLETE or state is QualificationCacheState.ABSENT:
        return
    if paths.definition.payload_kind is QualificationCachePayloadKind.DIRECTORY:
        stale = paths.payload_path
        files = [item for item in stale.rglob("*") if item.is_file()] if stale.is_dir() else []
        print(
            f"stale_candidate_manifest_present={'true' if (stale / 'manifest.json').is_file() else 'false'}",
            flush=True,
        )
        print(
            f"stale_candidate_chunk_count={sum(item.parent.name == 'chunks' for item in files)}",
            flush=True,
        )
        print(f"stale_candidate_total_bytes={sum(item.stat().st_size for item in files)}", flush=True)
        if stale.is_dir():
            import shutil

            shutil.rmtree(stale)
        else:
            stale.unlink(missing_ok=True)
    else:
        paths.payload_path.unlink(missing_ok=True)
    paths.metadata_path.unlink(missing_ok=True)


def canonical_json_text(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_json_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_text(value).encode("utf-8")).hexdigest()


def read_json_mapping(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON mapping: {path}")
    return value


def _temporary_path(path: Path, role: str) -> Path:
    return path.with_name(f".{path.name}.{role}.{uuid.uuid4().hex}.tmp")


def atomic_write_text(path: Path, text: str) -> None:
    temporary = _temporary_path(path, "write")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _remove_public_pair(paths: QualificationCachePaths) -> None:
    if paths.definition.payload_kind is QualificationCachePayloadKind.DIRECTORY:
        if paths.payload_path.is_dir():
            import shutil

            shutil.rmtree(paths.payload_path)
        else:
            paths.payload_path.unlink(missing_ok=True)
    else:
        paths.payload_path.unlink(missing_ok=True)
    paths.metadata_path.unlink(missing_ok=True)


def _metadata_text(context: QualificationCacheContext, kind: QualificationCacheKind, details: Mapping[str, object]) -> str:
    return canonical_json_text(qualification_cache_metadata(context, kind, details=details)) + "\n"


def publish_json_qualification_cache(
    paths: QualificationCachePaths,
    *,
    payload: Mapping[str, object],
    metadata_details: Mapping[str, object],
) -> dict[str, object]:
    if paths.definition.payload_kind is not QualificationCachePayloadKind.JSON:
        raise ValueError("JSON publication requires a JSON cache entry")
    payload_text = canonical_json_text(payload) + "\n"
    metadata_text = _metadata_text(paths.context, paths.definition.kind, metadata_details)
    payload_tmp = _temporary_path(paths.payload_path, "payload")
    metadata_tmp = _temporary_path(paths.metadata_path, "metadata")
    paths.context.case_dir.mkdir(parents=True, exist_ok=True)
    try:
        payload_tmp.write_text(payload_text, encoding="utf-8")
        metadata_tmp.write_text(metadata_text, encoding="utf-8")
        os.replace(payload_tmp, paths.payload_path)
        os.replace(metadata_tmp, paths.metadata_path)
    except BaseException:
        payload_tmp.unlink(missing_ok=True)
        metadata_tmp.unlink(missing_ok=True)
        _remove_public_pair(paths)
        raise
    return qualification_cache_metadata(
        paths.context, paths.definition.kind, details=metadata_details
    )


def publish_directory_qualification_cache(
    paths: QualificationCachePaths,
    *,
    staged_payload_path: Path,
    metadata_details: Mapping[str, object],
) -> dict[str, object]:
    if paths.definition.payload_kind is not QualificationCachePayloadKind.DIRECTORY:
        raise ValueError("directory publication requires a directory cache entry")
    metadata_text = _metadata_text(paths.context, paths.definition.kind, metadata_details)
    metadata_tmp = _temporary_path(paths.metadata_path, "metadata")
    paths.context.case_dir.mkdir(parents=True, exist_ok=True)
    try:
        os.replace(staged_payload_path, paths.payload_path)
        metadata_tmp.write_text(metadata_text, encoding="utf-8")
        os.replace(metadata_tmp, paths.metadata_path)
    except BaseException:
        metadata_tmp.unlink(missing_ok=True)
        _remove_public_pair(paths)
        staged_payload_path.unlink(missing_ok=True)
        raise
    return qualification_cache_metadata(
        paths.context, paths.definition.kind, details=metadata_details
    )


def promote_directory_qualification_cache(
    *,
    source_paths: QualificationCachePaths,
    destination_paths: QualificationCachePaths,
    destination_metadata_details: Mapping[str, object],
) -> dict[str, object]:
    if source_paths.definition.payload_kind is not QualificationCachePayloadKind.DIRECTORY:
        raise ValueError("promotion source must be a directory cache entry")
    if destination_paths.definition.payload_kind is not QualificationCachePayloadKind.DIRECTORY:
        raise ValueError("promotion destination must be a directory cache entry")
    if inspect_qualification_cache(source_paths) is not QualificationCacheState.COMPLETE:
        raise ValueError("promotion source is not complete")
    if inspect_qualification_cache(destination_paths) is not QualificationCacheState.ABSENT:
        raise ValueError("promotion destination is not absent")
    hidden = destination_paths.context.case_dir / f".{destination_paths.definition.payload_name}.promotion.{uuid.uuid4().hex}"
    metadata_tmp = _temporary_path(destination_paths.metadata_path, "metadata")
    destination_paths.context.case_dir.mkdir(parents=True, exist_ok=True)
    moved = False
    try:
        os.replace(source_paths.payload_path, hidden)
        moved = True
        metadata_tmp.write_text(
            _metadata_text(
                destination_paths.context,
                destination_paths.definition.kind,
                destination_metadata_details,
            ),
            encoding="utf-8",
        )
        os.replace(hidden, destination_paths.payload_path)
        moved = False
        os.replace(metadata_tmp, destination_paths.metadata_path)
        source_paths.metadata_path.unlink()
    except BaseException:
        metadata_tmp.unlink(missing_ok=True)
        if destination_paths.metadata_path.exists():
            destination_paths.metadata_path.unlink(missing_ok=True)
        if destination_paths.payload_path.exists():
            import shutil

            shutil.rmtree(destination_paths.payload_path)
        if moved and hidden.exists():
            os.replace(hidden, source_paths.payload_path)
        elif not source_paths.payload_path.exists() and destination_paths.payload_path.exists():
            os.replace(destination_paths.payload_path, source_paths.payload_path)
        raise
    return qualification_cache_metadata(
        destination_paths.context,
        destination_paths.definition.kind,
        details=destination_metadata_details,
    )
