"""Authoritative identity and storage vocabulary for slow qualification caches."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import os
from pathlib import Path
import subprocess
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
    if any(item.payload_kind not in QualificationCachePayloadKind for item in definitions):
        raise ValueError("unknown qualification cache payload kind")
    if set(payload_names) & set(metadata_names):
        raise ValueError("qualification cache payload and metadata names overlap")

