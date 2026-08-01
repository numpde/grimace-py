"""Revision-bound asset cache for the explicitly slow public qualification lane."""

from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import grimace

from grimace._south_star1.writer_continuation_asset import (
    open_writer_continuation_core,
)
from grimace._south_star1.writer_continuation_asset import (
    verify_writer_continuation_asset_consistency,
)
from tests.south_star1.default_writer_capability_ledger import (
    DefaultWriterCapabilityCase,
)


_METADATA_SCHEMA = "south_star1_slow_qualification_asset"
_METADATA_VERSION = 1
_ASSET_ROOT_ENV = "SOUTH_STAR1_SLOW_ASSET_ROOT"


@dataclass(frozen=True, slots=True)
class CachedQualificationAsset:
    case: DefaultWriterCapabilityCase
    asset_path: Path
    metadata_path: Path
    manifest_digest: str
    cache_reused: bool


def build_slow_qualification_asset(
    case: DefaultWriterCapabilityCase,
) -> CachedQualificationAsset:
    case_dir = _case_dir(case)
    asset_path = case_dir / "asset"
    metadata_path = case_dir / "metadata.json"
    if asset_path.exists() or metadata_path.exists():
        try:
            cached = require_slow_qualification_asset(case)
            return CachedQualificationAsset(
                cached.case,
                cached.asset_path,
                cached.metadata_path,
                cached.manifest_digest,
                True,
            )
        except Exception:
            shutil.rmtree(case_dir, ignore_errors=True)

    case_dir.mkdir(parents=True, exist_ok=True)
    digest = grimace.BuildMolToSmilesContinuationAsset(
        _mol(case),
        asset_path,
        rootedAtAtom=case.rooted_at_atom,
    )
    validation_started = time.monotonic()
    structural = verify_writer_continuation_asset_consistency(asset_path)
    print(
        f"cache_postwrite_validation_seconds={time.monotonic() - validation_started:.3f}",
        flush=True,
    )
    if not structural.accepted:
        raise AssertionError(structural.reason)
    opened = open_writer_continuation_core(asset_path)
    if opened.manifest_digest != digest:
        raise AssertionError("slow qualification asset manifest digest mismatch")
    metadata = _metadata(case, digest)
    metadata_path.write_text(
        json.dumps(metadata, sort_keys=True, separators=(",", ":")) + "\n"
    )
    return CachedQualificationAsset(case, asset_path, metadata_path, digest, False)


def require_slow_qualification_asset(
    case: DefaultWriterCapabilityCase,
) -> CachedQualificationAsset:
    case_dir = _case_dir(case)
    asset_path = case_dir / "asset"
    metadata_path = case_dir / "metadata.json"
    if not asset_path.is_dir() or not metadata_path.is_file():
        raise AssertionError(f"slow qualification asset is absent: {case.name}")
    metadata = json.loads(metadata_path.read_text())
    expected = _metadata(case, metadata.get("asset_manifest_digest"))
    if metadata != expected:
        raise AssertionError(f"slow qualification metadata mismatch: {case.name}")
    structural = verify_writer_continuation_asset_consistency(asset_path)
    if not structural.accepted:
        raise AssertionError(structural.reason)
    opened = open_writer_continuation_core(asset_path)
    if opened.manifest_digest != metadata["asset_manifest_digest"]:
        raise AssertionError(f"slow qualification manifest mismatch: {case.name}")
    return CachedQualificationAsset(
        case, asset_path, metadata_path, opened.manifest_digest, False
    )


def _case_dir(case: DefaultWriterCapabilityCase) -> Path:
    root = os.environ.get(_ASSET_ROOT_ENV)
    if not root:
        raise RuntimeError(f"{_ASSET_ROOT_ENV} is required for slow asset qualification")
    return Path(root) / case.name


def _metadata(case: DefaultWriterCapabilityCase, manifest_digest: str | None) -> dict:
    if not isinstance(manifest_digest, str):
        raise AssertionError(f"missing asset manifest digest: {case.name}")
    return {
        "schema": _METADATA_SCHEMA,
        "schema_version": _METADATA_VERSION,
        "git_head": _git_head(),
        "case_name": case.name,
        "source_smiles": case.smiles,
        "rooted_atom": case.rooted_at_atom,
        "expected_support_count": case.expected_support_count,
        "expected_completion_count": case.expected_completion_count,
        "expected_support_digest": case.expected_support_digest,
        "asset_manifest_digest": manifest_digest,
    }


def _git_head() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        text=True,
    ).strip()


def _mol(case: DefaultWriterCapabilityCase):
    from rdkit import Chem

    return Chem.MolFromSmiles(case.smiles)


__all__ = (
    "CachedQualificationAsset",
    "build_slow_qualification_asset",
    "require_slow_qualification_asset",
)
