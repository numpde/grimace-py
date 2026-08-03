"""Centralized guards for qualification-only producer boundaries."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import Enum
from unittest.mock import Mock, patch

from grimace._south_star1 import writer_continuation_asset
from grimace._south_star1 import writer_count_dag_envelope
from grimace._south_star1 import writer_frontier_count_envelope
from grimace._south_star1 import writer_snapshot
from grimace._south_star1 import writer_support
from grimace._south_star1 import writer_support_artifact_envelope
import grimace
from grimace import _runtime


class QualificationPath(Enum):
    PUBLIC_ASSET_BUILD = "public_asset_build"
    CONTINUATION_ASSET_WRITE = "continuation_asset_write"
    WHOLE_ASSET_RECERTIFICATION = "whole_asset_recertification"
    COUNT_ENVELOPE_BUILD = "count_envelope_build"
    COUNT_DAG_BUILD = "count_dag_build"
    RICH_SUPPORT_ARTIFACT_BUILD = "rich_support_artifact_build"
    SUPPORT_STRING_MATERIALIZATION = "support_string_materialization"
    LEGACY_SUPPORT_ENUMERATION = "legacy_support_enumeration"


@dataclass(frozen=True, slots=True)
class QualificationPathGuardReport:
    mocks: dict[QualificationPath, tuple[Mock, ...]]

    def call_counts(self) -> dict[str, int]:
        return {
            path.value: sum(mock.call_count for mock in mocks)
            for path, mocks in self.mocks.items()
        }

    def assert_unused(self, test_case) -> None:
        for path, mocks in self.mocks.items():
            test_case.assertEqual(
                sum(mock.call_count for mock in mocks),
                0,
                f"forbidden qualification path invoked: {path.value}",
            )


_TARGETS = {
    QualificationPath.PUBLIC_ASSET_BUILD: (
        (grimace, "BuildMolToSmilesContinuationAsset"),
        (writer_continuation_asset, "write_writer_continuation_asset"),
    ),
    QualificationPath.CONTINUATION_ASSET_WRITE: (
        (writer_continuation_asset, "write_writer_continuation_asset"),
    ),
    QualificationPath.WHOLE_ASSET_RECERTIFICATION: (
        (grimace, "VerifyMolToSmilesContinuationAsset"),
    ),
    QualificationPath.COUNT_ENVELOPE_BUILD: (
        (writer_frontier_count_envelope, "writer_frontier_count_envelope_for_snapshot"),
    ),
    QualificationPath.COUNT_DAG_BUILD: (
        (writer_count_dag_envelope, "writer_count_certificate_dag_envelope_for_product"),
    ),
    QualificationPath.RICH_SUPPORT_ARTIFACT_BUILD: (
        (writer_support_artifact_envelope, "writer_support_artifact_envelope_for_snapshot"),
        (writer_support_artifact_envelope, "_writer_support_artifact_envelope_for_snapshot_with_count_envelope"),
    ),
    QualificationPath.SUPPORT_STRING_MATERIALIZATION: (
        (writer_snapshot, "_iter_writer_snapshot_certified_support_strings"),
    ),
    QualificationPath.LEGACY_SUPPORT_ENUMERATION: (
        (writer_support, "enumerate_prepared_writer_shaped_support"),
        (_runtime, "mol_to_smiles_enum"),
    ),
}


QUALIFICATION_GUARD_PROFILES = {
    "public-runtime": frozenset({
        QualificationPath.PUBLIC_ASSET_BUILD,
        QualificationPath.WHOLE_ASSET_RECERTIFICATION,
        QualificationPath.CONTINUATION_ASSET_WRITE,
        QualificationPath.COUNT_ENVELOPE_BUILD,
        QualificationPath.COUNT_DAG_BUILD,
        QualificationPath.RICH_SUPPORT_ARTIFACT_BUILD,
        QualificationPath.SUPPORT_STRING_MATERIALIZATION,
        QualificationPath.LEGACY_SUPPORT_ENUMERATION,
    }),
    "public-recertification": frozenset({
        QualificationPath.PUBLIC_ASSET_BUILD,
        QualificationPath.CONTINUATION_ASSET_WRITE,
        QualificationPath.COUNT_ENVELOPE_BUILD,
        QualificationPath.COUNT_DAG_BUILD,
        QualificationPath.RICH_SUPPORT_ARTIFACT_BUILD,
        QualificationPath.SUPPORT_STRING_MATERIALIZATION,
        QualificationPath.LEGACY_SUPPORT_ENUMERATION,
    }),
    "public-proofs": frozenset({
        QualificationPath.PUBLIC_ASSET_BUILD,
        QualificationPath.WHOLE_ASSET_RECERTIFICATION,
        QualificationPath.CONTINUATION_ASSET_WRITE,
        QualificationPath.COUNT_ENVELOPE_BUILD,
        QualificationPath.COUNT_DAG_BUILD,
        QualificationPath.RICH_SUPPORT_ARTIFACT_BUILD,
        QualificationPath.SUPPORT_STRING_MATERIALIZATION,
        QualificationPath.LEGACY_SUPPORT_ENUMERATION,
    }),
    "cached-continuation-verification": frozenset({
        QualificationPath.PUBLIC_ASSET_BUILD,
        QualificationPath.CONTINUATION_ASSET_WRITE,
        QualificationPath.COUNT_ENVELOPE_BUILD,
        QualificationPath.COUNT_DAG_BUILD,
        QualificationPath.RICH_SUPPORT_ARTIFACT_BUILD,
        QualificationPath.SUPPORT_STRING_MATERIALIZATION,
        QualificationPath.LEGACY_SUPPORT_ENUMERATION,
    }),
    "slow-rich-artifact": frozenset({
        QualificationPath.PUBLIC_ASSET_BUILD,
        QualificationPath.CONTINUATION_ASSET_WRITE,
        QualificationPath.COUNT_ENVELOPE_BUILD,
        QualificationPath.COUNT_DAG_BUILD,
        QualificationPath.SUPPORT_STRING_MATERIALIZATION,
        QualificationPath.LEGACY_SUPPORT_ENUMERATION,
    }),
    "slow-stereo-audit": frozenset({
        QualificationPath.PUBLIC_ASSET_BUILD,
        QualificationPath.CONTINUATION_ASSET_WRITE,
        QualificationPath.COUNT_ENVELOPE_BUILD,
        QualificationPath.COUNT_DAG_BUILD,
        QualificationPath.SUPPORT_STRING_MATERIALIZATION,
        QualificationPath.LEGACY_SUPPORT_ENUMERATION,
    }),
}


@contextmanager
def forbid_qualification_paths(*paths: QualificationPath):
    with ExitStack() as stack:
        mocks: dict[QualificationPath, tuple[Mock, ...]] = {}
        for path in paths:
            targets = _TARGETS[path]
            path_mocks = tuple(stack.enter_context(patch.object(owner, name, side_effect=AssertionError(f"forbidden qualification path: {path.value}"))) for owner, name in targets)
            mocks[path] = path_mocks
        yield QualificationPathGuardReport(mocks)


def guard_profile(name: str) -> tuple[QualificationPath, ...]:
    try:
        return tuple(QUALIFICATION_GUARD_PROFILES[name])
    except KeyError as error:
        raise ValueError(f"unknown qualification guard profile: {name!r}") from error
