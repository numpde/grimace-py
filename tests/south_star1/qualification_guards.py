"""Single registry for producer paths forbidden by qualification gates."""

from __future__ import annotations

from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from enum import Enum
from unittest.mock import Mock, patch

import grimace
from grimace import _runtime
from grimace._south_star1 import public_continuation_asset
from grimace._south_star1 import writer_continuation_asset
from grimace._south_star1 import writer_count_dag_envelope
from grimace._south_star1 import writer_frontier_count_envelope
from grimace._south_star1 import writer_snapshot
from grimace._south_star1 import writer_support
from grimace._south_star1 import writer_support_artifact_envelope
from grimace._south_star1 import writer_support_image_envelope


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


_TARGETS: dict[QualificationPath, tuple[tuple[object, str], ...]] = {
    QualificationPath.PUBLIC_ASSET_BUILD: (
        (grimace, "BuildMolToSmilesContinuationAsset"),
        (public_continuation_asset, "build_mol_to_smiles_continuation_asset"),
    ),
    QualificationPath.CONTINUATION_ASSET_WRITE: (
        (writer_continuation_asset, "write_writer_continuation_asset"),
        (public_continuation_asset, "write_writer_continuation_asset"),
    ),
    QualificationPath.WHOLE_ASSET_RECERTIFICATION: (
        (grimace, "VerifyMolToSmilesContinuationAsset"),
        (public_continuation_asset, "verify_mol_to_smiles_continuation_asset"),
    ),
    QualificationPath.COUNT_ENVELOPE_BUILD: (
        (writer_frontier_count_envelope, "writer_frontier_count_envelope_for_snapshot"),
        (writer_support_artifact_envelope, "writer_frontier_count_envelope_for_snapshot"),
        (writer_support_image_envelope, "writer_frontier_count_envelope_for_snapshot"),
    ),
    QualificationPath.COUNT_DAG_BUILD: (
        (writer_count_dag_envelope, "writer_count_certificate_dag_envelope_for_product"),
        (writer_frontier_count_envelope, "writer_count_certificate_dag_envelope_for_product"),
    ),
    QualificationPath.RICH_SUPPORT_ARTIFACT_BUILD: (
        (writer_support_artifact_envelope, "writer_support_artifact_envelope_for_snapshot"),
        (writer_support_artifact_envelope, "_writer_support_artifact_envelope_for_snapshot_with_count_envelope"),
    ),
    QualificationPath.SUPPORT_STRING_MATERIALIZATION: (
        (writer_snapshot, "_iter_writer_snapshot_certified_support_strings"),
        (writer_support_image_envelope, "_iter_writer_snapshot_certified_support_strings"),
    ),
    QualificationPath.LEGACY_SUPPORT_ENUMERATION: (
        (writer_support, "enumerate_prepared_writer_shaped_support"),
        (_runtime, "mol_to_smiles_enum"),
    ),
}


NO_PUBLICATION = (
    QualificationPath.PUBLIC_ASSET_BUILD,
    QualificationPath.CONTINUATION_ASSET_WRITE,
)
NO_WHOLE_ASSET_RECERTIFICATION = (QualificationPath.WHOLE_ASSET_RECERTIFICATION,)
NO_COUNT_OR_MATERIALIZATION = (
    QualificationPath.COUNT_ENVELOPE_BUILD,
    QualificationPath.COUNT_DAG_BUILD,
    QualificationPath.RICH_SUPPORT_ARTIFACT_BUILD,
)
NO_LEGACY_SUPPORT = (
    QualificationPath.SUPPORT_STRING_MATERIALIZATION,
    QualificationPath.LEGACY_SUPPORT_ENUMERATION,
)


def _join(*groups: tuple[QualificationPath, ...]) -> tuple[QualificationPath, ...]:
    result: list[QualificationPath] = []
    for group in groups:
        for path in group:
            if path in result:
                raise ValueError(f"duplicate path in guard profile: {path.value}")
            result.append(path)
    return tuple(result)


QUALIFICATION_GUARD_PROFILES: dict[str, tuple[QualificationPath, ...]] = {
    "public-build-without-legacy-materialization": _join(NO_COUNT_OR_MATERIALIZATION, NO_LEGACY_SUPPORT),
    "continuation-asset-build-without-legacy-materialization": _join(NO_COUNT_OR_MATERIALIZATION, NO_LEGACY_SUPPORT),
    "public-runtime": _join(NO_PUBLICATION, NO_WHOLE_ASSET_RECERTIFICATION, NO_COUNT_OR_MATERIALIZATION, NO_LEGACY_SUPPORT),
    "public-recertification": _join(NO_PUBLICATION, NO_COUNT_OR_MATERIALIZATION, NO_LEGACY_SUPPORT),
    "public-proofs": _join(NO_PUBLICATION, NO_WHOLE_ASSET_RECERTIFICATION, NO_COUNT_OR_MATERIALIZATION, NO_LEGACY_SUPPORT),
    "cached-continuation-verification": _join(NO_PUBLICATION, NO_COUNT_OR_MATERIALIZATION, NO_LEGACY_SUPPORT),
    "slow-rich-artifact-build": _join(NO_PUBLICATION, NO_COUNT_OR_MATERIALIZATION),
    "slow-rich-artifact-live": _join(NO_PUBLICATION, NO_COUNT_OR_MATERIALIZATION, NO_LEGACY_SUPPORT),
    "slow-stereo-audit": _join(NO_PUBLICATION, NO_COUNT_OR_MATERIALIZATION, NO_LEGACY_SUPPORT),
    "stereo-audit-fast": _join(NO_COUNT_OR_MATERIALIZATION, NO_LEGACY_SUPPORT),
}


def validate_qualification_guard_registry() -> None:
    seen_targets: dict[tuple[int, str], QualificationPath] = {}
    for path, targets in _TARGETS.items():
        if not targets:
            raise ValueError(f"empty target path: {path.value}")
        for owner, name in targets:
            if not hasattr(owner, name):
                raise ValueError(f"missing guard target: {path.value}:{name}")
            key = (id(owner), name)
            previous = seen_targets.get(key)
            if previous is not None:
                raise ValueError(
                    f"guard target owned twice: {previous.value} and {path.value}: {name}"
                )
            seen_targets[key] = path
    for name, paths in QUALIFICATION_GUARD_PROFILES.items():
        if not paths:
            raise ValueError(f"empty guard profile: {name}")
        if len(paths) != len(set(paths)):
            raise ValueError(f"duplicate path in guard profile: {name}")
        if any(path not in _TARGETS for path in paths):
            raise ValueError(f"unknown path in guard profile: {name}")


validate_qualification_guard_registry()


@contextmanager
def forbid_qualification_paths(*paths: QualificationPath):
    if len(paths) != len(set(paths)):
        raise ValueError("duplicate path in guard context")
    with ExitStack() as stack:
        mocks: dict[QualificationPath, tuple[Mock, ...]] = {}
        for path in paths:
            path_mocks = tuple(
                stack.enter_context(
                    patch.object(
                        owner,
                        name,
                        side_effect=AssertionError(
                            f"forbidden qualification path: {path.value}"
                        ),
                    )
                )
                for owner, name in _TARGETS[path]
            )
            mocks[path] = path_mocks
        yield QualificationPathGuardReport(mocks)


@contextmanager
def forbid_qualification_profile(name: str):
    try:
        paths = QUALIFICATION_GUARD_PROFILES[name]
    except KeyError as error:
        raise ValueError(f"unknown qualification guard profile: {name!r}") from error
    with forbid_qualification_paths(*paths) as report:
        yield report
