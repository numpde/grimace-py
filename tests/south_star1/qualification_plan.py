"""Single source of truth for South Star qualification lanes."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass
from typing import Literal

from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
    BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES,
    DefaultWriterCapabilityCase,
)

PUBLIC_PROOF_SHARD_COUNT = 4


@dataclass(frozen=True, slots=True)
class SlowQualificationLayerDefinition:
    name: str
    test_ids: tuple[str, ...]
    kind: Literal["product", "diagnostic"]
    role: str


@dataclass(frozen=True, slots=True)
class SlowQualificationShardDefinition:
    name: str
    case_names: tuple[str, ...]
    product_layers: tuple[str, ...]


_CONTINUATION_PRODUCT_LAYERS = (
    "public-build",
    "public-certify",
    "public-runtime",
    "public-recertification",
    *(f"public-proofs-{index}" for index in range(PUBLIC_PROOF_SHARD_COUNT)),
    "support-reparse",
    "continuation",
    "stereo-audit",
)
_ZERO_H_PRODUCT_LAYERS = (
    "public-build",
    "public-certify",
    "public-runtime",
    "public-recertification",
    "offline-zero-h",
    "offline-adjacent",
    "support-zero-h",
    "support-adjacent",
    "support-reparse",
    "continuation",
    "stereo-audit",
)
_DIAGNOSTIC_LAYERS = (
    "count-dag-build",
    "count-dag-validate",
    "support-artifact-build",
    "support-artifact-live",
    "offline-complete",
)


def _proof_layer(index: int) -> SlowQualificationLayerDefinition:
    return SlowQualificationLayerDefinition(
        name=f"public-proofs-{index}",
        test_ids=(
            f"tests.south_star1.test_public_continuation_proofs."
            f"PublicContinuationProofTest.test_slow_coupled_public_proof_shard_{index}",
        ),
        kind="product",
        role="public proof retrieval shard",
    )


_LAYER_DEFINITIONS = (
    SlowQualificationLayerDefinition(
        "public-build",
        ("tests.south_star1.test_public_continuation_asset.PublicContinuationAssetTest.test_slow_coupled_cases_build_through_public_api",),
        "product",
        "public continuation asset construction",
    ),
    SlowQualificationLayerDefinition(
        "public-certify",
        ("tests.south_star1.test_public_continuation_asset.PublicContinuationAssetTest.test_slow_coupled_cases_certify_public_candidates",),
        "product",
        "public continuation asset certification",
    ),
    SlowQualificationLayerDefinition(
        "public-runtime",
        ("tests.south_star1.test_public_continuation_asset.PublicContinuationAssetTest.test_slow_coupled_cases_run_public_runtime",),
        "product",
        "public runtime language",
    ),
    SlowQualificationLayerDefinition(
        "public-recertification",
        ("tests.south_star1.test_public_continuation_asset_verification.PublicContinuationAssetVerificationTest.test_slow_coupled_cases_recertify_copied_assets",),
        "product",
        "public whole-asset recertification",
    ),
    *(_proof_layer(index) for index in range(PUBLIC_PROOF_SHARD_COUNT)),
    SlowQualificationLayerDefinition(
        "support-reparse",
        ("tests.south_star1.test_writer_default_parity_corpus.WriterDefaultParityCorpusTest.test_slow_coupled_corpus_reparses_to_isomorphic_facts",),
        "product",
        "support reparse",
    ),
    SlowQualificationLayerDefinition(
        "continuation",
        ("tests.south_star1.test_writer_default_continuation_corpus.WriterDefaultContinuationCorpusTest.test_slow_coupled_cases_cross_all_continuation_tiers",),
        "product",
        "continuation corpus",
    ),
    SlowQualificationLayerDefinition(
        "stereo-audit",
        ("tests.south_star1.test_writer_default_stereo_audit_fixture.WriterDefaultStereoAuditSlowTest",),
        "product",
        "stereo audit",
    ),
    SlowQualificationLayerDefinition(
        "offline-zero-h",
        ("tests.south_star1.test_writer_default_offline_complete.WriterDefaultOfflineCompleteTest.test_zero_h_tetrahedral_is_offline_complete",),
        "product",
        "zero-H offline materialized authority",
    ),
    SlowQualificationLayerDefinition(
        "offline-adjacent",
        ("tests.south_star1.test_writer_default_offline_complete.WriterDefaultOfflineCompleteTest.test_adjacent_specified_tetrahedral_is_offline_complete",),
        "product",
        "adjacent offline materialized authority",
    ),
    SlowQualificationLayerDefinition(
        "support-zero-h",
        ("tests.south_star1.test_writer_default_parity_corpus.WriterDefaultParityCorpusTest.test_zero_h_tetrahedral_support_artifact",),
        "product",
        "zero-H support artifact authority",
    ),
    SlowQualificationLayerDefinition(
        "support-adjacent",
        ("tests.south_star1.test_writer_default_parity_corpus.WriterDefaultParityCorpusTest.test_adjacent_specified_tetrahedral_support_artifact",),
        "product",
        "adjacent support artifact authority",
    ),
    SlowQualificationLayerDefinition(
        "count-dag-build",
        ("tests.south_star1.test_writer_count_dag_envelope.WriterCountDagEnvelopeTest.test_slow_coupled_count_dag_build",),
        "diagnostic",
        "count DAG construction diagnostic",
    ),
    SlowQualificationLayerDefinition(
        "count-dag-validate",
        ("tests.south_star1.test_writer_count_dag_envelope.WriterCountDagEnvelopeTest.test_slow_coupled_count_dag_validate",),
        "diagnostic",
        "count DAG validation diagnostic",
    ),
    SlowQualificationLayerDefinition(
        "support-artifact-build",
        ("tests.south_star1.test_slow_support_artifact_qualification.SlowSupportArtifactQualificationTest.test_slow_support_artifact_build",),
        "diagnostic",
        "rich support artifact construction diagnostic",
    ),
    SlowQualificationLayerDefinition(
        "support-artifact-live",
        ("tests.south_star1.test_slow_support_artifact_qualification.SlowSupportArtifactQualificationTest.test_slow_support_artifact_live",),
        "diagnostic",
        "rich support artifact live diagnostic",
    ),
    SlowQualificationLayerDefinition(
        "offline-complete",
        ("tests.south_star1.test_slow_support_artifact_qualification.SlowSupportArtifactQualificationTest.test_slow_support_artifact_offline_complete",),
        "diagnostic",
        "rich support artifact offline diagnostic",
    ),
)

QUALIFICATION_LAYERS = {definition.name: definition for definition in _LAYER_DEFINITIONS}
SLOW_QUALIFICATION_LAYERS = QUALIFICATION_LAYERS
CONTINUATION_AUTHORITY_PRODUCT_LAYERS = _CONTINUATION_PRODUCT_LAYERS
CONTINUATION_AUTHORITY_DIAGNOSTIC_LAYERS = _DIAGNOSTIC_LAYERS

_SHARD_DEFINITIONS = (
    SlowQualificationShardDefinition("zero-h-adjacent", ("zero_h_tetrahedral", "adjacent_specified_tetrahedral"), _ZERO_H_PRODUCT_LAYERS),
    SlowQualificationShardDefinition("remote-a", ("remote_coupled_tetrahedral_a",), _CONTINUATION_PRODUCT_LAYERS),
    SlowQualificationShardDefinition("remote-b", ("remote_coupled_tetrahedral_b",), _CONTINUATION_PRODUCT_LAYERS),
)
SLOW_QUALIFICATION_SHARDS = {shard.name: shard for shard in _SHARD_DEFINITIONS}
SLOW_COUPLED_CASE_NAMES = tuple(name for shard in _SHARD_DEFINITIONS for name in shard.case_names)

FAST_ACCEPTED_CASES = tuple(case for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES if case.name not in SLOW_COUPLED_CASE_NAMES)
SLOW_COUPLED_CASES = tuple(case for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES if case.name in SLOW_COUPLED_CASE_NAMES)
MATERIALIZED_ARTIFACT_QUALIFIED_CASES = tuple(case for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES if case.qualification_authority == "materialized_support_artifact")
CONTINUATION_PROOF_QUALIFIED_CASES = tuple(case for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES if case.qualification_authority == "continuation_proof_complete")

_SELECTED_SLOW_CASES: ContextVar[tuple[DefaultWriterCapabilityCase, ...] | None] = ContextVar("south_star1_selected_slow_cases", default=None)


def case_by_name(name: str) -> DefaultWriterCapabilityCase:
    for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES:
        if case.name == name:
            return case
    raise ValueError(f"unknown accepted writer case: {name!r}")


def slow_cases_for_shard(name: str) -> tuple[DefaultWriterCapabilityCase, ...]:
    try:
        names = SLOW_QUALIFICATION_SHARDS[name].case_names
    except KeyError as error:
        raise ValueError(f"unknown slow qualification shard: {name!r}") from error
    return tuple(case for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES if case.name in names)


def bind_slow_qualification_shard(name: str) -> object:
    return _SELECTED_SLOW_CASES.set(slow_cases_for_shard(name))


def reset_slow_qualification_shard(token: object) -> None:
    _SELECTED_SLOW_CASES.reset(token)


def selected_slow_qualification_cases() -> tuple[DefaultWriterCapabilityCase, ...]:
    selected = _SELECTED_SLOW_CASES.get()
    if selected is None:
        raise RuntimeError("slow qualification shard selection is required before loading slow tests")
    return selected


def validate_qualification_plan() -> None:
    if len(QUALIFICATION_LAYERS) != len(_LAYER_DEFINITIONS):
        raise ValueError("duplicate qualification layer name")
    if any(not definition.test_ids for definition in _LAYER_DEFINITIONS):
        raise ValueError("qualification layer has no test IDs")
    if len(SLOW_QUALIFICATION_SHARDS) != len(_SHARD_DEFINITIONS):
        raise ValueError("duplicate qualification shard name")
    accepted = {case.name for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES}
    blocked = {case.name for case in BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES}
    seen: set[str] = set()
    for shard in _SHARD_DEFINITIONS:
        if seen & set(shard.case_names):
            raise ValueError("case appears in two slow shards")
        seen.update(shard.case_names)
        if not set(shard.case_names) <= accepted or set(shard.case_names) & blocked:
            raise ValueError(f"invalid case in shard {shard.name}")
        if any(layer not in QUALIFICATION_LAYERS for layer in shard.product_layers):
            raise ValueError(f"unknown product layer in shard {shard.name}")
        if any(QUALIFICATION_LAYERS[layer].kind != "product" for layer in shard.product_layers):
            raise ValueError(f"diagnostic layer in product plan {shard.name}")
        expected = (
            _CONTINUATION_PRODUCT_LAYERS
            if shard.name.startswith("remote-")
            else _ZERO_H_PRODUCT_LAYERS
        )
        if shard.product_layers != expected:
            raise ValueError(f"wrong product plan for shard {shard.name}")
    if len(tuple(name for name in _CONTINUATION_PRODUCT_LAYERS if name.startswith("public-proofs-"))) != PUBLIC_PROOF_SHARD_COUNT:
        raise ValueError("wrong public proof layer count")
    if set(CONTINUATION_AUTHORITY_PRODUCT_LAYERS) & set(CONTINUATION_AUTHORITY_DIAGNOSTIC_LAYERS):
        raise ValueError("product and diagnostic layers overlap")
    if any(QUALIFICATION_LAYERS[name].kind != "diagnostic" for name in CONTINUATION_AUTHORITY_DIAGNOSTIC_LAYERS):
        raise ValueError("diagnostic registry contains product layer")
    if any(QUALIFICATION_LAYERS[name].kind != "product" for name in CONTINUATION_AUTHORITY_PRODUCT_LAYERS):
        raise ValueError("product registry contains diagnostic layer")
    if set(seen) != set(SLOW_COUPLED_CASE_NAMES):
        raise ValueError("slow shard flattening mismatch")
    if set(case.name for case in FAST_ACCEPTED_CASES) & set(SLOW_COUPLED_CASE_NAMES):
        raise ValueError("fast and slow accepted cases overlap")
    if set(case.name for case in FAST_ACCEPTED_CASES) | set(SLOW_COUPLED_CASE_NAMES) != accepted:
        raise ValueError("fast and slow accepted cases are incomplete")
    for shard in _SHARD_DEFINITIONS:
        authorities = {case_by_name(name).qualification_authority for name in shard.case_names}
        expected = {"continuation_proof_complete"} if shard.name.startswith("remote-") else {"materialized_support_artifact"}
        if authorities != expected:
            raise ValueError(f"wrong qualification authority in shard {shard.name}")
    if set(case.name for case in MATERIALIZED_ARTIFACT_QUALIFIED_CASES) & set(case.name for case in CONTINUATION_PROOF_QUALIFIED_CASES):
        raise ValueError("qualification authorities overlap")
