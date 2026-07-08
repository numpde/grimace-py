"""Facts-bound structural verifier for writer support artifact tables."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import MoleculeFacts
from .policy import SmilesPolicy
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import SouthStarWriterSurface
from .prepared_runtime import prepare_south_star_mol_from_facts
from .writer_envelope_terms import _identity_envelope
from .writer_envelope_terms import _runtime_options_terms
from .writer_envelope_work import WriterEnvelopeWorkBudget
from .writer_envelope_work import WriterEnvelopeWorkExceeded
from .writer_envelope_work import default_writer_envelope_work_budget
from .writer_envelope_work import writer_envelope_work_reason
from .writer_prepared_identity import writer_prepared_identity
from .writer_support_artifact_checker import verify_writer_support_artifact_consistency
from .writer_support_artifact_offline_verifier import OBJECT_KIND_OFFLINE_COVERAGE
from .writer_support_artifact_offline_verifier import (
    verify_writer_support_artifact_offline_replay,
)


@dataclass(frozen=True, slots=True)
class WriterSupportArtifactFactVerification:
    accepted: bool
    support_count: int | None = None
    witness_count: int | None = None
    reason: str | None = None
    structurally_checked: bool = False
    facts_identity_checked: bool = False
    offline_replay_complete: bool = False
    offline_checked_object_kinds: tuple[str, ...] = ()
    offline_unchecked_object_kinds: tuple[str, ...] = ()
    offline_checked_relation_families: tuple[str, ...] = ()


def verify_writer_support_artifact_for_facts(
    *,
    facts: MoleculeFacts,
    runtime_options: SouthStarRuntimeOptions,
    artifact: object,
    policy: SmilesPolicy | None = None,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> WriterSupportArtifactFactVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        structural = verify_writer_support_artifact_consistency(
            artifact,
            budget=budget,
        )
        if not structural.accepted:
            return WriterSupportArtifactFactVerification(
                accepted=False,
                support_count=structural.support_count,
                witness_count=structural.witness_count,
                reason=structural.reason,
                structurally_checked=False,
                facts_identity_checked=False,
            )
        if not isinstance(artifact, Mapping):
            _fact_verifier_violation("artifact_not_mapping")
        prepared = prepare_south_star_mol_from_facts(
            facts,
            writer_surface=SouthStarWriterSurface(),
            policy=policy,
        )
        expected_identity = _identity_envelope(
            writer_prepared_identity(prepared, runtime_options),
            budget=budget,
            operation="support_artifact_fact.prepared_identity.digest",
        )
        _check_prepared_identity(artifact, expected_identity)
        source_payload = _source_snapshot_payload(artifact)
        _check_source_snapshot_identity(
            source_payload,
            expected_identity=expected_identity,
            runtime_options=runtime_options,
        )
        _check_source_fields(artifact, source_payload)
        offline = verify_writer_support_artifact_offline_replay(
            facts=facts,
            artifact=artifact,
            budget=budget,
        )
        if not offline.accepted:
            return WriterSupportArtifactFactVerification(
                accepted=False,
                support_count=structural.support_count,
                witness_count=structural.witness_count,
                reason=offline.reason,
                structurally_checked=True,
                facts_identity_checked=True,
            )
        return WriterSupportArtifactFactVerification(
            accepted=True,
            support_count=structural.support_count,
            witness_count=structural.witness_count,
            structurally_checked=True,
            facts_identity_checked=True,
            offline_replay_complete=offline.offline_replay_complete,
            offline_checked_object_kinds=offline.checked_object_kinds,
            offline_unchecked_object_kinds=offline.unchecked_object_kinds,
            offline_checked_relation_families=offline.checked_relation_families,
        )
    except WriterEnvelopeWorkExceeded as exc:
        return WriterSupportArtifactFactVerification(
            accepted=False,
            reason=writer_envelope_work_reason(exc),
        )
    except SouthStarError as exc:
        return WriterSupportArtifactFactVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "verification_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterSupportArtifactFactVerification(
            accepted=False,
            reason=f"malformed_artifact:{type(exc).__name__}",
        )


def _check_prepared_identity(
    artifact: Mapping[str, object],
    expected_identity: Mapping[str, object],
) -> None:
    if artifact["prepared_identity"] != expected_identity:
        _fact_verifier_violation("prepared_identity_mismatch")


def _check_source_snapshot_identity(
    source_payload: Mapping[str, object],
    *,
    expected_identity: Mapping[str, object],
    runtime_options: SouthStarRuntimeOptions,
) -> None:
    if source_payload["prepared_identity_digest"] != expected_identity["digest"]:
        _fact_verifier_violation("source_prepared_identity_digest_mismatch")
    if source_payload["prepared_identity_terms"] != expected_identity["terms"]:
        _fact_verifier_violation("source_prepared_identity_terms_mismatch")
    if source_payload["runtime_options"] != _runtime_options_terms(runtime_options):
        _fact_verifier_violation("source_runtime_options_mismatch")


def _check_source_fields(
    artifact: Mapping[str, object],
    source_payload: Mapping[str, object],
) -> None:
    if artifact["source_kind"] == "snapshot":
        if artifact["source_snapshot"] != source_payload:
            _fact_verifier_violation("source_snapshot_object_mismatch")
        return
    prefix = artifact["prefix_read_envelope"]
    if not isinstance(prefix, Mapping):
        _fact_verifier_violation("prefix_read_envelope_not_mapping")
    if prefix["prepared_identity"] != artifact["prepared_identity"]:
        _fact_verifier_violation("prefix_prepared_identity_mismatch")
    if prefix["read_kind"] != "readable":
        _fact_verifier_violation("prefix_read_not_readable")
    if prefix["final_snapshot"] != source_payload:
        _fact_verifier_violation("prefix_final_snapshot_mismatch")


def _source_snapshot_payload(artifact: Mapping[str, object]) -> Mapping[str, object]:
    objects = {
        item["object_id"]: item
        for item in artifact["objects"]
        if isinstance(item, Mapping)
    }
    source = objects[artifact["roots"]["source_ref"]]
    if source["kind"] != "source_snapshot":
        _fact_verifier_violation("source_ref_kind_mismatch")
    payload = source["payload"]
    if not isinstance(payload, Mapping):
        _fact_verifier_violation("source_payload_not_mapping")
    return payload


def _fact_verifier_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer support artifact fact verifier violation: {kind}",
    )


__all__ = (
    "OBJECT_KIND_OFFLINE_COVERAGE",
    "WriterSupportArtifactFactVerification",
    "verify_writer_support_artifact_for_facts",
)
