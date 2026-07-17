"""Count-free durable artifacts for one checked writer EOS transition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_envelope_terms import _cursor_envelope
from .writer_envelope_terms import _digest_terms_bounded
from .writer_envelope_terms import _identity_digest
from .writer_envelope_terms import _identity_envelope
from .writer_envelope_terms import _runtime_options_from_terms
from .writer_envelope_terms import _snapshot_identity_envelope
from .writer_envelope_work import WriterEnvelopeWorkBudget
from .writer_envelope_work import WriterEnvelopeWorkExceeded
from .writer_envelope_work import default_writer_envelope_work_budget
from .writer_envelope_work import writer_envelope_work_reason
from .writer_frontier import WriterFrontierCursor
from .writer_frontier import _checked_writer_frontier_branch_supports
from .writer_snapshot import WriterDecoderBoundary
from .writer_snapshot import capture_writer_frontier_snapshot
from .writer_snapshot_closed_terms import writer_frontier_cursor_from_closed_terms
from .writer_snapshot_prefix_envelope import (
    _terminal_support_identity_envelope_from_certificate,
)
from .writer_support_artifact_checker import artifact_metrics
from .writer_support_artifact_envelope import _ObjectTable
from .writer_support_artifact_envelope import _add_terminal_support

SCHEMA_NAME = "writer_terminalization_artifact"
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class WriterTerminalizationArtifactVerification:
    accepted: bool
    object_count: int | None = None
    reason: str | None = None


def writer_terminalization_artifact_for_support(
    *, prepared, snapshot, support, budget: WriterEnvelopeWorkBudget | None = None
) -> Mapping[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    selected = _locate_live_terminal(
        prepared=prepared,
        snapshot=snapshot,
        support=support,
        budget=budget,
    )
    terminal = selected.checked_terminal_certificate
    table = _ObjectTable(budget)
    source_payload = _snapshot_identity_envelope(
        snapshot,
        budget=budget,
        operation="terminalization.source_snapshot.digest",
    )
    source_ref = table.add(
        "source_snapshot",
        source_payload,
        operation="terminalization.source_snapshot.object",
    )
    terminal_support_ref = _add_terminal_support(
        table,
        terminal=terminal,
        budget=budget,
    )
    terminal_identity = _terminal_support_identity_envelope_from_certificate(
        terminal,
        budget=budget,
    )
    finalized_cursor = WriterFrontierCursor(
        weighted_states=((selected.finalized_state, selected.parent_weight),)
    )
    projection_payload = {
        "source_cursor": _cursor_envelope(
            snapshot.cursor,
            budget=budget,
            operation="terminalization.projection.source_cursor",
        ),
        "finalized_cursor": _cursor_envelope(
            finalized_cursor,
            budget=budget,
            operation="terminalization.projection.finalized_cursor",
        ),
        "multiplicity": selected.parent_weight,
        "terminal_support_identity_digests": [terminal_identity["digest"]],
        "terminal_support_refs": [terminal_support_ref],
    }
    projection_payload["digest"] = _identity_digest(
        projection_payload,
        budget=budget,
        operation="terminalization.projection.digest",
    )
    terminal_projection_ref = table.add(
        "terminal_projection",
        projection_payload,
        operation="terminalization.projection.object",
    )
    objects = table.objects()
    roots = {
        "source_ref": source_ref,
        "terminal_projection_ref": terminal_projection_ref,
        "terminal_support_ref": terminal_support_ref,
    }
    metrics = {
        **artifact_metrics(objects),
        "reachable_object_count": 3,
        "unreferenced_object_count": 0,
    }
    artifact = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "prepared_identity": _identity_envelope(
            snapshot.prepared_identity,
            budget=budget,
            operation="terminalization.prepared_identity.digest",
        ),
        "source_kind": "snapshot",
        "source_snapshot": source_payload,
        "objects": objects,
        "roots": roots,
        "metrics": metrics,
    }
    artifact["digest"] = _digest_terms_bounded(
        terminalization_artifact_manifest(artifact),
        budget=budget,
        operation="terminalization.manifest.digest",
    )
    checked = verify_writer_terminalization_artifact_consistency(
        artifact,
        budget=budget,
    )
    if not checked.accepted:
        _violation(checked.reason or "terminalization_checker_rejected")
    return artifact


def verify_writer_terminalization_artifact_envelope(
    *, prepared, artifact, budget: WriterEnvelopeWorkBudget | None = None
) -> WriterTerminalizationArtifactVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        structural = verify_writer_terminalization_artifact_consistency(
            artifact,
            budget=budget,
        )
        if not structural.accepted:
            return structural
        snapshot = _source_snapshot_from_terminalization_artifact(
            prepared=prepared,
            artifact=artifact,
            budget=budget,
        )
        objects = {item["object_id"]: item for item in artifact["objects"]}
        payload = objects[artifact["roots"]["terminal_support_ref"]]["payload"]
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            snapshot.cursor,
            include_counts=False,
            include_frontier_certificate=True,
            include_count_certificate=False,
        )
        matches = tuple(
            support
            for support in batch.terminal_supports
            if _terminal_support_identity_envelope_from_certificate(
                support.checked_terminal_certificate,
                budget=budget,
            )["digest"] == payload["digest"]
        )
        if len(matches) != 1:
            _violation("live_terminal_identity_not_unique")
        expected = writer_terminalization_artifact_for_support(
            prepared=prepared,
            snapshot=snapshot,
            support=matches[0],
            budget=budget,
        )
        if expected != artifact:
            _violation("live_terminalization_artifact_mismatch")
        return WriterTerminalizationArtifactVerification(
            accepted=True,
            object_count=3,
        )
    except WriterEnvelopeWorkExceeded as exc:
        return WriterTerminalizationArtifactVerification(
            accepted=False,
            reason=writer_envelope_work_reason(exc),
        )
    except SouthStarError as exc:
        return WriterTerminalizationArtifactVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "verification_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterTerminalizationArtifactVerification(
            accepted=False,
            reason=f"malformed_terminalization_artifact:{type(exc).__name__}",
        )


def _locate_live_terminal(*, prepared, snapshot, support, budget):
    batch = _checked_writer_frontier_branch_supports(
        prepared,
        snapshot.cursor,
        include_counts=False,
        include_frontier_certificate=True,
        include_count_certificate=False,
    )
    requested = _terminal_support_identity_envelope_from_certificate(
        support.checked_terminal_certificate,
        budget=budget,
    )["digest"]
    matches = tuple(
        item
        for item in batch.terminal_supports
        if _terminal_support_identity_envelope_from_certificate(
            item.checked_terminal_certificate,
            budget=budget,
        )["digest"] == requested
    )
    if len(matches) != 1:
        _violation("selected_terminal_not_unique_live_member")
    return matches[0]


def _source_snapshot_from_terminalization_artifact(*, prepared, artifact, budget):
    terms = artifact["source_snapshot"]
    options = _runtime_options_from_terms(terms["runtime_options"])
    cursor = writer_frontier_cursor_from_closed_terms(terms["cursor"]["terms"])
    depth = terms["decoder_boundary"]["consumed_token_count"]
    snapshot = capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=cursor,
        decoder_boundary=WriterDecoderBoundary(consumed_token_count=depth),
    )
    expected = _snapshot_identity_envelope(
        snapshot,
        budget=budget,
        operation="terminalization.source_snapshot.reconstruct",
    )
    if expected != terms:
        _violation("source_snapshot_identity_mismatch")
    return snapshot


def terminalization_artifact_manifest(artifact):
    return {
        "schema_name": artifact["schema_name"],
        "schema_version": artifact["schema_version"],
        "prepared_identity_digest": artifact["prepared_identity"]["digest"],
        "source_kind": artifact["source_kind"],
        "roots": artifact["roots"],
        "metrics": artifact["metrics"],
        "objects": [
            {
                "object_id": item["object_id"],
                "kind": item["kind"],
                "digest": item["digest"],
            }
            for item in artifact["objects"]
        ],
    }


def _violation(kind):
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer terminalization artifact violation: {kind}",
    )


from .writer_terminalization_artifact_checker import (  # noqa: E402
    verify_writer_terminalization_artifact_consistency,
)


__all__ = (
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "WriterTerminalizationArtifactVerification",
    "terminalization_artifact_manifest",
    "verify_writer_terminalization_artifact_envelope",
    "writer_terminalization_artifact_for_support",
)
