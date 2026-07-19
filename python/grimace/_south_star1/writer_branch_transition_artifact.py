"""Count-free durable artifacts for one checked writer branch transition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_envelope_terms import _digest_terms_bounded
from .writer_envelope_terms import _identity_digest
from .writer_envelope_terms import _identity_envelope
from .writer_envelope_terms import _runtime_options_from_terms
from .writer_envelope_terms import _snapshot_identity_envelope
from .writer_envelope_work import WriterEnvelopeWorkBudget
from .writer_envelope_work import WriterEnvelopeWorkExceeded
from .writer_envelope_work import default_writer_envelope_work_budget
from .writer_envelope_work import writer_envelope_work_reason
from .writer_frontier import _checked_writer_frontier_branch_supports
from .writer_snapshot import WriterDecoderBoundary
from .writer_snapshot import capture_writer_frontier_snapshot
from .writer_snapshot_closed_terms import writer_frontier_cursor_from_closed_terms
from .writer_snapshot_prefix_envelope import _branch_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import _text_projection_certificate_identity_envelope
from .writer_support_artifact_checker import artifact_metrics
from .writer_support_artifact_envelope import _ObjectTable
from .writer_support_artifact_envelope import _add_branch_support

SCHEMA_NAME = "writer_branch_transition_artifact"
SCHEMA_VERSION = 3


@dataclass(frozen=True, slots=True)
class WriterBranchTransitionArtifactVerification:
    accepted: bool
    object_count: int | None = None
    reason: str | None = None


def writer_branch_transition_artifact_for_support(
    *, prepared, snapshot, support, budget: WriterEnvelopeWorkBudget | None = None
) -> Mapping[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    projection, branch = _locate_live_branch(
        prepared=prepared,
        snapshot=snapshot,
        support=support,
        budget=budget,
    )
    table = _ObjectTable(budget)
    source_payload = _snapshot_identity_envelope(
        snapshot,
        budget=budget,
        operation="branch_transition.source_snapshot.digest",
    )
    source_ref = table.add(
        "source_snapshot",
        source_payload,
        operation="branch_transition.source_snapshot.object",
    )
    full_projection = _text_projection_certificate_identity_envelope(
        projection,
        budget=budget,
    )
    branch_identity = _branch_certificate_identity_envelope(branch, budget=budget)
    projection_payload = {
        "source_cursor": full_projection["source_cursor"],
        "emitted_text": full_projection["emitted_text"],
        "successor_cursor": full_projection["successor_cursor"],
        "immediate_multiplicity": 1,
        "branch_certificate_digests": [branch_identity["digest"]],
    }
    projection_payload["digest"] = _identity_digest(
        projection_payload,
        budget=budget,
        operation="branch_transition.text_projection.digest",
    )
    branch_ref = _add_branch_support(
        table,
        branch=branch,
        text_projection=projection_payload,
        facts=prepared.facts,
        budget=budget,
    )
    projection_payload = {**projection_payload, "branch_support_refs": [branch_ref]}
    projection_ref = table.add(
        "text_projection",
        projection_payload,
        operation="branch_transition.text_projection.object",
    )
    objects = table.objects()
    roots = {
        "source_ref": source_ref,
        "text_projection_ref": projection_ref,
        "branch_support_ref": branch_ref,
    }
    metrics = artifact_metrics(objects)
    metrics = {**metrics, "reachable_object_count": 3, "unreferenced_object_count": 0}
    artifact = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "prepared_identity": _identity_envelope(
            snapshot.prepared_identity,
            budget=budget,
            operation="branch_transition.prepared_identity.digest",
        ),
        "source_kind": "snapshot",
        "source_snapshot": source_payload,
        "objects": objects,
        "roots": roots,
        "metrics": metrics,
    }
    artifact["digest"] = _digest_terms_bounded(
        branch_transition_artifact_manifest(artifact),
        budget=budget,
        operation="branch_transition.manifest.digest",
    )
    checked = verify_writer_branch_transition_artifact_consistency(
        artifact,
        budget=budget,
    )
    if not checked.accepted:
        _violation(checked.reason or "branch_transition_checker_rejected")
    return artifact


def verify_writer_branch_transition_artifact_envelope(
    *, prepared, artifact, budget: WriterEnvelopeWorkBudget | None = None
) -> WriterBranchTransitionArtifactVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        structural = verify_writer_branch_transition_artifact_consistency(
            artifact,
            budget=budget,
        )
        if not structural.accepted:
            return structural
        snapshot = _source_snapshot_from_branch_artifact(
            prepared=prepared,
            artifact=artifact,
            budget=budget,
        )
        objects = {item["object_id"]: item for item in artifact["objects"]}
        branch_payload = objects[artifact["roots"]["branch_support_ref"]]["payload"]
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            snapshot.cursor,
            include_counts=False,
            include_frontier_certificate=True,
            include_count_certificate=False,
        )
        matches = tuple(
            support
            for support in batch.supports
            if _branch_certificate_identity_envelope(
                support.checked_branch_certificate,
                budget=budget,
            )["digest"] == branch_payload["checked_branch_certificate_digest"]
        )
        if len(matches) != 1:
            _violation("live_branch_identity_not_unique")
        expected = writer_branch_transition_artifact_for_support(
            prepared=prepared,
            snapshot=snapshot,
            support=matches[0],
            budget=budget,
        )
        if expected != artifact:
            _violation("live_branch_artifact_mismatch")
        return WriterBranchTransitionArtifactVerification(accepted=True, object_count=3)
    except WriterEnvelopeWorkExceeded as exc:
        return WriterBranchTransitionArtifactVerification(
            accepted=False,
            reason=writer_envelope_work_reason(exc),
        )
    except SouthStarError as exc:
        return WriterBranchTransitionArtifactVerification(
            accepted=False,
            reason=exc.args[-1] if exc.args else "verification_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterBranchTransitionArtifactVerification(
            accepted=False,
            reason=f"malformed_branch_transition_artifact:{type(exc).__name__}",
        )


def _locate_live_branch(*, prepared, snapshot, support, budget):
    batch = _checked_writer_frontier_branch_supports(
        prepared,
        snapshot.cursor,
        include_counts=False,
        include_frontier_certificate=True,
        include_count_certificate=False,
    )
    requested = _branch_certificate_identity_envelope(
        support.checked_branch_certificate,
        budget=budget,
    )["digest"]
    matches = tuple(
        item
        for item in batch.supports
        if _branch_certificate_identity_envelope(
            item.checked_branch_certificate,
            budget=budget,
        )["digest"] == requested
    )
    if len(matches) != 1:
        _violation("selected_branch_not_unique_live_member")
    selected = matches[0]
    projections = tuple(
        projection
        for projection in batch.text_choice_projection_certificates
        if selected.checked_branch_certificate in projection.branch_certificates
    )
    if len(projections) != 1:
        _violation("selected_branch_projection_not_unique")
    return projections[0], selected.checked_branch_certificate


def _source_snapshot_from_branch_artifact(*, prepared, artifact, budget):
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
        operation="branch_transition.source_snapshot.reconstruct",
    )
    if expected != terms:
        _violation("source_snapshot_identity_mismatch")
    return snapshot


def branch_transition_artifact_manifest(artifact):
    return {
        "schema_name": artifact["schema_name"],
        "schema_version": artifact["schema_version"],
        "prepared_identity_digest": artifact["prepared_identity"]["digest"],
        "source_kind": artifact["source_kind"],
        "roots": artifact["roots"],
        "metrics": artifact["metrics"],
        "objects": [
            {"object_id": item["object_id"], "kind": item["kind"], "digest": item["digest"]}
            for item in artifact["objects"]
        ],
    }


def _violation(kind):
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer branch transition artifact violation: {kind}",
    )


from .writer_branch_transition_artifact_checker import (  # noqa: E402
    verify_writer_branch_transition_artifact_consistency,
)


__all__ = (
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "WriterBranchTransitionArtifactVerification",
    "branch_transition_artifact_manifest",
    "verify_writer_branch_transition_artifact_envelope",
    "writer_branch_transition_artifact_for_support",
)
