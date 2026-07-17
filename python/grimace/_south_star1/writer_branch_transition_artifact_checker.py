"""Structural checker for count-free writer branch transition artifacts."""

from __future__ import annotations

from collections.abc import Mapping

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_envelope_terms import _digest_terms_bounded
from .writer_envelope_terms import _identity_digest
from .writer_envelope_work import WriterEnvelopeWorkExceeded
from .writer_envelope_work import default_writer_envelope_work_budget
from .writer_envelope_work import writer_envelope_work_reason
from .writer_support_artifact_checker import _validate_object_payload_shape
from .writer_support_artifact_checker import artifact_metrics
from .writer_support_artifact_checker import support_artifact_object_identity_term

_TOP_LEVEL_FIELDS = frozenset((
    "schema_name", "schema_version", "prepared_identity", "source_kind",
    "source_snapshot", "objects", "roots", "metrics", "digest",
))
_ROOT_FIELDS = frozenset(("source_ref", "text_projection_ref", "branch_support_ref"))
_OBJECT_FIELDS = frozenset(("object_id", "kind", "payload", "digest"))
_KINDS = frozenset(("source_snapshot", "text_projection", "branch_support"))


def verify_writer_branch_transition_artifact_consistency(artifact, *, budget=None):
    from .writer_branch_transition_artifact import WriterBranchTransitionArtifactVerification
    from .writer_branch_transition_artifact import branch_transition_artifact_manifest

    try:
        budget = default_writer_envelope_work_budget(budget)
        if not isinstance(artifact, Mapping) or frozenset(artifact) != _TOP_LEVEL_FIELDS:
            _violation("top_level_fields_mismatch")
        if artifact["schema_name"] != "writer_branch_transition_artifact":
            _violation("unknown_schema_name")
        if artifact["schema_version"] != 1:
            _violation("unknown_schema_version")
        if artifact["source_kind"] != "snapshot":
            _violation("unsupported_source_kind")
        if not isinstance(artifact["roots"], Mapping) or frozenset(artifact["roots"]) != _ROOT_FIELDS:
            _violation("roots_fields_mismatch")
        if not isinstance(artifact["objects"], list) or len(artifact["objects"]) != 3:
            _violation("object_count_mismatch")
        objects = {}
        kinds = []
        for item in artifact["objects"]:
            if not isinstance(item, Mapping) or frozenset(item) != _OBJECT_FIELDS:
                _violation("object_fields_mismatch")
            if item["kind"] not in _KINDS:
                _violation("count_or_unknown_object_kind")
            expected = _identity_digest(
                support_artifact_object_identity_term(item["kind"], item["payload"]),
                budget=budget,
                operation="branch_transition.check.object",
            )
            if item["digest"] != expected or item["object_id"] != f"obj:{expected}":
                _violation("object_identity_mismatch")
            if item["object_id"] in objects:
                _violation("duplicate_object_id")
            _validate_object_payload_shape(item, budget=budget)
            objects[item["object_id"]] = item
            kinds.append(item["kind"])
        if sorted(kinds) != sorted(_KINDS):
            _violation("object_kind_cardinality_mismatch")
        source = _root(objects, artifact["roots"]["source_ref"], "source_snapshot")
        projection = _root(objects, artifact["roots"]["text_projection_ref"], "text_projection")
        branch = _root(objects, artifact["roots"]["branch_support_ref"], "branch_support")
        if artifact["source_snapshot"] != source["payload"]:
            _violation("source_snapshot_root_mismatch")
        if (
            source["payload"]["prepared_identity_digest"]
            != artifact["prepared_identity"]["digest"]
            or source["payload"]["prepared_identity_terms"]
            != artifact["prepared_identity"]["terms"]
        ):
            _violation("prepared_source_identity_mismatch")
        pp = projection["payload"]
        bp = branch["payload"]
        if pp["branch_support_refs"] != [branch["object_id"]]:
            _violation("projection_branch_ref_mismatch")
        projection_identity = {key: value for key, value in pp.items() if key not in ("digest", "branch_support_refs")}
        if pp["digest"] != _identity_digest(
            projection_identity,
            budget=budget,
            operation="branch_transition.check.projection",
        ):
            _violation("projection_digest_mismatch")
        if set(projection_identity) != {
            "source_cursor", "emitted_text", "successor_cursor",
            "immediate_multiplicity", "branch_certificate_digests",
        }:
            _violation("projection_fields_mismatch")
        if pp["immediate_multiplicity"] != 1 or len(pp["branch_certificate_digests"]) != 1:
            _violation("projection_branch_cardinality_mismatch")
        if (
            source["payload"]["cursor"] != pp["source_cursor"]
            or bp["source_cursor_digest"] != pp["source_cursor"]["digest"]
            or bp["successor_cursor_digest"] != pp["successor_cursor"]["digest"]
            or bp["emitted_text"] != pp["emitted_text"]
            or pp["branch_certificate_digests"] != [bp["checked_branch_certificate_digest"]]
        ):
            _violation("branch_projection_identity_mismatch")
        _require_unique_state_digest(pp["source_cursor"], bp["source_state_digest"], budget)
        _require_unique_state_digest(pp["successor_cursor"], bp["successor_state_digest"], budget)
        expected_metrics = artifact_metrics(list(artifact["objects"]))
        expected_metrics = {**expected_metrics, "reachable_object_count": 3, "unreferenced_object_count": 0}
        if artifact["metrics"] != expected_metrics:
            _violation("metrics_mismatch")
        expected_digest = _digest_terms_bounded(
            branch_transition_artifact_manifest(artifact),
            budget=budget,
            operation="branch_transition.check.manifest",
        )
        if artifact["digest"] != expected_digest:
            _violation("artifact_digest_mismatch")
        return WriterBranchTransitionArtifactVerification(accepted=True, object_count=3)
    except WriterEnvelopeWorkExceeded as exc:
        return WriterBranchTransitionArtifactVerification(accepted=False, reason=writer_envelope_work_reason(exc))
    except SouthStarError as exc:
        return WriterBranchTransitionArtifactVerification(accepted=False, reason=exc.args[-1] if exc.args else "verification_error")
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterBranchTransitionArtifactVerification(accepted=False, reason=f"malformed_branch_transition_artifact:{type(exc).__name__}")


def _root(objects, ref, kind):
    if ref not in objects or objects[ref]["kind"] != kind:
        _violation(f"{kind}_root_mismatch")
    return objects[ref]


def _require_unique_state_digest(cursor, expected, budget):
    terms = cursor["terms"]
    fields = dict(terms["fields"])
    weighted = fields["weighted_states"]
    matches = sum(
        _digest_terms_bounded(item[0], budget=budget, operation="branch_transition.check.state") == expected
        for item in weighted
    )
    if matches != 1:
        _violation("cursor_state_identity_not_unique")


def _violation(kind):
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer branch transition artifact checker violation: {kind}",
    )


__all__ = ("verify_writer_branch_transition_artifact_consistency",)
