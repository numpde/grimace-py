"""Structural checker for count-free writer terminalization artifacts."""

from __future__ import annotations

from collections.abc import Mapping

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_envelope_terms import _digest_terms_bounded
from .writer_envelope_terms import _identity_digest
from .writer_envelope_work import WriterEnvelopeWorkExceeded
from .writer_envelope_work import default_writer_envelope_work_budget
from .writer_envelope_work import writer_envelope_work_reason
from .writer_snapshot_closed_terms import writer_frontier_cursor_from_closed_terms
from .writer_support_artifact_checker import _validate_object_payload_shape
from .writer_support_artifact_checker import _validate_terminal_support_identity
from .writer_support_artifact_checker import artifact_metrics
from .writer_support_artifact_checker import support_artifact_object_identity_term

_TOP_LEVEL_FIELDS = frozenset((
    "schema_name", "schema_version", "prepared_identity", "source_kind",
    "source_snapshot", "objects", "roots", "metrics", "digest",
))
_ROOT_FIELDS = frozenset((
    "source_ref", "terminal_projection_ref", "terminal_support_ref",
))
_OBJECT_FIELDS = frozenset(("object_id", "kind", "payload", "digest"))
_KINDS = frozenset(("source_snapshot", "terminal_projection", "terminal_support"))
_PROJECTION_FIELDS = frozenset((
    "source_cursor", "finalized_cursor", "multiplicity",
    "terminal_support_identity_digests", "terminal_support_refs", "digest",
))


def verify_writer_terminalization_artifact_consistency(artifact, *, budget=None):
    from .writer_terminalization_artifact import (
        WriterTerminalizationArtifactVerification,
    )
    from .writer_terminalization_artifact import terminalization_artifact_manifest

    try:
        budget = default_writer_envelope_work_budget(budget)
        if not isinstance(artifact, Mapping) or frozenset(artifact) != _TOP_LEVEL_FIELDS:
            _violation("top_level_fields_mismatch")
        if artifact["schema_name"] != "writer_terminalization_artifact":
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
                operation="terminalization.check.object",
            )
            if item["digest"] != expected or item["object_id"] != f"obj:{expected}":
                _violation("object_identity_mismatch")
            if item["object_id"] in objects:
                _violation("duplicate_object_id")
            if item["kind"] == "terminal_projection":
                _validate_projection(item["payload"], budget=budget)
            elif item["kind"] == "terminal_support":
                _validate_terminal_support_identity(
                    item["payload"],
                    include_obligation_summary=True,
                )
            else:
                _validate_object_payload_shape(item, budget=budget)
            objects[item["object_id"]] = item
            kinds.append(item["kind"])
        if sorted(kinds) != sorted(_KINDS):
            _violation("object_kind_cardinality_mismatch")
        source = _root(objects, artifact["roots"]["source_ref"], "source_snapshot")
        projection = _root(
            objects,
            artifact["roots"]["terminal_projection_ref"],
            "terminal_projection",
        )
        terminal = _root(
            objects,
            artifact["roots"]["terminal_support_ref"],
            "terminal_support",
        )
        if artifact["source_snapshot"] != source["payload"]:
            _violation("source_snapshot_root_mismatch")
        if (
            source["payload"]["prepared_identity_digest"]
            != artifact["prepared_identity"]["digest"]
            or source["payload"]["prepared_identity_terms"]
            != artifact["prepared_identity"]["terms"]
        ):
            _violation("prepared_source_identity_mismatch")
        if source["payload"]["cursor"] != projection["payload"]["source_cursor"]:
            _violation("projection_source_cursor_mismatch")
        if projection["payload"]["terminal_support_refs"] != [terminal["object_id"]]:
            _violation("projection_terminal_support_ref_mismatch")
        if projection["payload"]["terminal_support_identity_digests"] != [
            terminal["payload"]["digest"]
        ]:
            _violation("projection_terminal_identity_mismatch")
        source_cursor = writer_frontier_cursor_from_closed_terms(
            projection["payload"]["source_cursor"]["terms"]
        )
        finalized_cursor = writer_frontier_cursor_from_closed_terms(
            projection["payload"]["finalized_cursor"]["terms"]
        )
        source_matches = tuple(
            weight
            for state, weight in source_cursor.weighted_states
            if _identity_digest(state) == terminal["payload"]["source_state_digest"]
        )
        finalized_matches = tuple(
            weight
            for state, weight in finalized_cursor.weighted_states
            if _identity_digest(state) == terminal["payload"]["finalized_state_digest"]
        )
        if len(source_matches) != 1:
            _violation("source_state_not_unique")
        if finalized_matches != (terminal["payload"]["parent_weight"],):
            _violation("finalized_state_not_unique")
        if source_matches[0] != terminal["payload"]["parent_weight"]:
            _violation("source_weight_mismatch")
        if projection["payload"]["multiplicity"] != terminal["payload"]["parent_weight"]:
            _violation("terminal_multiplicity_mismatch")
        if artifact["metrics"] != {
            **artifact_metrics(artifact["objects"]),
            "reachable_object_count": 3,
            "unreferenced_object_count": 0,
        }:
            _violation("metrics_mismatch")
        expected_digest = _digest_terms_bounded(
            terminalization_artifact_manifest(artifact),
            budget=budget,
            operation="terminalization.check.manifest",
        )
        if artifact["digest"] != expected_digest:
            _violation("artifact_digest_mismatch")
        return WriterTerminalizationArtifactVerification(accepted=True, object_count=3)
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
            reason=(
                f"malformed_terminalization_artifact:{type(exc).__name__}:"
                f"{exc}"
            ),
        )


def _validate_projection(payload, *, budget) -> None:
    if not isinstance(payload, Mapping) or frozenset(payload) != _PROJECTION_FIELDS:
        _violation("terminal_projection_fields_mismatch")
    for field in ("source_cursor", "finalized_cursor"):
        cursor = payload[field]
        if not isinstance(cursor, Mapping) or frozenset(cursor) != {"terms", "digest"}:
            _violation("terminal_projection_cursor_shape_mismatch")
        if cursor["digest"] != _digest_terms_bounded(
            cursor["terms"],
            budget=budget,
            operation="terminalization.check.cursor",
        ):
            _violation("terminal_projection_cursor_digest_mismatch")
        writer_frontier_cursor_from_closed_terms(cursor["terms"])
    if not isinstance(payload["multiplicity"], int) or payload["multiplicity"] <= 0:
        _violation("terminal_projection_multiplicity_mismatch")
    for field in ("terminal_support_identity_digests", "terminal_support_refs"):
        if (
            not isinstance(payload[field], list)
            or len(payload[field]) != 1
            or not isinstance(payload[field][0], str)
        ):
            _violation("terminal_projection_support_identity_mismatch")
    identity = {key: value for key, value in payload.items() if key != "digest"}
    if payload["digest"] != _identity_digest(identity):
        _violation("terminal_projection_digest_mismatch")


def _root(objects, ref, kind):
    if ref not in objects or objects[ref]["kind"] != kind:
        _violation(f"{kind}_root_mismatch")
    return objects[ref]


def _violation(kind):
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer terminalization artifact structural violation: {kind}",
    )


__all__ = ("verify_writer_terminalization_artifact_consistency",)
