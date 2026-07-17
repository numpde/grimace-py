"""Count-free durable artifacts for one checked writer branch transition."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from enum import Enum
import importlib
from typing import get_args
from typing import get_origin
from typing import get_type_hints

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
from .writer_snapshot_prefix_envelope import _branch_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import _text_projection_certificate_identity_envelope
from .writer_support_artifact_checker import artifact_metrics
from .writer_support_artifact_envelope import _ObjectTable
from .writer_support_artifact_envelope import _add_branch_support

SCHEMA_NAME = "writer_branch_transition_artifact"
SCHEMA_VERSION = 1


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
    cursor = _closed_value_from_term(terms["cursor"]["terms"])
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


def _closed_value_from_term(term, annotation=None):
    origin = get_origin(annotation)
    args = get_args(annotation)
    if term is None or isinstance(term, (str, bool, int)):
        return term
    if isinstance(term, list):
        if origin is frozenset:
            item_type = args[0] if args else None
            return frozenset(_closed_value_from_term(item, item_type) for item in term)
        if origin is dict:
            key_type, value_type = args if len(args) == 2 else (None, None)
            return {
                _closed_value_from_term(item[0], key_type): _closed_value_from_term(item[1], value_type)
                for item in term
            }
        item_type = args[0] if origin is tuple and len(args) == 2 and args[1] is Ellipsis else None
        return tuple(_closed_value_from_term(item, item_type) for item in term)
    if not isinstance(term, Mapping):
        _violation("closed_term_shape_mismatch")
    if "__enum__" in term:
        cls = _closed_term_class(term["__enum__"])
        if not issubclass(cls, Enum):
            _violation("closed_term_enum_class_mismatch")
        return cls(term["value"])
    if "__dataclass__" not in term or set(term) != {"__dataclass__", "fields"}:
        _violation("closed_term_dataclass_shape_mismatch")
    cls = _closed_term_class(term["__dataclass__"])
    if not is_dataclass(cls):
        _violation("closed_term_dataclass_class_mismatch")
    raw_fields = dict(term["fields"])
    if len(raw_fields) != len(term["fields"]) or set(raw_fields) != {field.name for field in fields(cls)}:
        _violation("closed_term_dataclass_fields_mismatch")
    hints = get_type_hints(cls)
    values = {
        field.name: _closed_value_from_term(raw_fields[field.name], hints.get(field.name))
        for field in fields(cls)
    }
    return cls(**values)


def _closed_term_class(path):
    if not isinstance(path, str) or not path.startswith("grimace._south_star1."):
        _violation("closed_term_class_path_mismatch")
    module_name, class_name = path.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), class_name, None)
    if not isinstance(cls, type):
        _violation("closed_term_class_missing")
    return cls


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
