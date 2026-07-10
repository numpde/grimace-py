"""Producer-free structural checker for writer support artifact tables."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_envelope_terms import _canonical_json
from .writer_envelope_terms import _digest_terms_bounded
from .writer_envelope_terms import _identity_digest
from .writer_envelope_work import WriterEnvelopeWorkBudget
from .writer_envelope_work import WriterEnvelopeWorkExceeded
from .writer_envelope_work import check_writer_envelope_work
from .writer_envelope_work import default_writer_envelope_work_budget
from .writer_envelope_work import writer_envelope_work_reason

SCHEMA_NAME = "writer_support_artifact"
SCHEMA_VERSION = 2
_TOP_LEVEL_FIELDS = frozenset((
    "schema_name",
    "schema_version",
    "prepared_identity",
    "source_kind",
    "source_snapshot",
    "prefix_read_envelope",
    "objects",
    "roots",
    "metrics",
    "digest",
))
_SOURCE_KINDS = frozenset(("snapshot", "prefix_read"))
_OBJECT_FIELDS = frozenset(("object_id", "kind", "payload", "digest"))


@dataclass(frozen=True, slots=True)
class WriterSupportArtifactCheckResult:
    accepted: bool
    support_count: int | None = None
    witness_count: int | None = None
    object_count: int | None = None
    reason: str | None = None


def verify_writer_support_artifact_consistency(
    artifact: object,
    *,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> WriterSupportArtifactCheckResult:
    try:
        budget = default_writer_envelope_work_budget(budget)
        _validate_shape(artifact)
        assert isinstance(artifact, Mapping)
        objects = _object_by_id(artifact, budget=budget)
        _validate_metrics(artifact, budget=budget)
        _validate_artifact_digest(artifact, budget=budget)
        _validate_object_table_closed(artifact, objects, budget=budget)
        root = _require_object(objects, artifact["roots"]["support_image_root"])
        if root["kind"] != "support_image":
            _artifact_violation("support_image_root_kind_mismatch")
        _validate_support_image_root(root, objects, budget=budget)
        return WriterSupportArtifactCheckResult(
            accepted=True,
            support_count=int(root["payload"]["distinct_count"]),
            witness_count=int(root["payload"]["witness_count"]),
            object_count=len(objects),
        )
    except WriterEnvelopeWorkExceeded as exc:
        return WriterSupportArtifactCheckResult(
            accepted=False,
            reason=writer_envelope_work_reason(exc),
        )
    except SouthStarError as exc:
        return WriterSupportArtifactCheckResult(
            accepted=False,
            reason=exc.args[-1] if exc.args else "verification_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterSupportArtifactCheckResult(
            accepted=False,
            reason=f"malformed_artifact:{type(exc).__name__}",
        )


def artifact_metrics(
    objects: list[dict[str, object]],
    *,
    roots: Mapping[str, object] | None = None,
) -> dict[str, object]:
    kind_counts: dict[str, int] = {}
    total_payload_bytes = 0
    largest_object_digest_bytes = 0
    for item in objects:
        kind = str(item["kind"])
        kind_counts[kind] = kind_counts.get(kind, 0) + 1
        size = len(_canonical_json(item["payload"]).encode("utf-8"))
        total_payload_bytes += size
        largest_object_digest_bytes = max(largest_object_digest_bytes, size)
    support_string_refs = []
    coverage_bucket_count = 0
    count_dag_node_count = None
    count_dag_edge_count = None
    reachable_object_count = 0
    unreferenced_object_count = 0
    raw_objects = {item["object_id"]: item for item in objects}
    if roots is not None and "support_image_root" in roots:
        root = raw_objects.get(roots["support_image_root"])
        if root is not None and root.get("kind") == "support_image":
            support_string_refs = list(root["payload"]["support_string_refs"])
            coverage = raw_objects.get(root["payload"]["coverage_ref"])
            if coverage is not None and coverage.get("kind") == "support_image_coverage":
                payload = coverage["payload"]
                coverage_bucket_count = len(payload["text_buckets"]) + (
                    0 if payload["terminal_bucket"] is None else 1
                )
            count = raw_objects.get(root["payload"]["count_ref"])
            if count is not None and count.get("kind") == "count_envelope":
                count_dag_node_count = count["payload"].get("count_dag_node_count")
                count_dag_edge_count = count["payload"].get("count_dag_edge_count")
        reachable = _reachable_object_ids(raw_objects, roots["support_image_root"])
        reachable_object_count = len(reachable)
        unreferenced_object_count = len(set(raw_objects) - reachable)
    return {
        "object_count": len(objects),
        "object_kind_counts": kind_counts,
        "reachable_object_count": reachable_object_count,
        "unreferenced_object_count": unreferenced_object_count,
        "support_string_count": len(support_string_refs),
        "coverage_bucket_count": coverage_bucket_count,
        "count_dag_node_count": count_dag_node_count,
        "count_dag_edge_count": count_dag_edge_count,
        "unique_replay_path_count": kind_counts.get("replay_path", 0),
        "unique_branch_support_count": kind_counts.get("branch_support", 0),
        "unique_text_projection_count": kind_counts.get("text_projection", 0),
        "unique_terminal_projection_count": kind_counts.get("terminal_projection", 0),
        "unique_terminal_support_count": kind_counts.get("terminal_support", 0),
        "total_payload_bytes": total_payload_bytes,
        "total_artifact_payload_bytes": total_payload_bytes,
        "largest_object_digest_bytes": largest_object_digest_bytes,
        "largest_object_digest_payload_bytes": largest_object_digest_bytes,
    }


def artifact_manifest(artifact: Mapping[str, object]) -> dict[str, object]:
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


def _validate_shape(artifact: object) -> None:
    if not isinstance(artifact, Mapping):
        _artifact_violation("artifact_not_mapping")
    if frozenset(artifact) != _TOP_LEVEL_FIELDS:
        _artifact_violation("top_level_fields_mismatch")
    if artifact["schema_name"] != SCHEMA_NAME:
        _artifact_violation("unknown_schema_name")
    if artifact["schema_version"] != SCHEMA_VERSION:
        _artifact_violation("unknown_schema_version")
    if artifact["source_kind"] not in _SOURCE_KINDS:
        _artifact_violation("unknown_source_kind")
    if artifact["source_kind"] == "snapshot":
        if artifact["source_snapshot"] is None:
            _artifact_violation("snapshot_source_missing")
        if artifact["prefix_read_envelope"] is not None:
            _artifact_violation("snapshot_source_has_prefix")
    else:
        if artifact["source_snapshot"] is not None:
            _artifact_violation("prefix_source_has_snapshot")
        if artifact["prefix_read_envelope"] is None:
            _artifact_violation("prefix_source_missing_prefix")
    if not isinstance(artifact["objects"], list):
        _artifact_violation("objects_not_list")
    if not isinstance(artifact["roots"], Mapping):
        _artifact_violation("roots_not_mapping")
    if not isinstance(artifact["metrics"], Mapping):
        _artifact_violation("metrics_not_mapping")


def _object_by_id(
    artifact: Mapping[str, object],
    *,
    budget: WriterEnvelopeWorkBudget,
) -> dict[str, Mapping[str, object]]:
    objects = {}
    for item in artifact["objects"]:
        if not isinstance(item, Mapping):
            _artifact_violation("object_not_mapping")
        if frozenset(item) != _OBJECT_FIELDS:
            _artifact_violation("object_fields_mismatch")
        object_id = item["object_id"]
        if object_id in objects:
            _artifact_violation("duplicate_object_id")
        expected_digest = _identity_digest(
            {"kind": item["kind"], "payload": item["payload"]},
            budget=budget,
            operation="support_artifact.object.digest",
        )
        if item["digest"] != expected_digest:
            _artifact_violation("object_digest_mismatch")
        if object_id != f"obj:{item['digest']}":
            _artifact_violation("object_id_digest_mismatch")
        objects[object_id] = item
    return objects


def _validate_metrics(
    artifact: Mapping[str, object],
    *,
    budget: WriterEnvelopeWorkBudget,
) -> None:
    metrics = artifact_metrics(
        artifact["objects"],
        roots=artifact["roots"],
    )
    if artifact["metrics"] != metrics:
        _artifact_violation("metrics_mismatch")
    check_writer_envelope_work(
        budget=budget,
        operation="support_artifact_check",
        metric="envelope_object_count",
        actual=metrics["object_count"],
        limit=budget.max_envelope_nodes,
    )


def _validate_artifact_digest(
    artifact: Mapping[str, object],
    *,
    budget: WriterEnvelopeWorkBudget,
) -> None:
    expected = _digest_terms_bounded(
        artifact_manifest(artifact),
        budget=budget,
        operation="support_artifact.manifest.digest",
    )
    if artifact["digest"] != expected:
        _artifact_violation("artifact_digest_mismatch")


def _validate_object_table_closed(
    artifact: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
    *,
    budget: WriterEnvelopeWorkBudget,
) -> None:
    roots = artifact["roots"]
    if frozenset(roots) != frozenset((
        "source_ref",
        "count_ref",
        "frontier_product_ref",
        "support_image_root",
    )):
        _artifact_violation("roots_fields_mismatch")
    for field in roots:
        _require_object(objects, roots[field])
    root = _require_object(objects, roots["support_image_root"])
    if root["kind"] != "support_image":
        _artifact_violation("support_image_root_kind_mismatch")
    payload = root["payload"]
    if payload["source_ref"] != roots["source_ref"]:
        _artifact_violation("root_source_ref_mismatch")
    if payload["count_ref"] != roots["count_ref"]:
        _artifact_violation("root_count_ref_mismatch")
    if payload["frontier_product_ref"] != roots["frontier_product_ref"]:
        _artifact_violation("root_frontier_product_ref_mismatch")
    for item in objects.values():
        _validate_object_payload_shape(item, budget=budget)
    reachable = _reachable_object_ids(objects, roots["support_image_root"])
    if reachable - set(objects):
        _artifact_violation("dangling_object_ref")
    if reachable != set(objects):
        _artifact_violation("unreferenced_object")


def _validate_object_payload_shape(
    item: Mapping[str, object],
    *,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> None:
    budget = default_writer_envelope_work_budget(budget)
    payload = item["payload"]
    kind = item["kind"]
    if not isinstance(payload, Mapping):
        _artifact_violation("object_payload_not_mapping")
    if kind == "source_snapshot":
        _require_exact_payload_fields(
            payload,
            (
                "serialization_language",
                "runtime_options",
                "prepared_identity_digest",
                "prepared_identity_terms",
                "cursor",
                "frame_stack_cursors",
                "decoder_boundary",
                "digest",
            ),
        )
        _require_mapping(payload["cursor"], "source_snapshot_cursor_not_mapping")
    elif kind == "count_envelope":
        _require_exact_payload_fields(
            payload,
            (
                "schema_name",
                "schema_version",
                "source_kind",
                "count_dag_ref",
                "frontier_snapshot_digest",
                "frontier_product_digest",
                "count_dag_digest",
                "support_count",
                "completion_count",
                "support_count_certificate_digest",
                "completion_count_certificate_digest",
                "count_dag_node_count",
                "count_dag_edge_count",
            ),
        )
        _require_int(payload["support_count"], "support_count_not_int")
        _require_int(payload["completion_count"], "completion_count_not_int")
        _require_int(payload["count_dag_node_count"], "count_dag_node_count_not_int")
        _require_int(payload["count_dag_edge_count"], "count_dag_edge_count_not_int")
        if not isinstance(payload["count_dag_ref"], str):
            _artifact_violation("count_dag_ref_not_string")
    elif kind == "count_dag":
        _require_mapping(payload, "count_dag_payload_not_mapping")
        for field in ("schema_name", "schema_version", "roots", "nodes", "metrics", "digest"):
            if field not in payload:
                _artifact_violation("count_dag_payload_fields_mismatch")
    elif kind == "frontier_product":
        _require_mapping(payload, "frontier_product_payload_not_mapping")
        if "kind" not in payload or "digest" not in payload:
            _artifact_violation("frontier_product_payload_fields_mismatch")
    elif kind == "replay_path":
        _require_exact_payload_fields(
            payload,
            (
                "source_ref",
                "emitted_texts",
                "text_projection_refs",
                "replay_certificate_digest",
                "final_cursor_digest",
                "final_snapshot_digest",
            ),
        )
        _require_string_list(payload["emitted_texts"], "replay_emitted_texts_not_strings")
        _require_string_list(payload["text_projection_refs"], "replay_text_projection_refs_not_strings")
        if not isinstance(payload["final_cursor_digest"], str):
            _artifact_violation("replay_final_cursor_digest_not_string")
    elif kind == "text_projection":
        _require_mapping(payload, "identity_payload_not_mapping")
        if "digest" not in payload:
            _artifact_violation("identity_payload_missing_digest")
        if "branch_support_refs" not in payload:
            _artifact_violation("text_projection_branch_support_refs_missing")
        _require_string_list(
            payload["branch_support_refs"],
            "branch_support_refs_not_strings",
        )
    elif kind == "branch_support":
        _require_exact_payload_fields(
            payload,
            (
                "emitted_text",
                "source_state_digest",
                "successor_state_digest",
                "source_cursor_digest",
                "successor_cursor_digest",
                "parent_weight",
                "branch_ordinal",
                "transition_kind",
                "graph_action_surface_digest",
                "successor_state_certificate_digest",
                "checked_branch_certificate_digest",
                "local_evidence",
                "graph_ring_delta",
                "obligation_summary",
                "obligation_manifests",
                "digest",
            ),
        )
        for field in (
            "emitted_text",
            "source_state_digest",
            "successor_state_digest",
            "source_cursor_digest",
            "successor_cursor_digest",
            "graph_action_surface_digest",
            "successor_state_certificate_digest",
            "checked_branch_certificate_digest",
            "digest",
        ):
            if not isinstance(payload[field], str):
                _artifact_violation("branch_support_string_field_mismatch")
        _require_int(payload["parent_weight"], "branch_support_parent_weight_not_int")
        _require_int(payload["branch_ordinal"], "branch_support_ordinal_not_int")
        _validate_local_evidence_payload(payload["local_evidence"], budget=budget)
        _validate_graph_ring_delta_payload(payload["graph_ring_delta"], budget=budget)
        _validate_obligation_summary(
            payload["obligation_summary"],
            (
                "residual_work_count",
                "finite_relation_work_count",
                "graph_obligation_work_count",
                "stereo_lifecycle_count",
                "residual_attachment_lifecycle_count",
                "closure_candidate_lifecycle_count",
                "directional_ring_closure_lifecycle_count",
            ),
        )
        _validate_obligation_manifests(
            payload["obligation_manifests"],
            payload["obligation_summary"],
            {
                "residual_work": "residual_work_count",
                "finite_relation_work": "finite_relation_work_count",
                "graph_obligation_work": "graph_obligation_work_count",
                "stereo_lifecycle": "stereo_lifecycle_count",
                "residual_attachment_lifecycle": "residual_attachment_lifecycle_count",
                "closure_candidate_lifecycle": "closure_candidate_lifecycle_count",
                "directional_ring_closure_lifecycle": (
                    "directional_ring_closure_lifecycle_count"
                ),
            },
        )
    elif kind == "terminal_projection":
        _require_exact_payload_fields(
            payload,
            (
                "source_cursor",
                "finalized_cursor",
                "multiplicity",
                "support_count",
                "completion_count",
                "terminal_support_identities",
                "terminal_certificate_digests",
                "digest",
            ),
        )
        _require_mapping(payload["source_cursor"], "terminal_source_cursor_not_mapping")
        _require_mapping(
            payload["finalized_cursor"],
            "terminal_finalized_cursor_not_mapping",
        )
        _require_int(payload["multiplicity"], "terminal_multiplicity_not_int")
        _require_int(payload["support_count"], "terminal_support_count_not_int")
        _require_int(payload["completion_count"], "terminal_completion_count_not_int")
        _validate_terminal_support_identities(
            payload["terminal_support_identities"],
            include_obligation_summary=False,
        )
        _require_string_list(
            payload["terminal_certificate_digests"],
            "terminal_certificate_digests_not_strings",
        )
        if not isinstance(payload["digest"], str):
            _artifact_violation("terminal_projection_digest_not_string")
    elif kind == "terminal_support":
        _validate_terminal_support_identity(payload, include_obligation_summary=True)
    elif kind == "support_string":
        _require_exact_payload_fields(
            payload,
            (
                "index",
                "string",
                "emitted_texts",
                "source_ref",
                "count_ref",
                "replay_path_ref",
                "text_projection_refs",
                "terminal_projection_ref",
                "terminal_support_refs",
            ),
        )
        _require_int(payload["index"], "support_string_index_not_int")
        if not isinstance(payload["string"], str):
            _artifact_violation("support_string_not_string")
        _require_string_list(payload["emitted_texts"], "support_string_emitted_texts_not_strings")
        _require_string_list(payload["text_projection_refs"], "text_projection_refs_not_strings")
        _require_string_list(payload["terminal_support_refs"], "terminal_support_refs_not_strings")
    elif kind == "support_image_coverage":
        _require_exact_payload_fields(
            payload,
            ("text_buckets", "terminal_bucket", "distinct_count", "support_count"),
        )
        if not isinstance(payload["text_buckets"], list):
            _artifact_violation("text_buckets_not_list")
        _require_int(payload["distinct_count"], "coverage_distinct_count_not_int")
        _require_int(payload["support_count"], "coverage_support_count_not_int")
    elif kind == "support_image":
        _require_exact_payload_fields(
            payload,
            (
                "source_ref",
                "count_ref",
                "frontier_product_ref",
                "support_string_refs",
                "coverage_ref",
                "support_strings",
                "distinct_count",
                "witness_count",
                "support_count_certificate_digest",
                "witness_count_certificate_digest",
            ),
        )
        _require_string_list(payload["support_string_refs"], "support_string_refs_not_strings")
        _require_string_list(payload["support_strings"], "support_strings_not_strings")
        _require_int(payload["distinct_count"], "distinct_count_not_int")
        _require_int(payload["witness_count"], "witness_count_not_int")
    else:
        _artifact_violation("unknown_object_kind")


def _validate_local_evidence_payload(
    evidence: object,
    *,
    budget: WriterEnvelopeWorkBudget,
) -> None:
    _require_mapping(evidence, "local_evidence_not_mapping")
    _require_exact_payload_fields(evidence, ("kind", "manifest", "digest"))
    if not isinstance(evidence["kind"], str):
        _artifact_violation("local_evidence_kind_not_string")
    if not isinstance(evidence["digest"], str):
        _artifact_violation("local_evidence_digest_not_string")
    _require_mapping(evidence["manifest"], "local_evidence_manifest_not_mapping")
    kind = evidence["kind"]
    manifest = evidence["manifest"]
    expected_digest = _identity_digest(
        {"kind": kind, "manifest": manifest},
        budget=budget,
        operation=f"support_artifact_check.local_evidence.{kind}.digest",
    )
    if evidence["digest"] != expected_digest:
        _artifact_violation("local_evidence_digest_mismatch")
    if kind == "other_structural":
        if manifest:
            _artifact_violation("other_structural_local_evidence_manifest_not_empty")
        return
    if kind == "plain_atom_text":
        _require_exact_payload_fields(
            manifest,
            (
                "atom_id",
                "element",
                "aromatic",
                "rendered_text",
                "bracket_required",
            ),
        )
        for field in ("element", "rendered_text"):
            if not isinstance(manifest[field], str):
                _artifact_violation("plain_atom_text_local_evidence_string_mismatch")
        if not isinstance(manifest["aromatic"], bool):
            _artifact_violation("plain_atom_text_local_evidence_aromatic_not_bool")
        if not isinstance(manifest["bracket_required"], bool):
            _artifact_violation("plain_atom_text_local_evidence_bracket_not_bool")
        return
    if kind == "bracket_atom_text":
        _require_exact_payload_fields(
            manifest,
            (
                "atom_id",
                "element",
                "isotope",
                "formal_charge",
                "hydrogen_count",
                "aromatic",
                "rendered_text",
                "bracket_required",
            ),
        )
        for field in ("element", "rendered_text"):
            if not isinstance(manifest[field], str):
                _artifact_violation("bracket_atom_text_local_evidence_string_mismatch")
        _require_int(
            manifest["formal_charge"],
            "bracket_atom_text_local_evidence_charge_not_int",
        )
        _require_int(
            manifest["hydrogen_count"],
            "bracket_atom_text_local_evidence_hydrogen_not_int",
        )
        if not isinstance(manifest["aromatic"], bool):
            _artifact_violation("bracket_atom_text_local_evidence_aromatic_not_bool")
        if not isinstance(manifest["bracket_required"], bool):
            _artifact_violation("bracket_atom_text_local_evidence_bracket_not_bool")
        return
    if kind == "closure_bond_text":
        _require_exact_payload_fields(manifest, ("items",))
        _validate_closure_evidence_items(manifest["items"])
        return
    if kind == "directional_ring_closure_bond_text":
        _require_exact_payload_fields(
            manifest,
            (
                "closure_bond_text",
                "directional_coupled_digests",
                "directional_coupled_count",
            ),
        )
        _validate_closure_evidence_items(manifest["closure_bond_text"])
        _require_string_list(
            manifest["directional_coupled_digests"],
            "directional_coupled_digests_not_strings",
        )
        _require_int(
            manifest["directional_coupled_count"],
            "directional_coupled_count_not_int",
        )
        if (
            manifest["directional_coupled_count"]
            != len(manifest["directional_coupled_digests"])
        ):
            _artifact_violation("directional_coupled_count_mismatch")
        return
    _artifact_violation("unknown_local_evidence_kind")


def _validate_graph_ring_delta_payload(
    delta: object,
    *,
    budget: WriterEnvelopeWorkBudget,
) -> None:
    _require_mapping(delta, "graph_ring_delta_not_mapping")
    _require_exact_payload_fields(delta, ("kind", "manifest", "digest"))
    if not isinstance(delta["kind"], str):
        _artifact_violation("graph_ring_delta_kind_not_string")
    if not isinstance(delta["digest"], str):
        _artifact_violation("graph_ring_delta_digest_not_string")
    _require_mapping(delta["manifest"], "graph_ring_delta_manifest_not_mapping")
    kind = delta["kind"]
    if kind not in {
        "atom_start",
        "atom_advance",
        "bond_advance",
        "branch_open",
        "branch_return",
        "ring_endpoint_open",
        "ring_endpoint_pair",
        "ring_endpoint_pair_non_single",
        "other_structural",
    }:
        _artifact_violation("unknown_graph_ring_delta_kind")
    manifest = delta["manifest"]
    _require_exact_payload_fields(
        manifest,
        (
            "source_state_digest",
            "successor_state_digest",
            "source_cursor_digest",
            "successor_cursor_digest",
            "transition_kind",
            "emitted_text",
            "graph_action_surface_digest",
            "successor_state_certificate_digest",
            "checked_branch_certificate_digest",
            "local_evidence_digest",
            "event_manifests",
        ),
    )
    for field in (
        "source_state_digest",
        "successor_state_digest",
        "source_cursor_digest",
        "successor_cursor_digest",
        "emitted_text",
        "graph_action_surface_digest",
        "successor_state_certificate_digest",
        "checked_branch_certificate_digest",
        "local_evidence_digest",
    ):
        if not isinstance(manifest[field], str):
            _artifact_violation("graph_ring_delta_string_field_mismatch")
    if not isinstance(manifest["event_manifests"], list):
        _artifact_violation("graph_ring_delta_events_not_list")
    for event in manifest["event_manifests"]:
        _require_mapping(event, "graph_ring_delta_event_not_mapping")
        if "kind" not in event or not isinstance(event["kind"], str):
            _artifact_violation("graph_ring_delta_event_kind_missing")
        _validate_graph_ring_delta_event_manifest(event)
    expected_digest = _identity_digest(
        {"kind": kind, "manifest": manifest},
        budget=budget,
        operation=f"support_artifact_check.graph_ring_delta.{kind}.digest",
    )
    if delta["digest"] != expected_digest:
        _artifact_violation("graph_ring_delta_digest_mismatch")


def _validate_graph_ring_delta_event_manifest(event: Mapping[str, object]) -> None:
    if event["kind"] != "local_order_closed":
        return
    if frozenset(event) == frozenset(("kind", "atom")):
        _require_int(event["atom"], "graph_ring_delta_local_order_atom_not_int")
        return
    _require_exact_payload_fields(
        event,
        (
            "kind",
            "atom",
            "site",
            "local_order",
            "reference_order",
            "source_local_order_record_digest",
            "successor_local_order_record_digest",
            "local_order_identity_digest",
        ),
    )
    _require_int(event["atom"], "graph_ring_delta_local_order_atom_not_int")
    _require_int(event["site"], "graph_ring_delta_local_order_site_not_int")
    for field in ("local_order", "reference_order"):
        values = event[field]
        if not isinstance(values, list) or not all(
            isinstance(item, int) for item in values
        ):
            _artifact_violation("graph_ring_delta_local_order_values_mismatch")
    for field in (
        "source_local_order_record_digest",
        "successor_local_order_record_digest",
        "local_order_identity_digest",
    ):
        if event[field] is not None and not isinstance(event[field], str):
            _artifact_violation("graph_ring_delta_local_order_digest_mismatch")


def _validate_closure_evidence_items(items: object) -> None:
    if not isinstance(items, list):
        _artifact_violation("closure_evidence_items_not_list")
    for item in items:
        _require_mapping(item, "closure_evidence_item_not_mapping")
        _require_exact_payload_fields(
            item,
            (
                "bond",
                "bond_order",
                "label",
                "opening_atom",
                "closing_atom",
                "opening_marker",
                "closing_marker",
                "marker_side",
                "event_kind",
                "closed_closure_record_digest",
            ),
        )
        for field in (
            "bond_order",
            "opening_marker",
            "closing_marker",
            "marker_side",
            "event_kind",
        ):
            if not isinstance(item[field], str):
                _artifact_violation("closure_evidence_string_field_mismatch")
        if item["bond_order"] not in ("double", "triple"):
            _artifact_violation("closure_evidence_unknown_bond_order")
        if item["marker_side"] not in ("opening", "closing"):
            _artifact_violation("closure_evidence_unknown_marker_side")
        if item["event_kind"] not in ("endpoint_emitted", "endpoint_paired"):
            _artifact_violation("closure_evidence_unknown_event_kind")
        digest = item["closed_closure_record_digest"]
        if digest is not None and not isinstance(digest, str):
            _artifact_violation("closure_evidence_closed_digest_not_string")


def _validate_terminal_support_identities(
    identities: object,
    *,
    include_obligation_summary: bool,
) -> None:
    if not isinstance(identities, list):
        _artifact_violation("terminal_support_identities_not_list")
    for identity in identities:
        _validate_terminal_support_identity(
            identity,
            include_obligation_summary=include_obligation_summary,
        )


def _validate_terminal_support_identity(
    identity: object,
    *,
    include_obligation_summary: bool,
) -> None:
    _require_mapping(identity, "terminal_support_identity_not_mapping")
    fields = (
        "source_state_digest",
        "finalized_state_digest",
        "parent_weight",
        "terminal_ordinal",
        "terminal_support_key_digest",
        "terminal_execution_capabilities_digest",
        "terminal_residual_work_evidence_digest",
        "terminal_stereo_lifecycle_evidence_digest",
        "graph_obligation_work_evidence_digest",
        "terminal_certificate_digests",
        "digest",
    )
    if include_obligation_summary:
        fields = (*fields[:-1], "obligation_summary", "obligation_manifests", fields[-1])
    _require_exact_payload_fields(
        identity,
        fields,
    )
    for field in (
        "source_state_digest",
        "finalized_state_digest",
        "terminal_support_key_digest",
        "terminal_execution_capabilities_digest",
        "terminal_residual_work_evidence_digest",
        "terminal_stereo_lifecycle_evidence_digest",
        "graph_obligation_work_evidence_digest",
        "digest",
    ):
        if not isinstance(identity[field], str):
            _artifact_violation("terminal_support_identity_string_field_mismatch")
    _require_int(identity["parent_weight"], "terminal_parent_weight_not_int")
    _require_int(identity["terminal_ordinal"], "terminal_ordinal_not_int")
    _require_string_list(
        identity["terminal_certificate_digests"],
        "terminal_support_certificate_digests_not_strings",
    )
    if include_obligation_summary:
        _validate_obligation_summary(
            identity["obligation_summary"],
            (
                "terminal_residual_work_count",
                "terminal_stereo_lifecycle_count",
                "graph_obligation_work_count",
            ),
        )
        _validate_obligation_manifests(
            identity["obligation_manifests"],
            identity["obligation_summary"],
            {
                "terminal_residual_work": "terminal_residual_work_count",
                "terminal_stereo_lifecycle": "terminal_stereo_lifecycle_count",
                "terminal_graph_obligation_work": "graph_obligation_work_count",
            },
        )


def _validate_obligation_summary(summary: object, fields: tuple[str, ...]) -> None:
    _require_mapping(summary, "obligation_summary_not_mapping")
    _require_exact_payload_fields(summary, fields)
    for field in fields:
        _require_int(summary[field], "obligation_summary_value_not_int")
        if summary[field] < 0:
            _artifact_violation("obligation_summary_value_negative")


def _validate_obligation_manifests(
    manifests: object,
    summary: Mapping[str, object],
    family_to_count_field: Mapping[str, str],
) -> None:
    _require_mapping(manifests, "obligation_manifests_not_mapping")
    _require_exact_payload_fields(manifests, tuple(family_to_count_field))
    for family, count_field in family_to_count_field.items():
        items = manifests[family]
        if not isinstance(items, list):
            _artifact_violation("obligation_manifest_items_not_list")
        if len(items) != summary[count_field]:
            _artifact_violation("obligation_manifest_count_mismatch")
        for item in items:
            _require_mapping(item, "obligation_manifest_not_mapping")
            _require_exact_payload_fields(
                item,
                (
                    "family",
                    "operation",
                    "source_digest",
                    "successor_digest",
                    "is_noop",
                    "is_empty",
                    "is_discharged",
                    "terminal_clean",
                    "ring_summary",
                    "evidence_digest",
                    "transition_term",
                    "transition_digest",
                    "linked_lifecycle_digests",
                    "linked_residual_work_digests",
                    "lifecycle_event_kind",
                    "lifecycle_capabilities",
                    "lifecycle_outcome_kind",
                    "residual_snapshot_changed",
                    "source_residual_snapshot_digest",
                    "successor_residual_snapshot_digest",
                    "local_orders_changed",
                    "residual_work_digests",
                    "residual_work_operations",
                    "certificate_kind",
                    "certificate_capability",
                    "certificate_lifecycle_digest",
                ),
            )
            if item["family"] != family:
                _artifact_violation("obligation_manifest_family_mismatch")
            for field in (
                "operation",
                "source_digest",
                "successor_digest",
                "evidence_digest",
            ):
                if not isinstance(item[field], str):
                    _artifact_violation("obligation_manifest_string_field_mismatch")
            _validate_residual_transition_manifest(item)
            for field in ("is_noop", "is_empty", "is_discharged", "terminal_clean"):
                if not isinstance(item[field], bool):
                    _artifact_violation("obligation_manifest_bool_field_mismatch")
            _require_string_list(
                item["linked_lifecycle_digests"],
                "obligation_manifest_link_digests_mismatch",
            )
            _require_string_list(
                item["linked_residual_work_digests"],
                "obligation_manifest_reverse_link_digests_mismatch",
            )
            _validate_lifecycle_provenance_manifest(item)
            _validate_ring_obligation_summary(item["ring_summary"])


def _validate_lifecycle_provenance_manifest(item: Mapping[str, object]) -> None:
    for field in (
        "lifecycle_event_kind",
        "lifecycle_outcome_kind",
        "source_residual_snapshot_digest",
        "successor_residual_snapshot_digest",
        "certificate_kind",
        "certificate_capability",
        "certificate_lifecycle_digest",
    ):
        if item[field] is not None and not isinstance(item[field], str):
            _artifact_violation("obligation_manifest_lifecycle_string_mismatch")
    _require_string_list(
        item["lifecycle_capabilities"],
        "obligation_manifest_lifecycle_capabilities_mismatch",
    )
    for field in ("residual_snapshot_changed", "local_orders_changed"):
        if not isinstance(item[field], bool):
            _artifact_violation("obligation_manifest_lifecycle_bool_mismatch")
    _require_string_list(
        item["residual_work_digests"],
        "obligation_manifest_residual_work_digests_mismatch",
    )
    _require_string_list(
        item["residual_work_operations"],
        "obligation_manifest_residual_work_operations_mismatch",
    )
    if item["family"] != "stereo_lifecycle":
        if (
            item["lifecycle_event_kind"] is not None
            or item["lifecycle_capabilities"]
            or item["lifecycle_outcome_kind"] is not None
            or item["residual_snapshot_changed"]
            or item["source_residual_snapshot_digest"] is not None
            or item["successor_residual_snapshot_digest"] is not None
            or item["local_orders_changed"]
            or item["residual_work_digests"]
            or item["residual_work_operations"]
            or item["certificate_kind"] is not None
            or item["certificate_capability"] is not None
            or item["certificate_lifecycle_digest"] is not None
        ):
            _artifact_violation("obligation_manifest_lifecycle_neutral_mismatch")


def _validate_residual_transition_manifest(item: Mapping[str, object]) -> None:
    term = item["transition_term"]
    digest = item["transition_digest"]
    if term is None:
        if digest is not None:
            _artifact_violation("obligation_manifest_transition_digest_mismatch")
        return
    if item["family"] != "residual_work":
        _artifact_violation("obligation_manifest_transition_family_mismatch")
    if not isinstance(digest, str):
        _artifact_violation("obligation_manifest_transition_digest_mismatch")
    _require_mapping(term, "obligation_manifest_transition_not_mapping")
    if frozenset(term) != frozenset(("__dataclass__", "fields")):
        _artifact_violation("obligation_manifest_transition_shape_mismatch")
    expected_path, expected_kind, expected_fields = (
        _expected_transition_manifest_shape(item["operation"])
    )
    if term["__dataclass__"] != expected_path:
        _artifact_violation("obligation_manifest_transition_class_mismatch")
    fields = term["fields"]
    if not isinstance(fields, list):
        _artifact_violation("obligation_manifest_transition_fields_mismatch")
    field_values = {}
    for field in fields:
        if (
            not isinstance(field, list)
            or len(field) != 2
            or not isinstance(field[0], str)
        ):
            _artifact_violation("obligation_manifest_transition_fields_mismatch")
        if field[0] in field_values:
            _artifact_violation("obligation_manifest_transition_fields_mismatch")
        field_values[field[0]] = field[1]
    if frozenset(field_values) != expected_fields:
        _artifact_violation("obligation_manifest_transition_fields_mismatch")
    kind = field_values["kind"]
    if (
        not isinstance(kind, Mapping)
        or frozenset(kind) != frozenset(("__enum__", "value"))
        or kind.get("__enum__")
        != (
            "grimace._south_star1.writer_residual_transition_terms."
            "WriterResidualTransitionKind"
        )
        or kind.get("value") != expected_kind
    ):
        _artifact_violation("obligation_manifest_transition_kind_mismatch")
    expected = _digest_terms_bounded(
        term,
        budget=default_writer_envelope_work_budget(None),
        operation="support_artifact.obligation.transition.check.digest",
    )
    if digest != expected:
        _artifact_violation("obligation_manifest_transition_digest_mismatch")


def _expected_transition_manifest_shape(operation: object) -> tuple[str, str, frozenset[str]]:
    common = frozenset(
        (
            "kind",
            "source_snapshot",
            "source_snapshot_digest",
            "atom",
            "site",
            "constraint_var",
            "constraint_value",
            "affected_variables",
            "affected_factor_keys",
            "propagation_result",
            "projected_variables",
            "discharged_factor_keys",
            "successor_snapshot",
            "successor_snapshot_digest",
        )
    )
    if operation == "tetrahedral atom-token restriction":
        return (
            "grimace._south_star1.writer_residual_transition_terms."
            "TetraAtomTokenRestrictionTransitionTerm",
            "tetra_atom_token_restriction",
            common | frozenset(("token",)),
        )
    if operation == "tetrahedral local-order factor closure":
        return (
            "grimace._south_star1.writer_residual_transition_terms."
            "TetraLocalOrderFactorClosureTransitionTerm",
            "tetra_local_order_factor_closure",
            common
            | frozenset(("local_order", "reference_order", "target_parity")),
        )
    _artifact_violation("obligation_manifest_transition_operation_mismatch")


def _validate_ring_obligation_summary(summary: object) -> None:
    if summary is None:
        return
    _require_mapping(summary, "ring_obligation_summary_not_mapping")
    _require_exact_payload_fields(
        summary,
        (
            "relation_kind",
            "operation",
            "bond",
            "endpoint_atom",
            "partner_atom",
            "ring_label",
            "side",
            "marker",
            "marker_count",
            "pending_before_count",
            "pending_after_count",
            "closed_before_count",
            "closed_after_count",
            "is_exact",
            "is_exhausted",
            "is_complete",
            "is_discharged",
        ),
    )
    for field in ("relation_kind", "operation", "side", "marker"):
        if not isinstance(summary[field], str):
            _artifact_violation("ring_obligation_summary_string_field_mismatch")
    for field in (
        "marker_count",
        "pending_before_count",
        "pending_after_count",
        "closed_before_count",
        "closed_after_count",
    ):
        _require_int(
            summary[field],
            "ring_obligation_summary_count_not_int",
        )
        if summary[field] < 0:
            _artifact_violation("ring_obligation_summary_count_negative")
    for field in ("is_exact", "is_exhausted", "is_complete", "is_discharged"):
        if not isinstance(summary[field], bool):
            _artifact_violation("ring_obligation_summary_bool_field_mismatch")


def _require_exact_payload_fields(payload: Mapping[str, object], fields: tuple[str, ...]) -> None:
    if frozenset(payload) != frozenset(fields):
        _artifact_violation("object_payload_fields_mismatch")


def _require_mapping(value: object, kind: str) -> None:
    if not isinstance(value, Mapping):
        _artifact_violation(kind)


def _require_int(value: object, kind: str) -> None:
    if not isinstance(value, int):
        _artifact_violation(kind)


def _require_string_list(value: object, kind: str) -> None:
    if not isinstance(value, list) or not all(isinstance(item, str) for item in value):
        _artifact_violation(kind)


def _validate_support_image_root(
    root: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
    *,
    budget: WriterEnvelopeWorkBudget,
) -> None:
    payload = root["payload"]
    _validate_object_payload_shape(root, budget=budget)
    count = _require_object(objects, payload["count_ref"])
    if count["kind"] != "count_envelope":
        _artifact_violation("count_ref_kind_mismatch")
    count_dag = _require_object(objects, count["payload"]["count_dag_ref"])
    if count_dag["kind"] != "count_dag":
        _artifact_violation("count_dag_ref_kind_mismatch")
    source = _require_object(objects, payload["source_ref"])
    if source["kind"] != "source_snapshot":
        _artifact_violation("source_ref_kind_mismatch")
    frontier = _require_object(objects, payload["frontier_product_ref"])
    if frontier["kind"] != "frontier_product":
        _artifact_violation("frontier_product_ref_kind_mismatch")
    if source["payload"]["digest"] != count["payload"]["frontier_snapshot_digest"]:
        _artifact_violation("source_count_snapshot_mismatch")
    if frontier["payload"]["digest"] != count["payload"]["frontier_product_digest"]:
        _artifact_violation("frontier_count_digest_mismatch")
    if count_dag["payload"]["digest"] != count["payload"]["count_dag_digest"]:
        _artifact_violation("count_dag_digest_mismatch")
    if count_dag["payload"]["metrics"]["node_count"] != count["payload"]["count_dag_node_count"]:
        _artifact_violation("count_dag_node_count_mismatch")
    if count_dag["payload"]["metrics"]["edge_count"] != count["payload"]["count_dag_edge_count"]:
        _artifact_violation("count_dag_edge_count_mismatch")
    if payload["distinct_count"] != count["payload"]["support_count"]:
        _artifact_violation("distinct_count_mismatch")
    if payload["witness_count"] != count["payload"]["completion_count"]:
        _artifact_violation("witness_count_mismatch")
    if (
        payload["support_count_certificate_digest"]
        != count["payload"]["support_count_certificate_digest"]
    ):
        _artifact_violation("support_count_certificate_mismatch")
    if (
        payload["witness_count_certificate_digest"]
        != count["payload"]["completion_count_certificate_digest"]
    ):
        _artifact_violation("witness_count_certificate_mismatch")
    string_refs = payload["support_string_refs"]
    if len(string_refs) != len(payload["support_strings"]):
        _artifact_violation("support_string_count_mismatch")
    if len(set(payload["support_strings"])) != len(payload["support_strings"]):
        _artifact_violation("duplicate_support_string")
    seen = []
    for index, ref in enumerate(string_refs):
        item = _require_object(objects, ref)
        if item["kind"] != "support_string":
            _artifact_violation("support_string_ref_kind_mismatch")
        if item["payload"]["index"] != index:
            _artifact_violation("support_string_index_mismatch")
        if item["payload"]["string"] != payload["support_strings"][index]:
            _artifact_violation("support_string_order_mismatch")
        if item["payload"]["source_ref"] != payload["source_ref"]:
            _artifact_violation("support_string_source_ref_mismatch")
        if item["payload"]["count_ref"] != payload["count_ref"]:
            _artifact_violation("support_string_count_ref_mismatch")
        _validate_support_string_refs(item, objects)
        seen.append(ref)
    coverage = _require_object(objects, payload["coverage_ref"])
    if coverage["kind"] != "support_image_coverage":
        _artifact_violation("coverage_ref_kind_mismatch")
    _validate_coverage(coverage, string_refs, objects)
    if Counter(seen) != Counter(string_refs):
        _artifact_violation("support_string_ref_mismatch")


def _validate_support_string_refs(
    item: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
) -> None:
    payload = item["payload"]
    replay = _require_object(objects, payload["replay_path_ref"])
    if replay["kind"] != "replay_path":
        _artifact_violation("replay_path_kind_mismatch")
    if replay["payload"]["source_ref"] != payload["source_ref"]:
        _artifact_violation("replay_path_source_ref_mismatch")
    if replay["payload"]["emitted_texts"] != payload["emitted_texts"]:
        _artifact_violation("replay_path_text_mismatch")
    for ref in payload["text_projection_refs"]:
        if _require_object(objects, ref)["kind"] != "text_projection":
            _artifact_violation("text_projection_ref_kind_mismatch")
    if _require_object(objects, payload["terminal_projection_ref"])["kind"] != "terminal_projection":
        _artifact_violation("terminal_projection_ref_kind_mismatch")
    for ref in payload["terminal_support_refs"]:
        if _require_object(objects, ref)["kind"] != "terminal_support":
            _artifact_violation("terminal_support_ref_kind_mismatch")


def _validate_coverage(
    coverage: Mapping[str, object],
    support_string_refs: list[str],
    objects: Mapping[str, Mapping[str, object]],
) -> None:
    payload = coverage["payload"]
    if payload["distinct_count"] != len(support_string_refs):
        _artifact_violation("coverage_distinct_count_mismatch")
    if payload["support_count"] != len(support_string_refs):
        _artifact_violation("coverage_support_count_mismatch")
    assigned = []
    for bucket in payload["text_buckets"]:
        if not isinstance(bucket, Mapping):
            _artifact_violation("text_bucket_not_mapping")
        if frozenset(bucket) != frozenset(("text_projection", "support_count", "string_refs")):
            _artifact_violation("text_bucket_fields_mismatch")
        refs = bucket["string_refs"]
        _require_string_list(refs, "text_bucket_refs_not_strings")
        if bucket["support_count"] != len(refs):
            _artifact_violation("text_bucket_count_mismatch")
        for ref in refs:
            if ref not in support_string_refs:
                _artifact_violation("text_bucket_unknown_string_ref")
            support_string = _require_object(objects, ref)
            if not support_string["payload"]["emitted_texts"]:
                _artifact_violation("empty_string_in_text_bucket")
        assigned.extend(refs)
    terminal = payload["terminal_bucket"]
    empty_refs = [
        ref
        for ref in support_string_refs
        if not _require_object(objects, ref)["payload"]["emitted_texts"]
    ]
    if terminal is None:
        if empty_refs:
            _artifact_violation("terminal_bucket_missing")
    else:
        if not isinstance(terminal, Mapping):
            _artifact_violation("terminal_bucket_not_mapping")
        if frozenset(terminal) != frozenset((
            "terminal_projection",
            "support_count",
            "string_ref",
        )):
            _artifact_violation("terminal_bucket_fields_mismatch")
        if not empty_refs:
            _artifact_violation("terminal_bucket_without_empty_string")
        if terminal["string_ref"] != empty_refs[0]:
            _artifact_violation("terminal_bucket_string_ref_mismatch")
        if terminal["support_count"] != len(empty_refs):
            _artifact_violation("terminal_bucket_count_mismatch")
        assigned.extend(empty_refs)
    if Counter(assigned) != Counter(support_string_refs):
        _artifact_violation("coverage_partition_mismatch")


def _reachable_object_ids(
    objects: Mapping[str, Mapping[str, object]],
    root_id: object,
) -> set[str]:
    reachable: set[str] = set()
    pending = [str(root_id)]
    while pending:
        object_id = pending.pop()
        if object_id in reachable:
            continue
        reachable.add(object_id)
        item = objects.get(object_id)
        if item is None:
            continue
        pending.extend(_object_refs(item))
    return reachable


def _object_refs(item: Mapping[str, object]) -> list[str]:
    payload = item["payload"]
    kind = item["kind"]
    if kind == "support_image":
        return [
            payload["source_ref"],
            payload["count_ref"],
            payload["frontier_product_ref"],
            payload["coverage_ref"],
            *payload["support_string_refs"],
        ]
    if kind == "count_envelope":
        return [payload["count_dag_ref"]]
    if kind == "support_string":
        return [
            payload["source_ref"],
            payload["count_ref"],
            payload["replay_path_ref"],
            payload["terminal_projection_ref"],
            *payload["text_projection_refs"],
            *payload["terminal_support_refs"],
        ]
    if kind == "replay_path":
        return [payload["source_ref"], *payload["text_projection_refs"]]
    if kind == "text_projection":
        return [*payload["branch_support_refs"]]
    if kind == "support_image_coverage":
        refs = [
            ref
            for bucket in payload["text_buckets"]
            for ref in bucket["string_refs"]
        ]
        terminal = payload["terminal_bucket"]
        if terminal is not None and terminal["string_ref"] is not None:
            refs.append(terminal["string_ref"])
        return refs
    return []


def _require_object(
    objects: Mapping[str, Mapping[str, object]],
    object_id: str,
) -> Mapping[str, object]:
    if object_id not in objects:
        _artifact_violation("missing_object_ref")
    return objects[object_id]


def _artifact_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer support artifact checker violation: {kind}",
    )


__all__ = (
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "WriterSupportArtifactCheckResult",
    "artifact_manifest",
    "artifact_metrics",
    "verify_writer_support_artifact_consistency",
)
