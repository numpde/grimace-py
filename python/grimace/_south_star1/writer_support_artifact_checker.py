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
SCHEMA_VERSION = 1
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
        _validate_object_table_closed(artifact, objects)
        root = _require_object(objects, artifact["roots"]["support_image_root"])
        if root["kind"] != "support_image":
            _artifact_violation("support_image_root_kind_mismatch")
        _validate_support_image_root(root, objects)
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
        _validate_object_payload_shape(item)
    reachable = _reachable_object_ids(objects, roots["support_image_root"])
    if reachable - set(objects):
        _artifact_violation("dangling_object_ref")
    if reachable != set(objects):
        _artifact_violation("unreferenced_object")


def _validate_object_payload_shape(item: Mapping[str, object]) -> None:
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
    elif kind in ("text_projection", "terminal_projection", "terminal_support"):
        _require_mapping(payload, "identity_payload_not_mapping")
        if "digest" not in payload:
            _artifact_violation("identity_payload_missing_digest")
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
) -> None:
    payload = root["payload"]
    _validate_object_payload_shape(root)
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
