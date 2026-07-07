"""Table-backed durable artifacts for complete writer support images."""
from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .prepared_runtime import SouthStarPreparedMol
from .writer_envelope_terms import _canonical_json
from .writer_envelope_terms import _digest_terms_bounded
from .writer_envelope_terms import _identity_digest
from .writer_envelope_terms import _identity_envelope
from .writer_envelope_terms import _snapshot_identity_envelope
from .writer_envelope_work import WriterEnvelopeWorkBudget
from .writer_envelope_work import WriterEnvelopeWorkExceeded
from .writer_envelope_work import check_writer_envelope_work
from .writer_envelope_work import default_writer_envelope_work_budget
from .writer_envelope_work import writer_envelope_work_reason
from .writer_frontier import _checked_writer_frontier_product
from .writer_frontier_count_envelope import writer_frontier_count_envelope_for_prefix_read
from .writer_frontier_count_envelope import writer_frontier_count_envelope_for_snapshot
from .writer_snapshot_envelope import _source_snapshot_from_envelope
from .writer_snapshot_prefix_envelope import _terminal_projection_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import _terminal_support_identity_envelope_from_certificate
from .writer_snapshot_prefix_envelope import _text_projection_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import verify_writer_snapshot_prefix_read_envelope
from .writer_support_image_envelope import _support_image_certificate_for_source
from .writer_support_image_envelope import _text_projection_bucket_key
from .writer_support_string_envelope import _support_string_replay_certificate_digest

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
class WriterSupportArtifactEnvelopeVerification:
    accepted: bool
    source_kind: str
    support_count: int | None = None
    witness_count: int | None = None
    reason: str | None = None


def writer_support_artifact_envelope_for_snapshot(
    *,
    prepared: SouthStarPreparedMol,
    snapshot,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> dict[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    product = _checked_product(prepared=prepared, snapshot=snapshot)
    count_envelope = writer_frontier_count_envelope_for_snapshot(
        prepared=prepared,
        snapshot=snapshot,
        budget=budget,
    )
    image = _support_image_certificate_for_source(
        prepared=prepared,
        snapshot=snapshot,
        product=product,
    )
    return _artifact_from_image(
        prepared=prepared,
        source_kind="snapshot",
        source_snapshot=snapshot,
        prefix_read_envelope=None,
        count_envelope=count_envelope,
        product=product,
        image=image,
        budget=budget,
    )


def writer_support_artifact_envelope_for_prefix_read(
    *,
    prepared: SouthStarPreparedMol,
    prefix_read_envelope: Mapping[str, object],
    budget: WriterEnvelopeWorkBudget | None = None,
) -> dict[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    prefix = verify_writer_snapshot_prefix_read_envelope(
        prepared=prepared,
        envelope=prefix_read_envelope,
        budget=budget,
    )
    if not prefix.accepted:
        _artifact_violation("prefix_read_envelope_rejected")
    if prefix.read_kind != "readable" or prefix.final_snapshot is None:
        _artifact_violation("prefix_read_envelope_not_readable")
    product = _checked_product(prepared=prepared, snapshot=prefix.final_snapshot)
    count_envelope = writer_frontier_count_envelope_for_prefix_read(
        prepared=prepared,
        prefix_read_envelope=prefix_read_envelope,
        budget=budget,
    )
    image = _support_image_certificate_for_source(
        prepared=prepared,
        snapshot=prefix.final_snapshot,
        product=product,
    )
    return _artifact_from_image(
        prepared=prepared,
        source_kind="prefix_read",
        source_snapshot=prefix.final_snapshot,
        prefix_read_envelope=prefix_read_envelope,
        count_envelope=count_envelope,
        product=product,
        image=image,
        budget=budget,
    )


def verify_writer_support_artifact_consistency(
    envelope: object,
    *,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> WriterSupportArtifactEnvelopeVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        _validate_shape(envelope)
        assert isinstance(envelope, Mapping)
        objects = _object_by_id(envelope, budget=budget)
        _validate_metrics(envelope, budget=budget)
        _validate_artifact_digest(envelope, budget=budget)
        _validate_object_table_closed(envelope, objects)
        root = _require_object(objects, envelope["roots"]["support_image_root"])
        if root["kind"] != "support_image":
            _artifact_violation("support_image_root_kind_mismatch")
        _validate_support_image_root(root, objects)
        return WriterSupportArtifactEnvelopeVerification(
            accepted=True,
            source_kind=str(envelope["source_kind"]),
            support_count=int(root["payload"]["distinct_count"]),
            witness_count=int(root["payload"]["witness_count"]),
        )
    except WriterEnvelopeWorkExceeded as exc:
        return WriterSupportArtifactEnvelopeVerification(
            accepted=False,
            source_kind=(
                envelope.get("source_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            reason=writer_envelope_work_reason(exc),
        )
    except SouthStarError as exc:
        return WriterSupportArtifactEnvelopeVerification(
            accepted=False,
            source_kind=(
                envelope.get("source_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            reason=exc.args[-1] if exc.args else "verification_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterSupportArtifactEnvelopeVerification(
            accepted=False,
            source_kind=(
                envelope.get("source_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            reason=f"malformed_envelope:{type(exc).__name__}",
        )


def verify_writer_support_artifact_envelope(
    *,
    prepared: SouthStarPreparedMol,
    envelope: object,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> WriterSupportArtifactEnvelopeVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        structural = verify_writer_support_artifact_consistency(
            envelope,
            budget=budget,
        )
        if not structural.accepted:
            return structural
        assert isinstance(envelope, Mapping)
        source_kind = str(envelope["source_kind"])
        source_snapshot = _source_snapshot_for_artifact(
            prepared=prepared,
            envelope=envelope,
            budget=budget,
        )
        if source_kind == "snapshot":
            expected = writer_support_artifact_envelope_for_snapshot(
                prepared=prepared,
                snapshot=source_snapshot,
                budget=budget,
            )
        else:
            expected = writer_support_artifact_envelope_for_prefix_read(
                prepared=prepared,
                prefix_read_envelope=envelope["prefix_read_envelope"],
                budget=budget,
            )
        if expected != envelope:
            return WriterSupportArtifactEnvelopeVerification(
                accepted=False,
                source_kind=source_kind,
                reason="artifact_terms_mismatch",
            )
        root = _require_object(
            _raw_object_by_id(envelope),
            envelope["roots"]["support_image_root"],
        )
        return WriterSupportArtifactEnvelopeVerification(
            accepted=True,
            source_kind=source_kind,
            support_count=int(root["payload"]["distinct_count"]),
            witness_count=int(root["payload"]["witness_count"]),
        )
    except WriterEnvelopeWorkExceeded as exc:
        return WriterSupportArtifactEnvelopeVerification(
            accepted=False,
            source_kind=(
                envelope.get("source_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            reason=writer_envelope_work_reason(exc),
        )
    except SouthStarError as exc:
        return WriterSupportArtifactEnvelopeVerification(
            accepted=False,
            source_kind=(
                envelope.get("source_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            reason=exc.args[-1] if exc.args else "verification_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterSupportArtifactEnvelopeVerification(
            accepted=False,
            source_kind=(
                envelope.get("source_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            reason=f"malformed_envelope:{type(exc).__name__}",
        )


def _artifact_from_image(
    *,
    prepared,
    source_kind: str,
    source_snapshot,
    prefix_read_envelope,
    count_envelope,
    product,
    image,
    budget: WriterEnvelopeWorkBudget,
) -> dict[str, object]:
    del product
    check_writer_envelope_work(
        budget=budget,
        operation="support_artifact_envelope",
        metric="support_string_count",
        actual=len(image.string_certificates),
        limit=budget.max_support_strings,
    )
    check_writer_envelope_work(
        budget=budget,
        operation="support_artifact_envelope",
        metric="total_emitted_text_bytes",
        actual=sum(
            len(text.encode("utf-8"))
            for certificate in image.string_certificates
            for text in certificate.emitted_texts
        ),
        limit=budget.max_total_emitted_text_bytes,
    )
    table = _ObjectTable(budget)
    source_identity = _snapshot_identity_envelope(
        source_snapshot,
        budget=budget,
        operation="support_artifact.source_snapshot.digest",
    )
    source_ref = table.add(
        "source_snapshot",
        source_identity,
        operation="support_artifact.source_snapshot.object",
    )
    count_ref = table.add(
        "count_envelope",
        _count_payload(count_envelope),
        operation="support_artifact.count.object",
    )
    frontier_ref = table.add(
        "frontier_product",
        count_envelope["frontier_product"],
        operation="support_artifact.frontier_product.object",
    )
    support_string_refs = []
    for index, certificate in enumerate(image.string_certificates):
        support_string_refs.append(
            _add_support_string(
                table,
                index=index,
                certificate=certificate,
                source_ref=source_ref,
                count_ref=count_ref,
                budget=budget,
            )
        )
    coverage_ref = _add_coverage(
        table,
        coverage=image.enumeration_coverage_certificate,
        support_string_refs=support_string_refs,
        budget=budget,
    )
    support_image_ref = table.add(
        "support_image",
        {
            "source_ref": source_ref,
            "count_ref": count_ref,
            "frontier_product_ref": frontier_ref,
            "support_string_refs": support_string_refs,
            "coverage_ref": coverage_ref,
            "support_strings": [certificate.string for certificate in image.string_certificates],
            "distinct_count": image.distinct_count,
            "witness_count": image.witness_count,
            "support_count_certificate_digest": count_envelope["support_count_certificate"]["digest"],
            "witness_count_certificate_digest": count_envelope["completion_count_certificate"]["digest"],
        },
        operation="support_artifact.support_image.object",
    )
    objects = table.objects()
    roots = {
        "source_ref": source_ref,
        "count_ref": count_ref,
        "frontier_product_ref": frontier_ref,
        "support_image_root": support_image_ref,
    }
    metrics = _artifact_metrics(objects, roots=roots)
    envelope = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "prepared_identity": _identity_envelope(
            source_snapshot.prepared_identity,
            budget=budget,
            operation="support_artifact.prepared_identity.digest",
        ),
        "source_kind": source_kind,
        "source_snapshot": source_identity if source_kind == "snapshot" else None,
        "prefix_read_envelope": prefix_read_envelope,
        "objects": objects,
        "roots": roots,
        "metrics": metrics,
    }
    envelope["digest"] = _digest_terms_bounded(
        _artifact_manifest(envelope),
        budget=budget,
        operation="support_artifact.manifest.digest",
    )
    _validate_shape(envelope)
    return envelope


def _add_support_string(
    table,
    *,
    index: int,
    certificate,
    source_ref: str,
    count_ref: str,
    budget: WriterEnvelopeWorkBudget,
) -> str:
    replay_ref = table.add(
        "replay_path",
        {
            "source_ref": source_ref,
            "emitted_texts": list(certificate.emitted_texts),
            "replay_certificate_digest": _support_string_replay_certificate_digest(
                certificate.replay_certificate,
                budget=budget,
            ),
            "final_snapshot_digest": _snapshot_identity_envelope(
                certificate.final_snapshot,
                budget=budget,
                operation="support_artifact.replay.final_snapshot.digest",
            )["digest"],
        },
        operation="support_artifact.replay_path.object",
    )
    text_projection_refs = [
        table.add(
            "text_projection",
            _text_projection_certificate_identity_envelope(
                projection,
                budget=budget,
            ),
            operation="support_artifact.text_projection.object",
        )
        for projection in certificate.text_projection_certificates
    ]
    terminal_projection = _terminal_projection_certificate_identity_envelope(
        certificate.terminal_projection_certificate,
        budget=budget,
    )
    terminal_projection_ref = table.add(
        "terminal_projection",
        terminal_projection,
        operation="support_artifact.terminal_projection.object",
    )
    terminal_support_refs = [
        table.add(
            "terminal_support",
            _terminal_support_identity_envelope_from_certificate(
                terminal,
                budget=budget,
            ),
            operation="support_artifact.terminal_support.object",
        )
        for terminal in certificate.terminal_projection_certificate.terminal_certificates
    ]
    return table.add(
        "support_string",
        {
            "index": index,
            "string": certificate.string,
            "emitted_texts": list(certificate.emitted_texts),
            "source_ref": source_ref,
            "count_ref": count_ref,
            "replay_path_ref": replay_ref,
            "text_projection_refs": text_projection_refs,
            "terminal_projection_ref": terminal_projection_ref,
            "terminal_support_refs": terminal_support_refs,
        },
        operation="support_artifact.support_string.object",
    )


def _add_coverage(
    table,
    *,
    coverage,
    support_string_refs: list[str],
    budget: WriterEnvelopeWorkBudget,
) -> str:
    text_buckets = []
    for bucket in coverage.text_buckets:
        indices = [
            coverage.string_certificates.index(certificate)
            for certificate in bucket.string_certificates
        ]
        text_buckets.append(
            {
                "text_projection": _text_projection_certificate_identity_envelope(
                    bucket.support_count_term.text_projection_certificate,
                    budget=budget,
                ),
                "support_count": bucket.support_count,
                "string_refs": [support_string_refs[index] for index in indices],
            }
        )
    terminal = coverage.terminal_bucket
    terminal_bucket = None
    if terminal is not None:
        terminal_bucket = {
            "terminal_projection": None
            if terminal.terminal_support_term is None
            else _terminal_projection_certificate_identity_envelope(
                terminal.terminal_support_term.terminal_projection_certificate,
                budget=budget,
            ),
            "support_count": terminal.support_count,
            "string_ref": None
            if terminal.string_certificate is None
            else support_string_refs[
                coverage.string_certificates.index(terminal.string_certificate)
            ],
        }
    return table.add(
        "support_image_coverage",
        {
            "text_buckets": text_buckets,
            "terminal_bucket": terminal_bucket,
            "distinct_count": coverage.distinct_count,
            "support_count": coverage.support_count,
        },
        operation="support_artifact.coverage.object",
    )


def _count_payload(count_envelope: Mapping[str, object]) -> dict[str, object]:
    dag_metrics = count_envelope["count_dag"]["metrics"]
    return {
        "schema_name": count_envelope["schema_name"],
        "schema_version": count_envelope["schema_version"],
        "source_kind": count_envelope["source_kind"],
        "frontier_snapshot_digest": count_envelope["frontier_snapshot"]["digest"],
        "frontier_product_digest": count_envelope["frontier_product"]["digest"],
        "count_dag_digest": count_envelope["count_dag"]["digest"],
        "support_count": count_envelope["support_count"],
        "completion_count": count_envelope["completion_count"],
        "support_count_certificate_digest": count_envelope["support_count_certificate"]["digest"],
        "completion_count_certificate_digest": count_envelope["completion_count_certificate"]["digest"],
        "count_dag_node_count": dag_metrics["node_count"],
        "count_dag_edge_count": dag_metrics["edge_count"],
    }


class _ObjectTable:
    def __init__(self, budget: WriterEnvelopeWorkBudget):
        self._budget = budget
        self._objects_by_id: dict[str, dict[str, object]] = {}

    def add(self, kind: str, payload, *, operation: str) -> str:
        digest = _identity_digest(
            {"kind": kind, "payload": payload},
            budget=self._budget,
            operation=operation,
        )
        object_id = f"obj:{digest}"
        if object_id not in self._objects_by_id:
            self._objects_by_id[object_id] = {
                "object_id": object_id,
                "kind": kind,
                "payload": payload,
                "digest": digest,
            }
        return object_id

    def objects(self) -> list[dict[str, object]]:
        return sorted(self._objects_by_id.values(), key=lambda item: item["object_id"])


def _artifact_metrics(
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


def _artifact_manifest(envelope: Mapping[str, object]) -> dict[str, object]:
    return {
        "schema_name": envelope["schema_name"],
        "schema_version": envelope["schema_version"],
        "prepared_identity_digest": envelope["prepared_identity"]["digest"],
        "source_kind": envelope["source_kind"],
        "roots": envelope["roots"],
        "metrics": envelope["metrics"],
        "objects": [
            {
                "object_id": item["object_id"],
                "kind": item["kind"],
                "digest": item["digest"],
            }
            for item in envelope["objects"]
        ],
    }


def _validate_shape(envelope: object) -> None:
    if not isinstance(envelope, Mapping):
        _artifact_violation("envelope_not_mapping")
    if frozenset(envelope) != _TOP_LEVEL_FIELDS:
        _artifact_violation("top_level_fields_mismatch")
    if envelope["schema_name"] != SCHEMA_NAME:
        _artifact_violation("unknown_schema_name")
    if envelope["schema_version"] != SCHEMA_VERSION:
        _artifact_violation("unknown_schema_version")
    if envelope["source_kind"] not in _SOURCE_KINDS:
        _artifact_violation("unknown_source_kind")
    if envelope["source_kind"] == "snapshot":
        if envelope["source_snapshot"] is None:
            _artifact_violation("snapshot_source_missing")
        if envelope["prefix_read_envelope"] is not None:
            _artifact_violation("snapshot_source_has_prefix")
    else:
        if envelope["source_snapshot"] is not None:
            _artifact_violation("prefix_source_has_snapshot")
        if envelope["prefix_read_envelope"] is None:
            _artifact_violation("prefix_source_missing_prefix")
    if not isinstance(envelope["objects"], list):
        _artifact_violation("objects_not_list")
    if not isinstance(envelope["roots"], Mapping):
        _artifact_violation("roots_not_mapping")
    if not isinstance(envelope["metrics"], Mapping):
        _artifact_violation("metrics_not_mapping")


def _object_by_id(
    envelope: Mapping[str, object],
    *,
    budget: WriterEnvelopeWorkBudget,
) -> dict[str, Mapping[str, object]]:
    objects = {}
    for item in envelope["objects"]:
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


def _validate_metrics(envelope: Mapping[str, object], *, budget: WriterEnvelopeWorkBudget) -> None:
    metrics = _artifact_metrics(
        envelope["objects"],
        roots=envelope["roots"],
    )
    if envelope["metrics"] != metrics:
        _artifact_violation("metrics_mismatch")
    check_writer_envelope_work(
        budget=budget,
        operation="support_artifact_verify",
        metric="envelope_object_count",
        actual=metrics["object_count"],
        limit=budget.max_envelope_nodes,
    )


def _validate_artifact_digest(envelope: Mapping[str, object], *, budget: WriterEnvelopeWorkBudget) -> None:
    expected = _digest_terms_bounded(
        _artifact_manifest(envelope),
        budget=budget,
        operation="support_artifact.manifest.digest",
    )
    if envelope["digest"] != expected:
        _artifact_violation("artifact_digest_mismatch")


def _validate_support_image_root(root: Mapping[str, object], objects: Mapping[str, Mapping[str, object]]) -> None:
    payload = root["payload"]
    _validate_object_payload_shape(root)
    count = _require_object(objects, payload["count_ref"])
    if count["kind"] != "count_envelope":
        _artifact_violation("count_ref_kind_mismatch")
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


def _validate_object_table_closed(
    envelope: Mapping[str, object],
    objects: Mapping[str, Mapping[str, object]],
) -> None:
    roots = envelope["roots"]
    for field in ("source_ref", "count_ref", "frontier_product_ref", "support_image_root"):
        if field not in roots:
            _artifact_violation("roots_fields_mismatch")
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
    missing = reachable - set(objects)
    if missing:
        _artifact_violation("dangling_object_ref")
    if reachable != set(objects):
        _artifact_violation("unreferenced_object")


def _validate_object_payload_shape(item: Mapping[str, object]) -> None:
    payload = item["payload"]
    kind = item["kind"]
    if not isinstance(payload, Mapping):
        _artifact_violation("object_payload_not_mapping")
    if kind == "source_snapshot":
        _require_payload_fields(payload, ("digest", "cursor"))
    elif kind == "count_envelope":
        _require_payload_fields(
            payload,
            (
                "schema_name",
                "schema_version",
                "source_kind",
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
    elif kind == "frontier_product":
        _require_payload_fields(payload, ("kind", "digest"))
    elif kind == "replay_path":
        _require_payload_fields(
            payload,
            ("source_ref", "emitted_texts", "replay_certificate_digest", "final_snapshot_digest"),
        )
    elif kind in ("text_projection", "terminal_projection", "terminal_support"):
        _require_payload_fields(payload, ("digest",))
    elif kind == "support_string":
        _require_payload_fields(
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
    elif kind == "support_image_coverage":
        _require_payload_fields(
            payload,
            ("text_buckets", "terminal_bucket", "distinct_count", "support_count"),
        )
    elif kind == "support_image":
        _require_payload_fields(
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
    else:
        _artifact_violation("unknown_object_kind")


def _require_payload_fields(payload: Mapping[str, object], fields: tuple[str, ...]) -> None:
    if not all(field in payload for field in fields):
        _artifact_violation("object_payload_fields_mismatch")


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
        return [payload["source_ref"]]
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


def _validate_support_string_refs(item: Mapping[str, object], objects: Mapping[str, Mapping[str, object]]) -> None:
    payload = item["payload"]
    replay = _require_object(objects, payload["replay_path_ref"])
    if replay["kind"] != "replay_path":
        _artifact_violation("replay_path_kind_mismatch")
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
    assigned = []
    for bucket in payload["text_buckets"]:
        refs = bucket["string_refs"]
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
    if empty_refs:
        if terminal is None:
            _artifact_violation("terminal_bucket_missing")
        if terminal["string_ref"] != empty_refs[0]:
            _artifact_violation("terminal_bucket_string_ref_mismatch")
        assigned.extend(empty_refs)
    elif terminal is not None:
        _artifact_violation("terminal_bucket_without_empty_string")
    if Counter(assigned) != Counter(support_string_refs):
        _artifact_violation("coverage_partition_mismatch")


def _raw_object_by_id(envelope: Mapping[str, object]) -> dict[str, Mapping[str, object]]:
    return {item["object_id"]: item for item in envelope["objects"]}


def _require_object(objects: Mapping[str, Mapping[str, object]], object_id: str) -> Mapping[str, object]:
    if object_id not in objects:
        _artifact_violation("missing_object_ref")
    return objects[object_id]


def _source_snapshot_for_artifact(*, prepared, envelope, budget):
    if envelope["source_kind"] == "snapshot":
        return _source_snapshot_from_envelope(
            prepared=prepared,
            envelope=envelope,
            budget=budget,
        )
    prefix = verify_writer_snapshot_prefix_read_envelope(
        prepared=prepared,
        envelope=envelope["prefix_read_envelope"],
        budget=budget,
    )
    if not prefix.accepted:
        _artifact_violation("prefix_read_envelope_rejected")
    if prefix.read_kind != "readable":
        _artifact_violation("prefix_read_envelope_not_readable")
    if prefix.final_snapshot is None:
        _artifact_violation("prefix_read_envelope_lacks_final_snapshot")
    return prefix.final_snapshot


def _checked_product(*, prepared, snapshot):
    return _checked_writer_frontier_product(
        prepared,
        snapshot.cursor,
        include_counts=True,
        include_frontier_certificate=True,
        include_count_certificate=True,
    )


def _artifact_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer support artifact envelope violation: {kind}",
    )


__all__ = (
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "WriterSupportArtifactEnvelopeVerification",
    "verify_writer_support_artifact_consistency",
    "verify_writer_support_artifact_envelope",
    "writer_support_artifact_envelope_for_prefix_read",
    "writer_support_artifact_envelope_for_snapshot",
)
