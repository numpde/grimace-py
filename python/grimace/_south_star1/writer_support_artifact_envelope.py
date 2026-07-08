"""Table-backed durable artifacts for complete writer support images."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .prepared_runtime import SouthStarPreparedMol
from .writer_envelope_terms import _digest_terms_bounded
from .writer_envelope_terms import _identity_digest
from .writer_envelope_terms import _identity_envelope
from .writer_envelope_terms import _snapshot_identity_envelope
from .writer_support_artifact_checker import SCHEMA_NAME
from .writer_support_artifact_checker import SCHEMA_VERSION
from .writer_support_artifact_checker import artifact_manifest
from .writer_support_artifact_checker import artifact_metrics
from .writer_support_artifact_checker import verify_writer_support_artifact_consistency as _check_writer_support_artifact_consistency
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
from .writer_snapshot_prefix_envelope import _branch_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import _text_projection_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import verify_writer_snapshot_prefix_read_envelope
from .writer_support_image_envelope import _support_image_certificate_for_source
from .writer_support_image_envelope import _text_projection_bucket_key
from .writer_support_string_envelope import _support_string_replay_certificate_digest


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
    result = _check_writer_support_artifact_consistency(envelope, budget=budget)
    return WriterSupportArtifactEnvelopeVerification(
        accepted=result.accepted,
        source_kind=(
            str(envelope.get("source_kind", "unknown"))
            if isinstance(envelope, Mapping)
            else "unknown"
        ),
        support_count=result.support_count,
        witness_count=result.witness_count,
        reason=result.reason,
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
        root = next(
            (
                item
                for item in envelope["objects"]
                if item["object_id"] == envelope["roots"]["support_image_root"]
            ),
            None,
        )
        if root is None:
            _artifact_violation("support_image_root_missing")
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
        _count_payload(
            count_envelope,
            count_dag_ref=table.add(
                "count_dag",
                count_envelope["count_dag"],
                operation="support_artifact.count_dag.object",
            ),
        ),
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
    metrics = artifact_metrics(objects, roots=roots)
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
        artifact_manifest(envelope),
        budget=budget,
        operation="support_artifact.manifest.digest",
    )
    checked = _check_writer_support_artifact_consistency(
        envelope,
        budget=budget,
    )
    if not checked.accepted:
        _artifact_violation(checked.reason or "artifact_checker_rejected")
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
    text_projection_refs = [
        _add_text_projection(
            table,
            projection=projection,
            budget=budget,
        )
        for projection in certificate.text_projection_certificates
    ]
    terminal_projection = _terminal_projection_certificate_identity_envelope(
        certificate.terminal_projection_certificate,
        budget=budget,
    )
    replay_ref = table.add(
        "replay_path",
        {
            "source_ref": source_ref,
            "emitted_texts": list(certificate.emitted_texts),
            "text_projection_refs": text_projection_refs,
            "replay_certificate_digest": _support_string_replay_certificate_digest(
                certificate.replay_certificate,
                budget=budget,
            ),
            "final_cursor_digest": terminal_projection["source_cursor"]["digest"],
            "final_snapshot_digest": _snapshot_identity_envelope(
                certificate.final_snapshot,
                budget=budget,
                operation="support_artifact.replay.final_snapshot.digest",
            )["digest"],
        },
        operation="support_artifact.replay_path.object",
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


def _add_text_projection(
    table,
    *,
    projection,
    budget: WriterEnvelopeWorkBudget,
) -> str:
    envelope = _text_projection_certificate_identity_envelope(
        projection,
        budget=budget,
    )
    branch_support_refs = [
        _add_branch_support(
            table,
            branch=branch,
            text_projection=envelope,
            budget=budget,
        )
        for branch in projection.branch_certificates
    ]
    return table.add(
        "text_projection",
        {
            **envelope,
            "branch_support_refs": branch_support_refs,
        },
        operation="support_artifact.text_projection.object",
    )


def _add_branch_support(
    table,
    *,
    branch,
    text_projection: Mapping[str, object],
    budget: WriterEnvelopeWorkBudget,
) -> str:
    envelope = _branch_certificate_identity_envelope(branch, budget=budget)
    return table.add(
        "branch_support",
        {
            "emitted_text": envelope["emitted_text"],
            "source_state_digest": envelope["source_state_digest"],
            "successor_state_digest": envelope["successor_state_digest"],
            "source_cursor_digest": text_projection["source_cursor"]["digest"],
            "successor_cursor_digest": text_projection["successor_cursor"]["digest"],
            "parent_weight": envelope["parent_weight"],
            "branch_ordinal": envelope["branch_ordinal"],
            "transition_kind": envelope["transition_kind"],
            "graph_action_surface_digest": envelope["graph_action_surface_digest"],
            "successor_state_certificate_digest": (
                envelope["successor_state_certificate_digest"]
            ),
            "checked_branch_certificate_digest": envelope["digest"],
            "digest": envelope["digest"],
        },
        operation="support_artifact.branch_support.object",
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


def _count_payload(
    count_envelope: Mapping[str, object],
    *,
    count_dag_ref: str,
) -> dict[str, object]:
    dag_metrics = count_envelope["count_dag"]["metrics"]
    return {
        "schema_name": count_envelope["schema_name"],
        "schema_version": count_envelope["schema_version"],
        "source_kind": count_envelope["source_kind"],
        "count_dag_ref": count_dag_ref,
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
