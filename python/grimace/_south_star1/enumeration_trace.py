"""Finite enumeration trace objects for South Star support completeness."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Literal
from typing import Mapping


EnumerationNodeKind = Literal[
    "skeleton",
    "prefix",
    "csp",
    "stereo_solution",
    "selected_solution",
    "witness",
]

RejectionReason = Literal[
    "invalid_skeleton",
    "invalid_prefix",
    "empty_tetra_domain",
    "empty_direction_domain",
    "empty_tetra_relation",
    "empty_mark_relation",
    "csp_unsatisfied",
    "annotation_not_selected",
    "render_duplicate",
    "policy_rejected",
    "internal_invariant",
]


@dataclass(frozen=True, slots=True)
class EnumerationNodeId:
    kind: EnumerationNodeKind
    key: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class RejectionCertificate:
    node: EnumerationNodeId
    reason: RejectionReason
    detail: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class AcceptanceCertificate:
    node: EnumerationNodeId
    witness_id: str
    rendered: str


@dataclass(frozen=True, slots=True)
class EnumerationTrace:
    accepted: tuple[AcceptanceCertificate, ...]
    rejected: tuple[RejectionCertificate, ...]

    skeleton_count: int
    prefix_count: int
    csp_count: int
    feasible_solution_count: int
    selected_solution_count: int
    witness_count: int
    support_count: int


@dataclass(frozen=True, slots=True)
class TraceIndex:
    accepted_by_node: dict[EnumerationNodeId, AcceptanceCertificate]
    rejected_by_node: dict[EnumerationNodeId, RejectionCertificate]

    def status(
        self,
        node: EnumerationNodeId,
    ) -> Literal["accepted", "rejected", "missing"]:
        if node in self.accepted_by_node:
            return "accepted"
        if node in self.rejected_by_node:
            return "rejected"
        return "missing"


class CompletenessTraceMode(Enum):
    FULL = "full"
    REJECTIONS_ONLY = "rejections_only"
    ACCEPTANCES_ONLY = "acceptances_only"


def build_trace_index(trace: EnumerationTrace) -> TraceIndex:
    accepted_by_node: dict[EnumerationNodeId, AcceptanceCertificate] = {}
    rejected_by_node: dict[EnumerationNodeId, RejectionCertificate] = {}

    for certificate in trace.accepted:
        if certificate.node in accepted_by_node:
            raise ValueError("duplicate acceptance certificate for node")
        accepted_by_node[certificate.node] = certificate

    for certificate in trace.rejected:
        if certificate.node in rejected_by_node:
            raise ValueError("duplicate rejection certificate for node")
        rejected_by_node[certificate.node] = certificate

    overlap = set(accepted_by_node) & set(rejected_by_node)
    if overlap:
        raise ValueError("node cannot be both accepted and rejected")

    if len(trace.accepted) != trace.witness_count:
        raise ValueError("accepted witness count does not match trace witness_count")
    unique_rendered = tuple(dict.fromkeys(cert.rendered for cert in trace.accepted))
    if len(unique_rendered) != trace.support_count:
        raise ValueError("unique accepted rendered count does not match support_count")

    return TraceIndex(
        accepted_by_node=accepted_by_node,
        rejected_by_node=rejected_by_node,
    )


def enumeration_trace_to_jsonable(trace: EnumerationTrace) -> dict[str, object]:
    return {
        "accepted": [
            _acceptance_to_jsonable(certificate)
            for certificate in trace.accepted
        ],
        "rejected": [
            _rejection_to_jsonable(certificate)
            for certificate in trace.rejected
        ],
        "skeleton_count": trace.skeleton_count,
        "prefix_count": trace.prefix_count,
        "csp_count": trace.csp_count,
        "feasible_solution_count": trace.feasible_solution_count,
        "selected_solution_count": trace.selected_solution_count,
        "witness_count": trace.witness_count,
        "support_count": trace.support_count,
    }


def enumeration_trace_from_jsonable(data: Mapping[str, object]) -> EnumerationTrace:
    return EnumerationTrace(
        accepted=tuple(
            _acceptance_from_jsonable(item)
            for item in _require_list(data["accepted"])
        ),
        rejected=tuple(
            _rejection_from_jsonable(item)
            for item in _require_list(data["rejected"])
        ),
        skeleton_count=int(data["skeleton_count"]),
        prefix_count=int(data["prefix_count"]),
        csp_count=int(data["csp_count"]),
        feasible_solution_count=int(data["feasible_solution_count"]),
        selected_solution_count=int(data["selected_solution_count"]),
        witness_count=int(data["witness_count"]),
        support_count=int(data["support_count"]),
    )


def _acceptance_to_jsonable(cert: AcceptanceCertificate) -> dict[str, object]:
    return {
        "node": _node_to_jsonable(cert.node),
        "witness_id": cert.witness_id,
        "rendered": cert.rendered,
    }


def _acceptance_from_jsonable(data: object) -> AcceptanceCertificate:
    mapping = _require_mapping(data)
    return AcceptanceCertificate(
        node=_node_from_jsonable(mapping["node"]),
        witness_id=str(mapping["witness_id"]),
        rendered=str(mapping["rendered"]),
    )


def _rejection_to_jsonable(cert: RejectionCertificate) -> dict[str, object]:
    return {
        "node": _node_to_jsonable(cert.node),
        "reason": cert.reason,
        "detail": _jsonable(cert.detail),
    }


def _rejection_from_jsonable(data: object) -> RejectionCertificate:
    mapping = _require_mapping(data)
    return RejectionCertificate(
        node=_node_from_jsonable(mapping["node"]),
        reason=_rejection_reason(str(mapping["reason"])),
        detail=_tuple_from_jsonable(mapping["detail"]),
    )


def _node_to_jsonable(node: EnumerationNodeId) -> dict[str, object]:
    return {
        "kind": node.kind,
        "key": _jsonable(node.key),
    }


def _node_from_jsonable(data: object) -> EnumerationNodeId:
    mapping = _require_mapping(data)
    return EnumerationNodeId(
        kind=_node_kind(str(mapping["kind"])),
        key=_tuple_from_jsonable(mapping["key"]),
    )


def _node_kind(value: str) -> EnumerationNodeKind:
    if value in {
        "skeleton",
        "prefix",
        "csp",
        "stereo_solution",
        "selected_solution",
        "witness",
    }:
        return value  # type: ignore[return-value]
    raise ValueError(f"unknown enumeration node kind: {value!r}")


def _rejection_reason(value: str) -> RejectionReason:
    if value in {
        "invalid_skeleton",
        "invalid_prefix",
        "empty_tetra_domain",
        "empty_direction_domain",
        "empty_tetra_relation",
        "empty_mark_relation",
        "csp_unsatisfied",
        "annotation_not_selected",
        "render_duplicate",
        "policy_rejected",
        "internal_invariant",
    }:
        return value  # type: ignore[return-value]
    raise ValueError(f"unknown enumeration rejection reason: {value!r}")


def _jsonable(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, frozenset):
        return [_jsonable(item) for item in sorted(value, key=repr)]
    return value


def _tuple_from_jsonable(value: object) -> tuple[object, ...]:
    if isinstance(value, list):
        return tuple(
            _tuple_from_jsonable(item) if isinstance(item, list) else item
            for item in value
        )
    raise TypeError(f"expected JSON list for tuple value: {value!r}")


def _require_list(value: object) -> list:
    if not isinstance(value, list):
        raise TypeError(f"expected list: {value!r}")
    return value


def _require_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping: {value!r}")
    return value


__all__ = (
    "AcceptanceCertificate",
    "CompletenessTraceMode",
    "EnumerationNodeId",
    "EnumerationTrace",
    "RejectionCertificate",
    "TraceIndex",
    "build_trace_index",
    "enumeration_trace_from_jsonable",
    "enumeration_trace_to_jsonable",
)
