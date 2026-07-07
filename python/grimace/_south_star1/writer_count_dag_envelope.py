"""Bounded DAG envelopes for recursive writer count certificates."""

from __future__ import annotations

from dataclasses import dataclass

from .writer_envelope_terms import _canonical_json
from .writer_envelope_terms import _cursor_envelope
from .writer_envelope_terms import _digest
from .writer_envelope_terms import _digest_bounded
from .writer_envelope_terms import _snapshot_identity_envelope
from .writer_envelope_terms import _term
from .writer_snapshot_prefix_envelope import (
    _branch_certificate_identity_envelope,
)
from .writer_snapshot_prefix_envelope import (
    _terminal_projection_certificate_identity_envelope,
)
from .writer_snapshot_prefix_envelope import (
    _text_projection_certificate_identity_envelope,
)


SCHEMA_NAME = "writer_count_certificate_dag"
SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class WriterEnvelopeWorkBudget:
    max_count_nodes: int = 10_000
    max_count_edges: int = 50_000
    max_count_depth: int = 1_000
    max_digest_term_bytes: int = 1_000_000
    max_envelope_nodes: int = 50_000


@dataclass(frozen=True, slots=True)
class WriterEnvelopeWorkViolation:
    operation: str
    metric: str
    actual: int
    limit: int


class WriterEnvelopeWorkExceeded(RuntimeError):
    def __init__(self, violation: WriterEnvelopeWorkViolation):
        self.violation = violation
        super().__init__(
            "WRITER_ENVELOPE_WORK_EXCEEDED: "
            f"operation={violation.operation!r}; "
            f"metric={violation.metric!r}; actual={violation.actual}; "
            f"limit={violation.limit}"
        )


class _CountDagBuilder:
    def __init__(self, *, budget: WriterEnvelopeWorkBudget):
        self._budget = budget
        self._nodes: dict[str, dict[str, object]] = {}
        self._depths: dict[str, int] = {}
        self._edge_count = 0

    def build(self, product) -> dict[str, object]:
        roots = {
            "support_count_root": self.support_count(
                product.support_count_certificate
            ),
            "completion_count_root": self.cursor_completion_count(
                product.count_certificate
            ),
            "choice_count_roots": [
                self.text_choice_count(certificate)
                for certificate in product.text_choice_count_certificates
            ],
            "terminal_choice_count_root": self.terminal_choice_count(
                product.terminal_choice_count_certificate
            ),
        }
        metrics = {
            "node_count": len(self._nodes),
            "edge_count": self._edge_count,
            "max_depth": max(self._depths.values(), default=0),
            "digest_input_bytes": sum(
                int(node["digest_input_bytes"])
                for node in self._nodes.values()
            ),
        }
        self._check("count_node_count", metrics["node_count"], self._budget.max_count_nodes)
        self._check("count_edge_count", metrics["edge_count"], self._budget.max_count_edges)
        self._check("count_depth", metrics["max_depth"], self._budget.max_count_depth)
        envelope = {
            "schema_name": SCHEMA_NAME,
            "schema_version": SCHEMA_VERSION,
            "roots": roots,
            "nodes": [
                self._nodes[node_id]
                for node_id in sorted(self._nodes)
            ],
            "metrics": metrics,
        }
        envelope["digest"] = self._digest(_term(envelope), "count_dag_envelope")
        return envelope

    def support_count(self, certificate) -> str | None:
        if certificate is None:
            return None
        child = self.state_support_count(certificate.state_support_count_certificate)
        return self._node(
            "writer_text_support_count",
            {
                "source_snapshot": _snapshot_or_cursor_envelope(
                    certificate.source_snapshot
                ),
                "cursor": _cursor_envelope(certificate.cursor),
                "state_support_count_node_id": child,
                "support_count": certificate.support_count,
            },
            [child],
        )

    def state_support_count(self, certificate) -> str | None:
        if certificate is None:
            return None
        choice_terms = [
            self.text_choice_support_count_term(term)
            for term in certificate.choice_terms
        ]
        return self._node(
            "writer_text_state_support_count",
            {
                "cursor": _cursor_envelope(certificate.cursor),
                "terminal_projection": (
                    _terminal_projection_certificate_identity_envelope(
                        certificate.terminal_projection_certificate
                    )
                ),
                "terminal_count": certificate.terminal_count,
                "choice_term_node_ids": choice_terms,
                "support_count": certificate.support_count,
            },
            choice_terms,
        )

    def text_choice_support_count_term(self, certificate) -> str:
        child = self.state_support_count(
            certificate.successor_support_count_certificate
        )
        return self._node(
            "writer_text_choice_support_count_term",
            {
                "text_projection": _text_projection_certificate_identity_envelope(
                    certificate.text_projection_certificate
                ),
                "successor_support_count_node_id": child,
                "support_count": certificate.support_count,
            },
            [child],
        )

    def cursor_completion_count(self, certificate) -> str | None:
        if certificate is None:
            return None
        entries = []
        children = []
        for state_key, weight, state_certificate in (
            certificate.state_count_certificates
        ):
            child = self.state_completion_count(state_certificate)
            children.append(child)
            entries.append(
                {
                    "state_key_digest": _digest(_term(state_key)),
                    "cursor_weight": weight,
                    "state_count_node_id": child,
                }
            )
        return self._node(
            "writer_cursor_completion_count",
            {
                "cursor": _cursor_envelope(certificate.cursor),
                "state_count_entries": entries,
                "completion_count": certificate.completion_count,
            },
            children,
        )

    def state_completion_count(self, certificate) -> str | None:
        if certificate is None:
            return None
        branch_terms = [
            self.branch_completion_term(term)
            for term in certificate.branch_terms
        ]
        return self._node(
            "writer_state_completion_count",
            {
                "state_key_digest": _digest(_term(certificate.state_key)),
                "terminal_projection": (
                    _terminal_projection_certificate_identity_envelope(
                        certificate.terminal_projection_certificate
                    )
                ),
                "terminal_count": certificate.terminal_count,
                "branch_term_node_ids": branch_terms,
                "completion_count": certificate.completion_count,
            },
            branch_terms,
        )

    def branch_completion_term(self, certificate) -> str:
        child = self.cursor_completion_count(
            certificate.successor_count_certificate
        )
        return self._node(
            "writer_branch_completion_term",
            {
                "branch_certificate": _branch_certificate_identity_envelope(
                    certificate.branch_certificate
                ),
                "successor_count_node_id": child,
                "successor_count": certificate.successor_count,
            },
            [child],
        )

    def text_choice_count(self, certificate) -> str:
        support_child = self.state_support_count(
            certificate.support_count_certificate
        )
        completion_child = self.cursor_completion_count(
            certificate.completion_count_certificate
        )
        return self._node(
            "writer_text_choice_count",
            {
                "text_projection": _text_projection_certificate_identity_envelope(
                    certificate.text_projection_certificate
                ),
                "support_count_node_id": support_child,
                "completion_count_node_id": completion_child,
                "emitted_text": certificate.emitted_text,
                "support_count": certificate.support_count,
                "completion_count": certificate.completion_count,
            },
            [support_child, completion_child],
        )

    def terminal_choice_count(self, certificate) -> str | None:
        if certificate is None:
            return None
        return self._node(
            "writer_terminal_choice_count",
            {
                "terminal_projection": (
                    _terminal_projection_certificate_identity_envelope(
                        certificate.terminal_projection_certificate
                    )
                ),
                "support_count": certificate.support_count,
                "completion_count": certificate.completion_count,
            },
            [],
        )

    def _node(
        self,
        kind: str,
        payload: dict[str, object],
        children: list[str | None],
    ) -> str:
        child_ids = [child for child in children if child is not None]
        term = {
            "kind": kind,
            "payload": payload,
            "children": child_ids,
        }
        digest = self._digest(_term(term), "count_dag_node")
        node_id = f"count:{digest}"
        if node_id in self._nodes:
            return node_id
        depth = 1 + max((self._depths[child] for child in child_ids), default=0)
        self._check("count_depth", depth, self._budget.max_count_depth)
        node = {
            "node_id": node_id,
            "kind": kind,
            **payload,
            "children": child_ids,
            "digest": digest,
            "digest_input_bytes": len(_term_jsonish(term)),
        }
        self._nodes[node_id] = node
        self._depths[node_id] = depth
        self._edge_count += len(child_ids)
        self._check("count_node_count", len(self._nodes), self._budget.max_count_nodes)
        self._check("count_edge_count", self._edge_count, self._budget.max_count_edges)
        return node_id

    def _check(self, metric: str, actual: int, limit: int) -> None:
        if actual > limit:
            raise WriterEnvelopeWorkExceeded(
                WriterEnvelopeWorkViolation(
                    operation="count_dag_envelope",
                    metric=metric,
                    actual=actual,
                    limit=limit,
                )
            )

    def _digest(self, term, operation: str) -> str:
        try:
            return _digest_bounded(
                term,
                budget=self._budget,
                operation=operation,
            )
        except ValueError as exc:
            message = str(exc)
            metric = "digest_term_bytes"
            actual = 0
            limit = self._budget.max_digest_term_bytes
            for item in message.split(";"):
                item = item.strip()
                if item.startswith("metric="):
                    metric = item.split("=", 1)[1].strip("'")
                elif item.startswith("actual="):
                    actual = int(item.split("=", 1)[1])
                elif item.startswith("limit="):
                    limit = int(item.split("=", 1)[1])
            raise WriterEnvelopeWorkExceeded(
                WriterEnvelopeWorkViolation(
                    operation=operation,
                    metric=metric,
                    actual=actual,
                    limit=limit,
                )
            ) from exc


def writer_count_certificate_dag_envelope_for_product(
    product,
    *,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> dict[str, object]:
    return _CountDagBuilder(
        budget=budget or WriterEnvelopeWorkBudget(),
    ).build(product)


def count_dag_node_by_id(dag: dict[str, object]) -> dict[str, dict[str, object]]:
    return {node["node_id"]: node for node in dag["nodes"]}


def validate_writer_count_certificate_dag_envelope(
    dag: object,
    *,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> None:
    if not isinstance(dag, dict):
        _dag_violation("count_dag_not_mapping")
    if frozenset(dag) != frozenset((
        "schema_name",
        "schema_version",
        "roots",
        "nodes",
        "metrics",
        "digest",
    )):
        _dag_violation("count_dag_fields_mismatch")
    if dag["schema_name"] != SCHEMA_NAME:
        _dag_violation("unknown_count_dag_schema")
    if dag["schema_version"] != SCHEMA_VERSION:
        _dag_violation("unknown_count_dag_version")
    roots = dag["roots"]
    if not isinstance(roots, dict):
        _dag_violation("count_dag_roots_not_mapping")
    if frozenset(roots) != frozenset((
        "support_count_root",
        "completion_count_root",
        "choice_count_roots",
        "terminal_choice_count_root",
    )):
        _dag_violation("count_dag_roots_fields_mismatch")
    nodes = dag["nodes"]
    if not isinstance(nodes, list):
        _dag_violation("count_dag_nodes_not_list")
    node_by_id: dict[str, dict[str, object]] = {}
    for node in nodes:
        if not isinstance(node, dict):
            _dag_violation("count_dag_node_not_mapping")
        node_id = node.get("node_id")
        if not isinstance(node_id, str):
            _dag_violation("count_dag_node_id_missing")
        if node_id in node_by_id:
            _dag_violation("duplicate_count_dag_node_id")
        node_by_id[node_id] = node
        _validate_node_digest(node)

    root_ids = [
        roots["support_count_root"],
        roots["completion_count_root"],
        roots["terminal_choice_count_root"],
        *roots["choice_count_roots"],
    ]
    for root_id in root_ids:
        if root_id is not None and root_id not in node_by_id:
            _dag_violation("missing_count_dag_root_node")

    edge_count = 0
    depths: dict[str, int] = {}
    visiting: set[str] = set()
    for node_id in sorted(node_by_id):
        edge_count += len(_children_for_node(node_by_id[node_id]))
        _depth(node_id, node_by_id, visiting, depths)

    metrics = dag["metrics"]
    if not isinstance(metrics, dict):
        _dag_violation("count_dag_metrics_not_mapping")
    actual_metrics = {
        "node_count": len(node_by_id),
        "edge_count": edge_count,
        "max_depth": max(depths.values(), default=0),
        "digest_input_bytes": sum(
            int(node["digest_input_bytes"])
            for node in node_by_id.values()
        ),
    }
    if metrics != actual_metrics:
        _dag_violation("count_dag_metrics_mismatch")
    if dag["digest"] != _digest(_term({k: v for k, v in dag.items() if k != "digest"})):
        _dag_violation("count_dag_digest_mismatch")
    _check_budget(actual_metrics, budget or WriterEnvelopeWorkBudget())


def _snapshot_or_cursor_envelope(value):
    if hasattr(value, "decoder_boundary"):
        return _snapshot_identity_envelope(value)
    if hasattr(value, "weighted_states"):
        return _cursor_envelope(value)
    terms = _term(value)
    return {"digest": _digest(terms), "terms": terms}


def _term_jsonish(term) -> bytes:
    # This is only a metric; the bounded digest above is the authority.
    return _canonical_json(_term(term)).encode("utf-8")


def _validate_node_digest(node: dict[str, object]) -> None:
    required = frozenset((
        "node_id",
        "kind",
        "children",
        "digest",
        "digest_input_bytes",
    ))
    if not required.issubset(node):
        _dag_violation("count_dag_node_fields_missing")
    children = _children_for_node(node)
    payload = {
        key: value
        for key, value in node.items()
        if key not in required
    }
    term = {
        "kind": node["kind"],
        "payload": payload,
        "children": children,
    }
    digest = _digest(_term(term))
    if node["digest"] != digest:
        _dag_violation("count_dag_node_digest_mismatch")
    if node["node_id"] != f"count:{digest}":
        _dag_violation("count_dag_node_id_digest_mismatch")
    if node["digest_input_bytes"] != len(_term_jsonish(term)):
        _dag_violation("count_dag_node_digest_size_mismatch")


def _children_for_node(node: dict[str, object]) -> list[str]:
    children = node.get("children")
    if not isinstance(children, list):
        _dag_violation("count_dag_node_children_not_list")
    if not all(isinstance(child, str) for child in children):
        _dag_violation("count_dag_node_child_not_string")
    return list(children)


def _depth(
    node_id: str,
    node_by_id: dict[str, dict[str, object]],
    visiting: set[str],
    depths: dict[str, int],
) -> int:
    if node_id in depths:
        return depths[node_id]
    if node_id in visiting:
        _dag_violation("count_dag_cycle")
    if node_id not in node_by_id:
        _dag_violation("missing_count_dag_child_node")
    visiting.add(node_id)
    children = _children_for_node(node_by_id[node_id])
    depth = 1 + max(
        (_depth(child, node_by_id, visiting, depths) for child in children),
        default=0,
    )
    visiting.remove(node_id)
    depths[node_id] = depth
    return depth


def _check_budget(
    metrics: dict[str, int],
    budget: WriterEnvelopeWorkBudget,
) -> None:
    checks = (
        ("count_node_count", metrics["node_count"], budget.max_count_nodes),
        ("count_edge_count", metrics["edge_count"], budget.max_count_edges),
        ("count_depth", metrics["max_depth"], budget.max_count_depth),
        (
            "digest_term_bytes",
            metrics["digest_input_bytes"],
            budget.max_digest_term_bytes,
        ),
    )
    for metric, actual, limit in checks:
        if actual > limit:
            raise WriterEnvelopeWorkExceeded(
                WriterEnvelopeWorkViolation(
                    operation="count_dag_envelope",
                    metric=metric,
                    actual=actual,
                    limit=limit,
                )
            )


def _dag_violation(kind: str) -> None:
    raise ValueError(f"writer count DAG envelope violation: {kind}")


__all__ = (
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "WriterEnvelopeWorkBudget",
    "WriterEnvelopeWorkExceeded",
    "WriterEnvelopeWorkViolation",
    "count_dag_node_by_id",
    "validate_writer_count_certificate_dag_envelope",
    "writer_count_certificate_dag_envelope_for_product",
)
