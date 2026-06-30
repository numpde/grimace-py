"""Private writer execution evidence records."""

from __future__ import annotations

from dataclasses import dataclass

from .ids import BondId
from .residual_constraints import _MAX_RESIDUAL_FACTOR_CANDIDATE_ROWS
from .residual_constraints import _MAX_RESIDUAL_FACTOR_SCOPE
from .residual_constraints import ResidualFactorKey
from .residual_constraints import ResidualPropagationKind
from .residual_constraints import ResidualPropagationResult
from .residual_constraints import VarId


@dataclass(frozen=True, slots=True)
class WriterResidualPropagationWorkEvidence:
    operation: str
    result_kind: ResidualPropagationKind
    component_variables: tuple[VarId, ...]
    component_factor_keys: tuple[ResidualFactorKey, ...]
    checked_candidate_rows: int
    largest_factor_scope: int
    largest_candidate_row_count: int

    @property
    def component_variable_count(self) -> int:
        return len(self.component_variables)

    @property
    def component_factor_count(self) -> int:
        return len(self.component_factor_keys)


@dataclass(frozen=True, slots=True)
class WriterResidualWorkEnvelope:
    max_component_variable_count: int | None = None
    max_component_factor_count: int | None = None
    max_checked_candidate_rows: int | None = None
    max_largest_factor_scope: int | None = _MAX_RESIDUAL_FACTOR_SCOPE
    max_largest_candidate_row_count: int | None = (
        _MAX_RESIDUAL_FACTOR_CANDIDATE_ROWS
    )


@dataclass(frozen=True, slots=True)
class WriterResidualWorkEnvelopeViolation:
    evidence: WriterResidualPropagationWorkEvidence
    metric: str
    actual: int
    limit: int


@dataclass(frozen=True, slots=True)
class WriterFiniteRelationWorkEvidence:
    operation: str
    relation_kind: str
    row_count: int
    total_candidate_count: int
    largest_candidate_count: int
    bond: BondId | None = None
    include_direction_marks: bool = False


@dataclass(frozen=True, slots=True)
class WriterGraphObligationWorkEvidence:
    operation: str
    component_index: int
    component_atom_count: int
    component_bond_count: int
    edge_obligation_count: int
    residual_attachment_count: int
    residual_attachment_action_count: int
    boundary_incidence_count: int
    closure_candidate_count: int
    live_branch_return_closure_candidate_count: int
    unsupported_closure_candidate_count: int
    open_closure_count: int
    closed_closure_count: int
    max_attachment_atom_count: int
    max_attachment_boundary_count: int
    max_attachment_cyclic_rank: int


@dataclass(frozen=True, slots=True)
class WriterGraphObligationWorkEnvelope:
    max_component_atom_count: int | None = None
    max_component_bond_count: int | None = None
    max_edge_obligation_count: int | None = None
    max_residual_attachment_count: int | None = None
    max_residual_attachment_action_count: int | None = None
    max_boundary_incidence_count: int | None = None
    max_closure_candidate_count: int | None = None
    max_live_branch_return_closure_candidate_count: int | None = None
    max_unsupported_closure_candidate_count: int | None = None
    max_open_closure_count: int | None = None
    max_closed_closure_count: int | None = None
    max_attachment_atom_count: int | None = None
    max_attachment_boundary_count: int | None = None
    max_attachment_cyclic_rank: int | None = None


@dataclass(frozen=True, slots=True)
class WriterGraphObligationWorkEnvelopeViolation:
    evidence: WriterGraphObligationWorkEvidence
    metric: str
    actual: int
    limit: int


@dataclass(frozen=True, slots=True)
class WriterFiniteRelationWorkEnvelope:
    max_row_count: int | None = None
    max_total_candidate_count: int | None = None
    max_largest_candidate_count: int | None = None


@dataclass(frozen=True, slots=True)
class WriterFiniteRelationWorkEnvelopeViolation:
    evidence: WriterFiniteRelationWorkEvidence
    metric: str
    actual: int
    limit: int


_PUBLIC_CLOSURE_ENDPOINT_RELATION_MAX_ROW_COUNT = 3
_PUBLIC_CLOSURE_ENDPOINT_RELATION_MAX_TOTAL_CANDIDATE_COUNT = 7
_PUBLIC_CLOSURE_ENDPOINT_RELATION_MAX_LARGEST_CANDIDATE_COUNT = 3

_PUBLIC_RESIDUAL_PROPAGATION_MAX_COMPONENT_VARIABLE_COUNT = 8
_PUBLIC_RESIDUAL_PROPAGATION_MAX_COMPONENT_FACTOR_COUNT = 9
_PUBLIC_RESIDUAL_PROPAGATION_MAX_CHECKED_CANDIDATE_ROWS = 197
_PUBLIC_RESIDUAL_PROPAGATION_MAX_LARGEST_FACTOR_SCOPE = (
    _MAX_RESIDUAL_FACTOR_SCOPE
)
_PUBLIC_RESIDUAL_PROPAGATION_MAX_LARGEST_CANDIDATE_ROW_COUNT = (
    _MAX_RESIDUAL_FACTOR_CANDIDATE_ROWS
)

_PUBLIC_GRAPH_OBLIGATION_MAX_COMPONENT_ATOM_COUNT = 10
_PUBLIC_GRAPH_OBLIGATION_MAX_COMPONENT_BOND_COUNT = 11
_PUBLIC_GRAPH_OBLIGATION_MAX_EDGE_OBLIGATION_COUNT = 11
_PUBLIC_GRAPH_OBLIGATION_MAX_RESIDUAL_ATTACHMENT_COUNT = 4
_PUBLIC_GRAPH_OBLIGATION_MAX_RESIDUAL_ATTACHMENT_ACTION_COUNT = 4
_PUBLIC_GRAPH_OBLIGATION_MAX_BOUNDARY_INCIDENCE_COUNT = 4
_PUBLIC_GRAPH_OBLIGATION_MAX_CLOSURE_CANDIDATE_COUNT = 1
_PUBLIC_GRAPH_OBLIGATION_MAX_LIVE_BRANCH_RETURN_CLOSURE_CANDIDATE_COUNT = 1
_PUBLIC_GRAPH_OBLIGATION_MAX_UNSUPPORTED_CLOSURE_CANDIDATE_COUNT = 0
_PUBLIC_GRAPH_OBLIGATION_MAX_OPEN_CLOSURE_COUNT = 2
_PUBLIC_GRAPH_OBLIGATION_MAX_CLOSED_CLOSURE_COUNT = 2
_PUBLIC_GRAPH_OBLIGATION_MAX_ATTACHMENT_ATOM_COUNT = 10
_PUBLIC_GRAPH_OBLIGATION_MAX_ATTACHMENT_BOUNDARY_COUNT = 3
_PUBLIC_GRAPH_OBLIGATION_MAX_ATTACHMENT_CYCLIC_RANK = 2

_PUBLIC_WRITER_RESIDUAL_WORK_ENVELOPE = WriterResidualWorkEnvelope(
    max_component_variable_count=(
        _PUBLIC_RESIDUAL_PROPAGATION_MAX_COMPONENT_VARIABLE_COUNT
    ),
    max_component_factor_count=(
        _PUBLIC_RESIDUAL_PROPAGATION_MAX_COMPONENT_FACTOR_COUNT
    ),
    max_checked_candidate_rows=(
        _PUBLIC_RESIDUAL_PROPAGATION_MAX_CHECKED_CANDIDATE_ROWS
    ),
    max_largest_factor_scope=(
        _PUBLIC_RESIDUAL_PROPAGATION_MAX_LARGEST_FACTOR_SCOPE
    ),
    max_largest_candidate_row_count=(
        _PUBLIC_RESIDUAL_PROPAGATION_MAX_LARGEST_CANDIDATE_ROW_COUNT
    ),
)
_PUBLIC_WRITER_FINITE_RELATION_WORK_ENVELOPE = (
    WriterFiniteRelationWorkEnvelope(
        max_row_count=_PUBLIC_CLOSURE_ENDPOINT_RELATION_MAX_ROW_COUNT,
        max_total_candidate_count=(
            _PUBLIC_CLOSURE_ENDPOINT_RELATION_MAX_TOTAL_CANDIDATE_COUNT
        ),
        max_largest_candidate_count=(
            _PUBLIC_CLOSURE_ENDPOINT_RELATION_MAX_LARGEST_CANDIDATE_COUNT
        ),
    )
)
_PUBLIC_WRITER_GRAPH_OBLIGATION_WORK_ENVELOPE = (
    WriterGraphObligationWorkEnvelope(
        max_component_atom_count=(
            _PUBLIC_GRAPH_OBLIGATION_MAX_COMPONENT_ATOM_COUNT
        ),
        max_component_bond_count=(
            _PUBLIC_GRAPH_OBLIGATION_MAX_COMPONENT_BOND_COUNT
        ),
        max_edge_obligation_count=(
            _PUBLIC_GRAPH_OBLIGATION_MAX_EDGE_OBLIGATION_COUNT
        ),
        max_residual_attachment_count=(
            _PUBLIC_GRAPH_OBLIGATION_MAX_RESIDUAL_ATTACHMENT_COUNT
        ),
        max_residual_attachment_action_count=(
            _PUBLIC_GRAPH_OBLIGATION_MAX_RESIDUAL_ATTACHMENT_ACTION_COUNT
        ),
        max_boundary_incidence_count=(
            _PUBLIC_GRAPH_OBLIGATION_MAX_BOUNDARY_INCIDENCE_COUNT
        ),
        max_closure_candidate_count=(
            _PUBLIC_GRAPH_OBLIGATION_MAX_CLOSURE_CANDIDATE_COUNT
        ),
        max_live_branch_return_closure_candidate_count=(
            _PUBLIC_GRAPH_OBLIGATION_MAX_LIVE_BRANCH_RETURN_CLOSURE_CANDIDATE_COUNT
        ),
        max_unsupported_closure_candidate_count=(
            _PUBLIC_GRAPH_OBLIGATION_MAX_UNSUPPORTED_CLOSURE_CANDIDATE_COUNT
        ),
        max_open_closure_count=(
            _PUBLIC_GRAPH_OBLIGATION_MAX_OPEN_CLOSURE_COUNT
        ),
        max_closed_closure_count=(
            _PUBLIC_GRAPH_OBLIGATION_MAX_CLOSED_CLOSURE_COUNT
        ),
        max_attachment_atom_count=(
            _PUBLIC_GRAPH_OBLIGATION_MAX_ATTACHMENT_ATOM_COUNT
        ),
        max_attachment_boundary_count=(
            _PUBLIC_GRAPH_OBLIGATION_MAX_ATTACHMENT_BOUNDARY_COUNT
        ),
        max_attachment_cyclic_rank=(
            _PUBLIC_GRAPH_OBLIGATION_MAX_ATTACHMENT_CYCLIC_RANK
        ),
    )
)


def writer_residual_propagation_work_evidence(
    *,
    operation: str,
    result: ResidualPropagationResult,
) -> WriterResidualPropagationWorkEvidence:
    stats = result.stats
    return WriterResidualPropagationWorkEvidence(
        operation=operation,
        result_kind=result.kind,
        component_variables=stats.component_variables,
        component_factor_keys=stats.component_factor_keys,
        checked_candidate_rows=stats.checked_candidate_rows,
        largest_factor_scope=stats.largest_factor_scope,
        largest_candidate_row_count=stats.largest_candidate_row_count,
    )


def writer_residual_work_envelope_violation(
    evidence: WriterResidualPropagationWorkEvidence,
    *,
    envelope: WriterResidualWorkEnvelope | None = None,
) -> WriterResidualWorkEnvelopeViolation | None:
    envelope = (
        _PUBLIC_WRITER_RESIDUAL_WORK_ENVELOPE
        if envelope is None
        else envelope
    )
    checks = (
        (
            "component_variable_count",
            evidence.component_variable_count,
            envelope.max_component_variable_count,
        ),
        (
            "component_factor_count",
            evidence.component_factor_count,
            envelope.max_component_factor_count,
        ),
        (
            "checked_candidate_rows",
            evidence.checked_candidate_rows,
            envelope.max_checked_candidate_rows,
        ),
        (
            "largest_factor_scope",
            evidence.largest_factor_scope,
            envelope.max_largest_factor_scope,
        ),
        (
            "largest_candidate_row_count",
            evidence.largest_candidate_row_count,
            envelope.max_largest_candidate_row_count,
        ),
    )

    for metric, actual, limit in checks:
        if limit is not None and actual > limit:
            return WriterResidualWorkEnvelopeViolation(
                evidence=evidence,
                metric=metric,
                actual=actual,
                limit=limit,
            )

    return None


def writer_finite_relation_work_envelope_violation(
    evidence: WriterFiniteRelationWorkEvidence,
    *,
    envelope: WriterFiniteRelationWorkEnvelope | None = None,
) -> WriterFiniteRelationWorkEnvelopeViolation | None:
    envelope = (
        _PUBLIC_WRITER_FINITE_RELATION_WORK_ENVELOPE
        if envelope is None
        else envelope
    )
    checks = (
        ("row_count", evidence.row_count, envelope.max_row_count),
        (
            "total_candidate_count",
            evidence.total_candidate_count,
            envelope.max_total_candidate_count,
        ),
        (
            "largest_candidate_count",
            evidence.largest_candidate_count,
            envelope.max_largest_candidate_count,
        ),
    )

    for metric, actual, limit in checks:
        if limit is not None and actual > limit:
            return WriterFiniteRelationWorkEnvelopeViolation(
                evidence=evidence,
                metric=metric,
                actual=actual,
                limit=limit,
            )

    return None


def writer_graph_obligation_work_envelope_violation(
    evidence: WriterGraphObligationWorkEvidence,
    *,
    envelope: WriterGraphObligationWorkEnvelope | None = None,
) -> WriterGraphObligationWorkEnvelopeViolation | None:
    envelope = (
        _PUBLIC_WRITER_GRAPH_OBLIGATION_WORK_ENVELOPE
        if envelope is None
        else envelope
    )
    checks = (
        (
            "component_atom_count",
            evidence.component_atom_count,
            envelope.max_component_atom_count,
        ),
        (
            "component_bond_count",
            evidence.component_bond_count,
            envelope.max_component_bond_count,
        ),
        (
            "edge_obligation_count",
            evidence.edge_obligation_count,
            envelope.max_edge_obligation_count,
        ),
        (
            "residual_attachment_count",
            evidence.residual_attachment_count,
            envelope.max_residual_attachment_count,
        ),
        (
            "residual_attachment_action_count",
            evidence.residual_attachment_action_count,
            envelope.max_residual_attachment_action_count,
        ),
        (
            "boundary_incidence_count",
            evidence.boundary_incidence_count,
            envelope.max_boundary_incidence_count,
        ),
        (
            "closure_candidate_count",
            evidence.closure_candidate_count,
            envelope.max_closure_candidate_count,
        ),
        (
            "live_branch_return_closure_candidate_count",
            evidence.live_branch_return_closure_candidate_count,
            envelope.max_live_branch_return_closure_candidate_count,
        ),
        (
            "unsupported_closure_candidate_count",
            evidence.unsupported_closure_candidate_count,
            envelope.max_unsupported_closure_candidate_count,
        ),
        (
            "open_closure_count",
            evidence.open_closure_count,
            envelope.max_open_closure_count,
        ),
        (
            "closed_closure_count",
            evidence.closed_closure_count,
            envelope.max_closed_closure_count,
        ),
        (
            "max_attachment_atom_count",
            evidence.max_attachment_atom_count,
            envelope.max_attachment_atom_count,
        ),
        (
            "max_attachment_boundary_count",
            evidence.max_attachment_boundary_count,
            envelope.max_attachment_boundary_count,
        ),
        (
            "max_attachment_cyclic_rank",
            evidence.max_attachment_cyclic_rank,
            envelope.max_attachment_cyclic_rank,
        ),
    )

    for metric, actual, limit in checks:
        if limit is not None and actual > limit:
            return WriterGraphObligationWorkEnvelopeViolation(
                evidence=evidence,
                metric=metric,
                actual=actual,
                limit=limit,
            )

    return None


def writer_closure_endpoint_relation_work_evidence(
    *,
    operation: str,
    bond: BondId,
    relation,
    include_direction_marks: bool,
) -> WriterFiniteRelationWorkEvidence:
    candidate_counts = tuple(len(seconds) for _first, seconds in relation.rows)
    return WriterFiniteRelationWorkEvidence(
        operation=operation,
        relation_kind="closure_endpoint",
        bond=bond,
        row_count=len(relation.rows),
        total_candidate_count=sum(candidate_counts),
        largest_candidate_count=max(candidate_counts, default=0),
        include_direction_marks=include_direction_marks,
    )


__all__ = [
    "WriterFiniteRelationWorkEvidence",
    "WriterFiniteRelationWorkEnvelope",
    "WriterFiniteRelationWorkEnvelopeViolation",
    "WriterGraphObligationWorkEvidence",
    "WriterGraphObligationWorkEnvelope",
    "WriterGraphObligationWorkEnvelopeViolation",
    "WriterResidualPropagationWorkEvidence",
    "WriterResidualWorkEnvelope",
    "WriterResidualWorkEnvelopeViolation",
    "writer_closure_endpoint_relation_work_evidence",
    "writer_finite_relation_work_envelope_violation",
    "writer_graph_obligation_work_envelope_violation",
    "writer_residual_propagation_work_evidence",
    "writer_residual_work_envelope_violation",
]
