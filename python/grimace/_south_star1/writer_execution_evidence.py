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


_PUBLIC_WRITER_RESIDUAL_WORK_ENVELOPE = WriterResidualWorkEnvelope()


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
    "WriterResidualPropagationWorkEvidence",
    "WriterResidualWorkEnvelope",
    "WriterResidualWorkEnvelopeViolation",
    "writer_closure_endpoint_relation_work_evidence",
    "writer_residual_propagation_work_evidence",
    "writer_residual_work_envelope_violation",
]
