"""Closure-candidate graph-liveness lifecycle validation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .writer_graph_obligations import WriterClosureCandidateResolution
from .writer_graph_obligations import WriterClosureCandidateResolutionKind
from .writer_graph_obligations import WriterEdgeObligationKind
from .writer_graph_obligations import build_writer_graph_obligation_context
from .writer_graph_obligations import writer_closure_candidate_resolutions


_SUPPORTED_CLOSURE_CANDIDATE_RESOLUTION_KINDS = frozenset(
    {
        WriterClosureCandidateResolutionKind.LIVE_BRANCH_RETURN,
        WriterClosureCandidateResolutionKind.DEFERRED_BRANCH_RETURN,
        WriterClosureCandidateResolutionKind.DEFERRED_CONTROL_LIVE,
    }
)


class WriterClosureCandidateLifecycleOutcomeKind(Enum):
    RETAINED_SUPPORTED = "retained_supported"
    OPENED = "opened"
    CLOSED = "closed"


@dataclass(frozen=True, slots=True)
class WriterClosureCandidateLifecycleEvidence:
    bond: object
    source_resolution: object
    outcome_kind: WriterClosureCandidateLifecycleOutcomeKind
    successor_resolution: object | None = None
    successor_obligation_kind: object | None = None


def validate_writer_closure_candidate_lifecycle_transition(
    *,
    prepared,
    source_state,
    successor_state,
    transition_kind,
    graph_action_surface,
) -> None:
    violations = writer_closure_candidate_lifecycle_transition_violations(
        prepared=prepared,
        source_state=source_state,
        successor_state=successor_state,
        transition_kind=transition_kind,
        graph_action_surface=graph_action_surface,
    )
    if violations:
        raise SouthStarError(
            SouthStarErrorKind.INTERNAL_INVARIANT,
            (
                "writer closure-candidate lifecycle transition violation: "
                f"{violations[0]}"
            ),
        )


def writer_closure_candidate_lifecycle_transition_violations(
    *,
    prepared,
    source_state,
    successor_state,
    transition_kind,
    graph_action_surface,
) -> tuple[str, ...]:
    del transition_kind
    try:
        writer_closure_candidate_lifecycle_evidence_for_transition(
            prepared=prepared,
            source_state=source_state,
            successor_state=successor_state,
            graph_action_surface=graph_action_surface,
        )
    except _ClosureCandidateLifecycleViolation as caught:
        return (caught.kind,)

    return ()


def writer_closure_candidate_lifecycle_evidence_for_transition(
    *,
    prepared,
    source_state,
    successor_state,
    graph_action_surface,
) -> tuple[WriterClosureCandidateLifecycleEvidence, ...]:
    source_context = build_writer_graph_obligation_context(
        prepared,
        source_state,
    )
    successor_context = build_writer_graph_obligation_context(
        prepared,
        successor_state,
    )
    source_resolutions = writer_closure_candidate_resolutions(
        source_state,
        source_context.edge_partition,
    )
    successor_resolutions = writer_closure_candidate_resolutions(
        successor_state,
        successor_context.edge_partition,
    )
    successor_resolution_by_bond = {
        resolution.bond: resolution
        for resolution in successor_resolutions
    }
    successor_obligation_by_bond = {
        obligation.bond: obligation.kind
        for obligation in successor_context.edge_partition.obligations
    }

    evidence: list[WriterClosureCandidateLifecycleEvidence] = []
    for resolution in source_resolutions:
        if (
            resolution.resolution_kind
            not in _SUPPORTED_CLOSURE_CANDIDATE_RESOLUTION_KINDS
        ):
            continue

        if _support_opens_resolution(
            resolution=resolution,
            graph_action_surface=graph_action_surface,
        ):
            successor_obligation = _require_successor_open_endpoint(
                resolution=resolution,
                successor_obligation_by_bond=successor_obligation_by_bond,
            )
            evidence.append(
                WriterClosureCandidateLifecycleEvidence(
                    bond=resolution.bond,
                    source_resolution=resolution,
                    outcome_kind=(
                        WriterClosureCandidateLifecycleOutcomeKind.OPENED
                    ),
                    successor_obligation_kind=successor_obligation,
                )
            )
            continue

        successor_resolution = successor_resolution_by_bond.get(
            resolution.bond
        )
        successor_obligation = successor_obligation_by_bond.get(
            resolution.bond
        )
        if successor_resolution is None:
            if successor_obligation is WriterEdgeObligationKind.OPEN_CLOSURE_ENDPOINT:
                evidence.append(
                    WriterClosureCandidateLifecycleEvidence(
                        bond=resolution.bond,
                        source_resolution=resolution,
                        outcome_kind=(
                            WriterClosureCandidateLifecycleOutcomeKind.OPENED
                        ),
                        successor_obligation_kind=successor_obligation,
                    )
                )
                continue
            if successor_obligation is WriterEdgeObligationKind.CLOSED_CLOSURE:
                evidence.append(
                    WriterClosureCandidateLifecycleEvidence(
                        bond=resolution.bond,
                        source_resolution=resolution,
                        outcome_kind=(
                            WriterClosureCandidateLifecycleOutcomeKind.CLOSED
                        ),
                        successor_obligation_kind=successor_obligation,
                    )
                )
                continue
            _lifecycle_violation("supported_candidate_disappeared")
            continue

        if (
            successor_resolution.resolution_kind
            not in _SUPPORTED_CLOSURE_CANDIDATE_RESOLUTION_KINDS
        ):
            _lifecycle_violation("supported_candidate_became_unsupported")
        evidence.append(
            WriterClosureCandidateLifecycleEvidence(
                bond=resolution.bond,
                source_resolution=resolution,
                outcome_kind=(
                    WriterClosureCandidateLifecycleOutcomeKind.RETAINED_SUPPORTED
                ),
                successor_resolution=successor_resolution,
            )
        )

    return tuple(evidence)


def _support_opens_resolution(
    *,
    resolution: WriterClosureCandidateResolution,
    graph_action_surface,
) -> bool:
    if graph_action_surface is None:
        return False

    return (
        graph_action_surface.bond == resolution.bond
        and graph_action_surface.boundary_atom == resolution.first_atom
        and graph_action_surface.partner_atom == resolution.second_atom
    )


def _require_successor_open_endpoint(
    *,
    resolution: WriterClosureCandidateResolution,
    successor_obligation_by_bond,
) -> WriterEdgeObligationKind:
    successor_obligation = successor_obligation_by_bond.get(resolution.bond)
    if successor_obligation is not WriterEdgeObligationKind.OPEN_CLOSURE_ENDPOINT:
        _lifecycle_violation("opened_candidate_lacks_successor_open_endpoint")
    return successor_obligation


def _lifecycle_violation(kind: str) -> None:
    raise _ClosureCandidateLifecycleViolation(kind)


class _ClosureCandidateLifecycleViolation(Exception):
    def __init__(self, kind: str) -> None:
        super().__init__(kind)
        self.kind = kind


__all__ = (
    "WriterClosureCandidateLifecycleEvidence",
    "WriterClosureCandidateLifecycleOutcomeKind",
    "validate_writer_closure_candidate_lifecycle_transition",
    "writer_closure_candidate_lifecycle_evidence_for_transition",
    "writer_closure_candidate_lifecycle_transition_violations",
)
