"""Closure-candidate graph-liveness lifecycle validation."""

from __future__ import annotations

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

    violations: list[str] = []
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
            _require_successor_open_endpoint(
                violations,
                resolution=resolution,
                successor_obligation_by_bond=successor_obligation_by_bond,
            )
            continue

        successor_resolution = successor_resolution_by_bond.get(
            resolution.bond
        )
        if successor_resolution is None:
            if successor_obligation_by_bond.get(resolution.bond) in (
                WriterEdgeObligationKind.OPEN_CLOSURE_ENDPOINT,
                WriterEdgeObligationKind.CLOSED_CLOSURE,
            ):
                continue
            violations.append("supported_candidate_disappeared")
            continue

        if (
            successor_resolution.resolution_kind
            not in _SUPPORTED_CLOSURE_CANDIDATE_RESOLUTION_KINDS
        ):
            violations.append("supported_candidate_became_unsupported")

    return tuple(violations)


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
    violations: list[str],
    *,
    resolution: WriterClosureCandidateResolution,
    successor_obligation_by_bond,
) -> None:
    if (
        successor_obligation_by_bond.get(resolution.bond)
        is not WriterEdgeObligationKind.OPEN_CLOSURE_ENDPOINT
    ):
        violations.append("opened_candidate_lacks_successor_open_endpoint")


__all__ = (
    "validate_writer_closure_candidate_lifecycle_transition",
    "writer_closure_candidate_lifecycle_transition_violations",
)
