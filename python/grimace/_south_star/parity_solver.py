from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.marker_equations import (
    SouthStarMarkerSlotParityEquation,
)
from grimace._south_star.marker_equations import expected_marker_from_equation


@dataclass(frozen=True, slots=True)
class SouthStarParitySolverAssignment:
    marker_by_slot: tuple[tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class SouthStarParitySolverDiagnostic:
    slot_count: int
    equation_count: int
    component_count: int
    affected_component_ids: tuple[str, ...]
    coupling_causes: tuple[str, ...]
    local_assignment_count: int
    estimated_product_size: int


@dataclass(frozen=True, slots=True)
class SouthStarParitySolverResult:
    assignments: tuple[SouthStarParitySolverAssignment, ...]
    diagnostic: SouthStarParitySolverDiagnostic


def solve_marker_slot_parity_equations(
    equations: tuple[SouthStarMarkerSlotParityEquation, ...],
) -> SouthStarParitySolverResult:
    marker_by_slot: dict[str, str] = {}
    for equation in equations:
        expected_marker = expected_marker_from_equation(equation)
        existing = marker_by_slot.setdefault(equation.slot_id, expected_marker)
        if existing != expected_marker:
            raise ValueError(
                f"conflicting parity equations for slot {equation.slot_id!r}"
            )

    assignments = (
        SouthStarParitySolverAssignment(
            marker_by_slot=tuple(sorted(marker_by_slot.items())),
        ),
    )
    return SouthStarParitySolverResult(
        assignments=assignments,
        diagnostic=_diagnostic(
            equations,
            assignment_count=len(assignments),
        ),
    )


def _diagnostic(
    equations: tuple[SouthStarMarkerSlotParityEquation, ...],
    *,
    assignment_count: int,
) -> SouthStarParitySolverDiagnostic:
    affected_component_ids = tuple(
        dict.fromkeys(
            component_id
            for equation in equations
            for component_id in equation.component_ids
        )
    )
    slot_ids = tuple(dict.fromkeys(equation.slot_id for equation in equations))
    coupling_causes = tuple(
        sorted(
            {
                "shared_carrier_edge"
                for equation in equations
                for term in equation.feature_terms
                if term.shared_carrier_incidence_count > 1
            }
        )
    )
    return SouthStarParitySolverDiagnostic(
        slot_count=len(slot_ids),
        equation_count=len(equations),
        component_count=len(affected_component_ids),
        affected_component_ids=affected_component_ids,
        coupling_causes=coupling_causes,
        local_assignment_count=assignment_count,
        estimated_product_size=assignment_count,
    )
