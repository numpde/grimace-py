from __future__ import annotations

import importlib.util
from dataclasses import dataclass

from tests.helpers.south_star_marker_equations import (
    SouthStarMarkerSlotParityEquation,
)


@dataclass(frozen=True, slots=True)
class SouthStarZ3MarkerAssignment:
    marker_by_slot: tuple[tuple[str, str], ...]


def z3_available() -> bool:
    return importlib.util.find_spec("z3") is not None


def z3_marker_assignments_for_equations(
    equations: tuple[SouthStarMarkerSlotParityEquation, ...],
) -> tuple[SouthStarZ3MarkerAssignment, ...]:
    z3 = _require_z3()
    slot_ids = tuple(dict.fromkeys(equation.slot_id for equation in equations))
    slot_is_backslash = {
        slot_id: z3.Bool(f"slot_{index}_is_backslash")
        for index, slot_id in enumerate(slot_ids)
    }

    solver = z3.Solver()
    for equation in equations:
        expected_marker = expected_marker_from_equation(equation)
        solver.add(
            slot_is_backslash[equation.slot_id]
            == z3.BoolVal(expected_marker == "\\")
        )

    assignments = []
    while solver.check() == z3.sat:
        model = solver.model()
        marker_by_slot = tuple(
            (
                slot_id,
                "\\"
                if bool(
                    model.eval(
                        slot_is_backslash[slot_id],
                        model_completion=True,
                    )
                )
                else "/",
            )
            for slot_id in slot_ids
        )
        assignments.append(SouthStarZ3MarkerAssignment(marker_by_slot=marker_by_slot))
        solver.add(
            z3.Or(
                *(
                    slot_is_backslash[slot_id]
                    != model.eval(
                        slot_is_backslash[slot_id],
                        model_completion=True,
                    )
                    for slot_id in slot_ids
                )
            )
            if slot_ids
            else z3.BoolVal(False)
        )
    return tuple(assignments)


def expected_marker_from_equation(
    equation: SouthStarMarkerSlotParityEquation,
) -> str:
    if equation.traversal_orientation_flip:
        return _flipped_marker(equation.graph_marker)
    return equation.graph_marker


def _require_z3():
    try:
        import z3
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "z3 is required for South Star Z3 oracle tests"
        ) from exc
    return z3


def _flipped_marker(marker: str) -> str:
    if marker == "/":
        return "\\"
    if marker == "\\":
        return "/"
    raise ValueError(f"unsupported South Star directional marker {marker!r}")
