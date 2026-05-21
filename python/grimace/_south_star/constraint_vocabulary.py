from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.annotation_policy import Edge


@dataclass(frozen=True, slots=True)
class SouthStarConstraintFamily:
    family_id: str
    description: str

    def __post_init__(self) -> None:
        _require_nonempty(self.family_id, "constraint family id")
        _require_nonempty(self.description, "constraint family description")


@dataclass(frozen=True, slots=True)
class SouthStarConstraintSyntaxSlot:
    family_id: str
    slot_id: str
    slot_kind: str
    syntax_position: str
    atom_idx: int | None = None
    edge: Edge | None = None

    def __post_init__(self) -> None:
        _require_nonempty(self.family_id, "constraint syntax-slot family id")
        _require_nonempty(self.slot_id, "constraint syntax-slot id")
        _require_nonempty(self.slot_kind, "constraint syntax-slot kind")
        _require_nonempty(self.syntax_position, "constraint syntax position")


@dataclass(frozen=True, slots=True)
class SouthStarConstraintObligation:
    family_id: str
    obligation_id: str
    subject_id: str
    required_fact_ids: tuple[str, ...]
    syntax_slot_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_nonempty(self.family_id, "constraint obligation family id")
        _require_nonempty(self.obligation_id, "constraint obligation id")
        _require_nonempty(self.subject_id, "constraint obligation subject")
        _require_all_nonempty(self.required_fact_ids, "required fact id")
        _require_all_nonempty(self.syntax_slot_ids, "syntax slot id")


@dataclass(frozen=True, slots=True)
class SouthStarConstraintEquation:
    family_id: str
    equation_id: str
    obligation_ids: tuple[str, ...]
    syntax_slot_ids: tuple[str, ...]

    def __post_init__(self) -> None:
        _require_nonempty(self.family_id, "constraint equation family id")
        _require_nonempty(self.equation_id, "constraint equation id")
        _require_all_nonempty(self.obligation_ids, "obligation id")
        _require_all_nonempty(self.syntax_slot_ids, "syntax slot id")


@dataclass(frozen=True, slots=True)
class SouthStarConstraintAssignment:
    family_id: str
    assignment_id: str
    syntax_slot_id: str
    value: str

    def __post_init__(self) -> None:
        _require_nonempty(self.family_id, "constraint assignment family id")
        _require_nonempty(self.assignment_id, "constraint assignment id")
        _require_nonempty(self.syntax_slot_id, "constraint assignment slot id")
        _require_nonempty(self.value, "constraint assignment value")


@dataclass(frozen=True, slots=True)
class SouthStarRendererInput:
    family_id: str
    syntax_slot_id: str
    token_family: str
    value: str

    def __post_init__(self) -> None:
        _require_nonempty(self.family_id, "renderer-input family id")
        _require_nonempty(self.syntax_slot_id, "renderer-input slot id")
        _require_nonempty(self.token_family, "renderer-input token family")
        _require_nonempty(self.value, "renderer-input value")


def _require_nonempty(value: str, field_name: str) -> None:
    if not value:
        raise ValueError(f"{field_name} must be nonempty")


def _require_all_nonempty(values: tuple[str, ...], field_name: str) -> None:
    for value in values:
        _require_nonempty(value, field_name)
