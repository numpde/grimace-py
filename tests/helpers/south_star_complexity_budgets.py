from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tests.helpers.south_star_exact_support import (
    load_south_star_exact_first_domain_cases,
    load_south_star_expanded_support_cases,
)


SOUTH_STAR_COMPLEXITY_BUDGET_POLICY = (
    "south_star_generation_diagnostics_budget_v1"
)
SOUTH_STAR_GENERATION_DIAGNOSTIC_BUDGET_FIELDS = (
    "fragment_count",
    "fragment_output_counts",
    "fragment_order_count",
    "stereo_component_count",
    "traversal_skeleton_count",
    "marker_slot_count",
    "local_assignment_count",
    "solved_assignment_count",
    "estimated_product_size",
    "raw_output_count",
    "output_count",
    "deduplication_drop_count",
)
COMPLEXITY_BUDGET_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "south_star_complexity_budgets"
    / "generation_diagnostics_v1.json"
)


@dataclass(frozen=True, slots=True)
class SouthStarGenerationDiagnosticsBudget:
    fragment_count: int
    fragment_output_counts: tuple[int, ...]
    fragment_order_count: int
    stereo_component_count: int
    traversal_skeleton_count: int
    marker_slot_count: int
    local_assignment_count: int
    solved_assignment_count: int
    estimated_product_size: int
    raw_output_count: int
    output_count: int
    deduplication_drop_count: int


@dataclass(frozen=True, slots=True)
class SouthStarComplexityBudgetCase:
    case_id: str
    evidence_notes: str
    expected_generation_diagnostics: SouthStarGenerationDiagnosticsBudget


def load_south_star_complexity_budget_cases(
    path: Path = COMPLEXITY_BUDGET_FIXTURE,
) -> tuple[SouthStarComplexityBudgetCase, ...]:
    raw = json.loads(path.read_text())
    if raw["schema_version"] != 1:
        raise ValueError(f"unsupported South Star complexity budget schema: {raw!r}")
    if raw["policy"] != SOUTH_STAR_COMPLEXITY_BUDGET_POLICY:
        raise ValueError(f"unsupported South Star complexity budget policy: {raw!r}")

    known_case_ids = {
        case.case_id
        for case in (
            load_south_star_exact_first_domain_cases()
            + load_south_star_expanded_support_cases()
        )
    }
    cases = tuple(_complexity_budget_case(case) for case in raw["cases"])
    case_ids = tuple(case.case_id for case in cases)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("duplicate South Star complexity budget case ids")
    unknown_case_ids = tuple(
        case_id for case_id in case_ids if case_id not in known_case_ids
    )
    if unknown_case_ids:
        raise ValueError(
            f"unknown South Star complexity budget case ids: {unknown_case_ids!r}"
        )
    return cases


def _complexity_budget_case(raw_case: object) -> SouthStarComplexityBudgetCase:
    if not isinstance(raw_case, dict):
        raise ValueError(
            f"South Star complexity budget case must be a dict: {raw_case!r}"
        )
    budget = _generation_diagnostics_budget(
        raw_case["expected_generation_diagnostics"]
    )
    if not raw_case["evidence_notes"]:
        raise ValueError(
            f"South Star complexity budget {raw_case['case_id']!r} needs evidence notes"
        )
    return SouthStarComplexityBudgetCase(
        case_id=raw_case["case_id"],
        evidence_notes=raw_case["evidence_notes"],
        expected_generation_diagnostics=budget,
    )


def _generation_diagnostics_budget(
    raw_budget: object,
) -> SouthStarGenerationDiagnosticsBudget:
    if not isinstance(raw_budget, dict):
        raise ValueError(
            f"South Star generation diagnostics budget must be a dict: {raw_budget!r}"
        )
    expected_fields = frozenset(SOUTH_STAR_GENERATION_DIAGNOSTIC_BUDGET_FIELDS)
    if frozenset(raw_budget) != expected_fields:
        raise ValueError(
            "South Star generation diagnostics budget fields must exactly match "
            f"{SOUTH_STAR_GENERATION_DIAGNOSTIC_BUDGET_FIELDS!r}: {raw_budget!r}"
        )
    return SouthStarGenerationDiagnosticsBudget(
        fragment_count=raw_budget["fragment_count"],
        fragment_output_counts=tuple(raw_budget["fragment_output_counts"]),
        fragment_order_count=raw_budget["fragment_order_count"],
        stereo_component_count=raw_budget["stereo_component_count"],
        traversal_skeleton_count=raw_budget["traversal_skeleton_count"],
        marker_slot_count=raw_budget["marker_slot_count"],
        local_assignment_count=raw_budget["local_assignment_count"],
        solved_assignment_count=raw_budget["solved_assignment_count"],
        estimated_product_size=raw_budget["estimated_product_size"],
        raw_output_count=raw_budget["raw_output_count"],
        output_count=raw_budget["output_count"],
        deduplication_drop_count=raw_budget["deduplication_drop_count"],
    )
