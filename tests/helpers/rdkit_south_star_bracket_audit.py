from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.helpers.pinned_rdkit_fixtures import (
    PINNED_SOUTH_STAR_BRACKET_AUDIT,
    load_pinned_rdkit_fixture_cases,
    normalized_unique_sorted_strings,
    optional_string,
    pinned_rdkit_fixture_root,
    required_positive_int,
    required_string,
)


@dataclass(frozen=True, slots=True)
class PinnedSouthStarBracketAuditCase:
    case_id: str
    source: str
    name: str
    smiles: str
    extraction_profile: str
    expected: str
    support_surface: str
    expected_support: tuple[str, ...]
    expected_support_count: int | None
    expected_completion_count: int | None
    blocker_phase: str | None
    blocker_kind: str | None
    blocker_error_kind: str | None
    blocker_message_contains: str | None
    note: str | None


_FIXTURE_ROOT = pinned_rdkit_fixture_root(PINNED_SOUTH_STAR_BRACKET_AUDIT)


def load_pinned_south_star_bracket_audit_cases(
    rdkit_version: str,
    *,
    fixture_root: Path = _FIXTURE_ROOT,
) -> tuple[PinnedSouthStarBracketAuditCase, ...]:
    cases = []
    for fixture_case in load_pinned_rdkit_fixture_cases(
        fixture_root=fixture_root,
        rdkit_version=rdkit_version,
        fixture_label="South Star bracket audit",
    ):
        raw_case = fixture_case.raw
        expected = required_string(
            raw_case,
            field_name="expected",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        raw_expected_support = raw_case.get("expected_support", [])
        if not isinstance(raw_expected_support, list):
            raise ValueError(
                f"fixture {fixture_case.fixture_path} case {fixture_case.case_id!r} "
                f"must define expected_support as a list"
            )
        if expected == "blocked":
            for field_name in ("expected_support_count", "expected_completion_count"):
                if field_name in raw_case:
                    raise ValueError(
                        f"fixture {fixture_case.fixture_path} case "
                        f"{fixture_case.case_id!r} must not define {field_name} "
                        "for a blocked bracket audit case"
                    )
        if expected == "accepted":
            for field_name in (
                "blocker_phase",
                "blocker_kind",
                "blocker_error_kind",
                "blocker_message_contains",
            ):
                if field_name in raw_case:
                    raise ValueError(
                        f"fixture {fixture_case.fixture_path} case "
                        f"{fixture_case.case_id!r} must not define {field_name} "
                        "for an accepted bracket audit case"
                    )
        expected_support = normalized_unique_sorted_strings(
            raw_expected_support,
            field_name="expected_support",
            fixture_path=fixture_case.fixture_path,
            case_id=fixture_case.case_id,
        )
        expected_support_count = (
            required_positive_int(
                raw_case,
                field_name="expected_support_count",
                fixture_path=fixture_case.fixture_path,
                case_id=fixture_case.case_id,
            )
            if expected == "accepted"
            else None
        )
        expected_completion_count = (
            required_positive_int(
                raw_case,
                field_name="expected_completion_count",
                fixture_path=fixture_case.fixture_path,
                case_id=fixture_case.case_id,
            )
            if expected == "accepted"
            else None
        )

        if expected == "blocked":
            blocker_phase = required_string(
                raw_case,
                field_name="blocker_phase",
                fixture_path=fixture_case.fixture_path,
                case_id=fixture_case.case_id,
            )
            blocker_kind = required_string(
                raw_case,
                field_name="blocker_kind",
                fixture_path=fixture_case.fixture_path,
                case_id=fixture_case.case_id,
            )
            blocker_error_kind = required_string(
                raw_case,
                field_name="blocker_error_kind",
                fixture_path=fixture_case.fixture_path,
                case_id=fixture_case.case_id,
            )
            blocker_message_contains = required_string(
                raw_case,
                field_name="blocker_message_contains",
                fixture_path=fixture_case.fixture_path,
                case_id=fixture_case.case_id,
            )
        else:
            blocker_phase = None
            blocker_kind = None
            blocker_error_kind = None
            blocker_message_contains = None

        cases.append(
            PinnedSouthStarBracketAuditCase(
                case_id=fixture_case.case_id,
                source=fixture_case.source,
                name=required_string(
                    raw_case,
                    field_name="name",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                smiles=required_string(
                    raw_case,
                    field_name="smiles",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                extraction_profile=required_string(
                    raw_case,
                    field_name="extraction_profile",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                expected=expected,
                support_surface=required_string(
                    raw_case,
                    field_name="support_surface",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
                expected_support=expected_support,
                expected_support_count=expected_support_count,
                expected_completion_count=expected_completion_count,
                blocker_phase=blocker_phase,
                blocker_kind=blocker_kind,
                blocker_error_kind=blocker_error_kind,
                blocker_message_contains=blocker_message_contains,
                note=optional_string(
                    raw_case,
                    field_name="note",
                    fixture_path=fixture_case.fixture_path,
                    case_id=fixture_case.case_id,
                ),
            )
        )

    seen_names: dict[str, str] = {}
    for case in cases:
        if case.expected not in {"accepted", "blocked"}:
            raise ValueError(
                f"South Star bracket audit case {case.case_id!r} has unsupported "
                f"expected value {case.expected!r}"
            )
        if case.name in seen_names:
            raise ValueError(
                f"South Star bracket audit case {case.case_id!r} duplicates "
                f"name {case.name!r} from {seen_names[case.name]!r}"
            )
        seen_names[case.name] = case.case_id
        if case.expected == "accepted" and not case.expected_support:
            raise ValueError(
                f"South Star bracket audit case {case.case_id!r} must define "
                "nonempty expected_support"
            )
        if case.expected == "blocked" and case.expected_support:
            raise ValueError(
                f"South Star bracket audit case {case.case_id!r} must not define "
                "expected_support"
            )

    return tuple(cases)
