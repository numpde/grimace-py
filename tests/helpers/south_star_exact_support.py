from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from tests.helpers.south_star_domain_manifest import (
    SOUTH_STAR_EXPANDED_SUPPORT_POLICY,
    SOUTH_STAR_FIRST_DOMAIN_POLICY,
    SOUTH_STAR_PRIVATE_DOMAIN,
)


EXACT_FIRST_DOMAIN_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "south_star_exact_first_domain"
    / "first_domain_v1.json"
)
EXPANDED_SUPPORT_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "south_star_expanded_support"
    / "expanded_domain_v1.json"
)

@dataclass(frozen=True, slots=True)
class SouthStarExactSupportCase:
    case_id: str
    source_smiles: str
    expected_support: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SouthStarExpandedSupportCase:
    case_id: str
    source_smiles: str
    feature_area: str
    support_authority: str
    evidence_notes: str
    expected_support: tuple[str, ...]


def load_south_star_exact_first_domain_cases(
    path: Path = EXACT_FIRST_DOMAIN_FIXTURE,
) -> tuple[SouthStarExactSupportCase, ...]:
    raw = json.loads(path.read_text())
    if raw["schema_version"] != 1:
        raise ValueError(f"unsupported South Star exact-support schema: {raw!r}")
    if raw["policy"] != SOUTH_STAR_FIRST_DOMAIN_POLICY:
        raise ValueError(f"unsupported South Star first-domain policy: {raw!r}")
    return tuple(
        SouthStarExactSupportCase(
            case_id=case["case_id"],
            source_smiles=case["source_smiles"],
            expected_support=tuple(case["expected_support"]),
        )
        for case in raw["cases"]
    )


def load_south_star_expanded_support_cases(
    path: Path = EXPANDED_SUPPORT_FIXTURE,
) -> tuple[SouthStarExpandedSupportCase, ...]:
    raw = json.loads(path.read_text())
    if raw["schema_version"] != 1:
        raise ValueError(f"unsupported South Star expanded-support schema: {raw!r}")
    if raw["policy"] != SOUTH_STAR_EXPANDED_SUPPORT_POLICY:
        raise ValueError(f"unsupported South Star expanded-support policy: {raw!r}")
    cases = tuple(_expanded_support_case(case) for case in raw["cases"])
    case_ids = tuple(case.case_id for case in cases)
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("duplicate South Star expanded-support case ids")
    return cases


def _expanded_support_case(raw_case: object) -> SouthStarExpandedSupportCase:
    if not isinstance(raw_case, dict):
        raise ValueError(
            f"South Star expanded-support case must be a dict: {raw_case!r}"
        )
    expected_support = tuple(raw_case["expected_support"])
    if not expected_support:
        raise ValueError(
            f"empty expected support in South Star case {raw_case['case_id']!r}"
        )
    if len(set(expected_support)) != len(expected_support):
        raise ValueError(
            f"duplicate expected support in South Star case {raw_case['case_id']!r}"
        )
    support_authority = raw_case["support_authority"]
    if support_authority not in SOUTH_STAR_PRIVATE_DOMAIN.support_authorities:
        raise ValueError(
            f"unsupported South Star expanded-support authority {support_authority!r}"
        )
    feature_area = raw_case["feature_area"]
    if feature_area not in SOUTH_STAR_PRIVATE_DOMAIN.expanded_feature_areas:
        raise ValueError(
            f"unsupported South Star expanded-support feature area {feature_area!r}"
        )
    return SouthStarExpandedSupportCase(
        case_id=raw_case["case_id"],
        source_smiles=raw_case["source_smiles"],
        feature_area=feature_area,
        support_authority=support_authority,
        evidence_notes=raw_case["evidence_notes"],
        expected_support=expected_support,
    )
