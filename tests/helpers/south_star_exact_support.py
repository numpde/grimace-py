from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


EXACT_FIRST_DOMAIN_FIXTURE = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "south_star_exact_first_domain"
    / "first_domain_v1.json"
)


@dataclass(frozen=True, slots=True)
class SouthStarExactSupportCase:
    case_id: str
    source_smiles: str
    expected_support: tuple[str, ...]


def load_south_star_exact_first_domain_cases(
    path: Path = EXACT_FIRST_DOMAIN_FIXTURE,
) -> tuple[SouthStarExactSupportCase, ...]:
    raw = json.loads(path.read_text())
    if raw["schema_version"] != 1:
        raise ValueError(f"unsupported South Star exact-support schema: {raw!r}")
    return tuple(
        SouthStarExactSupportCase(
            case_id=case["case_id"],
            source_smiles=case["source_smiles"],
            expected_support=tuple(case["expected_support"]),
        )
        for case in raw["cases"]
    )
