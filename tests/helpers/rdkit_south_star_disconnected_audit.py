"""Version-keyed loader for the fixed-order disconnected South Star audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from tests.helpers.fixture_paths import checked_in_fixture_path
from tests.helpers.pinned_rdkit_fixtures import normalized_unique_sorted_strings
from tests.helpers.pinned_rdkit_fixtures import PINNED_SOUTH_STAR_DISCONNECTED_AUDIT


@dataclass(frozen=True, slots=True)
class SouthStarDisconnectedAuditCase:
    case_id: str
    name: str
    source_smiles: str
    extraction_profile: str
    rooted_at_atom: int
    component_order: tuple[tuple[int, ...], ...]
    component_root_domains: tuple[tuple[int, ...], ...]
    expected_support: tuple[str, ...]
    support_count: int
    completion_count: int
    sorted_support_sha256: str


def load_pinned_south_star_disconnected_audit_cases(
    rdkit_version: str,
) -> tuple[SouthStarDisconnectedAuditCase, ...]:
    path = checked_in_fixture_path(PINNED_SOUTH_STAR_DISCONNECTED_AUDIT) / (
        f"{rdkit_version}.json"
    )
    raw = json.loads(path.read_text())
    if raw.get("rdkit_version") != rdkit_version:
        raise ValueError(f"disconnected audit fixture version mismatch: {path}")
    cases = []
    seen = set()
    for item in raw.get("cases", ()):
        case_id = item.get("id")
        if type(case_id) is not str or case_id in seen:
            raise ValueError(f"invalid disconnected audit case id: {case_id!r}")
        seen.add(case_id)
        support = normalized_unique_sorted_strings(
            item.get("expected_support"),
            field_name="expected_support",
            fixture_path=path,
            case_id=case_id,
        )
        digest = hashlib.sha256(
            json.dumps(
                support, sort_keys=True, separators=(",", ":"), ensure_ascii=True
            ).encode()
        ).hexdigest()
        if digest != item.get("sorted_support_sha256"):
            raise ValueError(f"disconnected audit digest mismatch: {case_id}")
        cases.append(
            SouthStarDisconnectedAuditCase(
                case_id=case_id,
                name=item["name"],
                source_smiles=item["source_smiles"],
                extraction_profile=item["extraction_profile"],
                rooted_at_atom=item["rooted_at_atom"],
                component_order=tuple(
                    tuple(component) for component in item["component_order"]
                ),
                component_root_domains=tuple(
                    tuple(domain) for domain in item["component_root_domains"]
                ),
                expected_support=support,
                support_count=item["support_count"],
                completion_count=item["completion_count"],
                sorted_support_sha256=item["sorted_support_sha256"],
            )
        )
    if not cases:
        raise ValueError(f"disconnected audit fixture has no cases: {path}")
    return tuple(cases)


__all__ = (
    "SouthStarDisconnectedAuditCase",
    "load_pinned_south_star_disconnected_audit_cases",
)
