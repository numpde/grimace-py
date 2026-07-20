"""Version-keyed loader for the ordinary aromatic South Star audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from tests.helpers.fixture_paths import checked_in_fixture_path
from tests.helpers.pinned_rdkit_fixtures import normalized_unique_sorted_strings
from tests.helpers.pinned_rdkit_fixtures import PINNED_SOUTH_STAR_AROMATIC_AUDIT


@dataclass(frozen=True, slots=True)
class SouthStarAromaticAuditCase:
    case_id: str
    source_smiles: str
    rooted_at_atom: int
    expected_support: tuple[str, ...]
    support_count: int
    completion_count: int
    witness_multiplicities: tuple[tuple[str, int], ...]
    sorted_support_sha256: str


def load_pinned_south_star_aromatic_audit_cases(
    rdkit_version: str,
) -> tuple[SouthStarAromaticAuditCase, ...]:
    path = checked_in_fixture_path(PINNED_SOUTH_STAR_AROMATIC_AUDIT) / (
        f"{rdkit_version}.json"
    )
    raw = json.loads(path.read_text())
    if raw.get("rdkit_version") != rdkit_version:
        raise ValueError(f"aromatic audit fixture version mismatch: {path}")
    cases = []
    seen = set()
    for item in raw.get("cases", ()):
        case_id = item.get("id")
        if type(case_id) is not str or case_id in seen:
            raise ValueError(f"invalid aromatic audit case id: {case_id!r}")
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
            raise ValueError(f"aromatic audit digest mismatch: {case_id}")
        multiplicities = tuple(
            (text, count) for text, count in item.get("witness_multiplicities", ())
        )
        if (
            tuple(text for text, _count in multiplicities) != support
            or any(type(count) is not int or count <= 0 for _text, count in multiplicities)
            or sum(count for _text, count in multiplicities)
            != item.get("completion_count")
        ):
            raise ValueError(f"aromatic audit multiplicity mismatch: {case_id}")
        cases.append(
            SouthStarAromaticAuditCase(
                case_id=case_id,
                source_smiles=item["source_smiles"],
                rooted_at_atom=item["rooted_at_atom"],
                expected_support=support,
                support_count=item["support_count"],
                completion_count=item["completion_count"],
                witness_multiplicities=multiplicities,
                sorted_support_sha256=item["sorted_support_sha256"],
            )
        )
    if not cases:
        raise ValueError(f"aromatic audit fixture has no cases: {path}")
    return tuple(cases)


__all__ = (
    "SouthStarAromaticAuditCase",
    "load_pinned_south_star_aromatic_audit_cases",
)
