"""Version-keyed loader for the specified-stereo South Star audit."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from tests.helpers.fixture_paths import checked_in_fixture_path
from tests.helpers.pinned_rdkit_fixtures import normalized_unique_sorted_strings
from tests.helpers.pinned_rdkit_fixtures import PINNED_SOUTH_STAR_STEREO_AUDIT


@dataclass(frozen=True, slots=True)
class SouthStarStereoAuditCase:
    case_id: str
    name: str
    source_smiles: str
    extraction_profile: str
    rooted_at_atom: int
    target_class: str
    target: str
    reference: tuple[int, ...]
    ligand_equivalence: str
    expected_support: tuple[str, ...]
    support_count: int
    completion_count: int
    sorted_support_sha256: str


def load_pinned_south_star_stereo_audit_cases(
    rdkit_version: str,
) -> tuple[SouthStarStereoAuditCase, ...]:
    path = checked_in_fixture_path(PINNED_SOUTH_STAR_STEREO_AUDIT) / (
        f"{rdkit_version}.json"
    )
    raw = json.loads(path.read_text())
    if raw.get("rdkit_version") != rdkit_version:
        raise ValueError(f"stereo audit fixture version mismatch: {path}")
    cases = []
    seen_ids = set()
    for item in raw.get("cases", ()):
        case_id = item.get("id")
        if type(case_id) is not str or case_id in seen_ids:
            raise ValueError(f"invalid stereo audit case id in {path}: {case_id!r}")
        seen_ids.add(case_id)
        support = normalized_unique_sorted_strings(
            item.get("expected_support"),
            field_name="expected_support",
            fixture_path=path,
            case_id=case_id,
        )
        digest = hashlib.sha256(
            json.dumps(
                support,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
            ).encode()
        ).hexdigest()
        if digest != item.get("sorted_support_sha256"):
            raise ValueError(f"stereo audit support digest mismatch: {case_id}")
        extraction_options = item.get("extraction_options")
        if type(extraction_options) is not dict:
            raise ValueError(
                f"stereo audit extraction options must be an object: {case_id}"
            )
        if (
            extraction_options.get("include_potential_sites") is not True
            or extraction_options.get("stereo_site_discovery_mode")
            != "specified_closure"
        ):
            raise ValueError(f"stereo audit extraction options mismatch: {case_id}")
        known_keys = {
            "include_potential_sites",
            "stereo_site_discovery_mode",
            "stereo_site_options",
        }
        if set(extraction_options) - known_keys:
            raise ValueError(f"stereo audit extraction options mismatch: {case_id}")
        stereo_site_options = extraction_options.get("stereo_site_options")
        if stereo_site_options is not None:
            if type(stereo_site_options) is not dict:
                raise ValueError(
                    f"stereo site options must be an object: {case_id}"
                )
            if set(stereo_site_options) != {"ligand_equivalence"}:
                raise ValueError(
                    f"stereo site options mismatch: {case_id}"
                )
            expected = stereo_site_options["ligand_equivalence"]
            if type(expected) is not str or not expected:
                raise ValueError(
                    f"stereo audit ligand equivalence invalid: {case_id}"
                )
        ligand_equivalence = item.get("ligand_equivalence")
        if type(ligand_equivalence) is not str or not ligand_equivalence:
            raise ValueError(
                f"stereo audit ligand equivalence invalid: {case_id}"
            )
        if ligand_equivalence == "immediate_color":
            if stereo_site_options is not None:
                raise ValueError(
                    f"stereo audit ligand equivalence mismatch: {case_id}"
                )
        else:
            if stereo_site_options is None:
                raise ValueError(
                    f"stereo audit ligand equivalence mismatch: {case_id}"
                )
            if stereo_site_options["ligand_equivalence"] != ligand_equivalence:
                raise ValueError(
                    f"stereo audit ligand equivalence mismatch: {case_id}"
                )
        cases.append(
            SouthStarStereoAuditCase(
                case_id=case_id,
                name=item["name"],
                source_smiles=item["source_smiles"],
                extraction_profile=item["extraction_profile"],
                rooted_at_atom=item["rooted_at_atom"],
                target_class=item["target_class"],
                target=item["target"],
                reference=tuple(item["reference"]),
                ligand_equivalence=ligand_equivalence,
                expected_support=support,
                support_count=item["support_count"],
                completion_count=item["completion_count"],
                sorted_support_sha256=item["sorted_support_sha256"],
            )
        )
    if not cases:
        raise ValueError(f"stereo audit fixture has no cases: {path}")
    return tuple(cases)


__all__ = (
    "SouthStarStereoAuditCase",
    "load_pinned_south_star_stereo_audit_cases",
)
