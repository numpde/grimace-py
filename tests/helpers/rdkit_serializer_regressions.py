from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PinnedSerializerRegressionCase:
    case_id: str
    source: str
    smiles: str
    rooted_at_atom: int
    isomeric_smiles: bool
    expected: tuple[str, ...]
    expected_inventory: tuple[str, ...]
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False
    rdkit_sample_draw_budget: int | None = None


_FIXTURE_ROOT = (
    Path(__file__).resolve().parents[1] / "fixtures" / "rdkit_serializer_regressions"
)


def _normalized_unique_sorted_strings(
    values: list[object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> tuple[str, ...]:
    normalized = tuple(str(value) for value in values)
    expected = tuple(sorted(set(normalized)))
    if normalized != expected:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} has non-canonical {field_name}; "
            f"expected sorted unique strings {expected!r}, got {normalized!r}"
        )
    return normalized


def load_pinned_serializer_regression_cases(
    rdkit_version: str,
) -> tuple[PinnedSerializerRegressionCase, ...]:
    fixture_path = _FIXTURE_ROOT / f"{rdkit_version}.json"
    if not fixture_path.is_file():
        raise FileNotFoundError(
            f"no pinned serializer-regression fixture for RDKit {rdkit_version}"
        )

    data = json.loads(fixture_path.read_text())
    if data.get("rdkit_version") != rdkit_version:
        raise ValueError(
            f"fixture {fixture_path} declares rdkit_version={data.get('rdkit_version')!r}, "
            f"expected {rdkit_version!r}"
        )

    cases = []
    seen_ids: set[str] = set()
    for raw_case in data["cases"]:
        case_id = str(raw_case["id"])
        if not case_id:
            raise ValueError(f"fixture {fixture_path} contains an empty case id")
        if case_id in seen_ids:
            raise ValueError(f"fixture {fixture_path} contains duplicate case id {case_id!r}")
        seen_ids.add(case_id)

        source = str(raw_case["source"])
        if not source:
            raise ValueError(f"fixture {fixture_path} case {case_id!r} contains an empty source")
        raw_draw_budget = raw_case.get("rdkit_sample_draw_budget")
        rdkit_sample_draw_budget = (
            None if raw_draw_budget is None else int(raw_draw_budget)
        )
        if rdkit_sample_draw_budget is not None and rdkit_sample_draw_budget <= 0:
            raise ValueError(
                f"fixture {fixture_path} case {case_id!r} has nonpositive "
                f"rdkit_sample_draw_budget={rdkit_sample_draw_budget}"
            )

        cases.append(
            PinnedSerializerRegressionCase(
                case_id=case_id,
                source=source,
                smiles=str(raw_case["smiles"]),
                rooted_at_atom=int(raw_case.get("rooted_at_atom", -1)),
                isomeric_smiles=bool(raw_case.get("isomeric_smiles", True)),
                expected=_normalized_unique_sorted_strings(
                    list(raw_case["expected"]),
                    field_name="expected",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                expected_inventory=_normalized_unique_sorted_strings(
                    list(raw_case["expected_inventory"]),
                    field_name="expected_inventory",
                    fixture_path=fixture_path,
                    case_id=case_id,
                ),
                kekule_smiles=bool(raw_case.get("kekule_smiles", False)),
                all_bonds_explicit=bool(raw_case.get("all_bonds_explicit", False)),
                all_hs_explicit=bool(raw_case.get("all_hs_explicit", False)),
                ignore_atom_map_numbers=bool(raw_case.get("ignore_atom_map_numbers", False)),
                rdkit_sample_draw_budget=rdkit_sample_draw_budget,
            )
        )

    return tuple(cases)
