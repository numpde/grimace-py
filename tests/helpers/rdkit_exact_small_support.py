from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


@dataclass(frozen=True, slots=True)
class PinnedExactSmallSupportCase:
    smiles: str
    rooted_at_atom: int
    isomeric_smiles: bool
    expected: tuple[str, ...]
    expected_inventory: tuple[str, ...]
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False


_FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "rdkit_exact_small_support"


def load_pinned_exact_small_support_cases(rdkit_version: str) -> tuple[PinnedExactSmallSupportCase, ...]:
    fixture_path = _FIXTURE_ROOT / f"{rdkit_version}.json"
    if not fixture_path.is_file():
        raise FileNotFoundError(f"no pinned exact small-support fixture for RDKit {rdkit_version}")

    data = json.loads(fixture_path.read_text())
    if data.get("rdkit_version") != rdkit_version:
        raise ValueError(
            f"fixture {fixture_path} declares rdkit_version={data.get('rdkit_version')!r}, "
            f"expected {rdkit_version!r}"
        )

    cases = []
    for raw_case in data["cases"]:
        cases.append(
            PinnedExactSmallSupportCase(
                smiles=str(raw_case["smiles"]),
                rooted_at_atom=int(raw_case["rooted_at_atom"]),
                isomeric_smiles=bool(raw_case["isomeric_smiles"]),
                expected=tuple(str(value) for value in raw_case["expected"]),
                expected_inventory=tuple(str(value) for value in raw_case["expected_inventory"]),
                kekule_smiles=bool(raw_case.get("kekule_smiles", False)),
                all_bonds_explicit=bool(raw_case.get("all_bonds_explicit", False)),
                all_hs_explicit=bool(raw_case.get("all_hs_explicit", False)),
                ignore_atom_map_numbers=bool(raw_case.get("ignore_atom_map_numbers", False)),
            )
        )

    return tuple(cases)
