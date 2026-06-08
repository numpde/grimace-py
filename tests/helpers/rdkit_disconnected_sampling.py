from __future__ import annotations

from pathlib import Path

from tests.helpers.fixture_paths import checked_in_fixture_path, read_fixture_json_object


_FIXTURE_PATH = checked_in_fixture_path(
    "rdkit_disconnected_sampling",
    "root_zero_smiles.json",
)


def load_disconnected_root_zero_smiles(
    fixture_path: Path = _FIXTURE_PATH,
) -> tuple[str, ...]:
    data = read_fixture_json_object(fixture_path)
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list):
        raise ValueError(f"fixture {fixture_path} must define a cases list")
    if not raw_cases or not all(type(case) is str and case for case in raw_cases):
        raise ValueError(
            f"fixture {fixture_path} must define nonempty SMILES strings"
        )
    cases = tuple(raw_cases)
    if cases != tuple(dict.fromkeys(cases)):
        raise ValueError(f"fixture {fixture_path} contains duplicate SMILES")
    return cases
