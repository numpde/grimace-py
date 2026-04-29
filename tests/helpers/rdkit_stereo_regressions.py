from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path


_STEROID_RING_COUPLED_COMPONENT_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "rdkit_stereo_regressions"
    / "steroid_ring_coupled_component.json"
)


@dataclass(frozen=True, slots=True)
class StereoMembershipRegression:
    input_smiles: str
    rooted_at_atom: int
    expected_member: str
    rejected_member: str


def load_steroid_ring_coupled_component_regression(
    fixture_path: Path = _STEROID_RING_COUPLED_COMPONENT_PATH,
) -> StereoMembershipRegression:
    data = json.loads(fixture_path.read_text())
    return StereoMembershipRegression(
        input_smiles=_required_string(data, "input_smiles", fixture_path),
        rooted_at_atom=_required_int(data, "rooted_at_atom", fixture_path),
        expected_member=_required_string(data, "expected_member", fixture_path),
        rejected_member=_required_string(data, "rejected_member", fixture_path),
    )


def _required_string(data: dict[str, object], field_name: str, fixture_path: Path) -> str:
    value = data.get(field_name)
    if type(value) is not str or not value:
        raise ValueError(f"fixture {fixture_path} must define nonempty {field_name}")
    return value


def _required_int(data: dict[str, object], field_name: str, fixture_path: Path) -> int:
    value = data.get(field_name)
    if type(value) is not int:
        raise ValueError(f"fixture {fixture_path} must define integer {field_name}")
    return value
