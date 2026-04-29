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
_ROOTED_MEMBERSHIP_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures"
    / "rdkit_stereo_regressions"
    / "rooted_membership.json"
)


@dataclass(frozen=True, slots=True)
class StereoMembershipRegression:
    input_smiles: str
    rooted_at_atom: int
    expected_member: str
    rejected_member: str


@dataclass(frozen=True, slots=True)
class StereoExpectedMemberRegression:
    case_id: str
    input_smiles: str
    rooted_at_atom: int
    expected_member: str
    validate_support: bool


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


def load_stereo_expected_member_regressions(
    fixture_path: Path = _ROOTED_MEMBERSHIP_PATH,
) -> tuple[StereoExpectedMemberRegression, ...]:
    data = json.loads(fixture_path.read_text())
    raw_cases = data["cases"]
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"fixture {fixture_path} must define nonempty cases")
    return tuple(
        StereoExpectedMemberRegression(
            case_id=_required_string(raw_case, "id", fixture_path),
            input_smiles=_required_string(raw_case, "input_smiles", fixture_path),
            rooted_at_atom=_required_int(raw_case, "rooted_at_atom", fixture_path),
            expected_member=_required_string(raw_case, "expected_member", fixture_path),
            validate_support=_required_bool(raw_case, "validate_support", fixture_path),
        )
        for raw_case in raw_cases
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


def _required_bool(data: dict[str, object], field_name: str, fixture_path: Path) -> bool:
    value = data.get(field_name)
    if type(value) is not bool:
        raise ValueError(f"fixture {fixture_path} must define boolean {field_name}")
    return value
