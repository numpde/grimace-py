from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from tests.helpers.fixture_paths import checked_in_fixture_path


SOUTH_STAR_SEMANTIC_FIXTURE_PATH = checked_in_fixture_path(
    "south_star_semantics",
    "basic.json",
)


@dataclass(frozen=True, slots=True)
class SouthStarNegativeSemanticSmiles:
    smiles: str
    reason: str


@dataclass(frozen=True, slots=True)
class SouthStarAnnotationPolicyExpectation:
    required_marker_edge_count: int


@dataclass(frozen=True, slots=True)
class SouthStarSemanticCase:
    case_id: str
    semantic_feature: str
    source_smiles: str
    eligible_carrier_edges: tuple[tuple[int, int], ...]
    maximal_eligible_carrier: SouthStarAnnotationPolicyExpectation
    rdkit_writer_membership_status: str
    rdkit_writer_membership_notes: str
    positive_semantic_smiles: tuple[str, ...]
    negative_semantic_smiles: tuple[SouthStarNegativeSemanticSmiles, ...]


def load_south_star_semantic_cases(
    fixture_path: Path = SOUTH_STAR_SEMANTIC_FIXTURE_PATH,
) -> tuple[SouthStarSemanticCase, ...]:
    data = json.loads(fixture_path.read_text())
    if data.get("schema_version") != 1:
        raise ValueError(f"fixture {fixture_path} must declare schema_version 1")

    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"fixture {fixture_path} must define nonempty cases")

    return tuple(_load_case(raw_case, fixture_path) for raw_case in raw_cases)


def _load_case(raw_case: object, fixture_path: Path) -> SouthStarSemanticCase:
    if not isinstance(raw_case, dict):
        raise ValueError(f"fixture {fixture_path} cases must be objects")

    case_id = _required_string(raw_case, "id", fixture_path)
    policy_expectations = _required_object(
        raw_case,
        "annotation_policy_expectations",
        fixture_path,
        case_id,
    )
    maximal_expectation = _required_object(
        policy_expectations,
        "maximal_eligible_carrier",
        fixture_path,
        case_id,
    )
    writer_membership = _required_object(
        raw_case,
        "rdkit_writer_membership",
        fixture_path,
        case_id,
    )

    return SouthStarSemanticCase(
        case_id=case_id,
        semantic_feature=_required_string(raw_case, "semantic_feature", fixture_path),
        source_smiles=_required_string(raw_case, "source_smiles", fixture_path),
        eligible_carrier_edges=_required_edge_tuple(
            raw_case,
            "eligible_carrier_edges",
            fixture_path,
            case_id,
        ),
        maximal_eligible_carrier=SouthStarAnnotationPolicyExpectation(
            required_marker_edge_count=_required_nonnegative_int(
                maximal_expectation,
                "required_marker_edge_count",
                fixture_path,
                case_id,
            ),
        ),
        rdkit_writer_membership_status=_required_string(
            writer_membership,
            "status",
            fixture_path,
        ),
        rdkit_writer_membership_notes=_required_string(
            writer_membership,
            "notes",
            fixture_path,
        ),
        positive_semantic_smiles=_required_string_tuple(
            raw_case,
            "positive_semantic_smiles",
            fixture_path,
            case_id,
        ),
        negative_semantic_smiles=tuple(
            _load_negative_semantic_smiles(raw_negative, fixture_path, case_id)
            for raw_negative in _required_list(
                raw_case,
                "negative_semantic_smiles",
                fixture_path,
                case_id,
            )
        ),
    )


def _load_negative_semantic_smiles(
    raw_negative: object,
    fixture_path: Path,
    case_id: str,
) -> SouthStarNegativeSemanticSmiles:
    if not isinstance(raw_negative, dict):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} negative semantic SMILES "
            "entries must be objects"
        )
    return SouthStarNegativeSemanticSmiles(
        smiles=_required_string(raw_negative, "smiles", fixture_path),
        reason=_required_string(raw_negative, "reason", fixture_path),
    )


def _required_object(
    data: dict[str, object],
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> dict[str, object]:
    value = data.get(field_name)
    if not isinstance(value, dict):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define object {field_name}"
        )
    return value


def _required_list(
    data: dict[str, object],
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> list[object]:
    value = data.get(field_name)
    if not isinstance(value, list) or not value:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define nonempty list "
            f"{field_name}"
        )
    return value


def _required_string(
    data: dict[str, object],
    field_name: str,
    fixture_path: Path,
) -> str:
    value = data.get(field_name)
    if type(value) is not str or not value:
        raise ValueError(f"fixture {fixture_path} must define nonempty {field_name}")
    return value


def _required_nonnegative_int(
    data: dict[str, object],
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> int:
    value = data.get(field_name)
    if type(value) is not int or value < 0:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define nonnegative "
            f"integer {field_name}"
        )
    return value


def _required_string_tuple(
    data: dict[str, object],
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> tuple[str, ...]:
    values = _required_list(data, field_name, fixture_path, case_id)
    if not all(type(value) is str and value for value in values):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define nonempty string "
            f"values for {field_name}"
        )
    unique = tuple(dict.fromkeys(values))
    if len(unique) != len(values):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} contains duplicate "
            f"{field_name}"
        )
    return unique


def _required_edge_tuple(
    data: dict[str, object],
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> tuple[tuple[int, int], ...]:
    raw_edges = _required_list(data, field_name, fixture_path, case_id)
    edges = []
    for raw_edge in raw_edges:
        if (
            not isinstance(raw_edge, list)
            or len(raw_edge) != 2
            or any(type(atom_idx) is not int or atom_idx < 0 for atom_idx in raw_edge)
        ):
            raise ValueError(
                f"fixture {fixture_path} case {case_id!r} must define {field_name} "
                "as nonnegative integer pairs"
            )
        edge = tuple(raw_edge)
        if edge[0] == edge[1]:
            raise ValueError(
                f"fixture {fixture_path} case {case_id!r} contains self-edge {edge}"
            )
        edges.append(edge)

    normalized = tuple(dict.fromkeys(tuple(sorted(edge)) for edge in edges))
    if len(normalized) != len(edges):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} contains duplicate "
            f"{field_name}"
        )
    return normalized
