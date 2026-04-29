from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

from tests.helpers.fixture_paths import checked_in_fixture_path


PINNED_RDKIT_EXACT_SMALL_SUPPORT = "rdkit_exact_small_support"
PINNED_RDKIT_ROOTED_RANDOM = "rdkit_rooted_random"
PINNED_RDKIT_SERIALIZER_REGRESSIONS = "rdkit_serializer_regressions"
PINNED_RDKIT_WRITER_MEMBERSHIP = "rdkit_writer_membership"
PINNED_RDKIT_PARITY_TARGETS: tuple[tuple[str, str], ...] = (
    (
        PINNED_RDKIT_EXACT_SMALL_SUPPORT,
        "tests.rdkit_serialization.test_exact_small_support",
    ),
    (
        PINNED_RDKIT_ROOTED_RANDOM,
        "tests.rdkit_serialization.test_rooted_random.RDKITRootedRandomWriterTests."
        "test_rdkit_rooted_random_generation_cases_are_in_grimace_support",
    ),
    (
        PINNED_RDKIT_SERIALIZER_REGRESSIONS,
        "tests.rdkit_serialization.test_serializer_regressions",
    ),
    (
        PINNED_RDKIT_WRITER_MEMBERSHIP,
        "tests.rdkit_serialization.test_writer_membership",
    ),
)
PINNED_RDKIT_PARITY_FIXTURE_FAMILIES = tuple(
    fixture_family for fixture_family, _module_name in PINNED_RDKIT_PARITY_TARGETS
)
PINNED_RDKIT_PARITY_MODULES = tuple(
    module_name for _fixture_family, module_name in PINNED_RDKIT_PARITY_TARGETS
)


@dataclass(frozen=True, slots=True)
class PinnedFixtureCase:
    case_id: str
    source: str
    raw: dict[str, object]
    fixture_path: Path


def pinned_rdkit_fixture_root(fixture_family: str) -> Path:
    return checked_in_fixture_path(fixture_family)


def pinned_rdkit_parity_fixture_roots() -> tuple[Path, ...]:
    return tuple(
        pinned_rdkit_fixture_root(fixture_family)
        for fixture_family in PINNED_RDKIT_PARITY_FIXTURE_FAMILIES
    )


def has_pinned_rdkit_fixture(fixture_root: Path, rdkit_version: str) -> bool:
    fixture_path = fixture_root / f"{rdkit_version}.json"
    fixture_dir = fixture_root / rdkit_version
    return fixture_path.is_file() or (
        fixture_dir.is_dir() and any(fixture_dir.glob("*.json"))
    )


def pinned_rdkit_fixture_versions(fixture_root: Path) -> tuple[str, ...]:
    versions = {
        path.stem
        for path in fixture_root.glob("*.json")
    } | {
        path.name
        for path in fixture_root.iterdir()
        if path.is_dir()
    }
    return tuple(sorted(versions))


def normalized_unique_sorted_strings(
    values: list[object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> tuple[str, ...]:
    if not all(type(value) is str for value in values):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define {field_name} "
            f"as strings; got {values!r}"
        )
    normalized = tuple(values)
    expected = tuple(sorted(set(normalized)))
    if normalized != expected:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} has non-canonical {field_name}; "
            f"expected sorted unique strings {expected!r}, got {normalized!r}"
        )
    return normalized


def required_string_tuple(
    values: list[object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> tuple[str, ...]:
    if not values or not all(type(value) is str for value in values):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define nonempty "
            f"{field_name} as strings; got {values!r}"
        )
    return tuple(values)


def optional_positive_int(
    raw_case: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> int | None:
    raw_value = raw_case.get(field_name)
    if raw_value is None:
        return None
    if type(raw_value) is not int or raw_value <= 0:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define positive "
            f"integer {field_name}; got {raw_value!r}"
        )
    return raw_value


def required_int(
    raw_case: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> int:
    raw_value = raw_case.get(field_name)
    if type(raw_value) is not int:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define integer "
            f"{field_name}; got {raw_value!r}"
        )
    return raw_value


def optional_int(
    raw_case: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> int | None:
    raw_value = raw_case.get(field_name)
    if raw_value is None:
        return None
    if type(raw_value) is not int:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define integer "
            f"{field_name}; got {raw_value!r}"
        )
    return raw_value


def required_bool(
    raw_case: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> bool:
    raw_value = raw_case.get(field_name)
    if type(raw_value) is not bool:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define boolean "
            f"{field_name}; got {raw_value!r}"
        )
    return raw_value


def optional_bool(
    raw_case: dict[str, object],
    *,
    field_name: str,
    default: bool,
    fixture_path: Path,
    case_id: str,
) -> bool:
    raw_value = raw_case.get(field_name, default)
    if type(raw_value) is not bool:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define boolean "
            f"{field_name}; got {raw_value!r}"
        )
    return raw_value


def required_string(
    raw_case: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> str:
    raw_value = raw_case.get(field_name)
    if type(raw_value) is not str or not raw_value:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define nonempty "
            f"string {field_name}; got {raw_value!r}"
        )
    return raw_value


def optional_string(
    raw_case: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> str | None:
    raw_value = raw_case.get(field_name)
    if raw_value is None:
        return None
    if type(raw_value) is not str or not raw_value:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define nonempty "
            f"string {field_name}; got {raw_value!r}"
        )
    return raw_value


def load_pinned_rdkit_fixture_cases(
    *,
    fixture_root: Path,
    rdkit_version: str,
    fixture_label: str,
) -> tuple[PinnedFixtureCase, ...]:
    fixture_path = fixture_root / f"{rdkit_version}.json"
    fixture_dir = fixture_root / rdkit_version
    if fixture_path.is_file():
        payloads = ((fixture_path, json.loads(fixture_path.read_text())),)
    elif fixture_dir.is_dir():
        payloads = tuple(
            (path, json.loads(path.read_text()))
            for path in sorted(fixture_dir.glob("*.json"))
        )
        if not payloads:
            raise FileNotFoundError(
                f"pinned {fixture_label} fixture directory for RDKit "
                f"{rdkit_version} contains no JSON shards: {fixture_dir}"
            )
    else:
        raise FileNotFoundError(
            f"no pinned {fixture_label} fixture for RDKit {rdkit_version}"
        )

    cases = []
    seen_ids: dict[str, Path] = {}
    for current_fixture_path, data in payloads:
        if data.get("rdkit_version") != rdkit_version:
            raise ValueError(
                f"fixture {current_fixture_path} declares "
                f"rdkit_version={data.get('rdkit_version')!r}, "
                f"expected {rdkit_version!r}"
            )
        for raw_case in data["cases"]:
            case_id = str(raw_case["id"])
            if not case_id:
                raise ValueError(
                    f"fixture {current_fixture_path} contains an empty case id"
                )
            if case_id in seen_ids:
                raise ValueError(
                    f"fixture {current_fixture_path} duplicates case id {case_id!r} "
                    f"from {seen_ids[case_id]}"
                )
            seen_ids[case_id] = current_fixture_path

            source = required_string(
                raw_case,
                field_name="source",
                fixture_path=current_fixture_path,
                case_id=case_id,
            )

            cases.append(
                PinnedFixtureCase(
                    case_id=case_id,
                    source=source,
                    raw=raw_case,
                    fixture_path=current_fixture_path,
                )
            )

    return tuple(cases)
