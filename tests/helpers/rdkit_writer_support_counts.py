from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tests.helpers.fixture_paths import read_fixture_json_object
from tests.helpers.pinned_rdkit_fixtures import (
    PINNED_RDKIT_WRITER_SUPPORT_COUNTS,
    optional_int,
    pinned_rdkit_fixture_root,
    required_int,
    required_string,
)
from tests.helpers.public_runtime import supported_public_kwargs


_FIXTURE_ROOT = pinned_rdkit_fixture_root(PINNED_RDKIT_WRITER_SUPPORT_COUNTS)
_SUPPORT_COUNT_FLAGS = (
    "isomericSmiles",
    "canonical",
    "doRandom",
    "kekuleSmiles",
    "allBondsExplicit",
    "allHsExplicit",
    "ignoreAtomMapNumbers",
)
_ADAPTIVE_SATURATION_METHOD = "rdkit_random_adaptive_saturation"
_ADAPTIVE_SATURATION_CRITERION_VERSION = 1
_FLOAT_TOLERANCE = 1e-12


@dataclass(frozen=True, slots=True)
class PinnedWriterSupportCountEvidenceRun:
    seed: int
    draw_count: int
    support_count: int
    consecutive_draws_without_new_variant: int
    singleton_count: int
    doubleton_count: int
    estimated_unseen_mass: float
    estimated_missing_variants: float


@dataclass(frozen=True, slots=True)
class PinnedWriterSupportCountEvidence:
    method: str
    criterion_version: int
    min_draws: int
    unseen_mass_threshold: float
    allowed_missing_variants: float
    runs: tuple[PinnedWriterSupportCountEvidenceRun, ...]


@dataclass(frozen=True, slots=True)
class PinnedWriterSupportCountCase:
    case_id: str
    source: str
    smiles: str
    rooted_at_atom: int
    isomeric_smiles: bool
    kekule_smiles: bool
    all_bonds_explicit: bool
    all_hs_explicit: bool
    ignore_atom_map_numbers: bool
    support_count: int
    evidence: PinnedWriterSupportCountEvidence
    fixture_path: Path

    def public_kwargs(self) -> dict[str, object]:
        return supported_public_kwargs(
            rootedAtAtom=self.rooted_at_atom,
            isomericSmiles=self.isomeric_smiles,
            kekuleSmiles=self.kekule_smiles,
            allBondsExplicit=self.all_bonds_explicit,
            allHsExplicit=self.all_hs_explicit,
            ignoreAtomMapNumbers=self.ignore_atom_map_numbers,
        )


def _required_positive_int(
    raw: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> int:
    value = required_int(
        raw,
        field_name=field_name,
        fixture_path=fixture_path,
        case_id=case_id,
    )
    if value <= 0:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define positive "
            f"integer {field_name}; got {value!r}"
        )
    return value


def _required_nonnegative_int(
    raw: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> int:
    value = required_int(
        raw,
        field_name=field_name,
        fixture_path=fixture_path,
        case_id=case_id,
    )
    if value < 0:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define nonnegative "
            f"integer {field_name}; got {value!r}"
        )
    return value


def _required_float(
    raw: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> float:
    value = raw.get(field_name)
    if type(value) not in (int, float) or isinstance(value, bool):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define number "
            f"{field_name}; got {value!r}"
        )
    return float(value)


def _required_nonnegative_float(
    raw: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> float:
    value = _required_float(
        raw,
        field_name=field_name,
        fixture_path=fixture_path,
        case_id=case_id,
    )
    if value < 0:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define nonnegative "
            f"number {field_name}; got {value!r}"
        )
    return value


def _required_probability(
    raw: dict[str, object],
    *,
    field_name: str,
    fixture_path: Path,
    case_id: str,
) -> float:
    value = _required_nonnegative_float(
        raw,
        field_name=field_name,
        fixture_path=fixture_path,
        case_id=case_id,
    )
    if value > 1:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define probability "
            f"{field_name}; got {value!r}"
        )
    return value


def _required_bool_flag(
    flags: dict[str, object],
    flag_name: str,
    fixture_path: Path,
) -> bool:
    value = flags.get(flag_name)
    if type(value) is not bool:
        raise ValueError(
            f"fixture {fixture_path} must define boolean flag {flag_name}; "
            f"got {value!r}"
        )
    return value


def _surface_name(flags: dict[str, object]) -> str:
    surface = "isomeric" if flags["isomericSmiles"] else "nonisomeric"
    modifiers = ["random"]
    if flags["kekuleSmiles"]:
        modifiers.append("kekule")
    if flags["allBondsExplicit"]:
        modifiers.append("all_bonds_explicit")
    if flags["allHsExplicit"]:
        modifiers.append("all_hs_explicit")
    if flags["ignoreAtomMapNumbers"]:
        modifiers.append("ignore_atom_maps")
    return f"{surface}__{'_'.join(modifiers)}"


def _adaptive_patience(support_count: int) -> int:
    return max(10_000, 20 * support_count)


def _estimated_missing_variants(singleton_count: int, doubleton_count: int) -> float:
    if singleton_count == 0:
        return 0.0
    if doubleton_count == 0:
        raise ValueError("missing-variant estimate is unstable without doubletons")
    return (singleton_count * singleton_count) / (2 * doubleton_count)


def _floats_match(left: float, right: float) -> bool:
    return abs(left - right) <= _FLOAT_TOLERANCE


def _load_flags(payload: dict[str, object], fixture_path: Path) -> dict[str, bool]:
    flags = payload.get("flags")
    if not isinstance(flags, dict):
        raise ValueError(f"fixture {fixture_path} must define object flags")
    if set(flags) != set(_SUPPORT_COUNT_FLAGS):
        raise ValueError(
            f"fixture {fixture_path} must define exactly flags "
            f"{_SUPPORT_COUNT_FLAGS!r}; got {tuple(sorted(flags))!r}"
        )

    parsed = {
        flag_name: _required_bool_flag(flags, flag_name, fixture_path)
        for flag_name in _SUPPORT_COUNT_FLAGS
    }
    if parsed["canonical"] is not False or parsed["doRandom"] is not True:
        raise ValueError(
            f"fixture {fixture_path} must describe random writer support with "
            "canonical=false and doRandom=true"
        )
    expected_stem = _surface_name(parsed)
    if fixture_path.stem != expected_stem:
        raise ValueError(
            f"fixture {fixture_path} filename must match flag surface "
            f"{expected_stem!r}"
        )
    return parsed


def _load_evidence_run(
    raw_run: object,
    *,
    fixture_path: Path,
    case_id: str,
    case_support_count: int,
    min_draws: int,
    unseen_mass_threshold: float,
    allowed_missing_variants: float,
) -> PinnedWriterSupportCountEvidenceRun:
    if not isinstance(raw_run, dict):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} evidence runs must be objects"
        )
    seed = _required_positive_int(
        raw_run,
        field_name="seed",
        fixture_path=fixture_path,
        case_id=case_id,
    )
    draw_count = _required_positive_int(
        raw_run,
        field_name="draw_count",
        fixture_path=fixture_path,
        case_id=case_id,
    )
    support_count = _required_positive_int(
        raw_run,
        field_name="support_count",
        fixture_path=fixture_path,
        case_id=case_id,
    )
    if support_count != case_support_count:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} evidence support_count "
            f"{support_count!r} does not match case support_count {case_support_count!r}"
        )
    if support_count > draw_count:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} support_count cannot exceed "
            f"draw_count"
        )
    if draw_count < min_draws:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} evidence draw_count "
            f"{draw_count!r} is below min_draws {min_draws!r}"
        )
    consecutive = _required_positive_int(
        raw_run,
        field_name="consecutive_draws_without_new_variant",
        fixture_path=fixture_path,
        case_id=case_id,
    )
    if consecutive >= draw_count:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} consecutive no-new "
            "draws must be less than draw_count"
        )
    patience = _adaptive_patience(support_count)
    if consecutive < patience:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} consecutive no-new "
            f"draws {consecutive!r} is below adaptive patience {patience!r}"
        )
    singleton_count = _required_nonnegative_int(
        raw_run,
        field_name="singleton_count",
        fixture_path=fixture_path,
        case_id=case_id,
    )
    doubleton_count = _required_nonnegative_int(
        raw_run,
        field_name="doubleton_count",
        fixture_path=fixture_path,
        case_id=case_id,
    )
    unseen_mass = _required_probability(
        raw_run,
        field_name="estimated_unseen_mass",
        fixture_path=fixture_path,
        case_id=case_id,
    )
    expected_unseen_mass = singleton_count / draw_count
    if not _floats_match(unseen_mass, expected_unseen_mass):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} estimated_unseen_mass "
            f"{unseen_mass!r} does not match singleton_count / draw_count "
            f"{expected_unseen_mass!r}"
        )
    if unseen_mass > unseen_mass_threshold:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} estimated_unseen_mass "
            f"{unseen_mass!r} exceeds threshold {unseen_mass_threshold!r}"
        )
    missing_variants = _required_nonnegative_float(
        raw_run,
        field_name="estimated_missing_variants",
        fixture_path=fixture_path,
        case_id=case_id,
    )
    try:
        expected_missing_variants = _estimated_missing_variants(
            singleton_count,
            doubleton_count,
        )
    except ValueError as exc:
        raise ValueError(f"fixture {fixture_path} case {case_id!r} {exc}") from exc
    if not _floats_match(missing_variants, expected_missing_variants):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} estimated_missing_variants "
            f"{missing_variants!r} does not match singleton/doubleton estimate "
            f"{expected_missing_variants!r}"
        )
    if missing_variants > allowed_missing_variants:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} estimated_missing_variants "
            f"{missing_variants!r} exceeds allowed {allowed_missing_variants!r}"
        )
    return PinnedWriterSupportCountEvidenceRun(
        seed=seed,
        draw_count=draw_count,
        support_count=support_count,
        consecutive_draws_without_new_variant=consecutive,
        singleton_count=singleton_count,
        doubleton_count=doubleton_count,
        estimated_unseen_mass=unseen_mass,
        estimated_missing_variants=missing_variants,
    )


def _load_evidence(
    raw_case: dict[str, object],
    *,
    fixture_path: Path,
    case_id: str,
    support_count: int,
) -> PinnedWriterSupportCountEvidence:
    raw_evidence = raw_case.get("evidence")
    if not isinstance(raw_evidence, dict):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} must define object evidence"
        )
    method = required_string(
        raw_evidence,
        field_name="method",
        fixture_path=fixture_path,
        case_id=case_id,
    )
    if method != _ADAPTIVE_SATURATION_METHOD:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} has unsupported evidence "
            f"method {method!r}"
        )
    criterion_version = _required_positive_int(
        raw_evidence,
        field_name="criterion_version",
        fixture_path=fixture_path,
        case_id=case_id,
    )
    if criterion_version != _ADAPTIVE_SATURATION_CRITERION_VERSION:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} has unsupported "
            f"criterion_version {criterion_version!r}"
        )
    min_draws = _required_positive_int(
        raw_evidence,
        field_name="min_draws",
        fixture_path=fixture_path,
        case_id=case_id,
    )
    unseen_mass_threshold = _required_probability(
        raw_evidence,
        field_name="unseen_mass_threshold",
        fixture_path=fixture_path,
        case_id=case_id,
    )
    if unseen_mass_threshold <= 0:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} unseen_mass_threshold "
            "must be positive"
        )
    allowed_missing_variants = _required_nonnegative_float(
        raw_evidence,
        field_name="allowed_missing_variants",
        fixture_path=fixture_path,
        case_id=case_id,
    )
    raw_runs = raw_evidence.get("runs")
    if not isinstance(raw_runs, list) or len(raw_runs) < 2:
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} adaptive evidence must "
            "define at least two runs"
        )
    runs = tuple(
        _load_evidence_run(
            raw_run,
            fixture_path=fixture_path,
            case_id=case_id,
            case_support_count=support_count,
            min_draws=min_draws,
            unseen_mass_threshold=unseen_mass_threshold,
            allowed_missing_variants=allowed_missing_variants,
        )
        for raw_run in raw_runs
    )
    seeds = [run.seed for run in runs]
    if len(seeds) != len(set(seeds)):
        raise ValueError(
            f"fixture {fixture_path} case {case_id!r} evidence run seeds "
            "must be unique"
        )
    return PinnedWriterSupportCountEvidence(
        method=method,
        criterion_version=criterion_version,
        min_draws=min_draws,
        unseen_mass_threshold=unseen_mass_threshold,
        allowed_missing_variants=allowed_missing_variants,
        runs=runs,
    )


def _payload_paths(fixture_root: Path, rdkit_version: str) -> tuple[Path, ...]:
    fixture_dir = fixture_root / rdkit_version
    if not fixture_dir.is_dir():
        raise FileNotFoundError(
            f"no pinned writer-support-count fixture for RDKit {rdkit_version}"
        )
    paths = tuple(sorted(fixture_dir.glob("*.json")))
    if not paths:
        raise FileNotFoundError(
            f"pinned writer-support-count fixture directory for RDKit "
            f"{rdkit_version} contains no JSON shards: {fixture_dir}"
        )
    return paths


def load_pinned_writer_support_count_cases(
    rdkit_version: str,
    *,
    fixture_root: Path = _FIXTURE_ROOT,
) -> tuple[PinnedWriterSupportCountCase, ...]:
    cases: list[PinnedWriterSupportCountCase] = []
    seen_ids: dict[str, Path] = {}
    for fixture_path in _payload_paths(fixture_root, rdkit_version):
        payload = read_fixture_json_object(fixture_path)
        if payload.get("rdkit_version") != rdkit_version:
            raise ValueError(
                f"fixture {fixture_path} declares "
                f"rdkit_version={payload.get('rdkit_version')!r}, "
                f"expected {rdkit_version!r}"
            )
        flags = _load_flags(payload, fixture_path)
        raw_cases = payload.get("cases")
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError(f"fixture {fixture_path} must define nonempty cases list")
        for raw_case in raw_cases:
            if not isinstance(raw_case, dict):
                raise ValueError(f"fixture {fixture_path} contains a non-object case")
            raw_case_id = raw_case.get("id")
            if type(raw_case_id) is not str or not raw_case_id:
                raise ValueError(f"fixture {fixture_path} contains an invalid case id")
            case_id = raw_case_id
            if case_id in seen_ids:
                raise ValueError(
                    f"fixture {fixture_path} duplicates case id {case_id!r} "
                    f"from {seen_ids[case_id]}"
                )
            seen_ids[case_id] = fixture_path
            rooted_at_atom = optional_int(
                raw_case,
                field_name="rooted_at_atom",
                fixture_path=fixture_path,
                case_id=case_id,
            )
            if rooted_at_atom is None:
                raise ValueError(
                    f"fixture {fixture_path} case {case_id!r} must define "
                    "integer rooted_at_atom"
                )
            support_count = _required_positive_int(
                raw_case,
                field_name="support_count",
                fixture_path=fixture_path,
                case_id=case_id,
            )
            cases.append(
                PinnedWriterSupportCountCase(
                    case_id=case_id,
                    source=required_string(
                        raw_case,
                        field_name="source",
                        fixture_path=fixture_path,
                        case_id=case_id,
                    ),
                    smiles=required_string(
                        raw_case,
                        field_name="smiles",
                        fixture_path=fixture_path,
                        case_id=case_id,
                    ),
                    rooted_at_atom=rooted_at_atom,
                    isomeric_smiles=flags["isomericSmiles"],
                    kekule_smiles=flags["kekuleSmiles"],
                    all_bonds_explicit=flags["allBondsExplicit"],
                    all_hs_explicit=flags["allHsExplicit"],
                    ignore_atom_map_numbers=flags["ignoreAtomMapNumbers"],
                    support_count=support_count,
                    evidence=_load_evidence(
                        raw_case,
                        fixture_path=fixture_path,
                        case_id=case_id,
                        support_count=support_count,
                    ),
                    fixture_path=fixture_path,
                )
            )
    return tuple(cases)
