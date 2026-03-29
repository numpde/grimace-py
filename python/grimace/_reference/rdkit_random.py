from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from rdkit import Chem, rdBase

from grimace._reference.policy import ReferencePolicy


SAMPLING_KEYS = {
    "seed",
    "draw_budget",
    "isomericSmiles",
    "kekuleSmiles",
    "rootedAtAtom",
    "canonical",
    "allBondsExplicit",
    "allHsExplicit",
    "doRandom",
    "ignoreAtomMapNumbers",
}

IDENTITY_KEYS = {
    "parse_with_rdkit",
    "canonical",
    "isomericSmiles",
    "kekuleSmiles",
    "rootedAtAtom",
    "allBondsExplicit",
    "allHsExplicit",
    "doRandom",
    "ignoreAtomMapNumbers",
}


@dataclass(frozen=True)
class ValidationIssue:
    sampled_smiles: str
    reason: str


@dataclass(frozen=True)
class RandomReferenceResult:
    sampled_smiles: tuple[str, ...]
    validation_issues: tuple[ValidationIssue, ...]

    @property
    def distinct_count(self) -> int:
        return len(self.sampled_smiles)

    @property
    def is_valid(self) -> bool:
        return not self.validation_issues


def _require_keys(section: Mapping[str, Any], expected: set[str], section_name: str) -> None:
    actual = set(section)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ValueError(
            f"{section_name} keys must match exactly; missing={missing}, extra={extra}"
        )


def _sampling_kwargs(policy: ReferencePolicy) -> tuple[int, int, int, dict[str, Any]]:
    sampling = policy.data["sampling"]
    if not isinstance(sampling, dict):
        raise TypeError("sampling policy must be a JSON object")
    _require_keys(sampling, SAMPLING_KEYS, "sampling")
    kwargs = {
        "isomericSmiles": sampling["isomericSmiles"],
        "kekuleSmiles": sampling["kekuleSmiles"],
        "canonical": sampling["canonical"],
        "allBondsExplicit": sampling["allBondsExplicit"],
        "allHsExplicit": sampling["allHsExplicit"],
        "doRandom": sampling["doRandom"],
        "ignoreAtomMapNumbers": sampling["ignoreAtomMapNumbers"],
    }
    return int(sampling["seed"]), int(sampling["draw_budget"]), int(sampling["rootedAtAtom"]), kwargs


def _identity_kwargs(policy: ReferencePolicy) -> dict[str, Any]:
    identity = policy.data["identity_check"]
    if not isinstance(identity, dict):
        raise TypeError("identity_check policy must be a JSON object")
    _require_keys(identity, IDENTITY_KEYS, "identity_check")
    if not identity["parse_with_rdkit"]:
        raise NotImplementedError("Only parse_with_rdkit=true is supported")
    return {
        "isomericSmiles": identity["isomericSmiles"],
        "kekuleSmiles": identity["kekuleSmiles"],
        "rootedAtAtom": identity["rootedAtAtom"],
        "canonical": identity["canonical"],
        "allBondsExplicit": identity["allBondsExplicit"],
        "allHsExplicit": identity["allHsExplicit"],
        "doRandom": identity["doRandom"],
        "ignoreAtomMapNumbers": identity["ignoreAtomMapNumbers"],
    }


def identity_smiles(mol: Chem.Mol, policy: ReferencePolicy) -> str:
    return Chem.MolToSmiles(Chem.Mol(mol), **_identity_kwargs(policy))


def sample_rdkit_random_smiles(mol: Chem.Mol, policy: ReferencePolicy) -> tuple[str, ...]:
    seed, draw_budget, rooted_at_atom, sampling_kwargs = _sampling_kwargs(policy)
    if draw_budget < 1:
        raise ValueError("draw_budget must be positive")
    rdBase.SeedRandomNumberGenerator(seed)
    sampled = {
        Chem.MolToSmiles(Chem.Mol(mol), rootedAtAtom=rooted_at_atom, **sampling_kwargs)
        for _ in range(draw_budget)
    }
    return tuple(sorted(sampled))


def sample_rdkit_random_smiles_from_root(
    mol: Chem.Mol,
    policy: ReferencePolicy,
    root_idx: int,
) -> tuple[str, ...]:
    seed, draw_budget, _, sampling_kwargs = _sampling_kwargs(policy)
    if draw_budget < 1:
        raise ValueError("draw_budget must be positive")
    rdBase.SeedRandomNumberGenerator(seed)
    sampled = {
        Chem.MolToSmiles(Chem.Mol(mol), rootedAtAtom=root_idx, **sampling_kwargs)
        for _ in range(draw_budget)
    }
    return tuple(sorted(sampled))


def sample_and_validate_rdkit_random(
    mol: Chem.Mol,
    policy: ReferencePolicy,
) -> RandomReferenceResult:
    target_identity = identity_smiles(mol, policy)
    sampled = sample_rdkit_random_smiles(mol, policy)

    issues: list[ValidationIssue] = []
    for sampled_smiles in sampled:
        parsed = Chem.MolFromSmiles(sampled_smiles)
        if parsed is None:
            issues.append(ValidationIssue(sampled_smiles=sampled_smiles, reason="failed to parse"))
            continue

        parsed_identity = identity_smiles(parsed, policy)
        if parsed_identity != target_identity:
            issues.append(
                ValidationIssue(
                    sampled_smiles=sampled_smiles,
                    reason=parsed_identity,
                )
            )

    return RandomReferenceResult(
        sampled_smiles=sampled,
        validation_issues=tuple(issues),
    )
