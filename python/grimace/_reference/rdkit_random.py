from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rdkit import Chem, rdBase

from grimace._mol_to_smiles_options import (
    MOL_TO_SMILES_OPTIONS,
)
from grimace._reference.policy import ReferencePolicy
from grimace._reference.policy_sections import identity_section, sampling_section


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


def _sampling_kwargs(policy: ReferencePolicy) -> tuple[int, int, int, dict[str, Any]]:
    sampling = sampling_section(policy)
    kwargs = {
        spec.public_name: sampling[spec.public_name]
        for spec in MOL_TO_SMILES_OPTIONS
        if spec.public_name != "rootedAtAtom"
    }
    return (
        int(sampling["seed"]),
        int(sampling["draw_budget"]),
        int(sampling["rootedAtAtom"]),
        kwargs,
    )


def _identity_kwargs(policy: ReferencePolicy) -> dict[str, Any]:
    identity = identity_section(policy)
    if not identity["parse_with_rdkit"]:
        raise NotImplementedError("Only parse_with_rdkit=true is supported")
    return {
        spec.public_name: identity[spec.public_name]
        for spec in MOL_TO_SMILES_OPTIONS
    }


def mol_to_identity_smiles(
    mol: Chem.Mol,
    *,
    isomericSmiles: bool,
    kekuleSmiles: bool,
    rootedAtAtom: int,
    canonical: bool,
    allBondsExplicit: bool,
    allHsExplicit: bool,
    doRandom: bool,
    ignoreAtomMapNumbers: bool,
) -> str:
    working_mol = Chem.Mol(mol)
    if ignoreAtomMapNumbers:
        for atom in working_mol.GetAtoms():
            atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(
        working_mol,
        isomericSmiles=isomericSmiles,
        kekuleSmiles=kekuleSmiles,
        rootedAtAtom=rootedAtAtom,
        canonical=canonical,
        allBondsExplicit=allBondsExplicit,
        allHsExplicit=allHsExplicit,
        doRandom=doRandom,
        ignoreAtomMapNumbers=ignoreAtomMapNumbers,
    )


def identity_smiles(mol: Chem.Mol, policy: ReferencePolicy) -> str:
    return mol_to_identity_smiles(Chem.Mol(mol), **_identity_kwargs(policy))


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
