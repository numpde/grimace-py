from __future__ import annotations

from rdkit import Chem, rdBase

import grimace


def grimace_support(
    mol: Chem.Mol,
    *,
    rooted_at_atom: int | None,
    isomeric_smiles: bool,
) -> set[str]:
    kwargs = dict(
        isomericSmiles=isomeric_smiles,
        canonical=False,
        doRandom=True,
    )
    if rooted_at_atom is not None:
        kwargs["rootedAtAtom"] = rooted_at_atom
    return set(grimace.MolToSmilesEnum(mol, **kwargs))


def sample_rdkit_random_support(
    mol: Chem.Mol,
    *,
    root_idx: int | None,
    isomeric_smiles: bool,
    draw_budget: int,
    seed: int = 12345,
) -> set[str]:
    rdBase.SeedRandomNumberGenerator(seed)
    kwargs = dict(
        isomericSmiles=isomeric_smiles,
        canonical=False,
        doRandom=True,
    )
    if root_idx is not None:
        kwargs["rootedAtAtom"] = root_idx
    return {Chem.MolToSmiles(Chem.Mol(mol), **kwargs) for _ in range(draw_budget)}
