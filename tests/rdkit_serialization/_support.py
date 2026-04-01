from __future__ import annotations

from rdkit import Chem, rdBase

import grimace
from tests.helpers.rdkit_writer_cases import ExactWriterCase


def grimace_support(
    mol: Chem.Mol,
    *,
    rooted_at_atom: int | None,
    isomeric_smiles: bool,
    kekule_smiles: bool = False,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> set[str]:
    kwargs = dict(
        isomericSmiles=isomeric_smiles,
        kekuleSmiles=kekule_smiles,
        canonical=False,
        allBondsExplicit=all_bonds_explicit,
        allHsExplicit=all_hs_explicit,
        doRandom=True,
        ignoreAtomMapNumbers=ignore_atom_map_numbers,
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


def rdkit_exact_writer_output(case: ExactWriterCase) -> str:
    mol = Chem.MolFromSmiles(case.smiles)
    kwargs = dict(
        isomericSmiles=case.isomeric_smiles,
        canonical=case.rdkit_canonical,
        doRandom=False,
        kekuleSmiles=case.kekule_smiles,
        allBondsExplicit=case.all_bonds_explicit,
        allHsExplicit=case.all_hs_explicit,
        ignoreAtomMapNumbers=case.ignore_atom_map_numbers,
    )
    if case.rooted_at_atom is not None:
        kwargs["rootedAtAtom"] = case.rooted_at_atom
    return Chem.MolToSmiles(Chem.Mol(mol), **kwargs)


def assert_exact_writer_case_in_grimace_support(test_case, case: ExactWriterCase) -> None:
    rdkit_out = rdkit_exact_writer_output(case)
    test_case.assertEqual(case.expected, rdkit_out)

    support = grimace_support(
        Chem.MolFromSmiles(case.smiles),
        rooted_at_atom=case.rooted_at_atom,
        isomeric_smiles=case.isomeric_smiles,
        kekule_smiles=case.kekule_smiles,
        all_bonds_explicit=case.all_bonds_explicit,
        all_hs_explicit=case.all_hs_explicit,
        ignore_atom_map_numbers=case.ignore_atom_map_numbers,
    )
    test_case.assertIn(case.expected, support)
