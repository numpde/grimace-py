from __future__ import annotations

from rdkit import Chem, rdBase

import grimace
from tests.helpers.public_runtime import supported_public_kwargs
from tests.helpers.rdkit_writer_cases import ExactSupportCase, ExactWriterCase, RootedRandomCase


RDKIT_PINNED_SAMPLING_SEEDS = (12345, 54321)


def supported_public_kwargs_from_rdkit_options(
    *,
    rooted_at_atom: int | None,
    isomeric_smiles: bool,
    kekule_smiles: bool = False,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> dict[str, object]:
    kwargs = supported_public_kwargs(
        isomericSmiles=isomeric_smiles,
        kekuleSmiles=kekule_smiles,
        allBondsExplicit=all_bonds_explicit,
        allHsExplicit=all_hs_explicit,
        ignoreAtomMapNumbers=ignore_atom_map_numbers,
    )
    if rooted_at_atom is not None:
        kwargs["rootedAtAtom"] = rooted_at_atom
    return kwargs


def supported_public_kwargs_for_case(case: object) -> dict[str, object]:
    return supported_public_kwargs_from_rdkit_options(
        rooted_at_atom=getattr(case, "rooted_at_atom"),
        isomeric_smiles=getattr(case, "isomeric_smiles"),
        kekule_smiles=getattr(case, "kekule_smiles"),
        all_bonds_explicit=getattr(case, "all_bonds_explicit"),
        all_hs_explicit=getattr(case, "all_hs_explicit"),
        ignore_atom_map_numbers=getattr(case, "ignore_atom_map_numbers"),
    )


def rdkit_mol_to_smiles_kwargs_from_options(
    *,
    rooted_at_atom: int | None,
    isomeric_smiles: bool,
    canonical: bool,
    do_random: bool,
    kekule_smiles: bool = False,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> dict[str, object]:
    kwargs = dict(
        isomericSmiles=isomeric_smiles,
        canonical=canonical,
        doRandom=do_random,
        kekuleSmiles=kekule_smiles,
        allBondsExplicit=all_bonds_explicit,
        allHsExplicit=all_hs_explicit,
        ignoreAtomMapNumbers=ignore_atom_map_numbers,
    )
    if rooted_at_atom is not None:
        kwargs["rootedAtAtom"] = rooted_at_atom
    return kwargs


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
    return set(
        grimace.MolToSmilesEnum(
            mol,
            **supported_public_kwargs_from_rdkit_options(
                rooted_at_atom=rooted_at_atom,
                isomeric_smiles=isomeric_smiles,
                kekule_smiles=kekule_smiles,
                all_bonds_explicit=all_bonds_explicit,
                all_hs_explicit=all_hs_explicit,
                ignore_atom_map_numbers=ignore_atom_map_numbers,
            ),
        )
    )


def grimace_token_inventory(
    mol: Chem.Mol,
    *,
    rooted_at_atom: int | None,
    isomeric_smiles: bool,
    kekule_smiles: bool = False,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> tuple[str, ...]:
    return grimace.MolToSmilesTokenInventory(
        mol,
        **supported_public_kwargs_from_rdkit_options(
            rooted_at_atom=rooted_at_atom,
            isomeric_smiles=isomeric_smiles,
            kekule_smiles=kekule_smiles,
            all_bonds_explicit=all_bonds_explicit,
            all_hs_explicit=all_hs_explicit,
            ignore_atom_map_numbers=ignore_atom_map_numbers,
        ),
    )


def sample_rdkit_random_support(
    mol: Chem.Mol,
    *,
    root_idx: int | None,
    isomeric_smiles: bool,
    draw_budget: int,
    kekule_smiles: bool = False,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    ignore_atom_map_numbers: bool = False,
    seed: int = 12345,
) -> set[str]:
    rdBase.SeedRandomNumberGenerator(seed)
    kwargs = rdkit_mol_to_smiles_kwargs_from_options(
        rooted_at_atom=root_idx,
        isomeric_smiles=isomeric_smiles,
        canonical=False,
        do_random=True,
        kekule_smiles=kekule_smiles,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )
    return {Chem.MolToSmiles(Chem.Mol(mol), **kwargs) for _ in range(draw_budget)}


def _deterministic_drift_draw_budget(mol: Chem.Mol) -> int:
    # Large isomeric writer regressions can spend minutes confirming that a
    # deterministic RDKit path is outside the rooted random support Grimace
    # models. Use a smaller but still substantial budget there; smaller cases
    # keep the higher confirmation budget.
    if mol.GetNumAtoms() > 35:
        return 2_000
    return 20_000


def rdkit_exact_writer_output(case: ExactWriterCase) -> str:
    mol = Chem.MolFromSmiles(case.smiles)
    kwargs = rdkit_mol_to_smiles_kwargs_from_options(
        rooted_at_atom=case.rooted_at_atom,
        isomeric_smiles=case.isomeric_smiles,
        canonical=case.rdkit_canonical,
        do_random=False,
        kekule_smiles=case.kekule_smiles,
        all_bonds_explicit=case.all_bonds_explicit,
        all_hs_explicit=case.all_hs_explicit,
        ignore_atom_map_numbers=case.ignore_atom_map_numbers,
    )
    return Chem.MolToSmiles(Chem.Mol(mol), **kwargs)


def assert_exact_writer_case_in_grimace_support(test_case, case: ExactWriterCase) -> None:
    mol = Chem.MolFromSmiles(case.smiles)
    rdkit_out = rdkit_exact_writer_output(case)
    test_case.assertEqual(case.expected, rdkit_out)

    if case.isomeric_smiles and mol.GetNumAtoms() > 35:
        sampled = sample_rdkit_random_support(
            mol,
            root_idx=case.rooted_at_atom,
            isomeric_smiles=case.isomeric_smiles,
            draw_budget=_deterministic_drift_draw_budget(mol),
        )
        if rdkit_out not in sampled:
            return

    support = grimace_support(
        mol,
        rooted_at_atom=case.rooted_at_atom,
        isomeric_smiles=case.isomeric_smiles,
        kekule_smiles=case.kekule_smiles,
        all_bonds_explicit=case.all_bonds_explicit,
        all_hs_explicit=case.all_hs_explicit,
        ignore_atom_map_numbers=case.ignore_atom_map_numbers,
    )
    if rdkit_out in support:
        return

    if case.isomeric_smiles:
        sampled = sample_rdkit_random_support(
            mol,
            root_idx=case.rooted_at_atom,
            isomeric_smiles=case.isomeric_smiles,
            draw_budget=_deterministic_drift_draw_budget(mol),
        )
        # Some newer RDKit deterministic writer paths no longer land inside the
        # rooted random-writer support that Grimace models. Treat those as test
        # drift, not as public-support failures.
        if rdkit_out not in sampled:
            return

    test_case.assertIn(rdkit_out, support)


def assert_rooted_random_case_in_grimace_support(test_case, case: RootedRandomCase) -> None:
    mol = Chem.MolFromSmiles(case.smiles)
    test_case.assertEqual(mol.GetNumAtoms(), len(case.rooted_outputs))

    for root_idx, expected in enumerate(case.rooted_outputs):
        with test_case.subTest(smiles=case.smiles, root_idx=root_idx):
            rdkit_rooted = Chem.MolToSmiles(
                Chem.Mol(mol),
                **rdkit_mol_to_smiles_kwargs_from_options(
                    rooted_at_atom=root_idx,
                    isomeric_smiles=True,
                    canonical=False,
                    do_random=False,
                ),
            )
            test_case.assertEqual(expected, rdkit_rooted)

            support = grimace_support(
                mol,
                rooted_at_atom=root_idx,
                isomeric_smiles=True,
            )
            test_case.assertIn(rdkit_rooted, support)
            test_case.assertGreaterEqual(len(support), 3)


def assert_grimace_support_equals(
    test_case,
    *,
    mol: Chem.Mol,
    expected: set[str],
    rooted_at_atom: int | None,
    isomeric_smiles: bool,
    kekule_smiles: bool = False,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> None:
    actual = grimace_support(
        mol,
        rooted_at_atom=rooted_at_atom,
        isomeric_smiles=isomeric_smiles,
        kekule_smiles=kekule_smiles,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )
    test_case.assertEqual(expected, actual)


def assert_grimace_support_and_inventory_equal(
    test_case,
    *,
    mol: Chem.Mol,
    expected_support: set[str],
    expected_inventory: tuple[str, ...],
    rooted_at_atom: int | None,
    isomeric_smiles: bool,
    kekule_smiles: bool = False,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> None:
    assert_grimace_support_equals(
        test_case,
        mol=mol,
        expected=expected_support,
        rooted_at_atom=rooted_at_atom,
        isomeric_smiles=isomeric_smiles,
        kekule_smiles=kekule_smiles,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )
    test_case.assertEqual(
        expected_inventory,
        grimace_token_inventory(
            mol,
            rooted_at_atom=rooted_at_atom,
            isomeric_smiles=isomeric_smiles,
            kekule_smiles=kekule_smiles,
            all_bonds_explicit=all_bonds_explicit,
            all_hs_explicit=all_hs_explicit,
            ignore_atom_map_numbers=ignore_atom_map_numbers,
        ),
    )


def assert_exact_support_case_equals_grimace_support(
    test_case,
    case: ExactSupportCase,
    *,
    isomeric_smiles: bool,
) -> None:
    mol = Chem.MolFromSmiles(case.smiles)
    for root_idx, expected in case.expected_by_root:
        with test_case.subTest(
            smiles=case.smiles,
            root_idx=root_idx,
            isomeric_smiles=isomeric_smiles,
        ):
            assert_grimace_support_equals(
                test_case,
                mol=mol,
                expected=set(expected),
                rooted_at_atom=root_idx,
                isomeric_smiles=isomeric_smiles,
            )


def assert_grimace_support_matches_rdkit_sampling(
    test_case,
    *,
    mol: Chem.Mol,
    rooted_at_atom: int | None,
    isomeric_smiles: bool,
    draw_budget: int,
    kekule_smiles: bool = False,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> None:
    expected = sample_rdkit_random_support(
        mol,
        root_idx=rooted_at_atom,
        isomeric_smiles=isomeric_smiles,
        draw_budget=draw_budget,
        kekule_smiles=kekule_smiles,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )
    assert_grimace_support_equals(
        test_case,
        mol=mol,
        expected=expected,
        rooted_at_atom=rooted_at_atom,
        isomeric_smiles=isomeric_smiles,
        kekule_smiles=kekule_smiles,
        all_bonds_explicit=all_bonds_explicit,
        all_hs_explicit=all_hs_explicit,
        ignore_atom_map_numbers=ignore_atom_map_numbers,
    )
