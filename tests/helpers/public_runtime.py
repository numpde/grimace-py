"""Shared helpers for public runtime integration tests."""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from collections.abc import Sequence

import grimace
from grimace import _runtime
from grimace._reference.prepared_graph import prepare_smiles_graph_from_mol_to_smiles_kwargs


def supported_public_kwargs(**kwargs: object) -> dict[str, object]:
    return {
        "canonical": False,
        "doRandom": True,
        **kwargs,
    }


def runtime_flags_from_public_kwargs(**kwargs: object) -> _runtime.MolToSmilesFlags:
    return _runtime._make_flags(
        isomeric_smiles=kwargs.get("isomericSmiles", True),
        kekule_smiles=kwargs.get("kekuleSmiles", False),
        rooted_at_atom=kwargs.get("rootedAtAtom", -1),
        canonical=kwargs.get("canonical", True),
        all_bonds_explicit=kwargs.get("allBondsExplicit", False),
        all_hs_explicit=kwargs.get("allHsExplicit", False),
        do_random=kwargs.get("doRandom", False),
        ignore_atom_map_numbers=kwargs.get("ignoreAtomMapNumbers", False),
    )


def public_enum_support(mol: object, **kwargs: object) -> frozenset[str]:
    return frozenset(grimace.MolToSmilesEnum(mol, **kwargs))


def public_token_inventory(mol: object, **kwargs: object) -> tuple[str, ...]:
    return grimace.MolToSmilesTokenInventory(mol, **kwargs)


def make_decoder(mol: object, **kwargs: object) -> grimace.MolToSmilesDecoder:
    return grimace.MolToSmilesDecoder(mol, **kwargs)


def make_determinized_decoder(
    mol: object,
    **kwargs: object,
) -> grimace.MolToSmilesDeterminizedDecoder:
    return grimace.MolToSmilesDeterminizedDecoder(mol, **kwargs)


def public_entrypoint_calls(
    mol: object,
    **kwargs: object,
) -> tuple[tuple[str, object], ...]:
    return (
        ("enum", lambda: tuple(grimace.MolToSmilesEnum(mol, **kwargs))),
        ("decoder", lambda: make_decoder(mol, **kwargs)),
        ("determinized_decoder", lambda: make_determinized_decoder(mol, **kwargs)),
        ("inventory", lambda: public_token_inventory(mol, **kwargs)),
    )


def choice_texts(decoder: object) -> tuple[str, ...]:
    return tuple(choice.text for choice in decoder.next_choices)


def unique_choice_texts(decoder: object) -> tuple[str, ...]:
    return tuple(sorted(set(choice_texts(decoder))))


def reachable_outputs_from_decoder(
    decoder: object,
    *,
    memo: dict[object, frozenset[str]] | None = None,
) -> frozenset[str]:
    return _runtime._reachable_terminal_prefixes(decoder._impl._state, memo=memo)


def exact_token_inventory_via_decoder(mol: object, **kwargs: object) -> tuple[str, ...]:
    decoder = make_decoder(mol, **kwargs)
    seen_state_keys: set[object] = set()
    stack = [decoder]
    inventory: set[str] = set()

    while stack:
        current = stack.pop()
        state_key = _runtime._state_cache_key(current._impl._state)
        if state_key in seen_state_keys:
            continue
        seen_state_keys.add(state_key)

        for choice in current.next_choices:
            inventory.add(choice.text)
            stack.append(choice.next_state)

    return tuple(sorted(inventory))


def public_enum_support_union_over_explicit_roots(
    mol: object,
    **kwargs: object,
) -> frozenset[str]:
    rooted_kwargs = dict(kwargs)
    rooted_kwargs.pop("rootedAtAtom", None)
    all_outputs: set[str] = set()

    for root_idx in range(mol.GetNumAtoms()):
        all_outputs.update(
            public_enum_support(
                mol,
                **supported_public_kwargs(rootedAtAtom=root_idx, **rooted_kwargs),
            )
        )

    return frozenset(all_outputs)


def public_token_inventory_union_over_explicit_roots(
    mol: object,
    **kwargs: object,
) -> tuple[str, ...]:
    rooted_kwargs = dict(kwargs)
    rooted_kwargs.pop("rootedAtAtom", None)
    inventory: set[str] = set()

    for root_idx in range(mol.GetNumAtoms()):
        inventory.update(
            public_token_inventory(
                mol,
                **supported_public_kwargs(rootedAtAtom=root_idx, **rooted_kwargs),
            )
        )

    return tuple(sorted(inventory))


def prepared_input_variants(
    mol: object,
    **kwargs: object,
) -> tuple[tuple[str, object], ...]:
    flags = runtime_flags_from_public_kwargs(**kwargs)
    surface_kind = _runtime._runtime_surface_kind(mol, flags=flags)
    reference_prepared = prepare_smiles_graph_from_mol_to_smiles_kwargs(
        mol,
        surface_kind=surface_kind,
        isomeric_smiles=bool(flags.isomeric_smiles),
        kekule_smiles=bool(flags.kekule_smiles),
        all_bonds_explicit=bool(flags.all_bonds_explicit),
        all_hs_explicit=bool(flags.all_hs_explicit),
        ignore_atom_map_numbers=bool(flags.ignore_atom_map_numbers),
    )
    core_prepared = _runtime.prepare_smiles_graph(reference_prepared, flags=flags)
    return (
        ("mol", mol),
        ("reference_prepared", reference_prepared),
        ("core_prepared", core_prepared),
    )


def assert_public_entrypoints_equivalent(
    test_case: unittest.TestCase,
    mol: object,
    *,
    provided_kwargs: Mapping[str, object],
    expected_kwargs: Mapping[str, object],
) -> None:
    test_case.assertEqual(
        public_enum_support(mol, **provided_kwargs),
        public_enum_support(mol, **expected_kwargs),
    )
    test_case.assertEqual(
        public_token_inventory(mol, **provided_kwargs),
        public_token_inventory(mol, **expected_kwargs),
    )

    decoder = make_decoder(mol, **provided_kwargs)
    expected_decoder = make_decoder(mol, **expected_kwargs)
    test_case.assertEqual(decoder.prefix, expected_decoder.prefix)
    test_case.assertEqual(choice_texts(decoder), choice_texts(expected_decoder))
    test_case.assertEqual(
        reachable_outputs_from_decoder(decoder),
        reachable_outputs_from_decoder(expected_decoder),
    )

    determinized = make_determinized_decoder(mol, **provided_kwargs)
    expected_determinized = make_determinized_decoder(mol, **expected_kwargs)
    test_case.assertEqual(determinized.prefix, expected_determinized.prefix)
    test_case.assertEqual(
        choice_texts(determinized),
        choice_texts(expected_determinized),
    )
    test_case.assertEqual(
        reachable_outputs_from_decoder(determinized),
        reachable_outputs_from_decoder(expected_determinized),
    )


def assert_public_entrypoints_raise(
    test_case: unittest.TestCase,
    mol: object,
    *,
    kwargs: Mapping[str, object],
    expected_exception: type[BaseException],
    expected_regex: str,
    included_entrypoints: Sequence[str] | None = None,
) -> None:
    allowed = None if included_entrypoints is None else set(included_entrypoints)
    for entrypoint_name, call in public_entrypoint_calls(mol, **kwargs):
        if allowed is not None and entrypoint_name not in allowed:
            continue
        with test_case.subTest(entrypoint=entrypoint_name):
            with test_case.assertRaisesRegex(expected_exception, expected_regex):
                call()
