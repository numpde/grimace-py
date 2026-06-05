"""Shared helpers for public runtime integration tests."""

from __future__ import annotations

import unittest
from collections.abc import Mapping
from collections.abc import Sequence

import grimace
import grimace._runtime_graphs as _runtime_graphs
import grimace._runtime_states as _runtime_states
from grimace._mol_to_smiles_options import (
    MOL_TO_SMILES_OPTIONS,
    MOL_TO_SMILES_PREPARED_OPTIONS,
    coerce_public_options,
)
from grimace._reference.prepared_graph import prepare_smiles_graph_from_mol_to_smiles_kwargs
from grimace._runtime_inputs import MolToSmilesFlags, make_flags


def supported_public_kwargs(**kwargs: object) -> dict[str, object]:
    return {
        "canonical": False,
        "doRandom": True,
        **kwargs,
    }


def runtime_flags_from_public_kwargs(**kwargs: object) -> MolToSmilesFlags:
    return make_flags(
        **coerce_public_options(
            MOL_TO_SMILES_OPTIONS,
            kwargs,
            context="Test MolToSmiles",
        )
    )


def prepared_writer_kwargs(kwargs: Mapping[str, object]) -> dict[str, object]:
    return {
        spec.public_name: kwargs[spec.public_name]
        for spec in MOL_TO_SMILES_PREPARED_OPTIONS
        if spec.public_name in kwargs
    }


def public_enum_support(mol: object, **kwargs: object) -> frozenset[str]:
    return frozenset(grimace.MolToSmilesEnum(mol, **kwargs))


def public_token_inventory(mol: object, **kwargs: object) -> tuple[str, ...]:
    return grimace.MolToSmilesTokenInventory(mol, **kwargs)


def public_token_inventory_superset(mol: object, **kwargs: object) -> tuple[str, ...]:
    return grimace.MolToSmilesTokenInventorySuperset(mol, **kwargs)


def public_sample(mol: object, **kwargs: object) -> grimace.SmilesSample:
    return grimace.MolToSmilesSample(mol, seed=0, **kwargs)


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
    # Candidate-taking APIs, such as MolToSmilesDeviation, are covered by their
    # own candidate-specific tests plus the shared public-signature contract.
    return (
        ("enum", lambda: tuple(grimace.MolToSmilesEnum(mol, **kwargs))),
        ("decoder", lambda: make_decoder(mol, **kwargs)),
        ("determinized_decoder", lambda: make_determinized_decoder(mol, **kwargs)),
        ("inventory", lambda: public_token_inventory(mol, **kwargs)),
        ("inventory_superset", lambda: public_token_inventory_superset(mol, **kwargs)),
        ("sample", lambda: public_sample(mol, **kwargs)),
    )


def choice_texts(decoder: object) -> tuple[str, ...]:
    return tuple(choice.text for choice in decoder.next_choices)


def unique_choice_texts(decoder: object) -> tuple[str, ...]:
    return tuple(sorted(set(choice_texts(decoder))))


def runtime_state_cache_key(state: object) -> object:
    return _runtime_states._state_cache_key(state)


def runtime_realized_branch_transitions(state: object) -> tuple[tuple[str, object], ...]:
    return _runtime_states._realize_state_transitions(
        state._branch_state_transitions()
    )


def runtime_realized_token_transitions(state: object) -> tuple[tuple[str, object], ...]:
    return _runtime_states._realize_state_transitions(
        state._token_state_transitions()
    )


def _runtime_transition_counts(
    transitions: _runtime_states._StateTransitions,
) -> tuple[tuple[str, int], ...]:
    return tuple(
        (transition.text, transition.branch_count)
        for transition in transitions
    )


def runtime_branch_transition_counts(state: object) -> tuple[tuple[str, int], ...]:
    return _runtime_transition_counts(state._branch_state_transitions())


def runtime_token_transition_counts(state: object) -> tuple[tuple[str, int], ...]:
    return _runtime_transition_counts(state._token_state_transitions())


def reachable_terminal_prefixes(
    state: object,
    *,
    memo: dict[object, frozenset[str]] | None = None,
) -> frozenset[str]:
    if memo is None:
        memo = {}

    key = runtime_state_cache_key(state)
    cached = memo.get(key)
    if cached is not None:
        return cached

    outputs: set[str] = set()
    if state.is_terminal():
        outputs.add(state.prefix())

    for _, successor in runtime_realized_branch_transitions(state):
        outputs.update(reachable_terminal_prefixes(successor, memo=memo))
    resolved = frozenset(outputs)
    memo[key] = resolved
    return resolved


def reachable_outputs_from_decoder(
    decoder: object,
    *,
    memo: dict[object, frozenset[str]] | None = None,
) -> frozenset[str]:
    return reachable_terminal_prefixes(decoder._state, memo=memo)


def exact_token_inventory_via_decoder(mol: object, **kwargs: object) -> tuple[str, ...]:
    decoder = make_decoder(mol, **kwargs)
    seen_state_keys: set[object] = set()
    stack = [decoder]
    inventory: set[str] = set()

    while stack:
        current = stack.pop()
        state_key = runtime_state_cache_key(current._state)
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
    surface_kind = _runtime_graphs.runtime_surface_kind(mol, flags=flags)
    reference_prepared = prepare_smiles_graph_from_mol_to_smiles_kwargs(
        mol,
        surface_kind=surface_kind,
        isomeric_smiles=bool(flags.isomeric_smiles),
        kekule_smiles=bool(flags.kekule_smiles),
        all_bonds_explicit=bool(flags.all_bonds_explicit),
        all_hs_explicit=bool(flags.all_hs_explicit),
        ignore_atom_map_numbers=bool(flags.ignore_atom_map_numbers),
    )
    core_prepared = _runtime_graphs.prepare_smiles_graph(reference_prepared, flags=flags)
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
    test_case.assertEqual(
        public_token_inventory_superset(mol, **provided_kwargs),
        public_token_inventory_superset(mol, **expected_kwargs),
    )
    test_case.assertEqual(
        public_sample(mol, **provided_kwargs),
        public_sample(mol, **expected_kwargs),
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
