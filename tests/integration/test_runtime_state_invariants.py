from __future__ import annotations

from collections.abc import Callable, Mapping
from collections import Counter
from dataclasses import dataclass
import unittest

import grimace
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import (
    reachable_terminal_prefixes,
    runtime_branch_transition_counts,
    runtime_realized_branch_transitions,
    runtime_realized_token_transitions,
    runtime_state_cache_key,
    runtime_token_transition_counts,
    supported_public_kwargs,
)


@dataclass(frozen=True, slots=True)
class RuntimeStateAuditCase:
    name: str
    smiles: str
    kwargs: Mapping[str, object]


def _audit_case(
    name: str,
    smiles: str,
    **kwargs: object,
) -> RuntimeStateAuditCase:
    return RuntimeStateAuditCase(
        name=name,
        smiles=smiles,
        kwargs=supported_public_kwargs(**kwargs),
    )


class RuntimeStateInvariantTests(unittest.TestCase):
    """Internal runtime-state invariants for the current decoder adapter model."""

    def _assert_state_graph_matches_outputs(
        self,
        *,
        initial_state: object,
        outputs: frozenset[str],
        successor_fn: Callable[[object], tuple[tuple[str, object], ...]],
        require_unique_choice_texts: bool = False,
    ) -> None:
        memo: dict[object, frozenset[str]] = {}
        self.assertEqual(
            outputs,
            reachable_terminal_prefixes(initial_state, memo=memo),
        )
        seen_state_keys: set[object] = set()
        stack = [initial_state]
        audited_state_count = 0

        while stack:
            state = stack.pop()
            state_key = runtime_state_cache_key(state)
            if state_key in seen_state_keys:
                continue
            seen_state_keys.add(state_key)
            audited_state_count += 1

            reachable = reachable_terminal_prefixes(
                state,
                memo=memo,
            )
            prefix = state.prefix()
            successor_states = successor_fn(state)

            self.assertTrue(reachable)
            self.assertTrue(reachable <= outputs)
            self.assertTrue(all(output.startswith(prefix) for output in reachable))

            if state.is_terminal():
                self.assertIn(prefix, reachable)

            if not successor_states:
                self.assertTrue(state.is_terminal())
                self.assertEqual(frozenset({prefix}), reachable)
                continue

            if require_unique_choice_texts:
                option_texts = tuple(text for text, _ in successor_states)
                self.assertEqual(len(set(option_texts)), len(option_texts))

            union_of_branch_outputs: set[str] = {prefix} if state.is_terminal() else set()
            for _, successor in successor_states:
                branch_outputs = reachable_terminal_prefixes(
                    successor,
                    memo=memo,
                )
                self.assertTrue(branch_outputs)
                self.assertTrue(branch_outputs <= reachable)
                self.assertTrue(
                    all(output.startswith(successor.prefix()) for output in branch_outputs)
                )
                union_of_branch_outputs.update(branch_outputs)
                stack.append(successor)

            self.assertEqual(reachable, frozenset(union_of_branch_outputs))

        self.assertGreater(audited_state_count, 0)

    def test_branch_transitions_count_each_exposed_branch_once(self) -> None:
        decoder = grimace.MolToSmilesDecoder(
            parse_smiles("CCO"),
            **supported_public_kwargs(isomericSmiles=False, rootedAtAtom=-1),
        )

        self.assertEqual(
            (("C", 1), ("C", 1), ("O", 1)),
            runtime_branch_transition_counts(decoder._state),
        )

    def test_token_transitions_sum_hidden_branch_counts(self) -> None:
        decoder = grimace.MolToSmilesDeterminizedDecoder(
            parse_smiles("CCO"),
            **supported_public_kwargs(isomericSmiles=False, rootedAtAtom=-1),
        )

        self.assertEqual(
            (("C", 2), ("O", 1)),
            runtime_token_transition_counts(decoder._state),
        )

    def test_lazy_all_roots_stereo_token_transitions_sum_hidden_branches(self) -> None:
        decoder = grimace.MolToSmilesDeterminizedDecoder(
            parse_smiles("F[C@H](Cl)Br"),
            **supported_public_kwargs(isomericSmiles=True, rootedAtAtom=-1),
        )

        self.assertEqual(
            (
                ("F", 1),
                ("[C@@H]", 3),
                ("[C@H]", 3),
                ("Cl", 1),
                ("Br", 1),
            ),
            runtime_token_transition_counts(decoder._state),
        )

    def test_disconnected_separator_transition_has_one_branch(self) -> None:
        decoder = grimace.MolToSmilesDeterminizedDecoder(
            parse_smiles("O.CCO"),
            **supported_public_kwargs(isomericSmiles=True, rootedAtAtom=-1),
        )
        oxygen_state = next(
            choice.next_state
            for choice in decoder.next_choices
            if choice.text == "O"
        )

        self.assertEqual(
            ((".", 1),),
            runtime_token_transition_counts(oxygen_state._state),
        )

    def test_token_branch_counts_match_branch_transition_text_counts(self) -> None:
        cases = (
            _audit_case("rooted_nonstereo", "CCO", rootedAtAtom=0, isomericSmiles=False),
            _audit_case("all_roots_nonstereo", "CCO", rootedAtAtom=-1, isomericSmiles=False),
            _audit_case("rooted_stereo", "F[C@H](Cl)Br", rootedAtAtom=0, isomericSmiles=True),
            _audit_case("all_roots_stereo", "F[C@H](Cl)Br", rootedAtAtom=-1, isomericSmiles=True),
            _audit_case(
                "disconnected",
                "F[C@H](Cl)Br.O",
                rootedAtAtom=-1,
                isomericSmiles=True,
            ),
        )

        for case in cases:
            with self.subTest(case=case.name):
                decoder = grimace.MolToSmilesDeterminizedDecoder(
                    parse_smiles(case.smiles),
                    **case.kwargs,
                )
                seen_state_keys: set[object] = set()
                stack = [decoder._state]

                while stack:
                    state = stack.pop()
                    state_key = runtime_state_cache_key(state)
                    if state_key in seen_state_keys:
                        continue
                    seen_state_keys.add(state_key)

                    branch_counts = Counter(
                        text for text, _ in runtime_realized_branch_transitions(state)
                    )
                    token_counts = dict(runtime_token_transition_counts(state))
                    self.assertEqual(dict(branch_counts), token_counts)

                    stack.extend(
                        successor
                        for _, successor in runtime_realized_branch_transitions(state)
                    )
                    stack.extend(
                        successor
                        for _, successor in runtime_realized_token_transitions(state)
                    )

    def test_determinized_decoder_state_audit_covers_all_reachable_states(self) -> None:
        cases = (
            _audit_case("rooted_nonstereo", "CCO", rootedAtAtom=0, isomericSmiles=False),
            _audit_case("rooted_stereo", "F[C@H](Cl)Br", rootedAtAtom=0, isomericSmiles=True),
            _audit_case(
                "unrooted_stereo",
                "F[C@H](Cl)Br",
                rootedAtAtom=-1,
                isomericSmiles=True,
            ),
            _audit_case("disconnected_rooted", "[Na+].C#N", rootedAtAtom=0, isomericSmiles=False),
            _audit_case(
                "disconnected_unrooted_stereo",
                "F[C@H](Cl)Br.O",
                rootedAtAtom=-1,
                isomericSmiles=True,
            ),
            _audit_case(
                "duplicate_same_text_connected",
                "C1CCC2=NN=NN2CC1",
                rootedAtAtom=2,
                isomericSmiles=False,
            ),
            _audit_case(
                "merged_then_visible_divergence",
                "CC(=O)Oc1ccccc1C(=O)O",
                rootedAtAtom=9,
                isomericSmiles=False,
            ),
            _audit_case(
                "explicit_hydrogens",
                "CO",
                rootedAtAtom=0,
                isomericSmiles=False,
                allHsExplicit=True,
            ),
            _audit_case(
                "kekule",
                "c1ccccc1",
                rootedAtAtom=0,
                isomericSmiles=False,
                kekuleSmiles=True,
            ),
            _audit_case(
                "atom_maps",
                "[CH3:7][OH:8]",
                rootedAtAtom=0,
                isomericSmiles=False,
                ignoreAtomMapNumbers=False,
            ),
        )

        for case in cases:
            mol = parse_smiles(case.smiles)
            with self.subTest(case=case.name, smiles=case.smiles):
                outputs = frozenset(grimace.MolToSmilesEnum(mol, **case.kwargs))
                decoder = grimace.MolToSmilesDeterminizedDecoder(mol, **case.kwargs)
                self._assert_state_graph_matches_outputs(
                    initial_state=decoder._state,
                    outputs=outputs,
                    successor_fn=runtime_realized_token_transitions,
                    require_unique_choice_texts=True,
                )

    def test_decoder_state_audit_covers_all_reachable_states(self) -> None:
        cases = (
            _audit_case("rooted_nonstereo", "CCO", rootedAtAtom=0, isomericSmiles=False),
            _audit_case("rooted_stereo", "F[C@H](Cl)Br", rootedAtAtom=0, isomericSmiles=True),
            _audit_case(
                "unrooted_stereo",
                "F[C@H](Cl)Br",
                rootedAtAtom=-1,
                isomericSmiles=True,
            ),
            _audit_case(
                "nonisomeric_explicit_bond_dirs",
                "F/C=C\\Cl",
                rootedAtAtom=0,
                isomericSmiles=False,
                allBondsExplicit=True,
            ),
            _audit_case("unrooted_connected", "CCO", rootedAtAtom=-1, isomericSmiles=False),
            _audit_case("disconnected_rooted", "[Na+].CC", rootedAtAtom=0, isomericSmiles=False),
            _audit_case("disconnected_unrooted", "O.CCO", rootedAtAtom=-1, isomericSmiles=True),
            _audit_case(
                "disconnected_unrooted_stereo",
                "F[C@H](Cl)Br.O",
                rootedAtAtom=-1,
                isomericSmiles=True,
            ),
            _audit_case(
                "duplicate_same_text_connected",
                "C1CCC2=NN=NN2CC1",
                rootedAtAtom=2,
                isomericSmiles=False,
            ),
            _audit_case(
                "explicit_hydrogens",
                "CO",
                rootedAtAtom=0,
                isomericSmiles=False,
                allHsExplicit=True,
            ),
            _audit_case(
                "kekule",
                "c1ccccc1",
                rootedAtAtom=0,
                isomericSmiles=False,
                kekuleSmiles=True,
            ),
            _audit_case(
                "atom_maps",
                "[CH3:7][OH:8]",
                rootedAtAtom=0,
                isomericSmiles=False,
                ignoreAtomMapNumbers=False,
            ),
        )

        for case in cases:
            mol = parse_smiles(case.smiles)
            with self.subTest(case=case.name, smiles=case.smiles):
                outputs = frozenset(grimace.MolToSmilesEnum(mol, **case.kwargs))
                decoder = grimace.MolToSmilesDecoder(mol, **case.kwargs)
                self._assert_state_graph_matches_outputs(
                    initial_state=decoder._state,
                    outputs=outputs,
                    successor_fn=runtime_realized_branch_transitions,
                )


if __name__ == "__main__":
    unittest.main()
