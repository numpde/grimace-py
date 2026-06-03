from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass
import unittest

import grimace
import grimace._runtime_states as _runtime_states
from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import supported_public_kwargs


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
            _runtime_states._reachable_terminal_prefixes(initial_state, memo=memo),
        )
        seen_state_keys: set[object] = set()
        stack = [initial_state]
        audited_state_count = 0

        while stack:
            state = stack.pop()
            state_key = _runtime_states._state_cache_key(state)
            if state_key in seen_state_keys:
                continue
            seen_state_keys.add(state_key)
            audited_state_count += 1

            reachable = _runtime_states._reachable_terminal_prefixes(
                state,
                memo=memo,
            )
            prefix = state.prefix()
            successor_states = successor_fn(state)

            self.assertTrue(reachable)
            self.assertTrue(reachable <= outputs)
            self.assertTrue(all(output.startswith(prefix) for output in reachable))

            if state.is_terminal():
                self.assertEqual((), successor_states)
                self.assertEqual(frozenset({prefix}), reachable)
                continue

            self.assertTrue(successor_states)
            if require_unique_choice_texts:
                option_texts = tuple(text for text, _ in successor_states)
                self.assertEqual(len(set(option_texts)), len(option_texts))

            union_of_branch_outputs: set[str] = set()
            for _, successor in successor_states:
                branch_outputs = _runtime_states._reachable_terminal_prefixes(
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
        )

        for case in cases:
            mol = parse_smiles(case.smiles)
            with self.subTest(case=case.name, smiles=case.smiles):
                outputs = frozenset(grimace.MolToSmilesEnum(mol, **case.kwargs))
                decoder = grimace.MolToSmilesDeterminizedDecoder(mol, **case.kwargs)
                self._assert_state_graph_matches_outputs(
                    initial_state=decoder._state,
                    outputs=outputs,
                    successor_fn=_runtime_states._grouped_successor_states,
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
        )

        for case in cases:
            mol = parse_smiles(case.smiles)
            with self.subTest(case=case.name, smiles=case.smiles):
                outputs = frozenset(grimace.MolToSmilesEnum(mol, **case.kwargs))
                decoder = grimace.MolToSmilesDecoder(mol, **case.kwargs)
                self._assert_state_graph_matches_outputs(
                    initial_state=decoder._state,
                    outputs=outputs,
                    successor_fn=_runtime_states._choice_successor_states,
                )


if __name__ == "__main__":
    unittest.main()
