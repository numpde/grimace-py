from __future__ import annotations

from dataclasses import dataclass
import unittest

import grimace
import grimace._runtime_graphs as _runtime_graphs
import grimace._runtime_states as _runtime_states
from rdkit import Chem
from grimace._runtime_inputs import MolToSmilesFlags
from tests.helpers.assertions import assert_prefix_options_match_outputs
from tests.helpers.mols import parse_smiles


@dataclass(frozen=True, slots=True)
class RuntimeStateAuditCase:
    name: str
    smiles: str
    rooted_at_atom: int | None
    isomeric_smiles: bool = True
    kekule_smiles: bool = False
    all_bonds_explicit: bool = False
    all_hs_explicit: bool = False
    ignore_atom_map_numbers: bool = False


def _public_kwargs(case: RuntimeStateAuditCase) -> dict[str, object]:
    kwargs: dict[str, object] = {
        "isomericSmiles": case.isomeric_smiles,
        "kekuleSmiles": case.kekule_smiles,
        "canonical": False,
        "allBondsExplicit": case.all_bonds_explicit,
        "allHsExplicit": case.all_hs_explicit,
        "doRandom": True,
        "ignoreAtomMapNumbers": case.ignore_atom_map_numbers,
    }
    if case.rooted_at_atom is not None:
        kwargs["rootedAtAtom"] = case.rooted_at_atom
    return kwargs


def _atom_tokens(case: RuntimeStateAuditCase) -> tuple[str, ...]:
    mol = parse_smiles(case.smiles)
    root = 0 if case.rooted_at_atom is None else case.rooted_at_atom
    flags = MolToSmilesFlags(
        isomeric_smiles=case.isomeric_smiles,
        kekule_smiles=case.kekule_smiles,
        rooted_at_atom=root,
        canonical=False,
        all_bonds_explicit=case.all_bonds_explicit,
        all_hs_explicit=case.all_hs_explicit,
        do_random=True,
        ignore_atom_map_numbers=case.ignore_atom_map_numbers,
    )
    if len(Chem.GetMolFrags(mol)) == 1:
        prepared = _runtime_graphs.prepare_smiles_graph(mol, flags=flags)
        return tuple(prepared.atom_tokens)

    atom_tokens: set[str] = set()
    for fragment_mol in Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False):
        prepared = _runtime_graphs.prepare_smiles_graph(
            fragment_mol,
            flags=flags.with_rooted_at_atom(0),
        )
        atom_tokens.update(prepared.atom_tokens)
    return tuple(sorted(atom_tokens))


class RuntimeStateInvariantTests(unittest.TestCase):
    """Internal runtime-state invariants for the current decoder adapter model."""

    def test_determinized_decoder_state_audit_covers_all_reachable_states(self) -> None:
        cases = (
            RuntimeStateAuditCase(
                name="rooted_nonstereo",
                smiles="CCO",
                rooted_at_atom=0,
                isomeric_smiles=False,
            ),
            RuntimeStateAuditCase(
                name="rooted_stereo",
                smiles="F[C@H](Cl)Br",
                rooted_at_atom=0,
                isomeric_smiles=True,
            ),
            RuntimeStateAuditCase(
                name="disconnected_rooted",
                smiles="[Na+].C#N",
                rooted_at_atom=0,
                isomeric_smiles=False,
            ),
            RuntimeStateAuditCase(
                name="duplicate_same_text_connected",
                smiles="C1CCC2=NN=NN2CC1",
                rooted_at_atom=2,
                isomeric_smiles=False,
            ),
            RuntimeStateAuditCase(
                name="merged_then_visible_divergence",
                smiles="CC(=O)Oc1ccccc1C(=O)O",
                rooted_at_atom=9,
                isomeric_smiles=False,
            ),
        )

        for case in cases:
            mol = parse_smiles(case.smiles)
            kwargs = _public_kwargs(case)
            outputs = frozenset(grimace.MolToSmilesEnum(mol, **kwargs))
            decoder = grimace.MolToSmilesDeterminizedDecoder(mol, **kwargs)
            memo: dict[object, frozenset[str]] = {}
            seen_state_keys: set[object] = set()
            stack = [decoder._state]
            audited_state_count = 0

            with self.subTest(case=case.name, smiles=case.smiles):
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
                    grouped_successors = _runtime_states._determinized_choice_successors(
                        state
                    )
                    option_texts = tuple(text for text, _ in grouped_successors)

                    self.assertTrue(reachable)
                    self.assertTrue(reachable <= outputs)
                    self.assertTrue(all(output.startswith(prefix) for output in reachable))

                    if state.is_terminal():
                        self.assertEqual((), grouped_successors)
                        self.assertEqual(frozenset({prefix}), reachable)
                        continue

                    self.assertTrue(grouped_successors)
                    assert_prefix_options_match_outputs(
                        self,
                        prefix,
                        option_texts,
                        reachable,
                        atom_tokens=_atom_tokens(case),
                    )

                    union_of_branch_outputs: set[str] = set()
                    for _, successor in grouped_successors:
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

    def test_decoder_state_audit_covers_all_reachable_states(self) -> None:
        cases = (
            RuntimeStateAuditCase(
                name="rooted_nonstereo",
                smiles="CCO",
                rooted_at_atom=0,
                isomeric_smiles=False,
            ),
            RuntimeStateAuditCase(
                name="rooted_stereo",
                smiles="F[C@H](Cl)Br",
                rooted_at_atom=0,
                isomeric_smiles=True,
            ),
            RuntimeStateAuditCase(
                name="nonisomeric_explicit_bond_dirs",
                smiles="F/C=C\\Cl",
                rooted_at_atom=0,
                isomeric_smiles=False,
                all_bonds_explicit=True,
            ),
            RuntimeStateAuditCase(
                name="unrooted_connected",
                smiles="CCO",
                rooted_at_atom=None,
                isomeric_smiles=False,
            ),
            RuntimeStateAuditCase(
                name="disconnected_rooted",
                smiles="[Na+].CC",
                rooted_at_atom=0,
                isomeric_smiles=False,
            ),
            RuntimeStateAuditCase(
                name="disconnected_unrooted",
                smiles="O.CCO",
                rooted_at_atom=None,
                isomeric_smiles=True,
            ),
            RuntimeStateAuditCase(
                name="duplicate_same_text_connected",
                smiles="C1CCC2=NN=NN2CC1",
                rooted_at_atom=2,
                isomeric_smiles=False,
            ),
        )

        for case in cases:
            mol = parse_smiles(case.smiles)
            kwargs = _public_kwargs(case)
            outputs = frozenset(grimace.MolToSmilesEnum(mol, **kwargs))
            decoder = grimace.MolToSmilesDecoder(mol, **kwargs)
            memo: dict[object, frozenset[str]] = {}
            seen_state_keys: set[object] = set()
            stack = [decoder._state]
            audited_state_count = 0

            with self.subTest(case=case.name, smiles=case.smiles):
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
                    successor_states = _runtime_states._choice_successor_states(state)

                    self.assertTrue(reachable)
                    self.assertTrue(reachable <= outputs)
                    self.assertTrue(all(output.startswith(prefix) for output in reachable))

                    if state.is_terminal():
                        self.assertEqual((), successor_states)
                        self.assertEqual(frozenset({prefix}), reachable)
                        continue

                    self.assertTrue(successor_states)
                    union_of_branch_outputs: set[str] = set()
                    for _, next_state in successor_states:
                        branch_outputs = _runtime_states._reachable_terminal_prefixes(
                            next_state,
                            memo=memo,
                        )
                        self.assertTrue(branch_outputs)
                        self.assertTrue(branch_outputs <= reachable)
                        self.assertTrue(
                            all(
                                output.startswith(next_state.prefix())
                                for output in branch_outputs
                            )
                        )
                        union_of_branch_outputs.update(branch_outputs)
                        stack.append(next_state)

                    self.assertEqual(reachable, frozenset(union_of_branch_outputs))

                self.assertGreater(audited_state_count, 0)


if __name__ == "__main__":
    unittest.main()
