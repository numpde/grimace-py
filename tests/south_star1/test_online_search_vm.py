"""Tests for the explicit-frame South Star online search VM."""

from __future__ import annotations

import ast
import unittest
from collections import Counter
from pathlib import Path

from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.online_decisions import OnlineDecision
from grimace._south_star1.online_render_sink import OnlineStringBuffer
from grimace._south_star1.online_search_vm import OnlineSearchFrame
from grimace._south_star1.online_search_vm import capture_residual_continuation
from grimace._south_star1.online_search_vm import iter_online_stereo_witness_strings_vm
from grimace._south_star1.online_search_vm import make_online_search_state
from grimace._south_star1.online_stereo_witness import iter_online_stereo_witness_strings
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.residual_constraints import VarId
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import tetrahedral_facts


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_SEARCH_VM_PATH = (
    REPO_ROOT / "python" / "grimace" / "_south_star1" / "online_search_vm.py"
)


class OnlineSearchVmTest(unittest.TestCase):
    def test_vm_online_tetra_multiset_matches_recursive_online(self) -> None:
        self._assert_vm_matches_recursive(tetrahedral_facts())

    def test_vm_online_directional_multiset_matches_recursive_online(self) -> None:
        self._assert_vm_matches_recursive(directional_facts())

    def test_vm_online_ring_tetra_multiset_matches_recursive_online(self) -> None:
        self._assert_vm_matches_recursive(ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"))

    def test_vm_online_disconnected_multiset_matches_recursive_online(self) -> None:
        self._assert_vm_matches_recursive(disconnected_facts())

    def test_vm_online_shared_carrier_multiset_matches_recursive_online(self) -> None:
        self._assert_vm_matches_recursive(directional_facts())

    def test_vm_snapshot_restores_output_buffer(self) -> None:
        state = _state(tetrahedral_facts())
        snapshot = state.checkpoint()

        self.assertTrue(state.output.append("C", token_text="C"))
        state.rollback(snapshot)

        self.assertEqual(state.output.value(), "")

    def test_vm_snapshot_restores_residual_store(self) -> None:
        state = _state(tetrahedral_facts())
        var = VarId("test", (0,))
        state.residual.add_var(var, ("a", "b"))
        snapshot = state.checkpoint()

        self.assertTrue(state.residual.assign(var, "a"))
        state.rollback(snapshot)

        self.assertIsNone(state.residual.assignment(var))

    def test_vm_snapshot_restores_ring_state(self) -> None:
        state = _state(tetrahedral_facts())
        snapshot = state.checkpoint()

        state.ring.labels = ((0, 1),)
        state.rollback(snapshot)

        self.assertEqual(state.ring.labels, ())

    def test_vm_snapshot_restores_decision_path(self) -> None:
        state = _state(tetrahedral_facts())
        snapshot = state.checkpoint()

        state.decisions.push(OnlineDecision("test", (1,)))
        state.rollback(snapshot)

        self.assertEqual(state.decisions.path().items, ())

    def test_vm_snapshot_restores_frame_stack(self) -> None:
        state = _state(tetrahedral_facts())
        state.frames.append(OnlineSearchFrame("root", (0,)))
        snapshot = state.checkpoint()

        state.frames.append(OnlineSearchFrame("child", (1,)))
        state.rollback(snapshot)

        self.assertEqual(state.frames, [OnlineSearchFrame("root", (0,))])

    def test_capture_residual_continuation_contains_snapshot(self) -> None:
        state = _state(tetrahedral_facts())
        state.frames.append(OnlineSearchFrame("root", (0,)))

        continuation = capture_residual_continuation(state, prefix="C")

        self.assertEqual(continuation.prefix, "C")
        self.assertEqual(continuation.snapshot.frame_stack, (OnlineSearchFrame("root", (0,)),))

    def test_online_search_vm_boundary_no_hidden_generator_or_artifact_imports(self) -> None:
        tree = ast.parse(ONLINE_SEARCH_VM_PATH.read_text(encoding="utf-8"))
        banned_modules = {
            "audit_rdkit",
            "finite_space_checker",
            "rdkit_adapter",
            "semantic_relation_checker",
            "support_artifact",
            "support_artifact_checker",
            "support_enumeration",
        }
        banned_calls = {
            "iter_online_stereo_witnesses_with_sink",
            "iter_online_stereo_witness_strings",
            "online_branch_preserving_choices",
            "online_determinized_choices",
        }
        imports: list[str] = []
        calls: list[str] = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.extend(
                    alias.name
                    for alias in node.names
                    if alias.name.split(".", 1)[0] in banned_modules
                )
            if isinstance(node, ast.ImportFrom):
                module = node.module or ""
                if module.split(".", 1)[0] in banned_modules:
                    imports.append(module)
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                if isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)

        self.assertEqual(imports, [])
        self.assertEqual(sorted(set(calls) & banned_calls), [])

    def _assert_vm_matches_recursive(self, facts: MoleculeFacts) -> None:
        self.assertEqual(_vm_counter(facts), _recursive_counter(facts))


def _state(facts: MoleculeFacts):
    return make_online_search_state(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
        sink=OnlineStringBuffer(),
    )


def _vm_counter(facts: MoleculeFacts) -> Counter[str]:
    return Counter(
        iter_online_stereo_witness_strings_vm(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )
    )


def _recursive_counter(facts: MoleculeFacts) -> Counter[str]:
    return Counter(
        iter_online_stereo_witness_strings(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )
    )


def disconnected_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "O")),
        bonds=(),
        components=(
            ComponentFacts(id=ComponentId(0), atoms=(AtomId(0),), bonds=()),
            ComponentFacts(id=ComponentId(1), atoms=(AtomId(1),), bonds=()),
        ),
    )


if __name__ == "__main__":
    unittest.main()
