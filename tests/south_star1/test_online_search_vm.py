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
from grimace._south_star1.online_residual_continuation import ResidualFrontierSink
from grimace._south_star1.online_search_vm import EventLoopFrame
from grimace._south_star1.online_search_vm import OnlineSearchFrame
from grimace._south_star1.online_search_vm import OnlineSearchVM
from grimace._south_star1.online_search_vm import ParentOrientationFrame
from grimace._south_star1.online_search_vm import capture_residual_continuation
from grimace._south_star1.online_search_vm import iter_online_stereo_witness_strings_vm
from grimace._south_star1.online_search_vm import make_online_search_state
from grimace._south_star1.online_stereo_witness import iter_online_stereo_witness_strings
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.residual_constraints import VarId
from grimace._south_star1.residual_constraints import direction_var
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import RingLabel
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

        self.assertTrue(
            state.ring.register_endpoint(
                bond=BondId(0),
                endpoint=0,
                label=RingLabel(1),
            )
        )
        state.rollback(snapshot)

        self.assertEqual(state.ring.endpoint_by_bond, {})
        self.assertEqual(state.ring.label_by_endpoint, {})
        self.assertEqual(state.ring.open_intervals, {})

    def test_vm_snapshot_restores_decision_path(self) -> None:
        state = _state(tetrahedral_facts())
        snapshot = state.checkpoint()

        state.decisions.push(OnlineDecision("test", (1,)))
        state.rollback(snapshot)

        self.assertEqual(state.decisions.path().items, ())

    def test_vm_snapshot_restores_frame_stack(self) -> None:
        state = _state(tetrahedral_facts())
        root_frame = OnlineSearchFrame(EventLoopFrame(("root",)))
        child_frame = OnlineSearchFrame(EventLoopFrame(("child",)))
        state.frames.append(root_frame)
        snapshot = state.checkpoint()

        state.frames.append(child_frame)
        state.rollback(snapshot)

        self.assertEqual(state.frames, [root_frame])

    def test_vm_snapshot_restores_real_traversal_state_after_root(self) -> None:
        state = _state(tetrahedral_facts())
        snapshot = state.checkpoint()

        state.traversal.component_index = 1
        state.traversal.roots.append(AtomId(0))
        state.traversal.parent[AtomId(0)] = None
        state.traversal.tree_bonds.add(BondId(0))
        state.traversal.visited_atoms.add(AtomId(0))
        state.traversal.active_atom_stack.append(AtomId(0))
        state.traversal.syntax_position = 3
        state.rollback(snapshot)

        self.assertEqual(state.traversal.component_index, 0)
        self.assertEqual(state.traversal.roots, [])
        self.assertEqual(state.traversal.parent, {})
        self.assertEqual(state.traversal.tree_bonds, set())
        self.assertEqual(state.traversal.visited_atoms, set())
        self.assertEqual(state.traversal.active_atom_stack, [])
        self.assertEqual(state.traversal.syntax_position, 0)

    def test_vm_snapshot_restores_ring_endpoint_state(self) -> None:
        state = _state(ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"))
        snapshot = state.checkpoint()

        self.assertTrue(
            state.ring.register_endpoint(
                bond=BondId(1),
                endpoint=0,
                label=RingLabel(1),
            )
        )
        state.rollback(snapshot)

        self.assertEqual(state.ring.endpoint_by_bond, {})
        self.assertEqual(state.ring.label_by_endpoint, {})
        self.assertEqual(state.ring.open_intervals, {})
        self.assertEqual(state.ring.next_endpoint_id, 0)

    def test_vm_snapshot_restores_direction_mark_assignment(self) -> None:
        state = _state(directional_facts())
        var = direction_var(0)
        state.residual.add_var(var, (DirectionMark.ABSENT, DirectionMark.FWD))
        snapshot = state.checkpoint()

        self.assertTrue(state.residual.assign(var, DirectionMark.FWD))
        state.rollback(snapshot)

        self.assertIsNone(state.residual.assignment(var))

    def test_from_snapshot_restores_residual_snapshot(self) -> None:
        facts = directional_facts()
        policy = ordinary_policy_for_facts(facts)
        semantics = OrdinarySmilesSemantics()
        state = make_online_search_state(
            facts=facts,
            policy=policy,
            semantics=semantics,
            sink=OnlineStringBuffer(),
        )
        var = direction_var(0)
        state.residual.add_var(var, (DirectionMark.ABSENT, DirectionMark.FWD))
        self.assertTrue(state.residual.assign(var, DirectionMark.FWD))
        snapshot = state.checkpoint()

        vm = OnlineSearchVM.from_snapshot(
            facts=facts,
            policy=policy,
            semantics=semantics,
            snapshot=snapshot,
            sink=OnlineStringBuffer(),
        )

        self.assertIsNot(vm.state.residual, state.residual)
        self.assertIs(vm.state.residual.assignment(var), DirectionMark.FWD)

    def test_resumed_snapshot_residual_store_mutation_does_not_mutate_producer_store(self) -> None:
        facts = directional_facts()
        policy = ordinary_policy_for_facts(facts)
        semantics = OrdinarySmilesSemantics()
        state = make_online_search_state(
            facts=facts,
            policy=policy,
            semantics=semantics,
            sink=OnlineStringBuffer(),
        )
        var = direction_var(0)
        state.residual.add_var(var, (DirectionMark.ABSENT, DirectionMark.FWD))
        self.assertTrue(state.residual.assign(var, DirectionMark.FWD))
        snapshot = state.checkpoint()
        extra = VarId("extra", (0,))

        vm = OnlineSearchVM.from_snapshot(
            facts=facts,
            policy=policy,
            semantics=semantics,
            snapshot=snapshot,
            sink=OnlineStringBuffer(),
        )
        vm.state.residual.add_var(extra, ("x",))
        self.assertTrue(vm.state.residual.assign(extra, "x"))

        with self.assertRaises(ValueError):
            state.residual.assign(extra, "x")
        self.assertIs(vm.state.residual.assignment(var), DirectionMark.FWD)
        self.assertIs(state.residual.assignment(var), DirectionMark.FWD)

    def test_vm_step_interface_yields_witness_then_exhausts(self) -> None:
        vm = OnlineSearchVM(
            facts=disconnected_facts(),
            policy=ordinary_policy_for_facts(disconnected_facts()),
            semantics=OrdinarySmilesSemantics(),
        )

        first = vm.step()

        self.assertEqual(first.kind, "yield_witness")
        self.assertIsNotNone(first.witness)
        while True:
            result = vm.step()
            if result.kind == "exhausted":
                break
        self.assertEqual(vm.step().kind, "exhausted")

    def test_capture_residual_continuation_contains_snapshot(self) -> None:
        state = _state(tetrahedral_facts())
        frame = OnlineSearchFrame(EventLoopFrame(("root",)))
        state.frames.append(frame)

        continuation = capture_residual_continuation(state, prefix="C")

        self.assertEqual(continuation.prefix, "C")
        self.assertEqual(continuation.snapshot.frame_stack, (frame,))

    def test_directional_candidate_rendering_restores_ring_state_between_candidates(self) -> None:
        facts = ring_directional_facts()
        vm = OnlineSearchVM(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )

        witnesses = [vm.run_until_witness_or_exhausted() for _ in range(2)]

        self.assertTrue(all(witness is not None for witness in witnesses))
        self.assertEqual(vm.state.ring.checkpoint(), ((), (), (), 0))

    def test_two_directional_candidates_on_same_ring_label_branch_both_render(self) -> None:
        facts = ring_directional_facts()
        rendered = tuple(
            iter_online_stereo_witness_strings_vm(
                facts=facts,
                policy=ordinary_policy_for_facts(facts),
                semantics=OrdinarySmilesSemantics(),
            )
        )

        self.assertGreater(len(rendered), len(set(rendered)))
        self.assertGreaterEqual(len(rendered), 2)

    def test_residual_continuation_captures_partial_ring_state_but_producer_ring_restores(self) -> None:
        facts = ring_directional_facts()
        sink = ResidualFrontierSink(required_prefix="C")
        vm = OnlineSearchVM(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
            sink_factory=lambda: sink,
        )
        sink.snapshot_provider = vm.checkpoint
        sink.decision_path_provider = vm.state.decisions.path

        while vm.run_until_witness_or_exhausted() is not None:
            pass

        self.assertEqual(vm.state.ring.checkpoint(), ((), (), (), 0))
        continuation = sink.completed_by_token["1"][0]
        self.assertNotEqual(continuation.snapshot.ring_state, ((), (), (), 0))

    def test_non_support_maximal_directional_candidates_stream_without_tuple_buffer(self) -> None:
        tree = ast.parse(ONLINE_SEARCH_VM_PATH.read_text(encoding="utf-8"))
        buffered_directional_candidates = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if not isinstance(node.func, ast.Name) or node.func.id != "tuple":
                continue
            if not node.args:
                continue
            arg = node.args[0]
            if isinstance(arg, ast.Call) and isinstance(arg.func, ast.Name):
                if arg.func.id == "_iter_directional_candidates":
                    buffered_directional_candidates.append(node.lineno)

        self.assertEqual(buffered_directional_candidates, [])

    def test_no_string_render_cursor_or_completion_frames_remain(self) -> None:
        text = ONLINE_SEARCH_VM_PATH.read_text(encoding="utf-8")

        self.assertNotIn('OnlineSearchFrame("', text)
        self.assertNotIn('OnlineSearchFrame("render-cursor"', text)
        self.assertNotIn('OnlineSearchFrame("completion"', text)

    def test_all_retained_frame_payloads_are_known_dataclasses(self) -> None:
        state = _state(tetrahedral_facts())
        state.frames.append(OnlineSearchFrame(ParentOrientationFrame(((AtomId(0), None),))))

        for frame in state.checkpoint().frame_stack:
            self.assertTrue(hasattr(frame.payload, "__dataclass_fields__"))

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
            "iter_online_traversal_traces",
            "online_branch_preserving_choices",
            "online_determinized_choices",
        }
        private_online_stereo_imports: list[str] = []
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
                if module == "online_stereo_witness":
                    private_online_stereo_imports.extend(
                        alias.name
                        for alias in node.names
                        if alias.name.startswith("_")
                    )
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.append(node.func.id)
                if isinstance(node.func, ast.Attribute):
                    calls.append(node.func.attr)

        self.assertEqual(imports, [])
        self.assertEqual(private_online_stereo_imports, [])
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


def ring_directional_facts() -> MoleculeFacts:
    return ordinary_molecule_facts_from_smiles("C/C=C/1CC1")


if __name__ == "__main__":
    unittest.main()
