"""Tests for online stereo witness enumeration."""

from __future__ import annotations

import ast
import unittest
from collections import Counter
from dataclasses import replace
from pathlib import Path

import grimace._south_star1.online_stereo_witness as online_stereo_witness_module
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import DirectionalValue
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.online_stereo_witness import iter_exhaustive_trace_ring_label_assignments
from grimace._south_star1.online_stereo_witness import iter_exhaustive_online_stereo_witness_strings
from grimace._south_star1.online_stereo_witness import iter_exhaustive_online_stereo_witnesses
from grimace._south_star1.online_stereo_witness import exhaustive_trace_slot_key
from grimace._south_star1.online_stereo_witness import exhaustive_trace_slot_view
from grimace._south_star1.online_traversal import iter_exhaustive_online_traversal_traces
from grimace._south_star1.online_traversal import trace_to_skeleton_like_key
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.proof_terms import skeleton_key
from grimace._south_star1.proof_terms import slot_key
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.slots import allocate_traversal_slots
from grimace._south_star1.support_enumeration import enumerate_exhaustive_stereo_witnesses
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


REPO_ROOT = Path(__file__).resolve().parents[2]
ONLINE_STEREO_WITNESS_PATH = (
    REPO_ROOT / "python" / "grimace" / "_south_star1" / "online_stereo_witness.py"
)


class OnlineStereoWitnessTest(unittest.TestCase):
    def test_online_tetra_multiset_matches_offline(self) -> None:
        self.assertEqual(
            _online_counter(tetrahedral_facts()),
            _offline_counter(tetrahedral_facts()),
        )

    def test_online_directional_multiset_matches_offline(self) -> None:
        self.assertEqual(
            _online_counter(directional_facts()),
            _offline_counter(directional_facts()),
        )

    def test_online_ring_tetra_multiset_matches_offline(self) -> None:
        facts = ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1")

        self.assertEqual(_online_counter(facts), _offline_counter(facts))

    def test_online_disconnected_multiset_matches_offline(self) -> None:
        facts = disconnected_facts()

        self.assertEqual(_online_counter(facts), _offline_counter(facts))

    def test_online_unspecified_directional_rejects_accidental_stereo(self) -> None:
        site = directional_facts().stereo.directional[0]
        facts = replace(
            directional_facts(),
            stereo=replace(
                directional_facts().stereo,
                directional=(
                    replace(
                        site,
                        status=SiteStatus.UNSPECIFIED,
                        target=DirectionalValue.NONE,
                    ),
                ),
            ),
        )

        online = tuple(_online_counter(facts))

        self.assertTrue(online)
        self.assertTrue(all("/" not in item and "\\" not in item for item in online))

    def test_online_shared_carrier_constraints_match_offline(self) -> None:
        self.assertEqual(
            _online_counter(directional_facts()),
            _offline_counter(directional_facts()),
        )

    def test_online_support_maximal_matches_offline_on_directional_case(self) -> None:
        facts = directional_facts()
        policy = ordinary_policy_for_facts(facts)
        self.assertIs(policy.annotation_mode, AnnotationMode.SUPPORT_MAXIMAL)

        self.assertEqual(_online_counter(facts), _offline_counter(facts))

    def test_online_witness_with_counts_does_not_deduplicate(self) -> None:
        witnesses = tuple(
            iter_exhaustive_online_stereo_witnesses(
                facts=ethane_facts(),
                policy=ordinary_policy_for_facts(ethane_facts()),
                semantics=OrdinarySmilesSemantics(),
            )
        )
        rendered = Counter(witness.rendered for witness in witnesses)

        self.assertGreater(sum(rendered.values()), len(rendered))
        self.assertEqual(rendered["CC"], 2)
        self.assertEqual(rendered["C(C)"], 2)

    def test_generic_online_stereo_witness_names_are_not_exported(self) -> None:
        prefix = "iter_" + "online_stereo_witness"
        generic_names = (
            prefix + "_strings",
            prefix + "es",
            prefix + "es_with_sink",
        )
        for name in generic_names:
            self.assertFalse(hasattr(online_stereo_witness_module, name))
            self.assertNotIn(name, online_stereo_witness_module.__all__)

    def test_exhaustive_trace_slot_keys_match_offline_slot_bundle_keys(self) -> None:
        facts = ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1")
        policy = ordinary_policy_for_facts(facts)
        offline_by_skeleton_key = {
            skeleton_key(skeleton): slot_key(allocate_traversal_slots(facts, skeleton))
            for skeleton in enumerate_traversal_skeletons(
                facts,
                build_graph_index(facts),
                policy,
            )
        }

        for trace in iter_exhaustive_online_traversal_traces(facts=facts, policy=policy):
            online_key = exhaustive_trace_slot_key(exhaustive_trace_slot_view(trace))
            offline_key = offline_by_skeleton_key[trace_to_skeleton_like_key(trace)]
            self.assertEqual(online_key[1:], offline_key[1:])

    def test_online_ring_label_assignments_match_offline_witnesses(self) -> None:
        facts = ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1")
        policy = ordinary_policy_for_facts(facts)
        trace = next(
            trace
            for trace in iter_exhaustive_online_traversal_traces(facts=facts, policy=policy)
            if any(event.__class__.__name__ == "OnlineRingEndpointEvent" for event in trace.events)
        )

        assignments = tuple(
            iter_exhaustive_trace_ring_label_assignments(trace=trace, policy=policy)
        )

        self.assertTrue(assignments)
        for assignment in assignments:
            labels = set(assignment.values())
            self.assertEqual(labels, {policy.ring_labels[0]})

    def test_online_stereo_witness_boundary_no_artifact_or_support_imports(self) -> None:
        tree = ast.parse(ONLINE_STEREO_WITNESS_PATH.read_text(encoding="utf-8"))
        banned_modules = {
            "audit_rdkit",
            "finite_space_checker",
            "rdkit_adapter",
            "semantic_relation_checker",
            "stereo_witness",
            "support_artifact",
            "support_artifact_checker",
            "support_enumeration",
        }
        banned_calls = {
            "compile_support_artifact",
            "enumerate_stereo_support",
            "enumerate_traversal_skeletons",
            "render_image_from_witnesses",
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

    def test_generic_trace_helper_names_are_absent(self) -> None:
        names = (
            "online_" + "slot_view_for_trace",
            "online_" + "slot_key",
            "iter_" + "online_ring_label_assignments",
            "online_" + "local_tetra_order",
        )
        for name in names:
            self.assertFalse(hasattr(online_stereo_witness_module, name))
            self.assertNotIn(name, online_stereo_witness_module.__all__)

    def test_generic_trace_artifact_names_are_absent(self) -> None:
        names = (
            "Online" + "Witness",
            "Online" + "BondSlot",
            "Online" + "CarrierSlot",
            "Online" + "RingEndpointSlot",
            "Online" + "SlotView",
            "directional_" + "templates_for_carrier",
        )
        for name in names:
            self.assertFalse(hasattr(online_stereo_witness_module, name))
            self.assertNotIn(name, online_stereo_witness_module.__all__)


def _online_counter(facts: MoleculeFacts) -> Counter[str]:
    return Counter(
        iter_exhaustive_online_stereo_witness_strings(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )
    )


def _offline_counter(facts: MoleculeFacts) -> Counter[str]:
    return Counter(
        witness.rendered
        for witness in enumerate_exhaustive_stereo_witnesses(
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


def ethane_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C")),
        bonds=(single_bond(0, 0, 1),),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1)),
                bonds=(BondId(0),),
            ),
        ),
    )


if __name__ == "__main__":
    unittest.main()
