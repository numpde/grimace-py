"""Tests for South Star support-completeness traces."""

from __future__ import annotations

import unittest
from dataclasses import replace

from grimace._south_star1.completeness_checker import (
    replay_support_completeness_certificate,
)
from grimace._south_star1.enumeration_trace import EnumerationNodeId
from grimace._south_star1.enumeration_trace import RejectionCertificate
from grimace._south_star1.enumeration_trace import build_trace_index
from grimace._south_star1.enumeration_trace import enumeration_trace_from_jsonable
from grimace._south_star1.enumeration_trace import enumeration_trace_to_jsonable
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.ordinary_stereo_sites import OrdinaryStereoSiteOptions
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.policy import AtomTextChoice
from grimace._south_star1.policy import AtomTextDomain
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.support_enumeration import TracedCertifiedSupportImage
from grimace._south_star1.support_enumeration import (
    enumerate_traced_certified_stereo_support,
)
from grimace._south_star1.support_enumeration import (
    traced_certified_support_from_jsonable,
)
from grimace._south_star1.support_enumeration import (
    traced_certified_support_to_jsonable,
)
from tests.south_star1.helpers import atom
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import single_bond
from tests.south_star1.helpers import tetrahedral_facts


class CompletenessCertificateTest(unittest.TestCase):
    def test_traced_support_replays_for_tetra_case(self) -> None:
        facts = tetrahedral_facts()
        result = _traced_result(facts)

        replay_support_completeness_certificate(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
            result=result,
        )

    def test_traced_support_replays_for_directional_case(self) -> None:
        facts = directional_facts()
        result = _traced_result(facts)

        replay_support_completeness_certificate(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
            result=result,
        )

    def test_traced_support_replays_for_ring_tetra_case(self) -> None:
        facts = ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1")
        result = _traced_result(facts)

        replay_support_completeness_certificate(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
            result=result,
        )

    def test_traced_support_manifest_counts_match_trace(self) -> None:
        facts = tetrahedral_facts()
        result = _traced_result(facts)

        self.assertEqual(result.manifest.skeleton_count, result.trace.skeleton_count)
        self.assertEqual(result.manifest.prefix_count, result.trace.prefix_count)
        self.assertEqual(result.manifest.csp_count, result.trace.csp_count)
        self.assertEqual(
            result.manifest.feasible_solution_count,
            result.trace.feasible_solution_count,
        )
        self.assertEqual(
            result.manifest.selected_solution_count,
            result.trace.selected_solution_count,
        )
        self.assertEqual(result.manifest.witness_count, result.trace.witness_count)
        self.assertEqual(result.manifest.support_count, result.trace.support_count)

    def test_trace_rejects_unselected_annotation_solution(self) -> None:
        facts = directional_facts()
        policy = replace(
            ordinary_policy_for_facts(facts),
            annotation_mode=AnnotationMode.CANONICAL,
        )
        result = enumerate_traced_certified_stereo_support(
            facts=facts,
            policy=policy,
            semantics=OrdinarySmilesSemantics(),
        )

        self.assertTrue(
            any(
                rejection.reason == "annotation_not_selected"
                for rejection in result.trace.rejected
            )
        )

    def test_trace_records_render_duplicate_as_quotient_rejection(self) -> None:
        facts = _ethane_facts()
        result = _traced_result(facts)

        self.assertLess(result.support.distinct_count, result.support.witness_count)
        self.assertTrue(
            any(
                rejection.reason == "render_duplicate"
                for rejection in result.trace.rejected
            )
        )

    def test_completeness_checker_rejects_missing_rejection(self) -> None:
        facts = directional_facts()
        policy = replace(
            ordinary_policy_for_facts(facts),
            annotation_mode=AnnotationMode.CANONICAL,
        )
        result = enumerate_traced_certified_stereo_support(
            facts=facts,
            policy=policy,
            semantics=OrdinarySmilesSemantics(),
        )
        self.assertTrue(result.trace.rejected)
        mutated = replace(
            result,
            trace=replace(result.trace, rejected=result.trace.rejected[:-1]),
        )

        with self.assertRaisesRegex(ValueError, "missing annotation rejection"):
            replay_support_completeness_certificate(
                facts=facts,
                policy=policy,
                semantics=OrdinarySmilesSemantics(),
                result=mutated,
            )

    def test_completeness_checker_rejects_missing_acceptance(self) -> None:
        facts = tetrahedral_facts()
        result = _traced_result(facts)
        mutated = replace(
            result,
            trace=replace(result.trace, accepted=result.trace.accepted[:-1]),
        )

        with self.assertRaisesRegex(ValueError, "accepted witness count"):
            replay_support_completeness_certificate(
                facts=facts,
                policy=ordinary_policy_for_facts(facts),
                semantics=OrdinarySmilesSemantics(),
                result=mutated,
            )

    def test_completeness_checker_rejects_mutated_manifest_count(self) -> None:
        facts = tetrahedral_facts()
        result = _traced_result(facts)
        mutated = replace(
            result,
            manifest=replace(
                result.manifest,
                witness_count=result.manifest.witness_count + 1,
            ),
        )

        with self.assertRaisesRegex(ValueError, "manifest witness count mismatch"):
            replay_support_completeness_certificate(
                facts=facts,
                policy=ordinary_policy_for_facts(facts),
                semantics=OrdinarySmilesSemantics(),
                result=mutated,
            )

    def test_enumeration_trace_json_roundtrip(self) -> None:
        facts = directional_facts()
        result = _traced_result(facts)

        encoded = enumeration_trace_to_jsonable(result.trace)
        decoded = enumeration_trace_from_jsonable(encoded)

        self.assertEqual(decoded, result.trace)

    def test_traced_support_jsonable_includes_manifest_and_trace(self) -> None:
        facts = tetrahedral_facts()
        result = _traced_result(facts)

        encoded = traced_certified_support_to_jsonable(result)

        self.assertIn("manifest", encoded)
        self.assertIn("trace", encoded)
        self.assertIn("certified_witnesses", encoded)

    def test_traced_support_json_roundtrip_replays(self) -> None:
        facts = tetrahedral_facts()
        policy = ordinary_policy_for_facts(facts)
        semantics = OrdinarySmilesSemantics()
        result = enumerate_traced_certified_stereo_support(
            facts=facts,
            policy=policy,
            semantics=semantics,
        )

        encoded = traced_certified_support_to_jsonable(result)
        decoded = traced_certified_support_from_jsonable(encoded)

        replay_support_completeness_certificate(
            facts=facts,
            policy=policy,
            semantics=semantics,
            result=decoded,
        )

    def test_trace_json_rejects_unknown_schema_version(self) -> None:
        facts = tetrahedral_facts()
        result = _traced_result(facts)
        encoded = enumeration_trace_to_jsonable(result.trace)
        encoded["schema_version"] = 999

        with self.assertRaisesRegex(ValueError, "schema version"):
            enumeration_trace_from_jsonable(encoded)

    def test_trace_json_rejects_unknown_rejection_reason(self) -> None:
        facts = _ethane_facts()
        result = _traced_result(facts)
        encoded = enumeration_trace_to_jsonable(result.trace)
        encoded["rejected"][0]["reason"] = "unknown"

        with self.assertRaisesRegex(ValueError, "rejection reason"):
            enumeration_trace_from_jsonable(encoded)

    def test_trace_json_rejects_missing_detail_field(self) -> None:
        facts = _ethane_facts()
        result = _traced_result(facts)
        encoded = enumeration_trace_to_jsonable(result.trace)
        del encoded["rejected"][0]["detail"]

        with self.assertRaises(KeyError):
            enumeration_trace_from_jsonable(encoded)

    def test_replay_rejects_json_roundtrip_with_changed_node_key(self) -> None:
        facts = tetrahedral_facts()
        policy = ordinary_policy_for_facts(facts)
        semantics = OrdinarySmilesSemantics()
        result = enumerate_traced_certified_stereo_support(
            facts=facts,
            policy=policy,
            semantics=semantics,
        )
        encoded = traced_certified_support_to_jsonable(result)
        encoded["trace"]["accepted"][0]["node"]["key"] = ["not-a-witness"]
        decoded = traced_certified_support_from_jsonable(encoded)

        with self.assertRaisesRegex(ValueError, "missing witness acceptance"):
            replay_support_completeness_certificate(
                facts=facts,
                policy=policy,
                semantics=semantics,
                result=decoded,
            )

    def test_source_ingestion_specified_closure_traced_support_records_manifest(
        self,
    ) -> None:
        options = RdkitOrdinaryExtractionOptions(
            stereo_site_options=OrdinaryStereoSiteOptions(
                ligand_equivalence="exact_stereochemical_graph_automorphism",
            ),
            stereo_site_discovery_mode="specified_closure",
        )
        facts = ordinary_molecule_facts_from_smiles(
            "[C@H](F)([C@H](F)Cl)[C@@H](F)Cl",
            options,
        )
        result = _traced_result(facts)

        self.assertEqual(result.support.distinct_count, 1216)
        self.assertEqual(result.trace.support_count, 1216)
        self.assertEqual(result.trace.witness_count, result.manifest.witness_count)

    def test_trace_index_rejects_node_with_acceptance_and_rejection(self) -> None:
        facts = tetrahedral_facts()
        result = _traced_result(facts)
        node = result.trace.accepted[0].node
        mutated = replace(
            result.trace,
            rejected=result.trace.rejected
            + (
                RejectionCertificate(
                    node=node,
                    reason="internal_invariant",
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "both accepted and rejected"):
            build_trace_index(mutated)

    def test_completeness_checker_rejects_missing_csp_unsatisfied_rejection(
        self,
    ) -> None:
        facts = tetrahedral_facts()
        policy = _tetra_rejecting_policy(facts)
        result = enumerate_traced_certified_stereo_support(
            facts=facts,
            policy=policy,
            semantics=OrdinarySmilesSemantics(),
        )
        self.assertTrue(result.trace.rejected)
        mutated = replace(
            result,
            trace=replace(result.trace, rejected=result.trace.rejected[:-1]),
        )

        with self.assertRaisesRegex(ValueError, "missing CSP rejection"):
            replay_support_completeness_certificate(
                facts=facts,
                policy=policy,
                semantics=OrdinarySmilesSemantics(),
                result=mutated,
            )

    def test_completeness_checker_rejects_false_empty_mark_relation_rejection(
        self,
    ) -> None:
        facts = tetrahedral_facts()
        policy = _tetra_rejecting_policy(facts)
        result = enumerate_traced_certified_stereo_support(
            facts=facts,
            policy=policy,
            semantics=OrdinarySmilesSemantics(),
        )
        rejection = result.trace.rejected[0]
        mutated_rejection = replace(rejection, reason="empty_mark_relation")
        mutated = replace(
            result,
            trace=replace(
                result.trace,
                rejected=(mutated_rejection,) + result.trace.rejected[1:],
            ),
        )

        with self.assertRaisesRegex(ValueError, "false empty_mark_relation"):
            replay_support_completeness_certificate(
                facts=facts,
                policy=policy,
                semantics=OrdinarySmilesSemantics(),
                result=mutated,
            )

    def test_completeness_checker_rejects_false_annotation_not_selected_rejection(
        self,
    ) -> None:
        facts = directional_facts()
        policy = replace(
            ordinary_policy_for_facts(facts),
            annotation_mode=AnnotationMode.CANONICAL,
        )
        result = enumerate_traced_certified_stereo_support(
            facts=facts,
            policy=policy,
            semantics=OrdinarySmilesSemantics(),
        )
        rejection = next(
            item
            for item in result.trace.rejected
            if item.reason == "annotation_not_selected"
        )
        mutated_rejection = replace(
            rejection,
            detail=rejection.detail[:-1] + (AnnotationMode.HARD.value,),
        )
        rejected = tuple(
            mutated_rejection if item == rejection else item
            for item in result.trace.rejected
        )
        mutated = replace(result, trace=replace(result.trace, rejected=rejected))

        with self.assertRaisesRegex(ValueError, "mode detail mismatch"):
            replay_support_completeness_certificate(
                facts=facts,
                policy=policy,
                semantics=OrdinarySmilesSemantics(),
                result=mutated,
            )

    def test_completeness_checker_rejects_missing_render_duplicate_rejection(
        self,
    ) -> None:
        facts = _ethane_facts()
        result = _traced_result(facts)
        rejected = tuple(
            item
            for item in result.trace.rejected
            if item.reason != "render_duplicate"
        )
        mutated = replace(result, trace=replace(result.trace, rejected=rejected))

        with self.assertRaisesRegex(ValueError, "missing render_duplicate"):
            replay_support_completeness_certificate(
                facts=facts,
                policy=ordinary_policy_for_facts(facts),
                semantics=OrdinarySmilesSemantics(),
                result=mutated,
            )

    def test_completeness_checker_rejects_trace_with_extra_unreachable_node(
        self,
    ) -> None:
        facts = tetrahedral_facts()
        result = _traced_result(facts)
        mutated = replace(
            result,
            trace=replace(
                result.trace,
                rejected=result.trace.rejected
                + (
                    RejectionCertificate(
                        node=EnumerationNodeId(kind="csp", key=("unknown",)),
                        reason="csp_unsatisfied",
                    ),
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "unreachable rejection node"):
            replay_support_completeness_certificate(
                facts=facts,
                policy=ordinary_policy_for_facts(facts),
                semantics=OrdinarySmilesSemantics(),
                result=mutated,
            )

    def test_completeness_checker_rejects_accepted_witness_not_selected_solution_set(
        self,
    ) -> None:
        facts = tetrahedral_facts()
        result = _traced_result(facts)
        first = result.trace.accepted[0]
        mutated_acceptance = replace(
            first,
            node=EnumerationNodeId(kind="witness", key=("not-selected",)),
        )
        mutated = replace(
            result,
            trace=replace(
                result.trace,
                accepted=(mutated_acceptance,) + result.trace.accepted[1:],
            ),
        )

        with self.assertRaisesRegex(ValueError, "missing witness acceptance"):
            replay_support_completeness_certificate(
                facts=facts,
                policy=ordinary_policy_for_facts(facts),
                semantics=OrdinarySmilesSemantics(),
                result=mutated,
            )

    def test_completeness_checker_can_compare_against_regeneration(self) -> None:
        facts = tetrahedral_facts()
        result = _traced_result(facts)

        replay_support_completeness_certificate(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
            result=result,
            compare_against_regeneration=True,
        )


def _traced_result(facts: MoleculeFacts) -> TracedCertifiedSupportImage:
    return enumerate_traced_certified_stereo_support(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
    )


def _ethane_facts() -> MoleculeFacts:
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


def _tetra_rejecting_policy(facts: MoleculeFacts):
    policy = ordinary_policy_for_facts(facts)
    center = facts.stereo.tetrahedral[0].center
    replacement = AtomTextDomain(
        atom=center,
        choices=(
            AtomTextChoice(
                name="nonchiral_C",
                text_by_tetra=((TetraToken.NONE, "C"),),
            ),
        ),
    )
    return replace(
        policy,
        atom_text_domains=tuple(
            replacement if domain.atom == center else domain
            for domain in policy.atom_text_domains
        ),
    )


if __name__ == "__main__":
    unittest.main()
