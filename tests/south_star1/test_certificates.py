"""Tests for South Star witness certificates."""

from __future__ import annotations

import unittest
from dataclasses import replace

from grimace._south_star1.certificate_checker import replay_witness_certificate
from grimace._south_star1.certificates import AnnotationSelectionCertificate
from grimace._south_star1.certificates import RelationCertificate
from grimace._south_star1.certificates import expected_relation_certificate_subjects
from grimace._south_star1.certificates import manifest_from_jsonable
from grimace._south_star1.certificates import manifest_to_jsonable
from grimace._south_star1.certificates import validate_stereo_solution_certificate
from grimace._south_star1.certificates import validate_witness_certificate
from grimace._south_star1.certificates import witness_certificate_from_jsonable
from grimace._south_star1.certificates import witness_certificate_to_jsonable
from grimace._south_star1.constraints import TraversalAssignment
from grimace._south_star1.graph_index import build_graph_index
from grimace._south_star1.ids import CarrierSlotId
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.ordinary_stereo_sites import OrdinaryStereoSiteOptions
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.policy import TetraToken
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.skeleton import enumerate_traversal_skeletons
from grimace._south_star1.slots import allocate_traversal_slots
from grimace._south_star1.stereo_csp import StereoSolution
from grimace._south_star1.stereo_csp import build_stereo_csp
from grimace._south_star1.stereo_csp import certify_stereo_solution
from grimace._south_star1.stereo_csp import select_stereo_solutions_with_certificates
from grimace._south_star1.stereo_csp import solve_stereo_csp
from grimace._south_star1.stereo_witness import enumerate_certified_stereo_witnesses_for_skeleton
from grimace._south_star1.stereo_witness import enumerate_presentation_prefixes
from grimace._south_star1.support_enumeration import enumerate_exhaustive_certified_stereo_support
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import tetrahedral_facts


class CertificateTest(unittest.TestCase):
    def test_support_maximal_annotation_certificate_accepts_inclusion_maximal_support(
        self,
    ) -> None:
        cert = AnnotationSelectionCertificate(
            mode=AnnotationMode.SUPPORT_MAXIMAL,
            selected_support=frozenset({CarrierSlotId(1), CarrierSlotId(2)}),
            feasible_supports=frozenset(
                {
                    frozenset({CarrierSlotId(1)}),
                    frozenset({CarrierSlotId(1), CarrierSlotId(2)}),
                }
            ),
            selected_supports=frozenset(
                {frozenset({CarrierSlotId(1), CarrierSlotId(2)})}
            ),
        )

        cert.validate()

    def test_support_maximal_annotation_certificate_rejects_dominated_support(
        self,
    ) -> None:
        cert = AnnotationSelectionCertificate(
            mode=AnnotationMode.SUPPORT_MAXIMAL,
            selected_support=frozenset({CarrierSlotId(1)}),
            feasible_supports=frozenset(
                {
                    frozenset({CarrierSlotId(1)}),
                    frozenset({CarrierSlotId(1), CarrierSlotId(2)}),
                }
            ),
            selected_supports=frozenset({frozenset({CarrierSlotId(1)})}),
        )

        with self.assertRaisesRegex(ValueError, "inclusion-maximal"):
            cert.validate()

    def test_cardinality_annotation_certificate_rejects_smaller_support(self) -> None:
        cert = AnnotationSelectionCertificate(
            mode=AnnotationMode.CARDINALITY_MAXIMAL,
            selected_support=frozenset({CarrierSlotId(1)}),
            feasible_supports=frozenset(
                {
                    frozenset({CarrierSlotId(1)}),
                    frozenset({CarrierSlotId(2), CarrierSlotId(3)}),
                }
            ),
            selected_supports=frozenset({frozenset({CarrierSlotId(1)})}),
        )

        with self.assertRaisesRegex(ValueError, "cardinality-maximal"):
            cert.validate()

    def test_canonical_annotation_certificate_rejects_nonminimal_solution(self) -> None:
        cert = AnnotationSelectionCertificate(
            mode=AnnotationMode.CANONICAL,
            selected_support=frozenset({CarrierSlotId(1)}),
            feasible_supports=frozenset({frozenset({CarrierSlotId(1)})}),
            selected_supports=frozenset({frozenset({CarrierSlotId(1)})}),
            canonical_key=("a",),
            selected_solution_key=("b",),
            selected_solution_keys=frozenset({("a",), ("b",)}),
        )

        with self.assertRaisesRegex(ValueError, "canonical-minimal"):
            cert.validate()

    def test_certified_tetra_witness_certificate_validates(self) -> None:
        csp, selected = _first_selected_solution(tetrahedral_facts())
        cert = certify_stereo_solution(
            csp=csp,
            solution=selected.solution,
            annotation_certificate=selected.certificate,
        )

        validate_stereo_solution_certificate(csp, selected.solution, cert)

    def test_relation_certificate_coverage_rejects_missing_relation(self) -> None:
        csp, selected = _first_selected_solution(tetrahedral_facts())
        cert = certify_stereo_solution(
            csp=csp,
            solution=selected.solution,
            annotation_certificate=selected.certificate,
        )
        mutated = replace(
            cert,
            relation_certificates=cert.relation_certificates[:-1],
        )

        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validate_stereo_solution_certificate(csp, selected.solution, mutated)

    def test_relation_certificate_coverage_rejects_extra_relation(self) -> None:
        csp, selected = _first_selected_solution(tetrahedral_facts())
        cert = certify_stereo_solution(
            csp=csp,
            solution=selected.solution,
            annotation_certificate=selected.certificate,
        )
        mutated = replace(
            cert,
            relation_certificates=cert.relation_certificates
            + (
                RelationCertificate(
                    kind=expected_relation_certificate_subjects(csp)[0][0],
                    subject="extra",
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "coverage mismatch"):
            validate_stereo_solution_certificate(csp, selected.solution, mutated)

    def test_certified_directional_witness_certificate_validates(self) -> None:
        csp, selected = _first_selected_solution(directional_facts())
        cert = certify_stereo_solution(
            csp=csp,
            solution=selected.solution,
            annotation_certificate=selected.certificate,
        )

        validate_stereo_solution_certificate(csp, selected.solution, cert)

    def test_certified_ring_tetra_witness_certificate_validates(self) -> None:
        facts = ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1")
        image = enumerate_exhaustive_certified_stereo_support(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )

        self.assertTrue(image.certified_witnesses)
        first = image.certified_witnesses[0]
        self.assertEqual(first.witness.id, first.certificate.witness_id)
        self.assertEqual(first.witness.rendered, first.certificate.rendered)

    def test_traversal_certificate_validates_atom_bond_ring_coverage(self) -> None:
        facts, skeleton, slots, policy, semantics, certified = _first_certified(
            "[C@H]1(F)CO1"
        )

        validate_witness_certificate(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            assignment=certified.assignment,
            policy=policy,
            semantics=semantics,
            certificate=certified.certificate,
        )

    def test_traversal_certificate_rejects_mutated_ring_label(self) -> None:
        facts, skeleton, slots, policy, semantics, certified = _first_certified(
            "[C@H]1(F)CO1"
        )
        mutated = TraversalAssignment(
            atom_text=certified.assignment.atom_text,
            tetra_tokens=certified.assignment.tetra_tokens,
            bond_text=certified.assignment.bond_text,
            ring_labels={},
            direction_marks=certified.assignment.direction_marks,
        )

        with self.assertRaises(ValueError):
            validate_witness_certificate(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                assignment=mutated,
                policy=policy,
                semantics=semantics,
                certificate=certified.certificate,
            )

    def test_standalone_replay_checker_accepts_generated_certificate(self) -> None:
        facts, skeleton, slots, policy, semantics, certified = _first_certified(
            "[C@H](F)(Cl)Br"
        )

        replay_witness_certificate(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            assignment=certified.assignment,
            policy=policy,
            semantics=semantics,
            certificate=certified.certificate,
        )

    def test_standalone_replay_checker_rejects_mutated_assignment(self) -> None:
        facts, skeleton, slots, policy, semantics, certified = _first_certified(
            "[C@H](F)(Cl)Br"
        )
        mutated_tokens = dict(certified.assignment.tetra_tokens)
        center = facts.stereo.tetrahedral[0].center
        mutated_tokens[center] = TetraToken.ATAT
        mutated = TraversalAssignment(
            atom_text=certified.assignment.atom_text,
            tetra_tokens=mutated_tokens,
            bond_text=certified.assignment.bond_text,
            ring_labels=certified.assignment.ring_labels,
            direction_marks=certified.assignment.direction_marks,
        )

        with self.assertRaises(ValueError):
            replay_witness_certificate(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                assignment=mutated,
                policy=policy,
                semantics=semantics,
                certificate=certified.certificate,
            )

    def test_certified_support_image_has_one_certificate_per_witness(self) -> None:
        facts = tetrahedral_facts()
        image = enumerate_exhaustive_certified_stereo_support(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )

        self.assertEqual(image.support.witness_count, len(image.certified_witnesses))

    def test_certified_support_manifest_counts_match_witnesses(self) -> None:
        facts = tetrahedral_facts()
        image = enumerate_exhaustive_certified_stereo_support(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )

        self.assertEqual(image.manifest.witness_count, len(image.certified_witnesses))
        self.assertEqual(image.manifest.support_count, image.support.distinct_count)
        self.assertEqual(
            image.manifest.selected_solution_count,
            len(image.certified_witnesses),
        )

    def test_certified_support_manifest_hash_is_stable(self) -> None:
        facts = directional_facts()
        kwargs = {
            "facts": facts,
            "policy": ordinary_policy_for_facts(facts),
            "semantics": OrdinarySmilesSemantics(),
        }

        first = enumerate_exhaustive_certified_stereo_support(**kwargs).manifest
        second = enumerate_exhaustive_certified_stereo_support(**kwargs).manifest

        self.assertEqual(first.support_hash, second.support_hash)
        self.assertEqual(first.witness_hash, second.witness_hash)

    def test_certified_support_image_support_strings_are_unique(self) -> None:
        facts = directional_facts()
        image = enumerate_exhaustive_certified_stereo_support(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )

        self.assertEqual(len(image.support.strings), len(set(image.support.strings)))

    def test_certificate_validation_rejects_mutated_direction_row(self) -> None:
        csp, selected = _first_selected_solution(directional_facts())
        cert = certify_stereo_solution(
            csp=csp,
            solution=selected.solution,
            annotation_certificate=selected.certificate,
        )
        mutated_marks = dict(selected.solution.direction_marks)
        carrier = next(
            carrier
            for carrier, mark in mutated_marks.items()
            if mark is not DirectionMark.ABSENT
        )
        mutated_marks[carrier] = DirectionMark.ABSENT
        mutated = StereoSolution(
            tetra_tokens=dict(selected.solution.tetra_tokens),
            direction_marks=mutated_marks,
        )

        with self.assertRaises(ValueError):
            validate_stereo_solution_certificate(csp, mutated, cert)

    def test_certificate_validation_rejects_mutated_tetra_token(self) -> None:
        csp, selected = _first_selected_solution(tetrahedral_facts())
        cert = certify_stereo_solution(
            csp=csp,
            solution=selected.solution,
            annotation_certificate=selected.certificate,
        )
        mutated_tokens = dict(selected.solution.tetra_tokens)
        mutated_tokens[tetrahedral_facts().stereo.tetrahedral[0].center] = (
            TetraToken.ATAT
        )
        mutated = StereoSolution(
            tetra_tokens=mutated_tokens,
            direction_marks=dict(selected.solution.direction_marks),
        )

        with self.assertRaises(ValueError):
            validate_stereo_solution_certificate(csp, mutated, cert)

    def test_source_ingestion_specified_closure_certified_support_validates(self) -> None:
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
        image = enumerate_exhaustive_certified_stereo_support(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )

        self.assertEqual(image.support.distinct_count, 1216)
        for certified in image.certified_witnesses[:8]:
            certified.certificate.stereo_solution.annotation_certificate.validate()
            self.assertEqual(certified.witness.id, certified.certificate.witness_id)

    def test_witness_certificate_json_roundtrip(self) -> None:
        _, _, _, _, _, certified = _first_certified("[C@H](F)(Cl)Br")

        encoded = witness_certificate_to_jsonable(certified.certificate)
        decoded = witness_certificate_from_jsonable(encoded)

        self.assertEqual(decoded, certified.certificate)

    def test_manifest_json_roundtrip(self) -> None:
        facts = tetrahedral_facts()
        image = enumerate_exhaustive_certified_stereo_support(
            facts=facts,
            policy=ordinary_policy_for_facts(facts),
            semantics=OrdinarySmilesSemantics(),
        )

        encoded = manifest_to_jsonable(image.manifest)
        decoded = manifest_from_jsonable(encoded)

        self.assertEqual(decoded, image.manifest)


def _first_selected_solution(facts):
    policy = ordinary_policy_for_facts(facts)
    semantics = OrdinarySmilesSemantics()
    skeleton = enumerate_traversal_skeletons(
        facts,
        build_graph_index(facts),
        policy,
    )[0]
    slots = allocate_traversal_slots(facts, skeleton)
    prefix = next(
        enumerate_presentation_prefixes(
            facts=facts,
            slots=slots,
            policy=policy,
        )
    )
    csp = build_stereo_csp(
        facts=facts,
        skeleton=skeleton,
        slots=slots,
        prefix=prefix,
        policy=policy,
        semantics=semantics,
    )
    solutions = tuple(solve_stereo_csp(csp))
    selected = select_stereo_solutions_with_certificates(
        csp=csp,
        solutions=solutions,
        mode=policy.annotation_mode,
    )
    return csp, selected[0]


def _first_certified(smiles: str):
    facts = ordinary_molecule_facts_from_smiles(smiles)
    policy = ordinary_policy_for_facts(facts)
    semantics = OrdinarySmilesSemantics()
    skeleton = enumerate_traversal_skeletons(
        facts,
        build_graph_index(facts),
        policy,
    )[0]
    slots = allocate_traversal_slots(facts, skeleton)
    certified = next(
        enumerate_certified_stereo_witnesses_for_skeleton(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            policy=policy,
            semantics=semantics,
        )
    )
    return facts, skeleton, slots, policy, semantics, certified


if __name__ == "__main__":
    unittest.main()
