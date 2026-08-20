"""Tests for compiled South Star support artifacts."""

from __future__ import annotations

import unittest
from dataclasses import replace

from grimace._south_star1.certificates import RelationCertificate
from grimace._south_star1.certificates import CertificateRelationKind
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.ordinary_stereo_sites import OrdinaryStereoSiteOptions
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.proof_terms import sequence_hash
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.support_artifact import ArtifactNode
from grimace._south_star1.support_artifact import SupportArtifact
from grimace._south_star1.support_artifact import compile_support_artifact
from grimace._south_star1.support_artifact import support_artifact_digest
from grimace._south_star1.support_artifact import support_artifact_from_jsonable
from grimace._south_star1.support_artifact import support_artifact_to_jsonable
from grimace._south_star1.support_artifact_checker import check_support_artifact
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import tetrahedral_facts


class SupportArtifactTest(unittest.TestCase):
    def test_support_artifact_roundtrips_json(self) -> None:
        artifact = _artifact_for_tetra()

        decoded = support_artifact_from_jsonable(
            support_artifact_to_jsonable(artifact),
        )

        self.assertEqual(decoded, artifact)
        self.assertEqual(
            support_artifact_digest(decoded),
            support_artifact_digest(artifact),
        )
        check_support_artifact(decoded)

    def test_support_artifact_checker_accepts_tetra_case(self) -> None:
        check_support_artifact(_artifact_for_tetra())

    def test_support_artifact_checker_accepts_directional_case(self) -> None:
        check_support_artifact(_artifact_for_facts(directional_facts()))

    def test_support_artifact_checker_accepts_ring_tetra_case(self) -> None:
        facts = ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1")

        check_support_artifact(_artifact_for_facts(facts))

    def test_support_artifact_checker_accepts_specified_closure_source_case(
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
        artifact = _artifact_for_facts(facts)

        self.assertEqual(artifact.traced_support.support.distinct_count, 1216)
        check_support_artifact(artifact)

    def test_support_artifact_checker_rejects_mutated_tetra_relation(self) -> None:
        artifact = _artifact_for_tetra()
        relation = next(item for item in artifact.relations if item.name == "tetra_site")
        mutated_relation = replace(
            relation,
            allowed_rows=(),
            row_hash=sequence_hash(()),
        )

        with self.assertRaisesRegex(ValueError, "semantic relation row"):
            check_support_artifact(
                _replace_relation(artifact, relation, mutated_relation),
            )

    def test_support_artifact_checker_rejects_mutated_mark_relation(self) -> None:
        artifact = _artifact_for_facts(directional_facts())
        relation = next(item for item in artifact.relations if item.name != "tetra_site")
        mutated_relation = replace(
            relation,
            allowed_rows=(),
            row_hash=sequence_hash(()),
        )

        with self.assertRaisesRegex(ValueError, "semantic relation row"):
            check_support_artifact(
                _replace_relation(artifact, relation, mutated_relation),
            )

    def test_support_artifact_checker_rejects_missing_skeleton_node(self) -> None:
        artifact = _artifact_for_tetra()
        mutated = replace(
            artifact,
            nodes=tuple(node for node in artifact.nodes if node.kind != "skeleton"),
        )

        with self.assertRaisesRegex(ValueError, "unknown child node|skeleton"):
            check_support_artifact(mutated)

    def test_support_artifact_checker_rejects_extra_prefix_node(self) -> None:
        artifact = _artifact_for_tetra()
        skeleton_key = artifact.traversal_space.skeleton_keys[0]
        mutated = replace(
            artifact,
            nodes=artifact.nodes
            + (ArtifactNode(kind="prefix", key=(skeleton_key, ("extra",))),),
        )

        with self.assertRaisesRegex(ValueError, "prefix node coverage"):
            check_support_artifact(mutated)

    def test_artifact_checker_rejects_missing_spanning_tree_skeleton(self) -> None:
        artifact = _artifact_for_tetra()
        mutated = replace(
            artifact,
            traversal_space=replace(
                artifact.traversal_space,
                skeleton_keys=artifact.traversal_space.skeleton_keys[:-1],
            ),
        )

        with self.assertRaisesRegex(ValueError, "skeleton node set"):
            check_support_artifact(mutated)

    def test_artifact_checker_rejects_extra_non_spanning_tree_skeleton(self) -> None:
        artifact = _artifact_for_tetra()
        extra = ("not-a-grammar-skeleton",)
        mutated = replace(
            artifact,
            nodes=artifact.nodes + (ArtifactNode(kind="skeleton", key=extra),),
            traversal_space=replace(
                artifact.traversal_space,
                skeleton_keys=artifact.traversal_space.skeleton_keys + (extra,),
            ),
            traversal_decisions=artifact.traversal_decisions
            + (replace(artifact.traversal_decisions[0], skeleton_key=extra),),
        )

        with self.assertRaisesRegex(ValueError, "traversal grammar skeleton coverage"):
            check_support_artifact(mutated)

    def test_artifact_checker_rejects_missing_local_event_order_skeleton(self) -> None:
        artifact = _artifact_for_tetra()
        decision = artifact.traversal_decisions[0]
        mutated_decision = replace(
            decision,
            local_event_orders=decision.local_event_orders[:-1],
        )

        with self.assertRaisesRegex(ValueError, "local-order table"):
            check_support_artifact(
                _replace_traversal_decision(artifact, decision, mutated_decision),
            )

    def test_support_artifact_checker_rejects_missing_feasible_solution(self) -> None:
        artifact = _artifact_for_tetra()
        space = artifact.csp_solution_spaces[0]
        mutated_space = replace(
            space,
            feasible_solution_keys=space.feasible_solution_keys[:-1],
        )

        with self.assertRaisesRegex(ValueError, "feasible solution coverage"):
            check_support_artifact(_replace_csp_space(artifact, space, mutated_space))

    def test_support_artifact_checker_rejects_false_annotation_selection(self) -> None:
        artifact = _artifact_for_facts(
            directional_facts(),
            annotation_mode=AnnotationMode.CANONICAL,
        )
        space = next(item for item in artifact.csp_solution_spaces if item.rejected_solution_keys)
        mutated_space = replace(
            space,
            selected_solution_keys=space.selected_solution_keys
            + (space.rejected_solution_keys[0],),
        )

        with self.assertRaisesRegex(ValueError, "selected solution coverage"):
            check_support_artifact(_replace_csp_space(artifact, space, mutated_space))

    def test_artifact_checker_rejects_missing_prefix_cartesian_product_element(
        self,
    ) -> None:
        artifact = _artifact_for_tetra()
        space = artifact.prefix_spaces[0]
        mutated_space = replace(space, prefix_keys=space.prefix_keys[:-1])

        with self.assertRaisesRegex(ValueError, "policy product"):
            check_support_artifact(_replace_prefix_space(artifact, space, mutated_space))

    def test_artifact_checker_rejects_extra_prefix_not_in_policy_product(self) -> None:
        artifact = _artifact_for_tetra()
        space = artifact.prefix_spaces[0]
        extra_prefix = (("extra",), (), ())
        mutated_space = replace(
            space,
            prefix_keys=space.prefix_keys + (extra_prefix,),
        )
        mutated = _replace_prefix_space(artifact, space, mutated_space)
        mutated = replace(
            mutated,
            nodes=mutated.nodes
            + (ArtifactNode(kind="prefix", key=(space.skeleton_key, extra_prefix)),),
        )

        with self.assertRaisesRegex(ValueError, "policy product"):
            check_support_artifact(mutated)

    def test_support_artifact_checker_rejects_render_program_mismatch(self) -> None:
        artifact = _artifact_for_tetra()
        program = artifact.render_programs[0]
        mutated_program = replace(program, rendered=program.rendered + "X")
        mutated = replace(
            artifact,
            render_programs=(mutated_program,) + artifact.render_programs[1:],
        )

        with self.assertRaisesRegex(ValueError, "render program mismatch"):
            check_support_artifact(mutated)

    def test_support_artifact_checker_rejects_support_hash_mismatch(self) -> None:
        artifact = _artifact_for_tetra()
        mutated = replace(
            artifact,
            traced_support=replace(
                artifact.traced_support,
                manifest=replace(
                    artifact.traced_support.manifest,
                    support_hash="bad",
                ),
            ),
        )

        with self.assertRaisesRegex(ValueError, "support hash mismatch"):
            check_support_artifact(mutated)

    def test_support_artifact_checker_rejects_unknown_schema_version(self) -> None:
        artifact = _artifact_for_tetra()
        mutated = replace(
            artifact,
            header=replace(artifact.header, schema_version=999),
        )

        with self.assertRaisesRegex(ValueError, "schema version"):
            check_support_artifact(mutated)

    def test_artifact_checker_rejects_parent_cycle(self) -> None:
        artifact = _artifact_for_tetra()
        decision = artifact.traversal_decisions[0]
        mutated_decision = replace(
            decision,
            roots=(2,),
            parent_items=((0, 1), (1, 0), (2, None), (3, 0)),
        )

        with self.assertRaisesRegex(ValueError, "cycle"):
            check_support_artifact(
                _replace_traversal_decision(artifact, decision, mutated_decision),
            )

    def test_artifact_checker_rejects_parent_nonedge(self) -> None:
        artifact = _artifact_for_tetra()
        decision = artifact.traversal_decisions[0]
        mutated_decision = replace(
            decision,
            parent_items=((0, None), (1, 2), (2, 0), (3, 0)),
        )

        with self.assertRaisesRegex(ValueError, "parent edge"):
            check_support_artifact(
                _replace_traversal_decision(artifact, decision, mutated_decision),
            )

    def test_artifact_checker_rejects_wrong_tree_ring_partition(self) -> None:
        artifact = _artifact_for_tetra()
        decision = artifact.traversal_decisions[0]
        mutated_decision = replace(
            decision,
            tree_bonds=decision.tree_bonds[:-1],
            ring_bonds=decision.ring_bonds + (decision.tree_bonds[-1],),
        )

        with self.assertRaisesRegex(ValueError, "tree bonds"):
            check_support_artifact(
                _replace_traversal_decision(artifact, decision, mutated_decision),
            )

    def test_artifact_checker_rejects_extra_local_event(self) -> None:
        artifact = _artifact_for_tetra()
        decision = artifact.traversal_decisions[0]
        atom, events = decision.local_event_orders[0]
        mutated_decision = replace(
            decision,
            local_event_orders=((atom, events + events[:1]),)
            + decision.local_event_orders[1:],
        )

        with self.assertRaisesRegex(ValueError, "child event coverage"):
            check_support_artifact(
                _replace_traversal_decision(artifact, decision, mutated_decision),
            )

    def test_artifact_checker_rejects_missing_ring_endpoint(self) -> None:
        artifact = _artifact_for_facts(ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"))
        decision = next(item for item in artifact.traversal_decisions if item.ring_bonds)
        mutated_orders = []
        removed = False
        for atom, events in decision.local_event_orders:
            if not removed:
                kept = tuple(event for event in events if event[0] != "ring")
                if len(kept) != len(events):
                    mutated_orders.append((atom, kept))
                    removed = True
                    continue
            mutated_orders.append((atom, events))
        self.assertTrue(removed)

        with self.assertRaisesRegex(ValueError, "ring endpoint coverage"):
            check_support_artifact(
                _replace_traversal_decision(
                    artifact,
                    decision,
                    replace(decision, local_event_orders=tuple(mutated_orders)),
                ),
            )

    def test_artifact_checker_rejects_prefix_atom_domain_not_in_policy(self) -> None:
        artifact = _artifact_for_tetra()
        space = artifact.prefix_spaces[0]
        atom, values = space.atom_text_domains[0]
        mutated_space = replace(
            space,
            atom_text_domains=((atom, values + ("not_in_policy",)),)
            + space.atom_text_domains[1:],
        )

        with self.assertRaisesRegex(ValueError, "atom domains"):
            check_support_artifact(_replace_prefix_space(artifact, space, mutated_space))

    def test_artifact_checker_rejects_prefix_bond_domain_not_in_policy(self) -> None:
        artifact = _artifact_for_tetra()
        space = artifact.prefix_spaces[0]
        slot, values = space.bond_text_domains[0]
        mutated_space = replace(
            space,
            bond_text_domains=((slot, values + ("not_in_policy",)),)
            + space.bond_text_domains[1:],
        )

        with self.assertRaisesRegex(ValueError, "bond domains"):
            check_support_artifact(_replace_prefix_space(artifact, space, mutated_space))

    def test_artifact_checker_rejects_non_least_free_ring_label(self) -> None:
        artifact = _artifact_for_facts(ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"))
        space = next(item for item in artifact.prefix_spaces if item.ring_label_assignments)
        assignment = space.ring_label_assignments[0]
        mutated_assignment = tuple((endpoint, 2) for endpoint, _ in assignment)
        mutated_space = replace(
            space,
            ring_label_assignments=(mutated_assignment,)
            + space.ring_label_assignments[1:],
        )

        with self.assertRaisesRegex(ValueError, "least-free"):
            check_support_artifact(_replace_prefix_space(artifact, space, mutated_space))

    def test_artifact_checker_rejects_missing_ring_label_assignment(self) -> None:
        artifact = _artifact_for_facts(ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"))
        space = next(item for item in artifact.prefix_spaces if item.ring_label_assignments)
        mutated_space = replace(space, ring_label_assignments=())

        with self.assertRaisesRegex(ValueError, "ring label assignment"):
            check_support_artifact(_replace_prefix_space(artifact, space, mutated_space))

    def test_artifact_checker_rejects_extra_overlapping_ring_label_assignment(
        self,
    ) -> None:
        artifact = _artifact_for_facts(ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"))
        space = next(item for item in artifact.prefix_spaces if item.ring_label_assignments)
        bad_assignment = ((999, 1),)
        mutated_space = replace(
            space,
            ring_label_assignments=space.ring_label_assignments + (bad_assignment,),
        )

        with self.assertRaisesRegex(ValueError, "endpoint coverage|policy product"):
            check_support_artifact(_replace_prefix_space(artifact, space, mutated_space))

    def test_artifact_checker_rejects_slot_bundle_not_induced_by_skeleton(self) -> None:
        artifact = _artifact_for_tetra()
        slot_node = next(node for node in artifact.nodes if node.kind == "slot_bundle")
        atom_slots, bond_slots, ring_endpoints, carrier_slots = slot_node.key[1]
        mutated_slot_key = (
            atom_slots,
            bond_slots[:-1],
            ring_endpoints,
            carrier_slots,
        )
        mutated_node = replace(slot_node, key=(slot_node.key[0], mutated_slot_key))

        with self.assertRaisesRegex(ValueError, "slot bundle"):
            check_support_artifact(_replace_node(artifact, slot_node, mutated_node))

    def test_artifact_checker_rejects_carrier_slot_not_induced_by_bond_slot(self) -> None:
        artifact = _artifact_for_tetra()
        slot_node = next(node for node in artifact.nodes if node.kind == "slot_bundle")
        atom_slots, bond_slots, ring_endpoints, carrier_slots = slot_node.key[1]
        carrier = carrier_slots[0]
        mutated_carriers = ((carrier[0], 999, carrier[2], carrier[3], carrier[4]),) + carrier_slots[1:]
        mutated_slot_key = (
            atom_slots,
            bond_slots,
            ring_endpoints,
            mutated_carriers,
        )
        mutated_node = replace(slot_node, key=(slot_node.key[0], mutated_slot_key))

        with self.assertRaisesRegex(ValueError, "slot bundle"):
            check_support_artifact(_replace_node(artifact, slot_node, mutated_node))

    def test_artifact_checker_rejects_atom_piece_with_wrong_tetra_token(self) -> None:
        artifact = _artifact_for_tetra()
        program = artifact.render_programs[0]
        pieces = _mutate_first_piece_arg(program.pieces, "atom", 2, "@@")
        mutated_program = replace(program, pieces=pieces)

        with self.assertRaisesRegex(ValueError, "render program mismatch|outside policy"):
            check_support_artifact(
                _replace_render_program(artifact, program, mutated_program),
            )

    def test_artifact_checker_rejects_bond_piece_with_wrong_direction_mark(self) -> None:
        artifact = _artifact_for_facts(directional_facts())
        program = artifact.render_programs[0]
        pieces = _mutate_first_piece_arg(program.pieces, "bond", 4, -1)
        mutated_program = replace(program, pieces=pieces)

        with self.assertRaisesRegex(ValueError, "render program mismatch|render policy"):
            check_support_artifact(
                _replace_render_program(artifact, program, mutated_program),
            )

    def test_artifact_checker_rejects_wrong_ring_label_piece(self) -> None:
        artifact = _artifact_for_facts(ordinary_molecule_facts_from_smiles("[C@H]1(F)CO1"))
        program = next(
            item
            for item in artifact.render_programs
            if any(piece[0] == "ring_label" for piece in item.pieces)
        )
        pieces = _mutate_first_piece_arg(program.pieces, "ring_label", 1, 2)
        mutated_program = replace(program, pieces=pieces)

        with self.assertRaisesRegex(ValueError, "render program mismatch"):
            check_support_artifact(
                _replace_render_program(artifact, program, mutated_program),
            )

    def test_artifact_checker_rejects_literal_render_program_for_nonliteral_witness(
        self,
    ) -> None:
        artifact = _artifact_for_tetra()
        program = artifact.render_programs[0]
        mutated_program = replace(
            program,
            pieces=(("literal", (program.rendered,)),),
        )

        with self.assertRaisesRegex(ValueError, "unsupported render-program piece"):
            check_support_artifact(
                _replace_render_program(artifact, program, mutated_program),
            )

    def test_artifact_checker_rejects_render_piece_order_swap(self) -> None:
        artifact = _artifact_for_tetra()
        program = artifact.render_programs[0]
        pieces = (program.pieces[1], program.pieces[0]) + program.pieces[2:]
        mutated_program = replace(program, pieces=pieces)

        with self.assertRaisesRegex(ValueError, "render program mismatch"):
            check_support_artifact(
                _replace_render_program(artifact, program, mutated_program),
            )

    def test_artifact_checker_rejects_witness_certificate_missing_relation(self) -> None:
        artifact = _artifact_for_tetra()
        certified = artifact.traced_support.certified_witnesses[0]
        stereo = certified.certificate.stereo_solution
        mutated_stereo = replace(
            stereo,
            relation_certificates=stereo.relation_certificates[:-1],
        )

        with self.assertRaisesRegex(ValueError, "relation certificate coverage"):
            check_support_artifact(
                _replace_certified_witness(
                    artifact,
                    certified,
                    replace(
                        certified,
                        certificate=replace(
                            certified.certificate,
                            stereo_solution=mutated_stereo,
                        ),
                    ),
                )
            )

    def test_artifact_checker_rejects_witness_certificate_extra_relation(self) -> None:
        artifact = _artifact_for_tetra()
        certified = artifact.traced_support.certified_witnesses[0]
        stereo = certified.certificate.stereo_solution
        extra = RelationCertificate(
            kind=CertificateRelationKind.TETRA_SITE,
            subject="extra",
            detail=("token", "@"),
        )
        mutated_stereo = replace(
            stereo,
            relation_certificates=stereo.relation_certificates + (extra,),
        )

        with self.assertRaisesRegex(ValueError, "relation certificate coverage"):
            check_support_artifact(
                _replace_certified_witness(
                    artifact,
                    certified,
                    replace(
                        certified,
                        certificate=replace(
                            certified.certificate,
                            stereo_solution=mutated_stereo,
                        ),
                    ),
                )
            )

    def test_artifact_checker_rejects_witness_certificate_row_not_in_relation(
        self,
    ) -> None:
        artifact = _artifact_for_tetra()
        certified = artifact.traced_support.certified_witnesses[0]
        stereo = certified.certificate.stereo_solution
        first_relation = stereo.relation_certificates[0]
        mutated_relation = replace(
            first_relation,
            detail=first_relation.detail[:-3] + ("token", ""),
        )
        mutated_stereo = replace(
            stereo,
            relation_certificates=(mutated_relation,)
            + stereo.relation_certificates[1:],
        )

        with self.assertRaisesRegex(ValueError, "row is outside artifact relation"):
            check_support_artifact(
                _replace_certified_witness(
                    artifact,
                    certified,
                    replace(
                        certified,
                        certificate=replace(
                            certified.certificate,
                            stereo_solution=mutated_stereo,
                        ),
                    ),
                )
            )

    def test_artifact_checker_rejects_tetra_domain_allowing_token_on_non_site(
        self,
    ) -> None:
        artifact = _artifact_for_facts(directional_facts())
        space = artifact.csp_solution_spaces[0]
        name, values = space.tetra_domains[0]
        mutated_space = replace(
            space,
            tetra_domains=((name, values + ("@",)),) + space.tetra_domains[1:],
        )

        with self.assertRaisesRegex(ValueError, "semantic tetra domain"):
            check_support_artifact(_replace_csp_space(artifact, space, mutated_space))

    def test_artifact_checker_rejects_tetra_domain_disallowing_specified_token(
        self,
    ) -> None:
        artifact = _artifact_for_tetra()
        space = artifact.csp_solution_spaces[0]
        name, _ = next(
            item for item in space.tetra_domains if item[0] == "tetra:0"
        )
        mutated_space = replace(
            space,
            tetra_domains=tuple(
                (name, ("@",)) if item[0] == name else item
                for item in space.tetra_domains
            ),
        )

        with self.assertRaisesRegex(ValueError, "semantic tetra domain"):
            check_support_artifact(_replace_csp_space(artifact, space, mutated_space))

    def test_artifact_checker_rejects_direction_domain_marker_on_ineligible_carrier(
        self,
    ) -> None:
        artifact = _artifact_for_tetra()
        space = artifact.csp_solution_spaces[0]
        name, values = space.direction_domains[0]
        mutated_space = replace(
            space,
            direction_domains=((name, values + (1,)),) + space.direction_domains[1:],
        )

        with self.assertRaisesRegex(ValueError, "semantic direction domain"):
            check_support_artifact(_replace_csp_space(artifact, space, mutated_space))

    def test_semantic_relation_checker_rejects_directional_scope_with_center_bond(
        self,
    ) -> None:
        artifact = _artifact_for_facts(directional_facts())
        relation = next(item for item in artifact.relations if item.name == "directional_site")
        mutated_relation = replace(
            relation,
            scope=("direction:0",) + relation.scope,
            allowed_rows=tuple((0,) + row for row in relation.allowed_rows),
            row_hash=sequence_hash(
                repr(tuple((0,) + row)) for row in relation.allowed_rows
            ),
        )

        with self.assertRaisesRegex(ValueError, "semantic relation scope"):
            check_support_artifact(
                _replace_relation(artifact, relation, mutated_relation),
            )

    def test_semantic_relation_checker_rejects_directional_scope_missing_carrier(
        self,
    ) -> None:
        artifact = _artifact_for_facts(directional_facts())
        relation = next(item for item in artifact.relations if item.name == "directional_site")
        mutated_relation = replace(
            relation,
            scope=relation.scope[:-1],
            allowed_rows=tuple(row[:-1] for row in relation.allowed_rows),
            row_hash=sequence_hash(
                repr(tuple(row[:-1])) for row in relation.allowed_rows
            ),
        )

        with self.assertRaisesRegex(ValueError, "semantic relation scope"):
            check_support_artifact(
                _replace_relation(artifact, relation, mutated_relation),
            )

    def test_semantic_relation_checker_rejects_directional_wrong_reference_row(
        self,
    ) -> None:
        artifact = _artifact_for_facts(directional_facts())
        relation = next(item for item in artifact.relations if item.name == "directional_site")
        mutated_rows = ((1, 1),)
        mutated_relation = replace(
            relation,
            allowed_rows=mutated_rows,
            row_hash=sequence_hash(repr(row) for row in mutated_rows),
        )

        with self.assertRaisesRegex(ValueError, "semantic relation row"):
            check_support_artifact(
                _replace_relation(artifact, relation, mutated_relation),
            )

    def test_semantic_relation_checker_rejects_tree_marker_on_double_bond(self) -> None:
        artifact = _artifact_for_facts(directional_facts())
        relation = next(
            item
            for item in artifact.relations
            if item.name == "tree_bond_decode" and item.subject.endswith("bond_slot:0")
        )
        mutated_rows = relation.allowed_rows + ((1,),)
        mutated_relation = replace(
            relation,
            allowed_rows=mutated_rows,
            row_hash=sequence_hash(repr(row) for row in mutated_rows),
        )

        with self.assertRaisesRegex(ValueError, "semantic relation row"):
            check_support_artifact(
                _replace_relation(artifact, relation, mutated_relation),
            )


def _artifact_for_tetra() -> SupportArtifact:
    return _artifact_for_facts(tetrahedral_facts())


def _artifact_for_facts(
    facts,
    *,
    annotation_mode: AnnotationMode | None = None,
) -> SupportArtifact:
    policy = ordinary_policy_for_facts(facts)
    if annotation_mode is not None:
        policy = replace(policy, annotation_mode=annotation_mode)
    return compile_support_artifact(
        facts=facts,
        policy=policy,
        semantics=OrdinarySmilesSemantics(),
    )


def _replace_relation(artifact, old, new):
    return replace(
        artifact,
        relations=tuple(new if item == old else item for item in artifact.relations),
    )


def _replace_csp_space(artifact, old, new):
    return replace(
        artifact,
        csp_solution_spaces=tuple(
            new if item == old else item for item in artifact.csp_solution_spaces
        ),
    )


def _replace_traversal_decision(artifact, old, new):
    return replace(
        artifact,
        traversal_decisions=tuple(
            new if item == old else item for item in artifact.traversal_decisions
        ),
    )


def _replace_prefix_space(artifact, old, new):
    return replace(
        artifact,
        prefix_spaces=tuple(
            new if item == old else item for item in artifact.prefix_spaces
        ),
    )


def _replace_render_program(artifact, old, new):
    return replace(
        artifact,
        render_programs=tuple(
            new if item == old else item for item in artifact.render_programs
        ),
    )


def _replace_certified_witness(artifact, old, new):
    traced = artifact.traced_support
    return replace(
        artifact,
        traced_support=replace(
            traced,
            certified_witnesses=tuple(
                new if item == old else item for item in traced.certified_witnesses
            ),
        ),
    )


def _replace_node(artifact, old, new):
    return replace(
        artifact,
        nodes=tuple(new if item == old else item for item in artifact.nodes),
        edges=tuple(
            replace(
                edge,
                parent=new if edge.parent == old else edge.parent,
                child=new if edge.child == old else edge.child,
            )
            for edge in artifact.edges
        ),
        domains=tuple(
            replace(domain, owner=new)
            if domain.owner == old
            else domain
            for domain in artifact.domains
        ),
    )


def _mutate_first_piece_arg(pieces, kind, arg_index, new_value):
    out = []
    changed = False
    for piece_kind, args in pieces:
        if not changed and piece_kind == kind:
            mutable = list(args)
            mutable[arg_index] = new_value
            out.append((piece_kind, tuple(mutable)))
            changed = True
            continue
        out.append((piece_kind, args))
    if not changed:
        raise AssertionError(f"no render piece of kind {kind!r}")
    return tuple(out)


if __name__ == "__main__":
    unittest.main()
