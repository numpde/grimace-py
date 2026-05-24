"""Tests for compiled South Star support artifacts."""

from __future__ import annotations

import unittest
from dataclasses import replace

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

        with self.assertRaisesRegex(ValueError, "feasible solution coverage"):
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

        with self.assertRaisesRegex(ValueError, "feasible solution coverage"):
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


if __name__ == "__main__":
    unittest.main()
