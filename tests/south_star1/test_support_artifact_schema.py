"""Schema tests for South Star support artifacts."""

from __future__ import annotations

import copy
import unittest
from dataclasses import replace

from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.ordinary_semantics import OrdinarySmilesSemantics
from grimace._south_star1.proof_terms import sequence_hash
from grimace._south_star1.support_artifact import ArtifactNode
from grimace._south_star1.support_artifact import compile_support_artifact
from grimace._south_star1.support_artifact import support_artifact_from_jsonable_checked
from grimace._south_star1.support_artifact import support_artifact_to_jsonable
from grimace._south_star1.support_artifact_checker import check_support_artifact
from grimace._south_star1.support_artifact_schema import validate_support_artifact_jsonable
from grimace._south_star1.support_artifact_schema import validate_support_artifact_schema
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import tetrahedral_facts


class SupportArtifactSchemaTest(unittest.TestCase):
    def test_schema_accepts_generated_artifact(self) -> None:
        validate_support_artifact_schema(_artifact_for_tetra())

    def test_checked_json_loader_roundtrips_and_checks(self) -> None:
        artifact = _artifact_for_tetra()

        loaded = support_artifact_from_jsonable_checked(
            support_artifact_to_jsonable(artifact),
        )

        self.assertEqual(loaded, artifact)
        check_support_artifact(loaded)

    def test_schema_rejects_unknown_top_level_field(self) -> None:
        data = _artifact_json_for_tetra()
        data["unexpected"] = True

        with self.assertRaisesRegex(ValueError, "unknown field"):
            validate_support_artifact_jsonable(data)

    def test_schema_rejects_unknown_node_kind(self) -> None:
        data = _artifact_json_for_tetra()
        data["nodes"][0]["kind"] = "not_a_node"

        with self.assertRaisesRegex(ValueError, "node kind"):
            validate_support_artifact_jsonable(data)

    def test_schema_rejects_unknown_relation_name(self) -> None:
        data = _artifact_json_for_tetra()
        data["relations"][0]["name"] = "not_a_relation"

        with self.assertRaisesRegex(ValueError, "relation name"):
            validate_support_artifact_jsonable(data)

    def test_schema_rejects_duplicate_node(self) -> None:
        artifact = _artifact_for_tetra()
        mutated = replace(artifact, nodes=artifact.nodes + (artifact.nodes[0],))

        with self.assertRaisesRegex(ValueError, "duplicate artifact node"):
            validate_support_artifact_schema(mutated)

    def test_schema_rejects_duplicate_relation(self) -> None:
        artifact = _artifact_for_tetra()
        mutated = replace(
            artifact,
            relations=artifact.relations + (artifact.relations[0],),
        )

        with self.assertRaisesRegex(ValueError, "duplicate artifact relation"):
            validate_support_artifact_schema(mutated)

    def test_schema_rejects_edge_to_unknown_node(self) -> None:
        artifact = _artifact_for_tetra()
        edge = artifact.edges[0]
        mutated_edge = replace(
            edge,
            child=ArtifactNode(kind="witness", key=("unknown",)),
        )
        mutated = replace(
            artifact,
            edges=(mutated_edge,) + artifact.edges[1:],
        )

        with self.assertRaisesRegex(ValueError, "unknown child node"):
            validate_support_artifact_schema(mutated)

    def test_schema_rejects_domain_owner_unknown_node(self) -> None:
        artifact = _artifact_for_tetra()
        domain = artifact.domains[0]
        mutated = replace(
            artifact,
            domains=(
                replace(
                    domain,
                    owner=ArtifactNode(kind="witness", key=("unknown",)),
                ),
            )
            + artifact.domains[1:],
        )

        with self.assertRaisesRegex(ValueError, "unknown owner node"):
            validate_support_artifact_schema(mutated)

    def test_schema_rejects_render_program_unknown_witness(self) -> None:
        artifact = _artifact_for_tetra()
        program = artifact.render_programs[0]
        mutated = replace(
            artifact,
            render_programs=(
                replace(
                    program,
                    witness_node=ArtifactNode(kind="witness", key=("unknown",)),
                ),
            )
            + artifact.render_programs[1:],
        )

        with self.assertRaisesRegex(ValueError, "unknown witness node"):
            validate_support_artifact_schema(mutated)

    def test_schema_rejects_missing_required_field(self) -> None:
        data = _artifact_json_for_tetra()
        del data["header"]

        with self.assertRaisesRegex(ValueError, "missing required field"):
            validate_support_artifact_jsonable(data)

    def test_schema_rejects_noncanonical_allowed_row_order(self) -> None:
        data = _artifact_json_for_directional()
        relation = next(
            item
            for item in data["relations"]
            if len(item["allowed_rows"]) > 1
        )
        reversed_rows = list(reversed(relation["allowed_rows"]))
        self.assertNotEqual(reversed_rows, relation["allowed_rows"])
        relation["allowed_rows"] = reversed_rows
        relation["row_hash"] = sequence_hash(
            repr(tuple(row)) for row in reversed_rows
        )

        with self.assertRaisesRegex(ValueError, "noncanonical relation row"):
            support_artifact_from_jsonable_checked(data)

    def test_schema_rejects_malformed_facts_json_unknown_atom_endpoint(self) -> None:
        artifact = _artifact_for_tetra()
        facts_json = copy.deepcopy(artifact.facts_json)
        facts_json["bonds"][0]["a"] = 999
        mutated = replace(artifact, facts_json=facts_json)

        with self.assertRaisesRegex(ValueError, "bond endpoint"):
            validate_support_artifact_schema(mutated)

    def test_schema_rejects_malformed_facts_json_bad_tetra_reference_order(
        self,
    ) -> None:
        artifact = _artifact_for_tetra()
        facts_json = copy.deepcopy(artifact.facts_json)
        tetra = facts_json["stereo"]["tetrahedral"][0]
        tetra["reference_order"] = tetra["reference_order"][:-1]
        mutated = replace(artifact, facts_json=facts_json)

        with self.assertRaisesRegex(ValueError, "tetra reference"):
            validate_support_artifact_schema(mutated)

    def test_schema_rejects_malformed_policy_json_duplicate_ring_label(self) -> None:
        artifact = _artifact_for_tetra()
        policy_json = copy.deepcopy(artifact.policy_json)
        policy_json["ring_labels"] = [1, 1]
        mutated = replace(artifact, policy_json=policy_json)

        with self.assertRaisesRegex(ValueError, "ring labels"):
            validate_support_artifact_schema(mutated)

    def test_schema_rejects_malformed_policy_json_unknown_annotation_mode(
        self,
    ) -> None:
        artifact = _artifact_for_tetra()
        policy_json = copy.deepcopy(artifact.policy_json)
        policy_json["annotation_mode"] = "not_a_mode"
        mutated = replace(artifact, policy_json=policy_json)

        with self.assertRaisesRegex(ValueError, "annotation mode"):
            validate_support_artifact_schema(mutated)

    def test_checked_json_loader_rejects_unknown_schema_version(self) -> None:
        data = _artifact_json_for_tetra()
        data["header"]["schema_version"] = 999

        with self.assertRaisesRegex(ValueError, "schema version"):
            support_artifact_from_jsonable_checked(data)


def _artifact_for_tetra():
    return _artifact_for_facts(tetrahedral_facts())


def _artifact_for_directional():
    return _artifact_for_facts(directional_facts())


def _artifact_for_facts(facts):
    return compile_support_artifact(
        facts=facts,
        policy=ordinary_policy_for_facts(facts),
        semantics=OrdinarySmilesSemantics(),
    )


def _artifact_json_for_tetra():
    return copy.deepcopy(support_artifact_to_jsonable(_artifact_for_tetra()))


def _artifact_json_for_directional():
    return copy.deepcopy(support_artifact_to_jsonable(_artifact_for_directional()))


if __name__ == "__main__":
    unittest.main()
