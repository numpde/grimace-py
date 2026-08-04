"""Bounded directional rich support-artifact replay contracts."""

from __future__ import annotations

import unittest
from tests.south_star1.writer_support_artifact_directional_test_support import directional_discharge_key_pairs, directional_transition_manifest, directional_transition_branch_and_manifest, bond_occurrence_terms_for_branch
from tests.south_star1.writer_support_artifact_fixtures import directional_support_artifact_fixture, shared_acyclic_directional_support_artifact_fixture
from tests.south_star1.writer_test_fixtures import shared_directional_ring_carrier_facts
from tests.south_star1.writer_support_artifact_queries import support_strings
from tests.south_star1.writer_artifact_test_support import closed_term_field
from grimace._south_star1.ids import BondId
from types import SimpleNamespace
from tests.south_star1.writer_test_context import prepare_writer_facts
import grimace._south_star1.writer_stereo as writer_stereo_module
from grimace._south_star1.writer_support_artifact_checker import verify_writer_support_artifact_consistency
from grimace._south_star1.writer_support_artifact_fact_verifier import verify_writer_support_artifact_for_facts

class WriterSupportArtifactDirectionalAcyclicTest(unittest.TestCase):

    def test_directional_rooted_acyclic_artifact_replays_complete(self) -> None:
                fixture = directional_support_artifact_fixture()
                facts, options, artifact = fixture.facts, fixture.runtime_options, fixture.artifact

                structural = verify_writer_support_artifact_consistency(artifact)
                verification = verify_writer_support_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=artifact,
                )

                self.assertTrue(structural.accepted, structural.reason)
                self.assertEqual(structural.support_count, 2)
                self.assertEqual(structural.witness_count, 2)
                self.assertEqual(
                    tuple(sorted(support_strings(artifact))),
                    ("F/C=C/Cl", "F\\C=C\\Cl"),
                )
                self.assertTrue(verification.accepted, verification.reason)
                self.assertTrue(verification.offline_replay_complete)
                first = directional_transition_manifest(artifact, bond=1)
                second = directional_transition_manifest(artifact, bond=2)
                self.assertEqual(
                    [closed_term_field(key, "kind") for key in closed_term_field(first["transition_term"], "discharged_factor_keys")],
                    ["directional_bond_emission"],
                )
                self.assertEqual(
                    [closed_term_field(key, "kind") for key in closed_term_field(second["transition_term"], "discharged_factor_keys")],
                    ["directional_bond_emission", "directional_site"],
                )

    def test_shared_acyclic_directional_artifact_replays_complete(self) -> None:
                fixture = shared_acyclic_directional_support_artifact_fixture()
                facts, options, artifact = fixture.facts, fixture.runtime_options, fixture.artifact

                structural = verify_writer_support_artifact_consistency(artifact)
                verification = verify_writer_support_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=artifact,
                )

                self.assertTrue(structural.accepted, structural.reason)
                self.assertEqual(structural.support_count, 2)
                self.assertEqual(structural.witness_count, 2)
                # The shared bridge relation forces equal normalized signs for the two
                # carrier variables, leaving exactly the all-forward and all-reverse
                # renderings. This is a normalized-sign fact, not an RDKit expectation.
                self.assertEqual(
                    tuple(sorted(support_strings(artifact))),
                    ("F/C=C/C=C/Cl", "F\\C=C\\C=C\\Cl"),
                )
                self.assertTrue(verification.accepted, verification.reason)
                self.assertTrue(verification.offline_replay_complete)
                self.assertEqual(verification.offline_unchecked_obligation_families, ())

                bond0 = directional_transition_manifest(artifact, bond=0)
                bond2 = directional_transition_manifest(artifact, bond=2)
                bond4 = directional_transition_manifest(artifact, bond=4)
                self.assertEqual(len(closed_term_field(bond0["transition_term"], "carrier_models")), 1)
                self.assertEqual(len(closed_term_field(bond0["transition_term"], "restrictions")), 1)
                self.assertEqual(len(closed_term_field(bond2["transition_term"], "carrier_models")), 2)
                self.assertEqual(len(closed_term_field(bond2["transition_term"], "restrictions")), 2)
                self.assertEqual(len(closed_term_field(bond4["transition_term"], "carrier_models")), 1)
                self.assertEqual(len(closed_term_field(bond4["transition_term"], "restrictions")), 1)
                self.assertEqual(
                    directional_discharge_key_pairs(bond0),
                    (("directional_bond_emission", (0,)),),
                )
                self.assertEqual(
                    directional_discharge_key_pairs(bond2),
                    (
                        ("directional_bond_emission", (2,)),
                        ("directional_site", (0,)),
                    ),
                )
                self.assertEqual(
                    directional_discharge_key_pairs(bond4),
                    (
                        ("directional_bond_emission", (4,)),
                        ("directional_site", (1,)),
                    ),
                )
                branch, _manifest = directional_transition_branch_and_manifest(
                    artifact,
                    bond=2,
                )
                source_records = bond_occurrence_terms_for_branch(
                    artifact,
                    branch,
                    cursor_name="source_cursor",
                    bond=2,
                )
                successor_records = bond_occurrence_terms_for_branch(
                    artifact,
                    branch,
                    cursor_name="successor_cursor",
                    bond=2,
                )
                self.assertEqual(source_records, ())
                self.assertEqual(len(successor_records), 1)

    def test_shared_ring_carrier_supports_ring_transition_terms(self) -> None:
                facts = shared_directional_ring_carrier_facts()
                prepared = prepare_writer_facts(facts)

                models = writer_stereo_module._directional_models_for_bond(
                    prepared,
                    BondId(1),
                )

                self.assertEqual(len(models), 2)
                self.assertTrue(
                    writer_stereo_module
                    ._supports_directional_bond_emission_transition_term(
                        prepared,
                        BondId(1),
                        models,
                    )
                )
                self.assertTrue(
                    writer_stereo_module
                    ._supports_directional_ring_endpoint_projection_transition_term(
                        prepared,
                        SimpleNamespace(bond=BondId(1), bond_text=""),
                        models,
                    )
                )
