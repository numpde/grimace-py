from __future__ import annotations

import unittest

from tests.south_star1.writer_support_artifact_directional_test_support import (
    directional_discharge_key_pairs, directional_transition_manifest,
    directional_transition_branch_and_manifest, bond_occurrence_terms_for_branch,
    mutate_directional_restriction_sign, mutate_directional_canonical_orientation,
    mutate_directional_model_field, remove_directional_model, remove_directional_restriction,
    duplicate_directional_model_site, mutate_directional_successor_snapshot,
    set_directional_discharges, set_directional_discharges_by_keys,
    remove_raw_lifecycle_capability, mutate_directional_term_mark, mutate_directional_term_bond,
    remove_directional_successor_bond_occurrence, duplicate_directional_successor_bond_occurrence,
    mutate_directional_successor_snapshot_unrelated,
)
from tests.south_star1.writer_support_artifact_fixtures import (
    directional_support_artifact_fixture, shared_acyclic_directional_support_artifact_fixture,
)
from tests.south_star1.writer_support_artifact_queries import (
    require_structurally_valid_support_artifact,
    support_strings,
)
from tests.south_star1.writer_artifact_test_support import closed_term_field
from grimace._south_star1.writer_support_artifact_checker import verify_writer_support_artifact_consistency
from grimace._south_star1.writer_support_artifact_fact_verifier import verify_writer_support_artifact_for_facts


class WriterSupportArtifactDirectionalForgeryTest(unittest.TestCase):

    def test_shared_acyclic_directional_coherent_forgeries_reject_semantically(
                self,
            ) -> None:
                cases = (
                    (
                        "remove_model",
                        lambda artifact: remove_directional_model(artifact, bond=2),
                        "directional_carrier_model_mismatch",
                    ),
                    (
                        "remove_restriction",
                        lambda artifact: remove_directional_restriction(artifact, bond=2),
                        "directional_carrier_restriction_mismatch",
                    ),
                    (
                        "wrong_site",
                        lambda artifact: mutate_directional_model_field(
                            artifact,
                            bond=2,
                            field="site",
                            value=99,
                        ),
                        "directional_carrier_model_mismatch",
                    ),
                    (
                        "wrong_side",
                        lambda artifact: mutate_directional_model_field(
                            artifact,
                            bond=2,
                            field="side",
                            value="right",
                            model_index=1,
                        ),
                        "directional_carrier_model_mismatch",
                    ),
                    (
                        "wrong_ligand_factor",
                        lambda artifact: mutate_directional_model_field(
                            artifact,
                            bond=2,
                            field="ligand_factor",
                            value=-1,
                            model_index=1,
                        ),
                        "directional_carrier_model_mismatch",
                    ),
                    (
                        "wrong_normalized_sign",
                        lambda artifact: mutate_directional_restriction_sign(
                            artifact,
                            bond=2,
                        ),
                        "directional_carrier_restriction_mismatch",
                    ),
                    (
                        "duplicate_site_model",
                        lambda artifact: duplicate_directional_model_site(
                            artifact,
                            bond=2,
                        ),
                        "directional_carrier_model_mismatch",
                    ),
                    (
                        "omit_shared_capability",
                        lambda artifact: remove_raw_lifecycle_capability(
                            artifact,
                            bond=2,
                            capability="shared_directional_carrier_restriction",
                        ),
                        "tetra_residual_lifecycle_capabilities_mismatch",
                    ),
                    (
                        "omit_site0_discharge",
                        lambda artifact: set_directional_discharges_by_keys(
                            artifact,
                            bond=2,
                            key_pairs=(("directional_bond_emission", (2,)),),
                        ),
                        "directional_carrier_discharge_factor_mismatch",
                    ),
                    (
                        "premature_site1_discharge",
                        lambda artifact: set_directional_discharges_by_keys(
                            artifact,
                            bond=2,
                            key_pairs=(
                                ("directional_bond_emission", (2,)),
                                ("directional_site", (0,)),
                                ("directional_site", (1,)),
                            ),
                        ),
                        "directional_carrier_discharge_factor_mismatch",
                    ),
                    (
                        "duplicate_bond_occurrence",
                        lambda artifact: duplicate_directional_successor_bond_occurrence(
                            artifact,
                            bond=2,
                        ),
                        "directional_carrier_successor_bond_occurrence_mismatch",
                    ),
                )
                for name, mutate, reason in cases:
                    with self.subTest(name=name):
                        fixture = shared_acyclic_directional_support_artifact_fixture()
                        facts, options, artifact = fixture.facts, fixture.runtime_options, fixture.artifact
                        mutate(artifact)
                        require_structurally_valid_support_artifact(artifact)

                        verification = verify_writer_support_artifact_for_facts(
                            facts=facts,
                            runtime_options=options,
                            artifact=artifact,
                        )

                        self.assertFalse(verification.accepted)
                        self.assertIn(reason, verification.reason)

    def test_directional_carrier_coherent_forgeries_reject_semantically(self) -> None:
                cases = (
                    (
                        "wrong_normalized_sign",
                        lambda artifact: mutate_directional_restriction_sign(artifact, bond=1),
                        "directional_carrier_restriction_mismatch",
                    ),
                    (
                        "wrong_canonical_orientation",
                        lambda artifact: mutate_directional_canonical_orientation(artifact, bond=1),
                        "directional_carrier_canonical_orientation_mismatch",
                    ),
                    (
                        "carrier_model_wrong_side",
                        lambda artifact: mutate_directional_model_field(artifact, bond=1, field="side", value="right"),
                        "directional_carrier_model_mismatch",
                    ),
                    (
                        "carrier_model_wrong_ligand_factor",
                        lambda artifact: mutate_directional_model_field(artifact, bond=1, field="ligand_factor", value=-1),
                        "directional_carrier_model_mismatch",
                    ),
                    (
                        "false_successor_snapshot",
                        lambda artifact: mutate_directional_successor_snapshot(artifact, bond=1),
                        "directional_carrier_successor_state_anchor_mismatch",
                    ),
                    (
                        "missing_bond_emission_discharge",
                        lambda artifact: set_directional_discharges(artifact, bond=1, kinds=()),
                        "directional_carrier_discharge_factor_mismatch",
                    ),
                    (
                        "premature_site_discharge",
                        lambda artifact: set_directional_discharges(
                            artifact,
                            bond=1,
                            kinds=("directional_bond_emission", "directional_site"),
                        ),
                        "directional_carrier_discharge_factor_mismatch",
                    ),
                    (
                        "missing_site_discharge",
                        lambda artifact: set_directional_discharges(
                            artifact,
                            bond=2,
                            kinds=("directional_bond_emission",),
                        ),
                        "directional_carrier_discharge_factor_mismatch",
                    ),
                    (
                        "successor_bond_occurrence_wrong_mark",
                        lambda artifact: mutate_directional_term_mark(artifact, bond=1, value=-1),
                        "directional_carrier_residual_mark_mismatch",
                    ),
                    (
                        "successor_bond_occurrence_absent",
                        lambda artifact: remove_directional_successor_bond_occurrence(artifact, bond=1),
                        "directional_carrier_successor_bond_occurrence_mismatch",
                    ),
                    (
                        "unrelated_residual_component_changed",
                        lambda artifact: mutate_directional_successor_snapshot_unrelated(artifact, bond=1),
                        "directional_carrier_successor_state_anchor_mismatch",
                    ),
                )
                for name, mutate, reason in cases:
                    with self.subTest(name=name):
                        fixture = directional_support_artifact_fixture()
                        facts, options, artifact = fixture.facts, fixture.runtime_options, fixture.artifact
                        mutate(artifact)
                        require_structurally_valid_support_artifact(artifact)

                        verification = verify_writer_support_artifact_for_facts(
                            facts=facts,
                            runtime_options=options,
                            artifact=artifact,
                        )

                        self.assertFalse(verification.accepted)
                        self.assertIn(reason, verification.reason)
