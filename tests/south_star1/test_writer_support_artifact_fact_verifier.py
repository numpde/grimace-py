"""Facts-bound writer support artifact verifier tests."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import os
from types import SimpleNamespace
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import StereoFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_envelope_terms import _digest_terms_bounded
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_events import WriterRingEndpointEmitted
from grimace._south_star1.writer_events import WriterRingEndpointPaired
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import WriterDecoderBoundary
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot_prefix_envelope import (
    writer_snapshot_prefix_read_envelope_for_emitted_texts,
)
import grimace._south_star1.writer_stereo as writer_stereo_module
import grimace._south_star1.writer_support_artifact_offline_verifier as offline_verifier_module
from grimace._south_star1.writer_support_artifact_checker import artifact_manifest
from grimace._south_star1.writer_support_artifact_checker import artifact_metrics
from grimace._south_star1.writer_support_artifact_checker import (
    support_artifact_object_identity_term,
)
from grimace._south_star1.writer_support_artifact_checker import (
    verify_writer_support_artifact_consistency,
)
from grimace._south_star1.writer_support_artifact_fact_verifier import (
    OBJECT_KIND_OFFLINE_COVERAGE,
)
from grimace._south_star1.writer_support_artifact_fact_verifier import (
    verify_writer_support_artifact_for_facts,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    classify_residual_stereo_obligations_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    validate_writer_bracket_atom_text_against_facts,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_branch_projection_identities_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_count_dag_arithmetic,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_graph_ring_branch_deltas_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_local_branch_successor_evidence_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_support_image_coverage_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_support_string_replay_paths_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_terminal_support_identities_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_writer_support_artifact_offline_replay,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    verify_writer_support_artifact_envelope,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_prefix_read,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_snapshot,
)
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import shared_acyclic_directional_facts
from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.test_writer_stereo_residual import (
    _directional_non_single_ring_carrier_facts,
)
from tests.south_star1.test_writer_stereo_residual import (
    _directional_ring_carrier_facts,
)
from tests.south_star1.test_writer_stereo_residual import (
    _shared_directional_ring_carrier_facts,
)
from tests.south_star1.test_writer_snapshot import two_atom_facts

RUN_SLOW_ENV = "SOUTH_STAR1_RUN_SLOW"


class WriterSupportArtifactFactVerifierTest(unittest.TestCase):
    def test_linked_lifecycle_requires_replayed_residual_work(self) -> None:
        lifecycle = {
            "family": "stereo_lifecycle",
            "evidence_digest": "lifecycle",
            "linked_residual_work_digests": ["residual"],
            "is_noop": False,
            "is_empty": False,
            "is_discharged": True,
            "terminal_clean": False,
        }

        self.assertFalse(
            offline_verifier_module._obligation_manifest_checked(
                lifecycle,
                replayed_residual_digests=set(),
                replayed_lifecycle_digests=set(),
                replayed_directional_ring_closure_digests=set(),
            )
        )
        self.assertTrue(
            offline_verifier_module._obligation_manifest_checked(
                lifecycle,
                replayed_residual_digests={"residual"},
                replayed_lifecycle_digests={"lifecycle"},
                replayed_directional_ring_closure_digests=set(),
            )
        )
        lifecycle["linked_residual_work_digests"] = []
        self.assertTrue(
            offline_verifier_module._obligation_manifest_checked(
                lifecycle,
                replayed_residual_digests=set(),
                replayed_lifecycle_digests=set(),
                replayed_directional_ring_closure_digests=set(),
            )
        )

    def test_snapshot_artifact_verifies_against_matching_facts(self) -> None:
        facts = cco_facts()
        prepared = _prepare(facts)
        options = _writer_options()
        artifact = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared, options),
        )

        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=artifact,
        )

        self.assertTrue(verification.accepted, verification.reason)
        self.assertTrue(verification.structurally_checked)
        self.assertTrue(verification.facts_identity_checked)
        self.assertTrue(
            verification.offline_replay_complete,
            verification.offline_unchecked_obligation_families,
        )
        self.assertIn("support_string", verification.offline_checked_object_kinds)
        self.assertIn("replay_path", verification.offline_checked_object_kinds)
        self.assertIn("branch_support", verification.offline_checked_object_kinds)
        self.assertIn("count_envelope", verification.offline_checked_object_kinds)
        self.assertIn("count_dag", verification.offline_checked_object_kinds)
        self.assertIn(
            "count_dag_arithmetic",
            verification.offline_checked_relation_families,
        )
        self.assertIn(
            "support_image_coverage",
            verification.offline_checked_relation_families,
        )
        self.assertIn(
            "support_string_replay_path",
            verification.offline_checked_relation_families,
        )
        self.assertIn(
            "branch_projection_identity",
            verification.offline_checked_relation_families,
        )
        self.assertIn(
            "graph_ring_branch_delta",
            verification.offline_checked_relation_families,
        )
        self.assertIn(
            "local_branch_successor_evidence",
            verification.offline_checked_relation_families,
        )
        self.assertIn(
            "terminal_support_identity",
            verification.offline_checked_relation_families,
        )
        self.assertIn("support_image", verification.offline_checked_object_kinds)
        self.assertIn(
            "support_image_coverage",
            verification.offline_checked_object_kinds,
        )
        self.assertEqual(verification.support_count, 4)

    def test_prefix_artifact_verifies_against_matching_facts(self) -> None:
        facts = two_atom_facts()
        prepared = _prepare(facts)
        options = _writer_options()
        prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared, options),
            emitted_texts=("C", "C"),
        )
        artifact = writer_support_artifact_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
        )

        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=artifact,
        )

        self.assertTrue(verification.accepted, verification.reason)
        self.assertTrue(verification.structurally_checked)
        self.assertTrue(verification.facts_identity_checked)
        self.assertTrue(verification.offline_replay_complete)

    def test_facts_bound_verifier_reports_bracket_atom_offline_check(self) -> None:
        for smiles in ("[N+]", "[NH+]", "[NH2+]", "[NH3+]", "[NH4+]", "[O-]", "[OH-]"):
            with self.subTest(smiles=smiles):
                verification = _rdkit_artifact_verification(smiles)

                self.assertTrue(verification.accepted, verification.reason)
                self.assertIn(
                    "bracket_atom_text",
                    verification.offline_checked_relation_families,
                )
                self.assertIn("text_projection", verification.offline_checked_object_kinds)
                self.assertTrue(verification.offline_replay_complete)

    def test_facts_bound_verifier_reports_isotope_atom_offline_check(self) -> None:
        verification = _rdkit_artifact_verification("[13CH4]")

        self.assertTrue(verification.accepted, verification.reason)
        self.assertIn(
            "bracket_atom_text",
            verification.offline_checked_relation_families,
        )
        self.assertIn("text_projection", verification.offline_checked_object_kinds)
        self.assertTrue(verification.offline_replay_complete)

    def test_facts_bound_verifier_reports_joint_double_closure_offline_check(
        self,
    ) -> None:
        verification = _rdkit_artifact_verification("C1=CC1")

        self.assertTrue(verification.accepted, verification.reason)
        self.assertIn(
            "closure_bond_text",
            verification.offline_checked_relation_families,
        )
        self.assertTrue(verification.offline_replay_complete)

    def test_facts_bound_verifier_reports_joint_triple_closure_offline_check(
        self,
    ) -> None:
        verification = _rdkit_artifact_verification("C1#CC1")

        self.assertTrue(verification.accepted, verification.reason)
        self.assertIn(
            "closure_bond_text",
            verification.offline_checked_relation_families,
        )
        self.assertTrue(verification.offline_replay_complete)

    def test_offline_bracket_atom_replay_rejects_wrong_facts(self) -> None:
        with self.assertRaisesRegex(SouthStarError, "bracket_atom_text_facts_mismatch"):
            validate_writer_bracket_atom_text_against_facts(
                facts=_rdkit_facts("[NH3+]"),
                rendered_text="[NH4+]",
            )

        with self.assertRaisesRegex(SouthStarError, "bracket_atom_text_facts_mismatch"):
            validate_writer_bracket_atom_text_against_facts(
                facts=_rdkit_facts("[12CH4]"),
                rendered_text="[13CH4]",
            )
        with self.assertRaisesRegex(SouthStarError, "bracket_atom_text_facts_mismatch"):
            validate_writer_bracket_atom_text_against_facts(
                facts=_rdkit_facts("[O-]"),
                rendered_text="[OH-]",
            )
        with self.assertRaisesRegex(SouthStarError, "bracket_atom_text_facts_mismatch"):
            validate_writer_bracket_atom_text_against_facts(
                facts=_rdkit_facts("[NH4+]"),
                rendered_text="[NH3+]",
            )

    def test_offline_tetra_bracket_atom_replay_rejects_wrong_facts(self) -> None:
        facts = tetrahedral_facts()
        validate_writer_bracket_atom_text_against_facts(
            facts=facts,
            rendered_text="[C@H]",
        )
        validate_writer_bracket_atom_text_against_facts(
            facts=facts,
            rendered_text="[C@@H]",
        )
        double_recorded_h_facts = replace(
            facts,
            atoms=(
                replace(facts.atoms[0], implicit_h_count=1),
                *facts.atoms[1:],
            ),
        )
        validate_writer_bracket_atom_text_against_facts(
            facts=double_recorded_h_facts,
            rendered_text="[C@H]",
        )

        cases = (
            replace(facts, stereo=StereoFacts()),
            replace(facts, ligand_occurrences=facts.ligand_occurrences[:-1]),
            _tetra_facts_with_implicit_h_only_outside_specified_site(facts),
            replace(
                facts,
                atoms=(
                    replace(facts.atoms[0], implicit_h_count=2),
                    *facts.atoms[1:],
                ),
            ),
        )
        for wrong_facts in cases:
            with self.subTest(facts=wrong_facts):
                with self.assertRaisesRegex(
                    SouthStarError,
                    "bracket_atom_text_facts_mismatch",
                ):
                    validate_writer_bracket_atom_text_against_facts(
                        facts=wrong_facts,
                        rendered_text="[C@H]",
                    )
        for rendered_text in ("[N@H]", "[13C@H]", "[C@H+]"):
            with self.subTest(rendered_text=rendered_text):
                with self.assertRaisesRegex(
                    SouthStarError,
                    "bracket_atom_text_facts_mismatch",
                ):
                    validate_writer_bracket_atom_text_against_facts(
                        facts=facts,
                        rendered_text=rendered_text,
                    )

    def test_offline_joint_closure_replay_rejects_wrong_facts(self) -> None:
        artifact = _rdkit_artifact("C1=CC1")
        verification = verify_writer_support_artifact_offline_replay(
            facts=_rdkit_facts("C1CC1"),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("graph_ring_bond_marker_mismatch", verification.reason)

    def test_graph_ring_branch_deltas_accept_default_relation_fixtures(self) -> None:
        for smiles in (
            "CCO",
            "CC(C)O",
            "C1CC1",
            "C1CCC1",
            "C1=CC1",
            "C1#CC1",
            "[NH4+]",
            "[13CH4]",
        ):
            with self.subTest(smiles=smiles):
                artifact = _rdkit_artifact(smiles)

                verification = _graph_ring_delta_verification(
                    _rdkit_facts(smiles),
                    artifact,
                )

                self.assertTrue(verification.accepted, verification.reason)
                self.assertGreater(verification.checked_branches, 0)

    def test_graph_ring_branch_delta_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        branch = _first_graph_ring_delta_branch(artifact, "bond_advance")
        event = _first_graph_ring_delta_event(branch, "bond_emitted")
        event["bond"] = "missing"
        _refresh_graph_ring_delta_digest(branch["payload"]["graph_ring_delta"])

        wrong_bond = _graph_ring_delta_verification(_rdkit_facts("CCO"), artifact)

        self.assertFalse(wrong_bond.accepted)
        self.assertIn("local_closure_bond_missing", wrong_bond.reason)

        artifact = _rdkit_artifact("CCO")
        branch = _first_graph_ring_delta_branch(artifact, "atom_start")
        event = _first_graph_ring_delta_event(branch, "atom_emitted")
        event["atom"] = "missing"
        _refresh_graph_ring_delta_digest(branch["payload"]["graph_ring_delta"])

        wrong_atom = _graph_ring_delta_verification(_rdkit_facts("CCO"), artifact)

        self.assertFalse(wrong_atom.accepted)
        self.assertIn("local_atom_text_atom_missing", wrong_atom.reason)

        artifact = _rdkit_artifact("C1=CC1")
        branch = _first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        event = _first_graph_ring_delta_event(branch, "ring_endpoint_paired")
        event["label"] = "wrong"
        _refresh_graph_ring_delta_digest(branch["payload"]["graph_ring_delta"])

        wrong_label = _graph_ring_delta_verification(_rdkit_facts("C1=CC1"), artifact)

        self.assertFalse(wrong_label.accepted)
        self.assertIn("graph_ring_endpoint_label_mismatch", wrong_label.reason)

        artifact = _rdkit_artifact("C1=CC1")
        branch = _first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        branch["payload"]["successor_state_digest"] = "wrong"

        wrong_state = _graph_ring_delta_verification(_rdkit_facts("C1=CC1"), artifact)

        self.assertFalse(wrong_state.accepted)
        self.assertIn("graph_ring_delta_successor_state_digest_mismatch", wrong_state.reason)

    def test_graph_ring_directional_carrier_text_is_replayed_by_direction_mark(
        self,
    ) -> None:
        facts = _directional_non_single_ring_carrier_facts()
        prepared = _prepare(facts)
        options = _writer_options(rooted_at_atom=0)
        artifact = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared, options),
        )

        accepted = _graph_ring_delta_verification(facts, artifact)

        self.assertTrue(accepted.accepted, accepted.reason)

        branch = _first_directional_bond_delta_branch(artifact)
        event = _first_graph_ring_delta_event(branch, "bond_emitted")
        event["text"] = "="
        _refresh_graph_ring_delta_digest(branch["payload"]["graph_ring_delta"])

        wrong_direction_text = _graph_ring_delta_verification(facts, artifact)

        self.assertFalse(wrong_direction_text.accepted)
        self.assertIn("graph_ring_bond_marker_mismatch", wrong_direction_text.reason)
        self.assertIn("expected_direction_text", wrong_direction_text.reason)
        self.assertIn("direction_mark", wrong_direction_text.reason)
        self.assertIn("successor_certificate", wrong_direction_text.reason)

    def test_directional_ring_carrier_root_zero_artifact_builds_with_default_budget(
        self,
    ) -> None:
        if os.environ.get(RUN_SLOW_ENV) != "1":
            self.skipTest(
                f"set {RUN_SLOW_ENV}=1 to run the directional ring carrier artifact probe"
            )
        facts = _directional_ring_carrier_facts()
        options = _writer_options(rooted_at_atom=0)
        prepared = _prepare(facts)
        budget = WriterEnvelopeWorkBudget()

        self.assertEqual(budget.max_digest_term_bytes, 25_000_000)

        artifact = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared, options),
            budget=budget,
        )
        structural = verify_writer_support_artifact_consistency(
            artifact,
            budget=budget,
        )
        live = verify_writer_support_artifact_envelope(
            prepared=prepared,
            envelope=artifact,
            budget=budget,
        )
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=artifact,
            budget=budget,
        )
        classification = _obligation_classification(artifact, facts=facts)

        self.assertTrue(structural.accepted, structural.reason)
        self.assertTrue(live.accepted, live.reason)
        self.assertTrue(verification.accepted, verification.reason)
        self.assertTrue(
            verification.offline_replay_complete,
            verification.offline_unchecked_obligation_families,
        )
        self.assertEqual(verification.offline_unchecked_obligation_families, ())
        self.assertIn(
            "stereo_lifecycle",
            verification.offline_checked_obligation_families,
        )
        self.assertTrue(classification.accepted, classification.reason)
        branch = _first_graph_ring_delta_branch(artifact, "ring_endpoint_open")
        event = _first_graph_ring_delta_event(branch, "ring_endpoint_emitted")
        self.assertEqual(event["bond"], 3)
        self.assertLessEqual(
            artifact["metrics"]["largest_object_identity_input_bytes"],
            budget.max_digest_term_bytes,
        )

    def test_reduced_directional_ring_opening_artifact_replays_semantically(self) -> None:
        facts, options, artifact = _directional_ring_opening_artifact()

        structural = verify_writer_support_artifact_consistency(artifact)
        live = verify_writer_support_artifact_envelope(
            prepared=_prepare(facts),
            envelope=artifact,
        )
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=artifact,
        )

        self.assertTrue(structural.accepted, structural.reason)
        self.assertTrue(live.accepted, live.reason)
        self.assertTrue(verification.accepted, verification.reason)
        manifests = [
            manifest
            for obj in artifact["objects"]
            if obj["kind"] == "branch_support"
            for manifest in obj["payload"]["obligation_manifests"]["residual_work"]
            if manifest["operation"] == "directional ring endpoint projection"
        ]
        self.assertTrue(manifests)
        snapshots = [
            (
                _term_field(manifest["transition_term"], "source_snapshot"),
                _term_field(manifest["transition_term"], "successor_snapshot"),
            )
            for manifest in manifests
        ]
        self.assertTrue(any(source == successor for source, successor in snapshots))
        self.assertTrue(any(source != successor for source, successor in snapshots))

    def test_reduced_directional_ring_pair_artifacts_replay_semantically(self) -> None:
        for first_mark in (DirectionMark.ABSENT, DirectionMark.FWD):
            with self.subTest(first_mark=first_mark):
                facts, options, artifact = _directional_ring_pair_artifact(first_mark)
                structural = verify_writer_support_artifact_consistency(artifact)
                live = verify_writer_support_artifact_envelope(
                    prepared=_prepare(facts),
                    envelope=artifact,
                )
                verification = verify_writer_support_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=artifact,
                )

                self.assertTrue(structural.accepted, structural.reason)
                self.assertTrue(live.accepted, live.reason)
                self.assertTrue(verification.accepted, verification.reason)
                self.assertTrue(verification.offline_replay_complete)
                self.assertEqual(verification.offline_unchecked_obligation_families, ())
                branch, manifest = _ring_pair_branch_and_manifest(artifact)
                self.assertEqual(
                    _term_field(manifest["transition_term"], "first_endpoint_direction_mark")["value"],
                    first_mark.value,
                )
                self.assertEqual(
                    branch["payload"]["graph_ring_delta"]["kind"],
                    "ring_endpoint_pair",
                )

    def test_directional_ring_pair_coherent_term_forgeries_are_rejected(self) -> None:
        cases = (
            ("missing_term", _forge_ring_pair_missing_term, "directional_ring_pair_transition_missing"),
            ("compatible_choices", _forge_ring_pair_compatible_choices, "directional_ring_pair_compatible_choices_mismatch"),
            ("first_mark", _forge_ring_pair_first_mark, "directional_ring_pair_event_first_endpoint_direction_mark_mismatch"),
            ("second_mark", _forge_ring_pair_second_mark, "directional_ring_pair_event_direction_mark_mismatch"),
            ("orientation", _forge_ring_pair_orientation, "directional_ring_pair_canonical_orientation_mismatch"),
            ("carrier", _forge_ring_pair_carrier, "directional_ring_pair_carrier_model_mismatch"),
            ("restriction", _forge_ring_pair_restriction, "directional_ring_pair_restriction_mismatch"),
            ("occurrence", _forge_ring_pair_occurrence, "directional_ring_pair_bond_occurrence_mismatch"),
            ("discharge", _forge_ring_pair_discharge, "directional_ring_pair_discharge_factor_mismatch"),
            ("successor", _forge_ring_pair_successor, "directional_ring_pair_successor_state_anchor_mismatch"),
        )
        facts, options, original = _directional_ring_pair_artifact(DirectionMark.ABSENT)
        for name, mutate, reason in cases:
            with self.subTest(name=name):
                artifact = deepcopy(original)
                mutate(artifact)
                _assert_structural_checker_accepts(self, artifact)
                verification = verify_writer_support_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=artifact,
                )
                self.assertFalse(verification.accepted)
                self.assertIn(reason, verification.reason)

    def test_directional_ring_opening_coherent_term_forgeries_are_rejected(self) -> None:
        facts, options, original = _directional_ring_opening_artifact()
        cases = (
            ("compatible_seconds", _forge_ring_compatible_seconds),
            ("domain_intersection", _forge_ring_domain_intersection),
            ("carrier_orientation", _forge_ring_carrier_orientation),
            ("event_mark_detached", _forge_ring_term_mark),
            ("false_noop", _forge_ring_false_noop),
            ("false_change", _forge_ring_false_change),
            ("factor_discharge", _forge_ring_factor_discharge),
            ("snapshot_detached", _forge_ring_source_snapshot),
            ("successor_open_endpoint", _forge_ring_successor_open_endpoint),
            ("bond_occurrence_added", _forge_ring_bond_occurrence_added),
            ("missing_term", _forge_ring_missing_term),
            ("lifecycle_operation", _forge_ring_lifecycle_operation),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                artifact = deepcopy(original)
                mutate(artifact)
                _assert_structural_checker_accepts(self, artifact)
                verification = verify_writer_support_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=artifact,
                )
                self.assertFalse(verification.accepted)

    def test_directional_rooted_acyclic_artifact_replays_complete(self) -> None:
        facts, options, artifact = _directional_rooted_artifact()

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
            tuple(sorted(_support_strings(artifact))),
            ("F/C=C/Cl", "F\\C=C\\Cl"),
        )
        self.assertTrue(verification.accepted, verification.reason)
        self.assertTrue(verification.offline_replay_complete)
        first = _directional_transition_manifest(artifact, bond=1)
        second = _directional_transition_manifest(artifact, bond=2)
        self.assertEqual(
            [_term_field(key, "kind") for key in _term_field(first["transition_term"], "discharged_factor_keys")],
            ["directional_bond_emission"],
        )
        self.assertEqual(
            [_term_field(key, "kind") for key in _term_field(second["transition_term"], "discharged_factor_keys")],
            ["directional_bond_emission", "directional_site"],
        )

    def test_shared_acyclic_directional_artifact_replays_complete(self) -> None:
        facts, options, artifact = _shared_acyclic_directional_artifact()

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
            tuple(sorted(_support_strings(artifact))),
            ("F/C=C/C=C/Cl", "F\\C=C\\C=C\\Cl"),
        )
        self.assertTrue(verification.accepted, verification.reason)
        self.assertTrue(verification.offline_replay_complete)
        self.assertEqual(verification.offline_unchecked_obligation_families, ())

        bond0 = _directional_transition_manifest(artifact, bond=0)
        bond2 = _directional_transition_manifest(artifact, bond=2)
        bond4 = _directional_transition_manifest(artifact, bond=4)
        self.assertEqual(len(_term_field(bond0["transition_term"], "carrier_models")), 1)
        self.assertEqual(len(_term_field(bond0["transition_term"], "restrictions")), 1)
        self.assertEqual(len(_term_field(bond2["transition_term"], "carrier_models")), 2)
        self.assertEqual(len(_term_field(bond2["transition_term"], "restrictions")), 2)
        self.assertEqual(len(_term_field(bond4["transition_term"], "carrier_models")), 1)
        self.assertEqual(len(_term_field(bond4["transition_term"], "restrictions")), 1)
        self.assertEqual(
            _directional_discharge_key_pairs(bond0),
            (("directional_bond_emission", (0,)),),
        )
        self.assertEqual(
            _directional_discharge_key_pairs(bond2),
            (
                ("directional_bond_emission", (2,)),
                ("directional_site", (0,)),
            ),
        )
        self.assertEqual(
            _directional_discharge_key_pairs(bond4),
            (
                ("directional_bond_emission", (4,)),
                ("directional_site", (1,)),
            ),
        )
        branch, _manifest = _directional_transition_branch_and_manifest(
            artifact,
            bond=2,
        )
        source_records = _bond_occurrence_terms_for_branch(
            artifact,
            branch,
            cursor_name="source_cursor",
            bond=2,
        )
        successor_records = _bond_occurrence_terms_for_branch(
            artifact,
            branch,
            cursor_name="successor_cursor",
            bond=2,
        )
        self.assertEqual(source_records, ())
        self.assertEqual(len(successor_records), 1)

    def test_shared_acyclic_directional_coherent_forgeries_reject_semantically(
        self,
    ) -> None:
        cases = (
            (
                "remove_model",
                lambda artifact: _remove_directional_model(artifact, bond=2),
                "directional_carrier_model_mismatch",
            ),
            (
                "remove_restriction",
                lambda artifact: _remove_directional_restriction(artifact, bond=2),
                "directional_carrier_restriction_mismatch",
            ),
            (
                "wrong_site",
                lambda artifact: _mutate_directional_model_field(
                    artifact,
                    bond=2,
                    field="site",
                    value=99,
                ),
                "directional_carrier_model_mismatch",
            ),
            (
                "wrong_side",
                lambda artifact: _mutate_directional_model_field(
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
                lambda artifact: _mutate_directional_model_field(
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
                lambda artifact: _mutate_directional_restriction_sign(
                    artifact,
                    bond=2,
                ),
                "directional_carrier_restriction_mismatch",
            ),
            (
                "duplicate_site_model",
                lambda artifact: _duplicate_directional_model_site(
                    artifact,
                    bond=2,
                ),
                "directional_carrier_model_mismatch",
            ),
            (
                "omit_shared_capability",
                lambda artifact: _remove_raw_lifecycle_capability(
                    artifact,
                    bond=2,
                    capability="shared_directional_carrier_restriction",
                ),
                "tetra_residual_lifecycle_capabilities_mismatch",
            ),
            (
                "omit_site0_discharge",
                lambda artifact: _set_directional_discharges_by_keys(
                    artifact,
                    bond=2,
                    key_pairs=(("directional_bond_emission", (2,)),),
                ),
                "directional_carrier_discharge_factor_mismatch",
            ),
            (
                "premature_site1_discharge",
                lambda artifact: _set_directional_discharges_by_keys(
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
                lambda artifact: _duplicate_directional_successor_bond_occurrence(
                    artifact,
                    bond=2,
                ),
                "directional_carrier_successor_bond_occurrence_mismatch",
            ),
        )
        for name, mutate, reason in cases:
            with self.subTest(name=name):
                facts, options, artifact = _shared_acyclic_directional_artifact()
                mutate(artifact)
                _assert_structural_checker_accepts(self, artifact)

                verification = verify_writer_support_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=artifact,
                )

                self.assertFalse(verification.accepted)
                self.assertIn(reason, verification.reason)

    def test_shared_ring_carrier_supports_ring_transition_terms(self) -> None:
        facts = _shared_directional_ring_carrier_facts()
        prepared = _prepare(facts)

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

    def test_directional_carrier_coherent_forgeries_reject_semantically(self) -> None:
        cases = (
            (
                "wrong_normalized_sign",
                lambda artifact: _mutate_directional_restriction_sign(artifact, bond=1),
                "directional_carrier_restriction_mismatch",
            ),
            (
                "wrong_canonical_orientation",
                lambda artifact: _mutate_directional_canonical_orientation(artifact, bond=1),
                "directional_carrier_canonical_orientation_mismatch",
            ),
            (
                "carrier_model_wrong_side",
                lambda artifact: _mutate_directional_model_field(artifact, bond=1, field="side", value="right"),
                "directional_carrier_model_mismatch",
            ),
            (
                "carrier_model_wrong_ligand_factor",
                lambda artifact: _mutate_directional_model_field(artifact, bond=1, field="ligand_factor", value=-1),
                "directional_carrier_model_mismatch",
            ),
            (
                "false_successor_snapshot",
                lambda artifact: _mutate_directional_successor_snapshot(artifact, bond=1),
                "directional_carrier_successor_state_anchor_mismatch",
            ),
            (
                "missing_bond_emission_discharge",
                lambda artifact: _set_directional_discharges(artifact, bond=1, kinds=()),
                "directional_carrier_discharge_factor_mismatch",
            ),
            (
                "premature_site_discharge",
                lambda artifact: _set_directional_discharges(
                    artifact,
                    bond=1,
                    kinds=("directional_bond_emission", "directional_site"),
                ),
                "directional_carrier_discharge_factor_mismatch",
            ),
            (
                "missing_site_discharge",
                lambda artifact: _set_directional_discharges(
                    artifact,
                    bond=2,
                    kinds=("directional_bond_emission",),
                ),
                "directional_carrier_discharge_factor_mismatch",
            ),
            (
                "successor_bond_occurrence_wrong_mark",
                lambda artifact: _mutate_directional_term_mark(artifact, bond=1, value=-1),
                "directional_carrier_residual_mark_mismatch",
            ),
            (
                "successor_bond_occurrence_absent",
                lambda artifact: _remove_directional_successor_bond_occurrence(artifact, bond=1),
                "directional_carrier_successor_bond_occurrence_mismatch",
            ),
            (
                "unrelated_residual_component_changed",
                lambda artifact: _mutate_directional_successor_snapshot_unrelated(artifact, bond=1),
                "directional_carrier_successor_state_anchor_mismatch",
            ),
        )
        for name, mutate, reason in cases:
            with self.subTest(name=name):
                facts, options, artifact = _directional_rooted_artifact()
                mutate(artifact)
                _assert_structural_checker_accepts(self, artifact)

                verification = verify_writer_support_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=artifact,
                )

                self.assertFalse(verification.accepted)
                self.assertIn(reason, verification.reason)

    def test_default_corpus_obligations_are_classified(self) -> None:
        cases = {
            "CCO": (),
            "CC(C)O": (),
            "C1CC1": (),
            "C1CCC1": (),
            "C1=CC1": (),
            "C1#CC1": (),
            "[NH4+]": (),
            "[13CH4]": (),
        }
        for smiles, unchecked_families in cases.items():
            with self.subTest(smiles=smiles):
                facts = _rdkit_facts(smiles)
                artifact = _rdkit_artifact(smiles)

                classification = _obligation_classification(artifact, facts=facts)
                verification = verify_writer_support_artifact_for_facts(
                    facts=facts,
                    runtime_options=_writer_options(),
                    artifact=artifact,
                )

                self.assertTrue(classification.accepted, classification.reason)
                self.assertEqual(classification.unchecked_families, unchecked_families)
                self.assertTrue(classification.stereo_obligations_present)
                self.assertEqual(
                    classification.graph_obligations_present,
                    True,
                )
                self.assertEqual(
                    classification.residual_obligations_present,
                    smiles.startswith("C1"),
                )
                self.assertIn("stereo_lifecycle", classification.checked_families)
                self.assertIn(
                    "residual_work_checked_empty",
                    classification.checked_empty_families,
                )
                self.assertTrue(verification.accepted, verification.reason)
                self.assertEqual(
                    verification.offline_replay_complete,
                    not unchecked_families,
                )
                self.assertEqual(verification.offline_unchecked_object_kinds, ())
                self.assertEqual(
                    verification.offline_unchecked_obligation_families,
                    unchecked_families,
                )
                self.assertIn(
                    "stereo_lifecycle",
                    verification.offline_checked_obligation_families,
                )
                self.assertIn(
                    "terminal_graph_obligation_work",
                    verification.offline_checked_obligation_families,
                )
                self.assertIn(
                    "terminal_stereo_lifecycle",
                    verification.offline_checked_obligation_families,
                )
                self.assertIn(
                    "residual_work_checked_empty",
                    verification.offline_empty_obligation_families,
                )

    def test_specified_tetra_raw_smiles_blocks_without_potential_sites(
        self,
    ) -> None:
        with self.assertRaisesRegex(
            SouthStarError,
            "raw tetrahedral stereo has no ordinary potential site",
        ) as raised:
            ordinary_molecule_facts_from_smiles(
                "[C@H](F)(Cl)Br",
                RdkitOrdinaryExtractionOptions(include_potential_sites=False),
            )

        self.assertIs(raised.exception.kind, SouthStarErrorKind.UNSUPPORTED_STEREO)

    def test_supported_specified_tetra_artifact_is_offline_complete(
        self,
    ) -> None:
        facts = tetrahedral_facts()
        site = facts.stereo.tetrahedral[0]
        self.assertIs(site.status, SiteStatus.SPECIFIED)
        self.assertIs(site.target, TetraValue.PLUS)
        prepared = _prepare(facts)
        options = _writer_options()
        artifact = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared, options),
        )

        structural = verify_writer_support_artifact_consistency(artifact)
        live = verify_writer_support_artifact_envelope(
            prepared=prepared,
            envelope=artifact,
        )
        classification = _obligation_classification(artifact, facts=facts)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=artifact,
        )

        self.assertTrue(structural.accepted, structural.reason)
        self.assertTrue(live.accepted, live.reason)
        self.assertEqual(structural.support_count, 12)
        self.assertEqual(
            tuple(_support_strings(artifact)),
            (
                "Br[C@@H](Cl)F",
                "Br[C@H](F)Cl",
                "Cl[C@@H](F)Br",
                "Cl[C@H](Br)F",
                "F[C@@H](Br)Cl",
                "F[C@H](Cl)Br",
                "[C@@H](Br)(Cl)F",
                "[C@@H](Cl)(F)Br",
                "[C@@H](F)(Br)Cl",
                "[C@H](Br)(F)Cl",
                "[C@H](Cl)(Br)F",
                "[C@H](F)(Cl)Br",
            ),
        )
        self.assertTrue(classification.accepted, classification.reason)
        self.assertTrue(classification.residual_obligations_present)
        self.assertTrue(classification.stereo_obligations_present)
        self.assertEqual(classification.unchecked_families, ())
        self.assertNotIn("residual_work", classification.unchecked_families)
        self.assertIn("residual_work", classification.checked_families)
        self.assertIn("stereo_lifecycle", classification.checked_families)
        self.assertIn("terminal_stereo_lifecycle", classification.checked_families)
        self.assertTrue(verification.accepted, verification.reason)
        self.assertTrue(verification.structurally_checked)
        self.assertTrue(verification.facts_identity_checked)
        self.assertTrue(verification.offline_replay_complete)
        self.assertEqual(verification.offline_unchecked_object_kinds, ())
        self.assertEqual(verification.offline_unchecked_obligation_families, ())
        self.assertIn(
            "residual_work",
            verification.offline_checked_obligation_families,
        )
        self.assertIn(
            "bracket_atom_text",
            verification.offline_checked_relation_families,
        )

    def test_specified_tetra_residual_manifest_digest_mismatch_is_rejected(
        self,
    ) -> None:
        facts = tetrahedral_facts()
        prepared = _prepare(facts)
        options = _writer_options()
        artifact = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared, options),
        )
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = branch["payload"]["obligation_manifests"]["residual_work"][0]
        manifest["source_digest"] = "wrong"

        classification = _obligation_classification(artifact, facts=facts)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=artifact,
        )

        self.assertFalse(classification.accepted)
        self.assertIn("tetra_residual_source_digest_mismatch", classification.reason)
        self.assertFalse(verification.accepted)
        self.assertIn("object_digest_mismatch", verification.reason)

    def test_specified_tetra_transition_rejects_state_detached_successor_token_domain(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        successor = _term_field(manifest["transition_term"], "successor_snapshot")
        domains = _term_field(successor, "domains")
        token_domain = next(
            domain
            for var, domain in domains
            if _term_field(var, "kind") == "tetra_token"
        )
        token = _term_field(manifest["transition_term"], "token")
        wrong_value = "@" if token["value"] == "@@" else "@@"
        token_domain[:] = [
            {
                "__enum__": "grimace._south_star1.policy.TetraToken",
                "value": wrong_value,
            }
        ]
        successor_digest = _closed_term_digest(successor)
        _set_term_field(
            manifest["transition_term"],
            "successor_snapshot_digest",
            successor_digest,
        )
        _refresh_linked_raw_lifecycle_residual_digest(
            branch,
            manifest=manifest,
            field="successor_residual_snapshot_digest",
            digest=successor_digest,
        )
        _refresh_transition_manifest_digest(manifest)
        _refresh_object_and_artifact_digest(artifact, branch)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_successor_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_flipped_local_order_parity(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral local-order factor closure"
        )
        target = _term_field(manifest["transition_term"], "target_parity")
        target["value"] = "odd" if target["value"] == "even" else "even"
        constraint = _term_field(manifest["transition_term"], "constraint_value")
        constraint["value"] = target["value"]
        _refresh_transition_manifest_digest(manifest)
        _refresh_object_and_artifact_digest(artifact, branch)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_target_parity_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_flipped_reference_order(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral local-order factor closure"
        )
        reference_order = _term_field(manifest["transition_term"], "reference_order")
        reference_order[:] = list(reversed(reference_order))
        _refresh_transition_manifest_digest(manifest)
        _refresh_object_and_artifact_digest(artifact, branch)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_reference_order_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_event_detached_local_order(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral local-order factor closure"
        )
        local_order = _term_field(manifest["transition_term"], "local_order")
        local_order[:] = list(reversed(local_order))
        _refresh_transition_manifest_digest(manifest)
        _refresh_object_and_artifact_digest(artifact, branch)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_event_order_mismatch",
            classification.reason,
        )

    def test_specified_tetra_local_order_rejects_coherent_detached_record_identity(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        event = _first_graph_ring_delta_event(branch, "local_order_closed")
        event["successor_local_order_record_digest"] = _different_local_order_digest(
            artifact,
            branch=branch,
            cursor_name="successor_cursor",
            atom=event["atom"],
        )
        _refresh_local_order_event_identity_digest(event)
        _refresh_graph_ring_delta_digest(branch["payload"]["graph_ring_delta"])
        _refresh_object_and_artifact_digest(artifact, branch)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_successor_record_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_local_order_rejects_coherent_detached_source_record_identity(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        event = _first_graph_ring_delta_event(branch, "local_order_closed")
        event["source_local_order_record_digest"] = _different_local_order_digest(
            artifact,
            branch=branch,
            cursor_name="source_cursor",
            atom=event["atom"],
        )
        _refresh_local_order_event_identity_digest(event)
        _refresh_graph_ring_delta_digest(branch["payload"]["graph_ring_delta"])
        _refresh_object_and_artifact_digest(artifact, branch)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_source_record_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_missing_factor_discharge(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral local-order factor closure"
        )
        _set_term_field(manifest["transition_term"], "discharged_factor_keys", [])
        _refresh_transition_manifest_digest(manifest)
        _refresh_object_and_artifact_digest(artifact, branch)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_discharge_factor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_state_detached_successor_discharge(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral local-order factor closure"
        )
        transition = manifest["transition_term"]
        source = _term_field(transition, "source_snapshot")
        successor = _term_field(transition, "successor_snapshot")
        discharged = _term_field(transition, "discharged_factor_keys")
        source_factor = next(
            factor
            for factor in _term_field(source, "factors")
            if _term_field(factor, "key") == discharged[0]
        )
        _term_field(successor, "factors").append(source_factor)
        successor_digest = _closed_term_digest(successor)
        _set_term_field(
            transition,
            "successor_snapshot_digest",
            successor_digest,
        )
        _refresh_linked_raw_lifecycle_residual_digest(
            branch,
            manifest=manifest,
            field="successor_residual_snapshot_digest",
            digest=successor_digest,
        )
        _refresh_transition_manifest_digest(manifest)
        _refresh_object_and_artifact_digest(artifact, branch)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_successor_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_state_detached_source_projection(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral local-order factor closure"
        )
        source = _term_field(manifest["transition_term"], "source_snapshot")
        constraint_var = _term_field(manifest["transition_term"], "constraint_var")
        domains = _term_field(source, "domains")
        domains[:] = [
            item
            for item in domains
            if item[0] != constraint_var
        ]
        source_digest = _closed_term_digest(source)
        _set_term_field(
            manifest["transition_term"],
            "source_snapshot_digest",
            source_digest,
        )
        _refresh_linked_raw_lifecycle_residual_digest(
            branch,
            manifest=manifest,
            field="source_residual_snapshot_digest",
            digest=source_digest,
        )
        _refresh_transition_manifest_digest(manifest)
        _refresh_object_and_artifact_digest(artifact, branch)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_source_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_state_detached_successor_component(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        successor = _term_field(manifest["transition_term"], "successor_snapshot")
        domains = _term_field(successor, "domains")
        domains.append(
            [
                {
                    "__dataclass__": "grimace._south_star1.residual_constraints.VarId",
                    "fields": [["kind", "unrelated_test_component"], ["key", [99]]],
                },
                [False, True],
            ]
        )
        successor_digest = _closed_term_digest(successor)
        _set_term_field(
            manifest["transition_term"],
            "successor_snapshot_digest",
            successor_digest,
        )
        _refresh_linked_raw_lifecycle_residual_digest(
            branch,
            manifest=manifest,
            field="successor_residual_snapshot_digest",
            digest=successor_digest,
        )
        _refresh_transition_manifest_digest(manifest)
        _refresh_object_and_artifact_digest(artifact, branch)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_successor_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_state_detached_wrong_successor(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        successor = _term_field(manifest["transition_term"], "successor_snapshot")
        assignments = _term_field(successor, "assignments")
        assignments[:] = []
        successor_digest = _closed_term_digest(successor)
        _set_term_field(
            manifest["transition_term"],
            "successor_snapshot_digest",
            successor_digest,
        )
        _refresh_linked_raw_lifecycle_residual_digest(
            branch,
            manifest=manifest,
            field="successor_residual_snapshot_digest",
            digest=successor_digest,
        )
        _refresh_transition_manifest_digest(manifest)
        _refresh_object_and_artifact_digest(artifact, branch)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_successor_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_transition_rejects_branch_detached_snapshots(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        successor = _term_field(manifest["transition_term"], "successor_snapshot")
        _set_term_field(manifest["transition_term"], "source_snapshot", successor)
        _set_term_field(
            manifest["transition_term"],
            "source_snapshot_digest",
            _closed_term_digest(successor),
        )
        _refresh_transition_manifest_digest(manifest)
        _refresh_object_and_artifact_digest(artifact, branch)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_source_lifecycle_mismatch",
            classification.reason,
        )

    def test_specified_tetra_atom_token_rejects_coherent_detached_residual_snapshots(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        for field in ("source_snapshot", "successor_snapshot"):
            snapshot = _term_field(manifest["transition_term"], field)
            _term_field(snapshot, "domains").append(
                [
                    {
                        "__dataclass__": "grimace._south_star1.residual_constraints.VarId",
                        "fields": [["kind", "detached_test_component"], ["key", [123]]],
                    },
                    [False, True],
                ]
            )
            digest = _closed_term_digest(snapshot)
            _set_term_field(
                manifest["transition_term"],
                f"{field}_digest",
                digest,
            )
            lifecycle_field = (
                "source_residual_snapshot_digest"
                if field == "source_snapshot"
                else "successor_residual_snapshot_digest"
            )
            _refresh_linked_raw_lifecycle_residual_digest(
                branch,
                manifest=manifest,
                field=lifecycle_field,
                digest=digest,
            )
        _refresh_transition_manifest_digest(manifest)
        _refresh_object_and_artifact_digest(artifact, branch)
        _assert_structural_checker_accepts(self, artifact)

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_transition_source_state_anchor_mismatch",
            classification.reason,
        )

    def test_specified_tetra_atom_token_residual_event_atom_is_bound(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        event = _first_graph_ring_delta_event(branch, "atom_emitted")
        event["atom"] = 1

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_atom_token_residual_atom_mismatch",
            classification.reason,
        )

    def test_specified_tetra_atom_token_residual_token_matches_text(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        event = _first_graph_ring_delta_event(branch, "atom_emitted")
        token = event["tetra_token"]
        token["value"] = "@@" if token["value"] == "@" else "@"

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_atom_token_residual_token_mismatch",
            classification.reason,
        )

    def test_specified_tetra_local_order_closed_atom_is_center(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        event = _first_graph_ring_delta_event(branch, "local_order_closed")
        event["atom"] = 1

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_residual_center_mismatch",
            classification.reason,
        )

    def test_specified_tetra_local_order_event_parent_is_center(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        event = _first_graph_ring_delta_event(branch, "atom_emitted")
        event["parent"] = event["atom"]

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_residual_parent_mismatch",
            classification.reason,
        )

    def test_specified_tetra_local_order_event_bond_connects_ligand(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral local-order factor closure",
        )
        event = _first_graph_ring_delta_event(branch, "atom_emitted")
        event["incoming_bond"] = 99

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_local_order_residual_bond_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_requires_exact_lifecycle_operations(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        lifecycle = branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
        branch_certificate = next(
            item
            for item in lifecycle
            if item["operation"] == "WriterStereoBranchCertificate"
        )
        branch_certificate["operation"] = "UnexpectedStereoCertificate"

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_lifecycle_operation_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_requires_lifecycle_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        manifest["linked_lifecycle_digests"] = []

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_no_second_authority_from_atom_token_text(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        event = _first_graph_ring_delta_event(branch, "atom_emitted")
        self.assertIn(event["tetra_token"]["value"], ("@", "@@"))
        self.assertTrue(
            any(event["tetra_token"]["value"] in text for text in _support_strings(artifact))
        )
        manifest["linked_lifecycle_digests"] = []

        classification = _obligation_classification(artifact, facts=facts)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )
        self.assertFalse(verification.accepted)
        self.assertIn("object_digest_mismatch", verification.reason)

    def test_specified_tetra_residual_rejects_wrong_lifecycle_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        manifest["linked_lifecycle_digests"] = ["wrong"]

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_unrelated_lifecycle_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = _append_unrelated_raw_lifecycle(branch, manifest=manifest)
        manifest["linked_lifecycle_digests"] = [unrelated["evidence_digest"]]

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_extra_lifecycle_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = _append_unrelated_raw_lifecycle(branch, manifest=manifest)
        manifest["linked_lifecycle_digests"].append(unrelated["evidence_digest"])

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_requires_reverse_lifecycle_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = next(
            item
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if item["evidence_digest"] in manifest["linked_lifecycle_digests"]
        )
        lifecycle["linked_residual_work_digests"].remove(manifest["evidence_digest"])

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_reverse_only_lifecycle_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = next(
            item
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if item["evidence_digest"] in manifest["linked_lifecycle_digests"]
        )
        manifest["linked_lifecycle_digests"].remove(lifecycle["evidence_digest"])

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_forward_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_bogus_reverse_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = next(
            item
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if item["evidence_digest"] in manifest["linked_lifecycle_digests"]
        )
        lifecycle["linked_residual_work_digests"].append("bogus")

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_rejects_unreciprocated_reverse_link(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        target = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        other = deepcopy(target)
        other["evidence_digest"] = "other_residual"
        other["linked_lifecycle_digests"] = []
        branch["payload"]["obligation_manifests"]["residual_work"].append(other)
        lifecycle = next(
            item
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if item["evidence_digest"] in target["linked_lifecycle_digests"]
        )
        lifecycle["linked_residual_work_digests"].append(other["evidence_digest"])

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_reciprocal_extra_link_is_rejected(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = _append_unrelated_raw_lifecycle(branch, manifest=manifest)
        manifest["linked_lifecycle_digests"].append(unrelated["evidence_digest"])
        unrelated["linked_residual_work_digests"].append(manifest["evidence_digest"])

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "residual_lifecycle_reverse_link_provenance_mismatch",
            classification.reason,
        )

    def test_specified_tetra_residual_lifecycle_provenance_mutations_are_rejected(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        cases = (
            (
                "raw_event_kind",
                "raw",
                "lifecycle_event_kind",
                "local_order_closed",
                "tetra_residual_lifecycle_event_kind_mismatch",
            ),
            (
                "raw_missing_capability",
                "raw",
                "lifecycle_capabilities",
                ["tetra_token_restriction"],
                "tetra_residual_lifecycle_capabilities_mismatch",
            ),
            (
                "raw_extra_capability",
                "raw",
                "lifecycle_capabilities",
                [
                    "residual_factor_discharge",
                    "residual_propagation",
                    "tetra_token_restriction",
                ],
                "tetra_residual_lifecycle_capabilities_mismatch",
            ),
            (
                "raw_outcome",
                "raw",
                "lifecycle_outcome_kind",
                "event_recorded",
                "tetra_residual_lifecycle_outcome_kind_mismatch",
            ),
            (
                "raw_change_flag",
                "raw",
                "residual_snapshot_changed",
                False,
                "tetra_residual_lifecycle_change_flag_mismatch",
            ),
            (
                "raw_work_digests",
                "raw",
                "residual_work_digests",
                [],
                "residual_lifecycle_reverse_link_provenance_mismatch",
            ),
            (
                "raw_work_operations",
                "raw",
                "residual_work_operations",
                ["wrong"],
                "tetra_residual_lifecycle_work_operation_mismatch",
            ),
            (
                "certificate_kind",
                "certificate",
                "certificate_kind",
                "tetra_local_order_restricted",
                "tetra_residual_lifecycle_evidence_missing",
            ),
            (
                "certificate_capability",
                "certificate",
                "certificate_capability",
                "tetra_local_order_restriction",
                "tetra_residual_certificate_capability_mismatch",
            ),
            (
                "certificate_lifecycle_digest",
                "certificate",
                "certificate_lifecycle_digest",
                "wrong",
                "tetra_residual_certificate_lifecycle_digest_mismatch",
            ),
        )
        for name, lifecycle_kind, field, value, reason in cases:
            with self.subTest(name=name):
                mutated = deepcopy(artifact)
                branch = _first_residual_work_branch(
                    mutated,
                    operation="tetrahedral atom-token restriction",
                )
                manifest = next(
                    item
                    for item in (
                        branch["payload"]["obligation_manifests"]["residual_work"]
                    )
                    if item["operation"] == "tetrahedral atom-token restriction"
                )
                lifecycle = _linked_tetra_lifecycle_manifest(
                    branch=branch,
                    manifest=manifest,
                    lifecycle_kind=lifecycle_kind,
                    certificate_kind="tetra_token_restricted",
                )
                lifecycle[field] = value

                classification = _obligation_classification(mutated, facts=facts)

                self.assertFalse(classification.accepted)
                self.assertIn(reason, classification.reason)

    def test_specified_tetra_no_second_authority_from_stale_lifecycle_digest(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = _linked_tetra_lifecycle_manifest(
            branch=branch,
            manifest=manifest,
            lifecycle_kind="raw",
            certificate_kind="tetra_token_restricted",
        )
        lifecycle["source_digest"] = "stale_source_digest"

        classification = _obligation_classification(artifact, facts=facts)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_lifecycle_source_mismatch",
            classification.reason,
        )
        self.assertFalse(verification.accepted)
        self.assertIn("object_digest_mismatch", verification.reason)

    def test_specified_tetra_no_second_authority_from_final_support_strings(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        lifecycle = _linked_tetra_lifecycle_manifest(
            branch=branch,
            manifest=manifest,
            lifecycle_kind="certificate",
            certificate_kind="tetra_token_restricted",
        )
        self.assertTrue(any("@" in text for text in _support_strings(artifact)))
        lifecycle["certificate_capability"] = "tetra_local_order_restriction"

        classification = _obligation_classification(artifact, facts=facts)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_certificate_capability_mismatch",
            classification.reason,
        )
        self.assertFalse(verification.accepted)
        self.assertIn("object_digest_mismatch", verification.reason)

    def test_specified_tetra_residual_rejects_coherently_forged_link_projection(
        self,
    ) -> None:
        facts, artifact = _manual_tetra_artifact()
        branch = _first_residual_work_branch(
            artifact,
            operation="tetrahedral atom-token restriction",
        )
        manifest = next(
            item
            for item in branch["payload"]["obligation_manifests"]["residual_work"]
            if item["operation"] == "tetrahedral atom-token restriction"
        )
        unrelated = _append_unrelated_raw_lifecycle(branch, manifest=manifest)
        unrelated["linked_residual_work_digests"].append(manifest["evidence_digest"])
        unrelated["residual_work_digests"].append(manifest["evidence_digest"])
        manifest["linked_lifecycle_digests"] = [
            item["evidence_digest"]
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if manifest["evidence_digest"] in item["residual_work_digests"]
        ]

        # Public verification rejects this hand-edited artifact structurally via
        # stale object digests; this isolates the offline classifier precondition.
        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn(
            "tetra_residual_lifecycle_evidence_missing",
            classification.reason,
        )

    def test_terminal_clean_obligation_manifests_are_checked(self) -> None:
        facts = _rdkit_facts("CCO")
        artifact = _rdkit_artifact("CCO")
        classification = _obligation_classification(artifact, facts=facts)
        terminal = _first_terminal_support_object(artifact)

        self.assertTrue(classification.accepted, classification.reason)
        self.assertNotIn(
            "terminal_graph_obligation_work",
            classification.unchecked_families,
        )
        self.assertNotIn(
            "terminal_stereo_lifecycle",
            classification.unchecked_families,
        )
        self.assertIn(
            "terminal_graph_obligation_work",
            classification.checked_families,
        )
        self.assertIn(
            "terminal_stereo_lifecycle",
            classification.checked_families,
        )
        self.assertTrue(
            terminal["payload"]["obligation_manifests"][
                "terminal_graph_obligation_work"
            ][0]["terminal_clean"]
        )
        self.assertTrue(
            terminal["payload"]["obligation_manifests"][
                "terminal_stereo_lifecycle"
            ][0]["terminal_clean"]
        )

    def test_terminal_manifest_flags_are_reconstructed_not_credited(self) -> None:
        facts = _rdkit_facts("CCO")
        artifact = _rdkit_artifact("CCO")
        terminal = _first_terminal_support_object(artifact)
        for family in (
            "terminal_graph_obligation_work",
            "terminal_stereo_lifecycle",
        ):
            manifest = terminal["payload"]["obligation_manifests"][family][0]
            manifest["terminal_clean"] = False
            manifest["is_noop"] = False
            manifest["is_empty"] = False
            manifest["is_discharged"] = False

        classification = _obligation_classification(artifact, facts=facts)

        self.assertFalse(classification.accepted)
        self.assertIn("terminal_graph_manifest_mismatch", classification.reason)

    def test_terminal_support_identity_forgeries_reject_after_redigest(self) -> None:
        facts = _rdkit_facts("CCO")
        cases = (
            (
                "support_key",
                lambda payload: payload.__setitem__(
                    "terminal_support_key_digest", "0" * 64
                ),
                "terminal_support_terminal_support_key_digest_mismatch",
            ),
            (
                "certificate",
                lambda payload: payload["terminal_certificate_digests"].reverse(),
                "terminal_support_terminal_certificate_digests_mismatch",
            ),
            (
                "lifecycle_tuple",
                lambda payload: payload.__setitem__(
                    "terminal_stereo_lifecycle_evidence_digest", "0" * 64
                ),
                "terminal_support_terminal_stereo_lifecycle_evidence_digest_mismatch",
            ),
            (
                "discharged_flag",
                lambda payload: payload["obligation_manifests"]
                ["terminal_graph_obligation_work"][0].__setitem__(
                    "is_discharged", True
                ),
                "terminal_graph_manifest_mismatch",
            ),
        )
        for name, mutate, reason in cases:
            with self.subTest(name=name):
                artifact = deepcopy(_rdkit_artifact("CCO"))
                terminal = _first_terminal_support_object(artifact)
                mutate(terminal["payload"])
                identity = {
                    key: value
                    for key, value in terminal["payload"].items()
                    if key not in (
                        "terminalization_term",
                        "terminalization_term_digest",
                        "obligation_summary",
                        "obligation_manifests",
                    )
                }
                for item in artifact["objects"]:
                    if item["kind"] != "terminal_projection":
                        continue
                    for index, candidate in enumerate(
                        item["payload"]["terminal_support_identities"]
                    ):
                        if candidate["digest"] == identity["digest"]:
                            item["payload"]["terminal_support_identities"][index] = (
                                deepcopy(identity)
                            )
                _refresh_object_and_artifact_digest(artifact, terminal)
                structural = verify_writer_support_artifact_consistency(artifact)
                checked = verify_writer_support_artifact_for_facts(
                    facts=facts,
                    runtime_options=_writer_options(),
                    artifact=artifact,
                )
                self.assertTrue(structural.accepted, structural.reason)
                self.assertFalse(checked.accepted)
                self.assertIn(reason, checked.reason)

    def test_terminal_obligation_manifest_count_mismatch_is_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        terminal = _first_terminal_support_object(artifact)
        terminal["payload"]["obligation_summary"]["graph_obligation_work_count"] += 1

        verification = verify_writer_support_artifact_for_facts(
            facts=_rdkit_facts("CCO"),
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("obligation_manifest_count_mismatch", verification.reason)

    def test_terminal_obligation_manifest_unknown_family_is_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        terminal = _first_terminal_support_object(artifact)
        terminal["payload"]["obligation_manifests"]["unknown_terminal_family"] = []

        verification = verify_writer_support_artifact_for_facts(
            facts=_rdkit_facts("CCO"),
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("object_payload_fields_mismatch", verification.reason)

    def test_ring_finite_relation_and_graph_obligation_are_checked(self) -> None:
        for smiles in ("C1CC1", "C1CCC1", "C1=CC1", "C1#CC1"):
            with self.subTest(smiles=smiles):
                facts = _rdkit_facts(smiles)
                artifact = _rdkit_artifact(smiles)
                classification = _obligation_classification(artifact, facts=facts)
                verification = verify_writer_support_artifact_for_facts(
                    facts=facts,
                    runtime_options=_writer_options(),
                    artifact=artifact,
                )

                self.assertTrue(classification.accepted, classification.reason)
                self.assertNotIn(
                    "finite_relation_work",
                    classification.unchecked_families,
                )
                self.assertNotIn(
                    "graph_obligation_work",
                    classification.unchecked_families,
                )
                self.assertIn("finite_relation_work", classification.checked_families)
                self.assertIn("graph_obligation_work", classification.checked_families)
                self.assertTrue(verification.accepted, verification.reason)
                self.assertTrue(verification.offline_replay_complete)

    def test_ring_obligation_manifest_mutations_are_classified(self) -> None:
        facts = _rdkit_facts("C1=CC1")
        artifact = _rdkit_artifact("C1=CC1")
        branch = _first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        finite = branch["payload"]["obligation_manifests"]["finite_relation_work"][0]
        finite["ring_summary"]["is_exact"] = False

        not_exact = _obligation_classification(artifact, facts=facts)

        self.assertTrue(not_exact.accepted, not_exact.reason)
        self.assertIn("finite_relation_work", not_exact.unchecked_families)

        artifact = _rdkit_artifact("C1=CC1")
        branch = _first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        graph = branch["payload"]["obligation_manifests"]["graph_obligation_work"][0]
        graph["ring_summary"]["is_complete"] = False

        not_complete = _obligation_classification(artifact, facts=facts)

        self.assertTrue(not_complete.accepted, not_complete.reason)
        self.assertIn("graph_obligation_work", not_complete.unchecked_families)

    def test_ring_obligation_cross_link_mutations_are_rejected(self) -> None:
        facts = _rdkit_facts("C1=CC1")
        artifact = _rdkit_artifact("C1=CC1")
        branch = _first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        graph = branch["payload"]["obligation_manifests"]["graph_obligation_work"][0]
        graph["ring_summary"]["bond"] = "wrong"

        wrong_bond = _obligation_classification(artifact, facts=facts)

        self.assertFalse(wrong_bond.accepted)
        self.assertIn("ring_obligation_bond_mismatch", wrong_bond.reason)

        artifact = _rdkit_artifact("C1=CC1")
        branch = _first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        finite = branch["payload"]["obligation_manifests"]["finite_relation_work"][0]
        finite["operation"] = "unknown closure operation"

        wrong_operation = _obligation_classification(artifact, facts=facts)

        self.assertFalse(wrong_operation.accepted)
        self.assertIn("ring_obligation_operation_mismatch", wrong_operation.reason)

        artifact = _rdkit_artifact("C1=CC1")
        branch = _first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        graph = branch["payload"]["obligation_manifests"]["graph_obligation_work"][0]
        graph["ring_summary"]["marker"] = "#"

        wrong_marker = _obligation_classification(artifact, facts=facts)

        self.assertFalse(wrong_marker.accepted)
        self.assertIn("ring_obligation_marker_mismatch", wrong_marker.reason)

    def test_ring_obligation_manifest_count_mismatch_is_rejected(self) -> None:
        artifact = _rdkit_artifact("C1=CC1")
        branch = _first_graph_ring_delta_branch(
            artifact,
            "ring_endpoint_pair_non_single",
        )
        branch["payload"]["obligation_summary"]["finite_relation_work_count"] += 1

        verification = verify_writer_support_artifact_for_facts(
            facts=_rdkit_facts("C1=CC1"),
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("obligation_manifest_count_mismatch", verification.reason)

    def test_synthetic_stereo_obligation_is_reported_unchecked(self) -> None:
        facts = _rdkit_facts("CCO")
        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        manifest = branch["payload"]["obligation_manifests"]["stereo_lifecycle"][0]
        manifest["is_discharged"] = False
        manifest["is_noop"] = False
        manifest["is_empty"] = False

        classification = _obligation_classification(artifact, facts=facts)

        self.assertTrue(classification.accepted, classification.reason)
        self.assertTrue(classification.stereo_obligations_present)
        self.assertIn("stereo_lifecycle", classification.unchecked_families)

    def test_obligation_summary_mutation_is_structurally_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        branch["payload"]["obligation_summary"]["stereo_lifecycle_count"] += 1

        verification = verify_writer_support_artifact_for_facts(
            facts=_rdkit_facts("CCO"),
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("obligation_manifest_count_mismatch", verification.reason)

    def test_count_dag_arithmetic_accepts_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "CC(C)O", "C1CC1", "C1=CC1", "[NH4+]", "[13CH4]"):
            with self.subTest(smiles=smiles):
                artifact = _rdkit_artifact(smiles)
                count = _object(artifact, artifact["roots"]["count_ref"])
                count_dag = _object(artifact, count["payload"]["count_dag_ref"])

                verification = verify_count_dag_arithmetic(
                    count_dag=count_dag["payload"],
                    count_object=count["payload"],
                )

                self.assertTrue(verification.accepted, verification.reason)
                self.assertEqual(verification.support_count, count["payload"]["support_count"])
                self.assertEqual(
                    verification.completion_count,
                    count["payload"]["completion_count"],
                )

    def test_count_dag_arithmetic_rejects_changed_count_object_totals(self) -> None:
        artifact = _rdkit_artifact("CCO")
        count = deepcopy(_object(artifact, artifact["roots"]["count_ref"]))
        count_dag = _object(artifact, count["payload"]["count_dag_ref"])
        count["payload"]["support_count"] += 1

        verification = verify_count_dag_arithmetic(
            count_dag=count_dag["payload"],
            count_object=count["payload"],
        )

        self.assertFalse(verification.accepted)
        self.assertIn("count_dag_support_count_mismatch", verification.reason)

    def test_count_dag_arithmetic_rejects_changed_root_node_count(self) -> None:
        artifact = _rdkit_artifact("CCO")
        count = _object(artifact, artifact["roots"]["count_ref"])
        count_dag = deepcopy(_object(artifact, count["payload"]["count_dag_ref"]))
        root_id = count_dag["payload"]["roots"]["support_count_root"]
        root = next(
            node
            for node in count_dag["payload"]["nodes"]
            if node["node_id"] == root_id
        )
        root["support_count"] += 1

        verification = verify_count_dag_arithmetic(
            count_dag=count_dag["payload"],
            count_object=count["payload"],
        )

        self.assertFalse(verification.accepted)

    def test_count_dag_arithmetic_rejects_missing_child_and_cycle(self) -> None:
        artifact = _rdkit_artifact("CCO")
        count = _object(artifact, artifact["roots"]["count_ref"])
        count_dag = deepcopy(_object(artifact, count["payload"]["count_dag_ref"]))
        child_id = next(
            node["children"][0]
            for node in count_dag["payload"]["nodes"]
            if node["children"]
        )
        count_dag["payload"]["nodes"] = [
            node for node in count_dag["payload"]["nodes"] if node["node_id"] != child_id
        ]

        missing = verify_count_dag_arithmetic(
            count_dag=count_dag["payload"],
            count_object=count["payload"],
        )

        self.assertFalse(missing.accepted)

        count_dag = deepcopy(_object(artifact, count["payload"]["count_dag_ref"]))
        count_dag["payload"]["nodes"][0]["children"].append(
            count_dag["payload"]["nodes"][0]["node_id"]
        )
        cycle = verify_count_dag_arithmetic(
            count_dag=count_dag["payload"],
            count_object=count["payload"],
        )

        self.assertFalse(cycle.accepted)

    def test_support_image_coverage_accepts_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "C1CC1", "C1=CC1", "[NH4+]", "[13CH4]"):
            with self.subTest(smiles=smiles):
                artifact = _rdkit_artifact(smiles)

                verification = _coverage_verification(artifact)

                self.assertTrue(verification.accepted, verification.reason)
                root = _object(artifact, artifact["roots"]["support_image_root"])
                self.assertEqual(verification.support_count, root["payload"]["distinct_count"])
                self.assertEqual(verification.witness_count, root["payload"]["witness_count"])

    def test_support_image_coverage_accepts_terminal_bucket(self) -> None:
        artifact = _two_atom_completed_prefix_artifact()

        verification = _coverage_verification(artifact)

        self.assertTrue(verification.accepted, verification.reason)
        self.assertEqual(verification.support_count, 1)
        self.assertEqual(verification.witness_count, 2)

    def test_coverage_missing_or_extra_text_bucket_is_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        coverage = _coverage_object(artifact)
        coverage["payload"]["text_buckets"] = []

        missing = _coverage_verification(artifact)

        self.assertFalse(missing.accepted)
        self.assertIn("coverage_partition_mismatch", missing.reason)

        artifact = _rdkit_artifact("CCO")
        coverage = _coverage_object(artifact)
        coverage["payload"]["text_buckets"].append(
            deepcopy(coverage["payload"]["text_buckets"][0])
        )

        extra = _coverage_verification(artifact)

        self.assertFalse(extra.accepted)
        self.assertIn("coverage_duplicate_assignment", extra.reason)

    def test_coverage_wrong_text_projection_ref_is_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        coverage = _coverage_object(artifact)
        coverage["payload"]["text_buckets"][0]["text_projection"] = {
            **coverage["payload"]["text_buckets"][0]["text_projection"],
            "emitted_text": "N",
        }

        verification = _coverage_verification(artifact)

        self.assertFalse(verification.accepted)
        self.assertIn("coverage_text_projection_mismatch", verification.reason)

    def test_coverage_wrong_duplicate_and_unassigned_refs_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        coverage = _coverage_object(artifact)
        coverage["payload"]["text_buckets"][0]["string_refs"] = ["missing"]
        coverage["payload"]["text_buckets"][0]["support_count"] = 1

        wrong = _coverage_verification(artifact)

        self.assertFalse(wrong.accepted)
        self.assertIn("coverage_text_bucket_unknown_ref", wrong.reason)

        artifact = _rdkit_artifact("CC(C)O")
        root = _object(artifact, artifact["roots"]["support_image_root"])
        coverage = _coverage_object(artifact)
        first_ref = root["payload"]["support_string_refs"][0]
        coverage["payload"]["text_buckets"][0]["string_refs"] = [first_ref, first_ref]
        coverage["payload"]["text_buckets"][0]["support_count"] = 2

        duplicate = _coverage_verification(artifact)

        self.assertFalse(duplicate.accepted)
        self.assertIn("coverage_duplicate_assignment", duplicate.reason)

    def test_coverage_support_and_witness_totals_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        root = _object(artifact, artifact["roots"]["support_image_root"])
        root["payload"]["distinct_count"] += 1

        distinct = _coverage_verification(artifact)

        self.assertFalse(distinct.accepted)
        self.assertIn("support_image_distinct_count_mismatch", distinct.reason)

        artifact = _rdkit_artifact("CCO")
        root = _object(artifact, artifact["roots"]["support_image_root"])
        root["payload"]["witness_count"] += 1

        witness = _coverage_verification(artifact)

        self.assertFalse(witness.accepted)
        self.assertIn("coverage_count_completion_total_mismatch", witness.reason)

        artifact = _rdkit_artifact("CCO")
        count = _object(artifact, artifact["roots"]["count_ref"])
        count["payload"]["support_count"] += 1

        count_support = _coverage_verification(artifact)

        self.assertFalse(count_support.accepted)
        self.assertIn("coverage_count_support_total_mismatch", count_support.reason)

        artifact = _rdkit_artifact("CCO")
        count = _object(artifact, artifact["roots"]["count_ref"])
        count["payload"]["completion_count"] += 1

        count_completion = _coverage_verification(artifact)

        self.assertFalse(count_completion.accepted)
        self.assertIn(
            "coverage_count_completion_total_mismatch",
            count_completion.reason,
        )

    def test_coverage_terminal_bucket_mutations_are_rejected(self) -> None:
        artifact = _two_atom_completed_prefix_artifact()
        coverage = _coverage_object(artifact)
        coverage["payload"]["terminal_bucket"] = None

        missing = _coverage_verification(artifact)

        self.assertFalse(missing.accepted)
        self.assertIn("coverage_terminal_bucket_missing", missing.reason)

        artifact = _two_atom_completed_prefix_artifact()
        coverage = _coverage_object(artifact)
        coverage["payload"]["text_buckets"].append(
            {
                "text_projection": {},
                "support_count": 1,
                "string_refs": [
                    _object(
                        artifact,
                        artifact["roots"]["support_image_root"],
                    )["payload"]["support_string_refs"][0]
                ],
            }
        )

        text_bucket = _coverage_verification(artifact)

        self.assertFalse(text_bucket.accepted)
        self.assertIn("coverage_empty_string_in_text_bucket", text_bucket.reason)

        artifact = _two_atom_completed_prefix_artifact()
        coverage = _coverage_object(artifact)
        coverage["payload"]["terminal_bucket"]["terminal_projection"] = {
            **coverage["payload"]["terminal_bucket"]["terminal_projection"],
            "digest": "0" * 64,
        }

        wrong_projection = _coverage_verification(artifact)

        self.assertFalse(wrong_projection.accepted)
        self.assertIn("coverage_terminal_projection_mismatch", wrong_projection.reason)

    def test_support_string_replay_paths_accept_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "C1CC1", "C1=CC1", "[NH4+]", "[13CH4]"):
            with self.subTest(smiles=smiles):
                artifact = _rdkit_artifact(smiles)

                verification = _replay_path_verification(artifact)

                self.assertTrue(verification.accepted, verification.reason)
                self.assertGreater(verification.checked_support_strings, 0)
                self.assertIn(
                    "support_string_replay_path",
                    verification.relation_families,
                )

    def test_support_string_replay_paths_accept_empty_terminal_path(self) -> None:
        artifact = _two_atom_completed_prefix_artifact()

        verification = _replay_path_verification(artifact)

        self.assertTrue(verification.accepted, verification.reason)
        self.assertEqual(verification.checked_support_strings, 1)
        self.assertEqual(verification.checked_projection_steps, 0)

    def test_replay_path_wrong_emitted_texts_and_join_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        replay = _object(artifact, support["payload"]["replay_path_ref"])
        replay["payload"]["emitted_texts"] = ["C"]

        wrong_replay = _replay_path_verification(artifact)

        self.assertFalse(wrong_replay.accepted)
        self.assertIn("replay_path_emitted_texts_mismatch", wrong_replay.reason)

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        support["payload"]["string"] = "bad"

        wrong_join = _replay_path_verification(artifact)

        self.assertFalse(wrong_join.accepted)
        self.assertIn("replay_path_support_string_join_mismatch", wrong_join.reason)

    def test_replay_path_missing_and_extra_projection_refs_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        support["payload"]["text_projection_refs"] = (
            support["payload"]["text_projection_refs"][:-1]
        )

        missing = _replay_path_verification(artifact)

        self.assertFalse(missing.accepted)
        self.assertIn("replay_path_text_projection_count_mismatch", missing.reason)

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        support["payload"]["text_projection_refs"] = [
            *support["payload"]["text_projection_refs"],
            support["payload"]["text_projection_refs"][0],
        ]

        extra = _replay_path_verification(artifact)

        self.assertFalse(extra.accepted)
        self.assertIn("replay_path_text_projection_count_mismatch", extra.reason)

    def test_replay_path_projection_chain_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        projection = _first_text_projection_object(artifact)
        projection["payload"]["emitted_text"] = "N"

        wrong_text = _replay_path_verification(artifact)

        self.assertFalse(wrong_text.accepted)
        self.assertIn("replay_path_projection_text_mismatch", wrong_text.reason)

        artifact = _rdkit_artifact("CCO")
        projection = _first_text_projection_object(artifact)
        projection["payload"]["source_cursor"] = (
            projection["payload"]["successor_cursor"]
        )

        wrong_source = _replay_path_verification(artifact)

        self.assertFalse(wrong_source.accepted)
        self.assertIn(
            "replay_path_projection_source_cursor_mismatch",
            wrong_source.reason,
        )

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        second = _object(artifact, support["payload"]["text_projection_refs"][1])
        second["payload"]["source_cursor"] = second["payload"]["successor_cursor"]

        broken_chain = _replay_path_verification(artifact)

        self.assertFalse(broken_chain.accepted)
        self.assertIn(
            "replay_path_projection_source_cursor_mismatch",
            broken_chain.reason,
        )

    def test_replay_path_terminal_and_final_cursor_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        replay = _object(artifact, support["payload"]["replay_path_ref"])
        replay["payload"]["final_cursor_digest"] = "0" * 64

        final_cursor = _replay_path_verification(artifact)

        self.assertFalse(final_cursor.accepted)
        self.assertIn("replay_path_final_cursor_mismatch", final_cursor.reason)

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        terminal = _object(artifact, support["payload"]["terminal_projection_ref"])
        terminal["payload"]["source_cursor"] = terminal["payload"]["finalized_cursor"]

        terminal_source = _replay_path_verification(artifact)

        self.assertFalse(terminal_source.accepted)
        self.assertIn(
            "replay_path_terminal_source_cursor_mismatch",
            terminal_source.reason,
        )

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        support["payload"]["terminal_projection_ref"] = (
            support["payload"]["replay_path_ref"]
        )

        missing_terminal = _replay_path_verification(artifact)

        self.assertFalse(missing_terminal.accepted)
        self.assertIn(
            "replay_path_terminal_projection_ref_kind_mismatch",
            missing_terminal.reason,
        )

    def test_replay_path_empty_and_terminal_support_mutations_are_rejected(self) -> None:
        artifact = _two_atom_completed_prefix_artifact()
        support = _first_support_string_object(artifact)
        support["payload"]["text_projection_refs"] = ["missing"]

        empty_text = _replay_path_verification(artifact)

        self.assertFalse(empty_text.accepted)
        self.assertIn("replay_path_text_projection_count_mismatch", empty_text.reason)

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        support["payload"]["terminal_support_refs"] = [
            support["payload"]["terminal_projection_ref"]
        ]

        wrong_support = _replay_path_verification(artifact)

        self.assertFalse(wrong_support.accepted)
        self.assertIn(
            "replay_path_terminal_support_ref_kind_mismatch",
            wrong_support.reason,
        )

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        terminal_support = _object(artifact, support["payload"]["terminal_support_refs"][0])
        terminal_support["payload"]["digest"] = "0" * 64

        stale_support = _replay_path_verification(artifact)

        self.assertFalse(stale_support.accepted)
        self.assertIn(
            "replay_path_terminal_support_identity_mismatch",
            stale_support.reason,
        )

    def test_terminal_support_identities_accept_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "C1CC1", "C1=CC1", "[NH4+]", "[13CH4]"):
            with self.subTest(smiles=smiles):
                artifact = _rdkit_artifact(smiles)

                verification = _terminal_identity_verification(artifact)

                self.assertTrue(verification.accepted, verification.reason)
                self.assertGreater(verification.checked_terminal_projections, 0)
                self.assertGreater(verification.checked_terminal_supports, 0)
                self.assertGreater(verification.checked_terminal_paths, 0)

    def test_terminal_support_identities_accept_empty_terminal_bucket(self) -> None:
        artifact = _two_atom_completed_prefix_artifact()

        verification = _terminal_identity_verification(artifact)

        self.assertTrue(verification.accepted, verification.reason)
        self.assertEqual(verification.checked_terminal_paths, 1)

    def test_terminal_projection_cursor_and_support_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        terminal = _first_terminal_projection_object(artifact)
        terminal["payload"].pop("source_cursor")

        missing_source = _terminal_identity_verification(artifact)

        self.assertFalse(missing_source.accepted)

        artifact = _rdkit_artifact("CCO")
        terminal = _first_terminal_projection_object(artifact)
        terminal["payload"]["source_cursor"] = terminal["payload"]["finalized_cursor"]

        wrong_source = _terminal_identity_verification(artifact)

        self.assertFalse(wrong_source.accepted)
        self.assertIn("terminal_projection_source_cursor_mismatch", wrong_source.reason)

        artifact = _rdkit_artifact("CCO")
        terminal = _first_terminal_projection_object(artifact)
        terminal["payload"]["finalized_cursor"] = terminal["payload"]["source_cursor"]
        terminal["payload"]["terminal_support_identities"][0][
            "terminal_support_key_digest"
        ] = "0" * 64

        wrong_key = _terminal_identity_verification(artifact)

        self.assertFalse(wrong_key.accepted)
        self.assertIn("terminal_support_identity_mismatch", wrong_key.reason)

    def test_terminal_support_ordinal_and_key_mutations_are_rejected(self) -> None:
        artifact = _two_atom_completed_prefix_artifact()
        support = _first_terminal_support_object(artifact)
        support["payload"]["terminal_ordinal"] = -1
        terminal = _first_terminal_projection_object(artifact)
        terminal["payload"]["terminal_support_identities"][0]["terminal_ordinal"] = -1

        wrong_ordinal = _terminal_identity_verification(artifact)

        self.assertFalse(wrong_ordinal.accepted)
        self.assertIn("terminal_support_ordinal_negative", wrong_ordinal.reason)

        artifact = _two_atom_completed_prefix_artifact()
        terminal = _first_terminal_projection_object(artifact)
        identities = terminal["payload"]["terminal_support_identities"]
        identities[1]["terminal_ordinal"] = identities[0]["terminal_ordinal"]
        support = _first_support_string_object(artifact)
        second_support = _object(artifact, support["payload"]["terminal_support_refs"][1])
        second_support["payload"]["terminal_ordinal"] = identities[0]["terminal_ordinal"]

        duplicate_ordinal = _terminal_identity_verification(artifact)

        self.assertFalse(duplicate_ordinal.accepted)
        self.assertIn("terminal_projection_duplicate_ordinal", duplicate_ordinal.reason)

        artifact = _two_atom_completed_prefix_artifact()
        terminal = _first_terminal_projection_object(artifact)
        identities = terminal["payload"]["terminal_support_identities"]
        identities[1]["terminal_support_key_digest"] = identities[0][
            "terminal_support_key_digest"
        ]
        support = _first_support_string_object(artifact)
        second_support = _object(artifact, support["payload"]["terminal_support_refs"][1])
        second_support["payload"]["terminal_support_key_digest"] = identities[0][
            "terminal_support_key_digest"
        ]

        duplicate_key = _terminal_identity_verification(artifact)

        self.assertFalse(duplicate_key.accepted)
        self.assertIn("terminal_projection_duplicate_key_digest", duplicate_key.reason)

        artifact = _rdkit_artifact("CCO")
        support = _first_terminal_support_object(artifact)
        support["payload"]["parent_weight"] = 0
        terminal = _first_terminal_projection_object(artifact)
        terminal["payload"]["terminal_support_identities"][0]["parent_weight"] = 0

        parent_weight = _terminal_identity_verification(artifact)

        self.assertFalse(parent_weight.accepted)
        self.assertIn(
            "terminal_support_parent_weight_nonpositive",
            parent_weight.reason,
        )

    def test_terminal_support_ref_and_bucket_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        support["payload"]["terminal_support_refs"] = []

        missing_ref = _terminal_identity_verification(artifact)

        self.assertFalse(missing_ref.accepted)
        self.assertIn("terminal_support_refs_missing", missing_ref.reason)

        artifact = _rdkit_artifact("CCO")
        support = _first_support_string_object(artifact)
        support["payload"]["terminal_support_refs"] = [
            support["payload"]["terminal_projection_ref"]
        ]

        wrong_ref_kind = _terminal_identity_verification(artifact)

        self.assertFalse(wrong_ref_kind.accepted)
        self.assertIn("terminal_support_ref_kind_mismatch", wrong_ref_kind.reason)

        artifact = _rdkit_artifact("CCO")
        terminal_support = _first_terminal_support_object(artifact)
        terminal_support["payload"]["digest"] = "0" * 64

        stale_ref = _terminal_identity_verification(artifact)

        self.assertFalse(stale_ref.accepted)
        self.assertIn("terminal_support_not_in_projection", stale_ref.reason)

        artifact = _two_atom_completed_prefix_artifact()
        coverage = _coverage_object(artifact)
        coverage["payload"]["terminal_bucket"]["terminal_projection"] = {}

        wrong_bucket_projection = _terminal_identity_verification(artifact)

        self.assertFalse(wrong_bucket_projection.accepted)
        self.assertIn("terminal_bucket_projection_mismatch", wrong_bucket_projection.reason)

        artifact = _two_atom_completed_prefix_artifact()
        support = _first_support_string_object(artifact)
        support["payload"]["terminal_support_refs"] = support["payload"][
            "terminal_support_refs"
        ][:-1]

        wrong_bucket_support = _terminal_identity_verification(artifact)

        self.assertFalse(wrong_bucket_support.accepted)
        self.assertIn("terminal_projection_support_set_mismatch", wrong_bucket_support.reason)

    def test_branch_projection_identities_accept_default_relation_fixtures(self) -> None:
        for smiles in ("CCO", "CC(C)O", "C1=CC1"):
            with self.subTest(smiles=smiles):
                artifact = _rdkit_artifact(smiles)

                verification = _branch_projection_verification(artifact)

                self.assertTrue(verification.accepted, verification.reason)
                self.assertGreater(verification.checked_text_projections, 0)
                self.assertGreater(verification.checked_branch_supports, 0)

    def test_branch_projection_identity_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("CCO")
        projection = _first_text_projection_object(artifact)
        projection["payload"]["branch_support_refs"] = []

        missing = _branch_projection_verification(artifact)

        self.assertFalse(missing.accepted)
        self.assertIn("branch_projection_support_refs_missing", missing.reason)

        artifact = _rdkit_artifact("CCO")
        projection = _first_text_projection_object(artifact)
        branch = _first_branch_support_object(artifact)
        branch["payload"]["emitted_text"] = "N"

        wrong_text = _branch_projection_verification(artifact)

        self.assertFalse(wrong_text.accepted)
        self.assertIn("branch_projection_emitted_text_mismatch", wrong_text.reason)

        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        branch["payload"]["source_cursor_digest"] = "0" * 64

        wrong_source = _branch_projection_verification(artifact)

        self.assertFalse(wrong_source.accepted)
        self.assertIn("branch_projection_source_cursor_mismatch", wrong_source.reason)

        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        branch["payload"]["successor_cursor_digest"] = "0" * 64

        wrong_successor = _branch_projection_verification(artifact)

        self.assertFalse(wrong_successor.accepted)
        self.assertIn(
            "branch_projection_successor_cursor_mismatch",
            wrong_successor.reason,
        )

    def test_branch_projection_multiplicity_and_digest_mutations_are_rejected(
        self,
    ) -> None:
        artifact = _rdkit_artifact("CC(C)O")
        projection = _first_text_projection_object(artifact)
        projection["payload"]["branch_support_refs"] = (
            projection["payload"]["branch_support_refs"][:-1]
        )

        count_mismatch = _branch_projection_verification(artifact)

        self.assertFalse(count_mismatch.accepted)
        self.assertIn("branch_projection_multiplicity_mismatch", count_mismatch.reason)

        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        branch["payload"]["checked_branch_certificate_digest"] = "0" * 64

        stale_digest = _branch_projection_verification(artifact)

        self.assertFalse(stale_digest.accepted)
        self.assertIn(
            "branch_projection_certificate_digest_mismatch",
            stale_digest.reason,
        )

        artifact = _rdkit_artifact("CCO")
        projection = _first_text_projection_object(artifact)
        projection["payload"]["branch_support_refs"] = [
            projection["payload"]["branch_support_refs"][0],
            projection["payload"]["branch_support_refs"][0],
        ]

        duplicate_ref = _branch_projection_verification(artifact)

        self.assertFalse(duplicate_ref.accepted)
        self.assertIn("branch_projection_duplicate_support_ref", duplicate_ref.reason)

    def test_local_branch_successor_evidence_accepts_relation_fixtures(self) -> None:
        for smiles, plain_atom_count, bracket_atom_count, closure_count in (
            ("[NH4+]", 0, 1, 0),
            ("[13CH4]", 0, 1, 0),
            ("C1=CC1", 1, 0, 1),
            ("C1#CC1", 1, 0, 1),
            ("CCO", 3, 0, 0),
        ):
            with self.subTest(smiles=smiles):
                artifact = _rdkit_artifact(smiles)

                verification = _local_branch_evidence_verification(
                    _rdkit_facts(smiles),
                    artifact,
                )

                self.assertTrue(verification.accepted, verification.reason)
                self.assertGreaterEqual(
                    verification.checked_plain_atom_text_branches,
                    plain_atom_count,
                )
                self.assertGreaterEqual(
                    verification.checked_bracket_atom_text_branches,
                    bracket_atom_count,
                )
                self.assertGreaterEqual(
                    verification.checked_closure_bond_text_branches,
                    closure_count,
                )

    def test_local_atom_text_evidence_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("[NH4+]")
        evidence = _first_local_evidence(artifact, "bracket_atom_text")
        evidence["manifest"]["rendered_text"] = "[NH3+]"
        _refresh_local_evidence_digest(evidence)

        wrong_text = _local_branch_evidence_verification(_rdkit_facts("[NH4+]"), artifact)

        self.assertFalse(wrong_text.accepted)
        self.assertIn("local_bracket_atom_text_rendered_text_mismatch", wrong_text.reason)

        artifact = _rdkit_artifact("[NH4+]")
        evidence = _first_local_evidence(artifact, "bracket_atom_text")
        evidence["manifest"]["formal_charge"] = 0
        _refresh_local_evidence_digest(evidence)

        wrong_charge = _local_branch_evidence_verification(_rdkit_facts("[NH4+]"), artifact)

        self.assertFalse(wrong_charge.accepted)
        self.assertIn("local_bracket_atom_text_charge_mismatch", wrong_charge.reason)

        artifact = _rdkit_artifact("[13CH4]")
        evidence = _first_local_evidence(artifact, "bracket_atom_text")
        evidence["manifest"]["isotope"] = 12
        _refresh_local_evidence_digest(evidence)

        wrong_isotope = _local_branch_evidence_verification(
            _rdkit_facts("[13CH4]"),
            artifact,
        )

        self.assertFalse(wrong_isotope.accepted)
        self.assertIn("local_bracket_atom_text_isotope_mismatch", wrong_isotope.reason)

        artifact = _rdkit_artifact("[13CH4]")
        evidence = _first_local_evidence(artifact, "bracket_atom_text")
        evidence["manifest"]["hydrogen_count"] = 3
        _refresh_local_evidence_digest(evidence)

        wrong_h_count = _local_branch_evidence_verification(
            _rdkit_facts("[13CH4]"),
            artifact,
        )

        self.assertFalse(wrong_h_count.accepted)
        self.assertIn("local_bracket_atom_text_hydrogen_count_mismatch", wrong_h_count.reason)

        artifact = _rdkit_artifact("CCO")
        evidence = _first_local_evidence(artifact, "plain_atom_text")
        evidence["manifest"]["element"] = "N"
        _refresh_local_evidence_digest(evidence)

        wrong_plain = _local_branch_evidence_verification(_rdkit_facts("CCO"), artifact)

        self.assertFalse(wrong_plain.accepted)
        self.assertIn("local_plain_atom_text_element_mismatch", wrong_plain.reason)

    def test_local_closure_bond_text_evidence_mutations_are_rejected(self) -> None:
        artifact = _rdkit_artifact("C1#CC1")
        item = _first_closure_evidence_item(artifact)
        item["bond_order"] = "double"
        _refresh_local_evidence_digest(_first_local_evidence(artifact, "closure_bond_text"))

        wrong_order = _local_branch_evidence_verification(_rdkit_facts("C1#CC1"), artifact)

        self.assertFalse(wrong_order.accepted)
        self.assertIn("local_closure_bond_order_mismatch", wrong_order.reason)

        artifact = _rdkit_artifact("C1#CC1")
        item = _first_closure_evidence_item(artifact)
        item["opening_marker"] = ""
        item["closing_marker"] = ""
        _refresh_local_evidence_digest(_first_local_evidence(artifact, "closure_bond_text"))

        missing_marker = _local_branch_evidence_verification(
            _rdkit_facts("C1#CC1"),
            artifact,
        )

        self.assertFalse(missing_marker.accepted)
        self.assertIn("local_closure_marker_missing", missing_marker.reason)

        artifact = _rdkit_artifact("C1#CC1")
        item = _first_closure_evidence_item(artifact)
        item["opening_marker"] = "#"
        item["closing_marker"] = "#"
        _refresh_local_evidence_digest(_first_local_evidence(artifact, "closure_bond_text"))

        duplicate_marker = _local_branch_evidence_verification(
            _rdkit_facts("C1#CC1"),
            artifact,
        )

        self.assertFalse(duplicate_marker.accepted)
        self.assertIn("local_closure_marker_duplicate", duplicate_marker.reason)

        artifact = _rdkit_artifact("C1#CC1")
        item = _first_closure_evidence_item(artifact)
        item["opening_marker"] = "="
        item["closing_marker"] = "="
        _refresh_local_evidence_digest(_first_local_evidence(artifact, "closure_bond_text"))

        wrong_marker = _local_branch_evidence_verification(
            _rdkit_facts("C1#CC1"),
            artifact,
        )

        self.assertFalse(wrong_marker.accepted)
        self.assertIn("local_closure_marker_missing", wrong_marker.reason)

        artifact = _rdkit_artifact("C1#CC1")
        item = _first_closure_evidence_item(artifact)
        item["bond"] = "missing"
        _refresh_local_evidence_digest(_first_local_evidence(artifact, "closure_bond_text"))

        wrong_bond = _local_branch_evidence_verification(_rdkit_facts("C1#CC1"), artifact)

        self.assertFalse(wrong_bond.accepted)
        self.assertIn("local_closure_bond_missing", wrong_bond.reason)

    def test_local_branch_evidence_rejects_wrong_facts(self) -> None:
        artifact = _rdkit_artifact("[NH4+]")

        atom = _local_branch_evidence_verification(_rdkit_facts("[13CH4]"), artifact)

        self.assertFalse(atom.accepted)
        self.assertIn("local_bracket_atom_text_element_mismatch", atom.reason)

        artifact = _rdkit_artifact("[13CH4]")

        isotope = _local_branch_evidence_verification(_rdkit_facts("C"), artifact)

        self.assertFalse(isotope.accepted)
        self.assertIn("local_bracket_atom_text_isotope_mismatch", isotope.reason)

        artifact = _rdkit_artifact("C1=CC1")

        closure = _local_branch_evidence_verification(_rdkit_facts("C1CC1"), artifact)

        self.assertFalse(closure.accepted)
        self.assertIn("local_closure_bond_order_unsupported", closure.reason)

    def test_local_branch_evidence_unknown_kind_rejected(self) -> None:
        artifact = _rdkit_artifact("[NH4+]")
        evidence = _first_local_evidence(artifact, "bracket_atom_text")
        evidence["kind"] = "unknown"
        _refresh_local_evidence_digest(evidence)

        verification = _local_branch_evidence_verification(
            _rdkit_facts("[NH4+]"),
            artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("local_branch_unknown_evidence_kind", verification.reason)

    def test_local_branch_identity_fields_are_checked(self) -> None:
        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        branch["payload"]["source_cursor_digest"] = "wrong"

        wrong_source = _branch_projection_verification(artifact)

        self.assertFalse(wrong_source.accepted)
        self.assertIn("branch_projection_source_cursor_mismatch", wrong_source.reason)

        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        branch["payload"]["successor_cursor_digest"] = "wrong"

        wrong_successor = _branch_projection_verification(artifact)

        self.assertFalse(wrong_successor.accepted)
        self.assertIn("branch_projection_successor_cursor_mismatch", wrong_successor.reason)

        artifact = _rdkit_artifact("CCO")
        branch = _first_branch_support_object(artifact)
        branch["payload"]["checked_branch_certificate_digest"] = ""

        missing_digest = _local_branch_evidence_verification(_rdkit_facts("CCO"), artifact)

        self.assertFalse(missing_digest.accepted)
        self.assertIn(
            "local_branch_checked_certificate_digest_missing",
            missing_digest.reason,
        )

    def test_wrong_facts_are_rejected(self) -> None:
        artifact = _snapshot_artifact(cco_facts())

        verification = verify_writer_support_artifact_for_facts(
            facts=two_atom_facts(),
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("prepared_identity_mismatch", verification.reason)

    def test_wrong_runtime_options_are_rejected(self) -> None:
        facts = cco_facts()
        artifact = _snapshot_artifact(facts)
        wrong_options = _writer_options(rooted_at_atom=0)

        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=wrong_options,
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("prepared_identity_mismatch", verification.reason)

    def test_wrong_explicit_policy_is_rejected(self) -> None:
        facts = two_atom_facts()
        artifact = _snapshot_artifact(facts)
        wrong_policy = ordinary_policy_for_facts(
            facts,
            OrdinaryPolicyOptions(single_bond_mode="both"),
        )

        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=_writer_options(),
            artifact=artifact,
            policy=wrong_policy,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("prepared_identity_mismatch", verification.reason)

    def test_mutated_prepared_identity_is_rejected(self) -> None:
        artifact = _snapshot_artifact(cco_facts())
        artifact["prepared_identity"] = deepcopy(artifact["prepared_identity"])
        artifact["prepared_identity"]["digest"] = "0" * 64
        _refresh_artifact_digest(artifact)

        verification = verify_writer_support_artifact_for_facts(
            facts=cco_facts(),
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertIn("prepared_identity_mismatch", verification.reason)

    def test_mutated_source_prepared_identity_is_rejected(self) -> None:
        artifact = _snapshot_artifact(cco_facts())
        source = _object(artifact, artifact["roots"]["source_ref"])
        source["payload"]["prepared_identity_digest"] = "0" * 64

        verification = verify_writer_support_artifact_for_facts(
            facts=cco_facts(),
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)

    def test_structurally_invalid_artifact_is_rejected(self) -> None:
        artifact = _snapshot_artifact(cco_facts())
        artifact["objects"].pop()

        verification = verify_writer_support_artifact_for_facts(
            facts=cco_facts(),
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertFalse(verification.structurally_checked)

    def test_unknown_object_kind_is_rejected_by_structural_checker(self) -> None:
        artifact = _snapshot_artifact(cco_facts())
        root = _object(artifact, artifact["roots"]["support_image_root"])
        root["kind"] = "unknown"

        verification = verify_writer_support_artifact_for_facts(
            facts=cco_facts(),
            runtime_options=_writer_options(),
            artifact=artifact,
        )

        self.assertFalse(verification.accepted)
        self.assertFalse(verification.structurally_checked)

    def test_offline_coverage_ledger_classifies_partial_replay(self) -> None:
        self.assertEqual(
            OBJECT_KIND_OFFLINE_COVERAGE,
            {
                "source_snapshot": "identity_checked",
                "count_envelope": "arithmetic_checked",
                "count_dag": "arithmetic_checked",
                "frontier_product": "structurally_checked",
                "replay_path": "partially_offline_checked",
                "branch_support": "partially_offline_checked",
                "text_projection": "partially_offline_checked",
                "terminal_projection": "partially_offline_checked",
                "terminal_support": "partially_offline_checked",
                "support_string": "partially_offline_checked",
                "support_image_coverage": "structurally_checked",
                "support_image": "structurally_checked",
            },
        )

    def test_offline_coverage_ledger_covers_artifact_object_kinds(self) -> None:
        artifact = _rdkit_artifact("C1=CC1")
        kinds = {item["kind"] for item in artifact["objects"]}

        self.assertLessEqual(kinds, set(OBJECT_KIND_OFFLINE_COVERAGE))


def _snapshot_artifact(facts):
    options = _writer_options()
    prepared = _prepare(facts)
    return deepcopy(
        writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared, options),
        )
    )


def _rdkit_facts(smiles: str):
    return ordinary_molecule_facts_from_smiles(
        smiles,
        RdkitOrdinaryExtractionOptions(include_potential_sites=False),
    )


def _rdkit_artifact(smiles: str):
    return _snapshot_artifact(_rdkit_facts(smiles))


def _rdkit_artifact_verification(smiles: str):
    facts = _rdkit_facts(smiles)
    return verify_writer_support_artifact_for_facts(
        facts=facts,
        runtime_options=_writer_options(),
        artifact=_snapshot_artifact(facts),
    )


def _two_atom_completed_prefix_artifact():
    facts = two_atom_facts()
    prepared = _prepare(facts)
    options = _writer_options()
    prefix = writer_snapshot_prefix_read_envelope_for_emitted_texts(
        prepared=prepared,
        snapshot=_initial_snapshot(prepared, options),
        emitted_texts=("C", "C"),
    )
    return deepcopy(
        writer_support_artifact_envelope_for_prefix_read(
            prepared=prepared,
            prefix_read_envelope=prefix,
        )
    )


def _coverage_object(artifact):
    root = _object(artifact, artifact["roots"]["support_image_root"])
    return _object(artifact, root["payload"]["coverage_ref"])


def _coverage_verification(artifact):
    return verify_support_image_coverage_offline(
        artifact=artifact,
        objects={item["object_id"]: item for item in artifact["objects"]},
    )


def _replay_path_verification(artifact):
    return verify_support_string_replay_paths_offline(
        artifact=artifact,
        objects={item["object_id"]: item for item in artifact["objects"]},
    )


def _branch_projection_verification(artifact):
    return verify_branch_projection_identities_offline(
        artifact=artifact,
        objects={item["object_id"]: item for item in artifact["objects"]},
    )


def _graph_ring_delta_verification(facts, artifact):
    return verify_graph_ring_branch_deltas_offline(
        facts=facts,
        artifact=artifact,
        objects={item["object_id"]: item for item in artifact["objects"]},
    )


def _obligation_classification(artifact, *, facts):
    return classify_residual_stereo_obligations_offline(
        facts=facts,
        artifact=artifact,
        objects={item["object_id"]: item for item in artifact["objects"]},
    )


def _local_branch_evidence_verification(facts, artifact):
    return verify_local_branch_successor_evidence_offline(
        facts=facts,
        artifact=artifact,
        objects={item["object_id"]: item for item in artifact["objects"]},
    )


def _terminal_identity_verification(artifact):
    return verify_terminal_support_identities_offline(
        artifact=artifact,
        objects={item["object_id"]: item for item in artifact["objects"]},
    )


def _manual_tetra_artifact():
    facts = tetrahedral_facts()
    prepared = _prepare(facts)
    return (
        facts,
        writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared, _writer_options()),
        ),
    )


def _directional_rooted_artifact():
    facts = directional_facts()
    options = _writer_options(rooted_at_atom=2)
    prepared = _prepare(facts)
    return (
        facts,
        options,
        writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared, options),
        ),
    )


def _shared_acyclic_directional_artifact():
    facts = shared_acyclic_directional_facts()
    options = _writer_options(rooted_at_atom=0)
    prepared = _prepare(facts)
    return (
        facts,
        options,
        writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=_initial_snapshot(prepared, options),
        ),
    )


@lru_cache(maxsize=1)
def _directional_ring_opening_artifact():
    facts = _directional_ring_carrier_facts()
    options = _writer_options(rooted_at_atom=0)
    prepared = _prepare(facts)
    initial = _initial_snapshot(prepared, options)
    frontier = [(initial.cursor, 0)]
    seen = set()
    opening_sources = []
    while frontier:
        cursor, depth = frontier.pop(0)
        cursor_key = repr(cursor)
        if cursor_key in seen:
            continue
        seen.add(cursor_key)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=False,
            include_count_certificate=False,
        )
        for support in batch.supports:
            if any(
                isinstance(event, WriterRingEndpointEmitted)
                and event.bond == BondId(3)
                for event in support.events
            ):
                opening_sources.append((cursor, depth))
            frontier.append((support.successor_cursor, depth + 1))
    if not opening_sources:
        raise AssertionError("missing cursor before BondId(3) ring opening")
    source, source_depth = max(opening_sources, key=lambda item: item[1])
    snapshot = capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=source,
        decoder_boundary=WriterDecoderBoundary(consumed_token_count=source_depth),
    )
    artifact = writer_support_artifact_envelope_for_snapshot(
        prepared=prepared,
        snapshot=snapshot,
    )
    return facts, options, artifact


@lru_cache(maxsize=2)
def _directional_ring_pair_artifact(first_mark: DirectionMark):
    facts = _directional_ring_carrier_facts()
    options = _writer_options(rooted_at_atom=0)
    prepared = _prepare(facts)
    initial = _initial_snapshot(prepared, options)
    frontier = [(initial.cursor, 0)]
    seen = set()
    source = None
    source_depth = None
    while frontier and source is None:
        cursor, depth = frontier.pop(0)
        cursor_key = repr(cursor)
        if cursor_key in seen:
            continue
        seen.add(cursor_key)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=False,
            include_count_certificate=False,
        )
        for support in batch.supports:
            if any(
                isinstance(event, WriterRingEndpointPaired)
                and event.bond == BondId(3)
                and event.first_endpoint_direction_mark is first_mark
                for event in support.events
            ):
                source = cursor
                source_depth = depth
                break
            frontier.append((support.successor_cursor, depth + 1))
    if source is None or source_depth is None:
        raise AssertionError(
            f"missing cursor before BondId(3) pair with first mark {first_mark}"
        )
    snapshot = capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=source,
        decoder_boundary=WriterDecoderBoundary(consumed_token_count=source_depth),
    )
    artifact = writer_support_artifact_envelope_for_snapshot(
        prepared=prepared,
        snapshot=snapshot,
    )
    return facts, options, artifact


def _first_support_string_object(artifact):
    root = _object(artifact, artifact["roots"]["support_image_root"])
    return _object(artifact, root["payload"]["support_string_refs"][0])


def _first_text_projection_object(artifact):
    support = _first_support_string_object(artifact)
    return _object(artifact, support["payload"]["text_projection_refs"][0])


def _first_branch_support_object(artifact):
    projection = _first_text_projection_object(artifact)
    return _object(artifact, projection["payload"]["branch_support_refs"][0])


def _first_graph_ring_delta_branch(artifact, kind: str):
    for item in artifact["objects"]:
        if item["kind"] != "branch_support":
            continue
        if item["payload"]["graph_ring_delta"]["kind"] == kind:
            return item
    raise AssertionError(f"missing graph/ring delta kind: {kind}")


def _first_residual_work_branch(artifact, *, operation: str):
    for item in artifact["objects"]:
        if item["kind"] != "branch_support":
            continue
        manifests = item["payload"]["obligation_manifests"]["residual_work"]
        if any(manifest["operation"] == operation for manifest in manifests):
            return item
    raise AssertionError(f"missing residual work operation: {operation}")


def _directional_transition_branch_and_manifest(artifact, *, bond: int):
    for branch in artifact["objects"]:
        if branch["kind"] != "branch_support":
            continue
        for manifest in branch["payload"]["obligation_manifests"]["residual_work"]:
            if manifest["operation"] != "directional carrier-mark restriction":
                continue
            if _term_field(manifest["transition_term"], "bond") == bond:
                return branch, manifest
    raise AssertionError(f"missing directional carrier transition for bond {bond}")


def _directional_transition_manifest(artifact, *, bond: int):
    return _directional_transition_branch_and_manifest(artifact, bond=bond)[1]


def _ring_projection_branch_and_manifest(artifact, *, changed: bool | None = None):
    for branch in artifact["objects"]:
        if branch["kind"] != "branch_support":
            continue
        for manifest in branch["payload"]["obligation_manifests"]["residual_work"]:
            if manifest["operation"] != "directional ring endpoint projection":
                continue
            term = manifest["transition_term"]
            is_changed = _term_field(term, "source_snapshot") != _term_field(
                term, "successor_snapshot"
            )
            if changed is None or changed == is_changed:
                return branch, manifest
    raise AssertionError("missing directional ring projection transition")


def _ring_pair_branch_and_manifest(artifact):
    for branch in artifact["objects"]:
        if branch["kind"] != "branch_support":
            continue
        for manifest in branch["payload"]["obligation_manifests"]["residual_work"]:
            if manifest["operation"] == "directional ring pair restriction":
                return branch, manifest
    raise AssertionError("missing directional ring pair transition")


def _refresh_ring_pair_term(artifact, branch, manifest) -> None:
    _refresh_transition_manifest_digest(manifest)
    _refresh_object_and_artifact_digest(artifact, branch)


def _forge_ring_pair_missing_term(artifact) -> None:
    branch, manifest = _ring_pair_branch_and_manifest(artifact)
    manifest["transition_term"] = None
    manifest["transition_digest"] = None
    _refresh_object_and_artifact_digest(artifact, branch)


def _forge_ring_pair_compatible_choices(artifact) -> None:
    branch, manifest = _ring_pair_branch_and_manifest(artifact)
    del _term_field(manifest["transition_term"], "compatible_second_endpoint_choices")[-1]
    _refresh_ring_pair_term(artifact, branch, manifest)


def _forge_ring_pair_first_mark(artifact) -> None:
    branch, manifest = _ring_pair_branch_and_manifest(artifact)
    mark = _term_field(manifest["transition_term"], "first_endpoint_direction_mark")
    mark["value"] = 1 if mark["value"] != 1 else -1
    _refresh_ring_pair_term(artifact, branch, manifest)


def _forge_ring_pair_second_mark(artifact) -> None:
    branch, manifest = _ring_pair_branch_and_manifest(artifact)
    mark = _term_field(manifest["transition_term"], "second_endpoint_direction_mark")
    mark["value"] = -1 if mark["value"] != -1 else 1
    _refresh_ring_pair_term(artifact, branch, manifest)


def _forge_ring_pair_orientation(artifact) -> None:
    branch, manifest = _ring_pair_branch_and_manifest(artifact)
    term = manifest["transition_term"]
    value = _term_field(term, "second_canonical_orientation")
    _set_term_field(term, "second_canonical_orientation", -value)
    _refresh_ring_pair_term(artifact, branch, manifest)


def _forge_ring_pair_carrier(artifact) -> None:
    branch, manifest = _ring_pair_branch_and_manifest(artifact)
    model = _term_field(manifest["transition_term"], "carrier_models")[0]
    value = _term_field(model, "ligand_factor")
    _set_term_field(model, "ligand_factor", -value)
    _refresh_ring_pair_term(artifact, branch, manifest)


def _forge_ring_pair_restriction(artifact) -> None:
    branch, manifest = _ring_pair_branch_and_manifest(artifact)
    sign = _term_field(manifest["transition_term"], "restrictions")[0][1]
    sign["value"] = "negative" if sign["value"] == "positive" else "positive"
    _refresh_ring_pair_term(artifact, branch, manifest)


def _forge_ring_pair_occurrence(artifact) -> None:
    branch, manifest = _ring_pair_branch_and_manifest(artifact)
    term = manifest["transition_term"]
    parent = _term_field(term, "bond_occurrence_parent")
    child = _term_field(term, "bond_occurrence_child")
    _set_term_field(term, "bond_occurrence_parent", child)
    _set_term_field(term, "bond_occurrence_child", parent)
    _refresh_ring_pair_term(artifact, branch, manifest)


def _forge_ring_pair_discharge(artifact) -> None:
    branch, manifest = _ring_pair_branch_and_manifest(artifact)
    del _term_field(manifest["transition_term"], "discharged_factor_keys")[-1]
    _refresh_ring_pair_term(artifact, branch, manifest)


def _forge_ring_pair_successor(artifact) -> None:
    branch, manifest = _ring_pair_branch_and_manifest(artifact)
    term = manifest["transition_term"]
    successor = deepcopy(_term_field(term, "source_snapshot"))
    digest = _closed_term_digest(successor)
    _set_term_field(term, "successor_snapshot", successor)
    _set_term_field(term, "successor_snapshot_digest", digest)
    _refresh_linked_raw_lifecycle_residual_digest(
        branch,
        manifest=manifest,
        field="successor_residual_snapshot_digest",
        digest=digest,
    )
    _refresh_ring_pair_term(artifact, branch, manifest)


def _refresh_ring_projection_term(artifact, branch, manifest) -> None:
    _refresh_transition_manifest_digest(manifest)
    _refresh_object_and_artifact_digest(artifact, branch)


def _forge_ring_compatible_seconds(artifact) -> None:
    branch, manifest = _ring_projection_branch_and_manifest(artifact)
    choices = _term_field(manifest["transition_term"], "compatible_second_endpoint_choices")
    del choices[-1]
    _refresh_ring_projection_term(artifact, branch, manifest)


def _forge_ring_domain_intersection(artifact) -> None:
    branch, manifest = _ring_projection_branch_and_manifest(artifact, changed=True)
    values = _term_field(manifest["transition_term"], "domain_intersections")[0][1]
    values[0]["value"] = "negative" if values[0]["value"] == "positive" else "positive"
    _refresh_ring_projection_term(artifact, branch, manifest)


def _forge_ring_carrier_orientation(artifact) -> None:
    branch, manifest = _ring_projection_branch_and_manifest(artifact)
    model = _term_field(manifest["transition_term"], "carrier_model")
    orientation = _term_field(model, "endpoint_orientation_factor")
    _set_term_field(model, "endpoint_orientation_factor", -orientation)
    _refresh_ring_projection_term(artifact, branch, manifest)


def _forge_ring_term_mark(artifact) -> None:
    branch, manifest = _ring_projection_branch_and_manifest(artifact)
    mark = _term_field(manifest["transition_term"], "direction_mark")
    mark["value"] = -1 if mark["value"] != -1 else 1
    _refresh_ring_projection_term(artifact, branch, manifest)


def _forge_ring_false_noop(artifact) -> None:
    branch, manifest = _ring_projection_branch_and_manifest(artifact, changed=True)
    term = manifest["transition_term"]
    source = deepcopy(_term_field(term, "source_snapshot"))
    _set_term_field(term, "successor_snapshot", source)
    digest = _closed_term_digest(source)
    _set_term_field(term, "successor_snapshot_digest", digest)
    _refresh_linked_raw_lifecycle_residual_digest(
        branch,
        manifest=manifest,
        field="successor_residual_snapshot_digest",
        digest=digest,
    )
    _refresh_ring_projection_term(artifact, branch, manifest)


def _forge_ring_false_change(artifact) -> None:
    branch, manifest = _ring_projection_branch_and_manifest(artifact, changed=False)
    _other_branch, other = _ring_projection_branch_and_manifest(artifact, changed=True)
    term = manifest["transition_term"]
    successor = deepcopy(_term_field(other["transition_term"], "successor_snapshot"))
    _set_term_field(term, "successor_snapshot", successor)
    digest = _closed_term_digest(successor)
    _set_term_field(term, "successor_snapshot_digest", digest)
    _refresh_linked_raw_lifecycle_residual_digest(
        branch,
        manifest=manifest,
        field="successor_residual_snapshot_digest",
        digest=digest,
    )
    _refresh_ring_projection_term(artifact, branch, manifest)


def _forge_ring_factor_discharge(artifact) -> None:
    branch, manifest = _ring_projection_branch_and_manifest(artifact)
    term = manifest["transition_term"]
    source = _term_field(term, "source_snapshot")
    factor = _term_field(_term_field(source, "factors")[0], "key")
    _set_term_field(term, "discharged_factor_keys", [factor])
    _refresh_ring_projection_term(artifact, branch, manifest)


def _forge_ring_source_snapshot(artifact) -> None:
    branch, manifest = _ring_projection_branch_and_manifest(artifact)
    term = manifest["transition_term"]
    source = _term_field(term, "source_snapshot")
    domains = _term_field(source, "domains")
    domains[:] = list(reversed(domains))
    digest = _closed_term_digest(source)
    _set_term_field(term, "source_snapshot_digest", digest)
    _refresh_linked_raw_lifecycle_residual_digest(
        branch,
        manifest=manifest,
        field="source_residual_snapshot_digest",
        digest=digest,
    )
    _refresh_ring_projection_term(artifact, branch, manifest)


def _forge_ring_missing_term(artifact) -> None:
    branch, manifest = _ring_projection_branch_and_manifest(artifact)
    manifest["transition_term"] = None
    manifest["transition_digest"] = None
    _refresh_object_and_artifact_digest(artifact, branch)


def _forge_ring_lifecycle_operation(artifact) -> None:
    branch, manifest = _ring_projection_branch_and_manifest(artifact)
    lifecycle = _linked_tetra_lifecycle_manifest(
        branch=branch,
        manifest=manifest,
        lifecycle_kind="raw",
        certificate_kind="",
    )
    lifecycle["residual_work_operations"] = ["wrong"]
    _refresh_object_and_artifact_digest(artifact, branch)


def _forge_ring_successor_open_endpoint(artifact) -> None:
    branch, manifest = _ring_projection_branch_and_manifest(artifact)
    projection = _text_projection_for_branch(artifact, branch)
    cursor = projection["payload"]["successor_cursor"]
    state = _cursor_state_by_digest(
        cursor,
        branch["payload"]["successor_state_digest"],
    )
    ring_state = _term_field(state, "ring_state")
    endpoint = next(
        endpoint
        for endpoint in _term_field(ring_state, "open_endpoints")
        if int(_term_field(endpoint, "bond")) == 3
    )
    _set_term_field(endpoint, "first_endpoint_text", "%01")
    _refresh_ring_successor_cursor_change(
        artifact=artifact,
        branch=branch,
        manifest=manifest,
        projection=projection,
        cursor=cursor,
        state=state,
    )


def _forge_ring_bond_occurrence_added(artifact) -> None:
    branch, manifest = _ring_projection_branch_and_manifest(artifact)
    projection = _text_projection_for_branch(artifact, branch)
    cursor = projection["payload"]["successor_cursor"]
    state = _cursor_state_by_digest(
        cursor,
        branch["payload"]["successor_state_digest"],
    )
    stereo = _term_field(state, "stereo_state")
    _term_field(stereo, "bond_occurrences").append(
        {
            "__dataclass__": "grimace._south_star1.writer_stereo.WriterBondOccurrenceRecord",
            "fields": [
                ["bond", 3],
                ["parent", 0],
                ["child", 2],
                ["mark", {"__enum__": "grimace._south_star1.policy.DirectionMark", "value": 0}],
            ],
        }
    )
    _refresh_ring_successor_cursor_change(
        artifact=artifact,
        branch=branch,
        manifest=manifest,
        projection=projection,
        cursor=cursor,
        state=state,
    )


def _refresh_ring_successor_cursor_change(
    *, artifact, branch, manifest, projection, cursor, state
) -> None:
    old_cursor_digest = branch["payload"]["successor_cursor_digest"]
    _refresh_cursor_digest(cursor)
    successor_state_digest = _closed_term_digest(state)
    _propagate_text_projection_cursor_change(
        artifact,
        old_cursor_digest=old_cursor_digest,
        new_cursor=cursor,
    )
    branch["payload"]["successor_state_digest"] = successor_state_digest
    branch["payload"]["graph_ring_delta"]["manifest"]["successor_state_digest"] = (
        successor_state_digest
    )
    manifest["successor_digest"] = successor_state_digest
    for lifecycle in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]:
        if manifest["evidence_digest"] in lifecycle["linked_residual_work_digests"]:
            lifecycle["successor_digest"] = successor_state_digest
    branch["payload"]["successor_cursor_digest"] = cursor["digest"]
    branch["payload"]["graph_ring_delta"]["manifest"]["successor_cursor_digest"] = (
        cursor["digest"]
    )
    _refresh_graph_ring_delta_digest(branch["payload"]["graph_ring_delta"])
    projection["payload"]["digest"] = _text_projection_identity_digest(
        projection["payload"]
    )
    _refresh_object_and_artifact_digest(artifact, branch)


def _directional_discharge_key_pairs(manifest):
    return tuple(
        (
            _term_field(key, "kind"),
            tuple(_term_field(key, "key")),
        )
        for key in _term_field(
            manifest["transition_term"],
            "discharged_factor_keys",
        )
    )


def _bond_occurrence_terms_for_branch(
    artifact,
    branch,
    *,
    cursor_name: str,
    bond: int,
):
    projection = _text_projection_for_branch(artifact, branch)
    state = _single_cursor_state(projection["payload"][cursor_name])
    stereo = _term_field(state, "stereo_state")
    return tuple(
        occurrence
        for occurrence in _term_field(stereo, "bond_occurrences")
        if int(_term_field(occurrence, "bond")) == bond
    )


def _mutate_directional_restriction_sign(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    sign = _term_field(manifest["transition_term"], "restrictions")[0][1]
    sign["value"] = "negative" if sign["value"] == "positive" else "positive"
    _refresh_transition_manifest_digest(manifest)
    _refresh_object_and_artifact_digest(artifact, branch)


def _mutate_directional_canonical_orientation(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    value = _term_field(manifest["transition_term"], "canonical_orientation")
    _set_term_field(manifest["transition_term"], "canonical_orientation", -value)
    _refresh_transition_manifest_digest(manifest)
    _refresh_object_and_artifact_digest(artifact, branch)


def _mutate_directional_model_field(
    artifact,
    *,
    bond: int,
    field: str,
    value,
    model_index: int = 0,
) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    model = _term_field(manifest["transition_term"], "carrier_models")[model_index]
    _set_term_field(model, field, value)
    _refresh_transition_manifest_digest(manifest)
    _refresh_object_and_artifact_digest(artifact, branch)


def _remove_directional_model(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    models = _term_field(manifest["transition_term"], "carrier_models")
    del models[-1]
    _refresh_transition_manifest_digest(manifest)
    _refresh_object_and_artifact_digest(artifact, branch)


def _remove_directional_restriction(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    restrictions = _term_field(manifest["transition_term"], "restrictions")
    del restrictions[-1]
    _refresh_transition_manifest_digest(manifest)
    _refresh_object_and_artifact_digest(artifact, branch)


def _duplicate_directional_model_site(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    models = _term_field(manifest["transition_term"], "carrier_models")
    duplicate_site = _term_field(models[0], "site")
    _set_term_field(models[1], "site", duplicate_site)
    _refresh_transition_manifest_digest(manifest)
    _refresh_object_and_artifact_digest(artifact, branch)


def _mutate_directional_successor_snapshot(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    successor = _term_field(manifest["transition_term"], "successor_snapshot")
    domains = _term_field(successor, "domains")
    domains[:] = list(reversed(domains))
    digest = _closed_term_digest(successor)
    _set_term_field(manifest["transition_term"], "successor_snapshot_digest", digest)
    _refresh_linked_raw_lifecycle_residual_digest(
        branch,
        manifest=manifest,
        field="successor_residual_snapshot_digest",
        digest=digest,
    )
    _refresh_transition_manifest_digest(manifest)
    _refresh_object_and_artifact_digest(artifact, branch)


def _set_directional_discharges(
    artifact,
    *,
    bond: int,
    kinds: tuple[str, ...],
) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    source = _term_field(manifest["transition_term"], "source_snapshot")
    factor_by_kind = {
        _term_field(_term_field(factor, "key"), "kind"): _term_field(factor, "key")
        for factor in _term_field(source, "factors")
    }
    _set_term_field(
        manifest["transition_term"],
        "discharged_factor_keys",
        [factor_by_kind[kind] for kind in kinds],
    )
    _refresh_transition_manifest_digest(manifest)
    _refresh_object_and_artifact_digest(artifact, branch)


def _set_directional_discharges_by_keys(
    artifact,
    *,
    bond: int,
    key_pairs: tuple[tuple[str, tuple[int, ...]], ...],
) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    source = _term_field(manifest["transition_term"], "source_snapshot")
    factor_by_pair = {
        (
            _term_field(_term_field(factor, "key"), "kind"),
            tuple(_term_field(_term_field(factor, "key"), "key")),
        ): _term_field(factor, "key")
        for factor in _term_field(source, "factors")
    }
    _set_term_field(
        manifest["transition_term"],
        "discharged_factor_keys",
        [factor_by_pair[key_pair] for key_pair in key_pairs],
    )
    _refresh_transition_manifest_digest(manifest)
    _refresh_object_and_artifact_digest(artifact, branch)


def _remove_raw_lifecycle_capability(
    artifact,
    *,
    bond: int,
    capability: str,
) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    lifecycle = _linked_tetra_lifecycle_manifest(
        branch=branch,
        manifest=manifest,
        lifecycle_kind="raw",
        certificate_kind="directional_carrier_restricted",
    )
    capabilities = lifecycle["lifecycle_capabilities"]
    capabilities.remove(capability)
    _refresh_object_and_artifact_digest(artifact, branch)


def _mutate_directional_term_mark(artifact, *, bond: int, value: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    _term_field(manifest["transition_term"], "direction_mark")["value"] = value
    _refresh_transition_manifest_digest(manifest)
    _refresh_object_and_artifact_digest(artifact, branch)


def _mutate_directional_term_bond(artifact, *, bond: int, value: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    _set_term_field(manifest["transition_term"], "bond", value)
    _refresh_transition_manifest_digest(manifest)
    _refresh_object_and_artifact_digest(artifact, branch)


def _remove_directional_successor_bond_occurrence(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    projection = _text_projection_for_branch(artifact, branch)
    cursor = projection["payload"]["successor_cursor"]
    state = _single_cursor_state(cursor)
    stereo = _term_field(state, "stereo_state")
    occurrences = _term_field(stereo, "bond_occurrences")
    kept = [
        occurrence
        for occurrence in occurrences
        if int(_term_field(occurrence, "bond")) != bond
    ]
    if len(kept) == len(occurrences):
        raise AssertionError(f"missing successor bond occurrence for bond {bond}")
    occurrences[:] = kept
    _refresh_cursor_digest(cursor)
    successor_state_digest = _closed_term_digest(state)
    _propagate_text_projection_cursor_change(
        artifact,
        old_cursor_digest=branch["payload"]["successor_cursor_digest"],
        new_cursor=cursor,
    )
    branch["payload"]["successor_state_digest"] = successor_state_digest
    branch["payload"]["graph_ring_delta"]["manifest"]["successor_state_digest"] = (
        successor_state_digest
    )
    manifest["successor_digest"] = successor_state_digest
    for lifecycle in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]:
        if manifest["evidence_digest"] in lifecycle["linked_residual_work_digests"]:
            lifecycle["successor_digest"] = successor_state_digest
    branch["payload"]["successor_cursor_digest"] = cursor["digest"]
    branch["payload"]["graph_ring_delta"]["manifest"]["successor_cursor_digest"] = (
        cursor["digest"]
    )
    _refresh_graph_ring_delta_digest(branch["payload"]["graph_ring_delta"])
    projection["payload"]["digest"] = _text_projection_identity_digest(
        projection["payload"]
    )
    _refresh_object_and_artifact_digest(artifact, branch)


def _duplicate_directional_successor_bond_occurrence(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    projection = _text_projection_for_branch(artifact, branch)
    cursor = projection["payload"]["successor_cursor"]
    state = _single_cursor_state(cursor)
    stereo = _term_field(state, "stereo_state")
    occurrences = _term_field(stereo, "bond_occurrences")
    matches = [
        occurrence
        for occurrence in occurrences
        if int(_term_field(occurrence, "bond")) == bond
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one successor bond occurrence for bond {bond}")
    occurrences.append(deepcopy(matches[0]))
    _refresh_cursor_digest(cursor)
    successor_state_digest = _closed_term_digest(state)
    _propagate_text_projection_cursor_change(
        artifact,
        old_cursor_digest=branch["payload"]["successor_cursor_digest"],
        new_cursor=cursor,
    )
    branch["payload"]["successor_state_digest"] = successor_state_digest
    branch["payload"]["graph_ring_delta"]["manifest"]["successor_state_digest"] = (
        successor_state_digest
    )
    manifest["successor_digest"] = successor_state_digest
    for lifecycle in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]:
        if manifest["evidence_digest"] in lifecycle["linked_residual_work_digests"]:
            lifecycle["successor_digest"] = successor_state_digest
    branch["payload"]["successor_cursor_digest"] = cursor["digest"]
    branch["payload"]["graph_ring_delta"]["manifest"]["successor_cursor_digest"] = (
        cursor["digest"]
    )
    _refresh_graph_ring_delta_digest(branch["payload"]["graph_ring_delta"])
    projection["payload"]["digest"] = _text_projection_identity_digest(
        projection["payload"]
    )
    _refresh_object_and_artifact_digest(artifact, branch)


def _mutate_directional_successor_snapshot_unrelated(artifact, *, bond: int) -> None:
    branch, manifest = _directional_transition_branch_and_manifest(artifact, bond=bond)
    successor = _term_field(manifest["transition_term"], "successor_snapshot")
    _term_field(successor, "domains").append(
        [
            {
                "__dataclass__": "grimace._south_star1.residual_constraints.VarId",
                "fields": [["kind", "unrelated_directional_test"], ["key", [99]]],
            },
            [False, True],
        ]
    )
    digest = _closed_term_digest(successor)
    _set_term_field(manifest["transition_term"], "successor_snapshot_digest", digest)
    _refresh_linked_raw_lifecycle_residual_digest(
        branch,
        manifest=manifest,
        field="successor_residual_snapshot_digest",
        digest=digest,
    )
    _refresh_transition_manifest_digest(manifest)
    _refresh_object_and_artifact_digest(artifact, branch)


def _linked_tetra_lifecycle_manifest(
    *,
    branch,
    manifest,
    lifecycle_kind: str,
    certificate_kind: str,
):
    operation = (
        "WriterStereoLifecycleEvidence"
        if lifecycle_kind == "raw"
        else "WriterStereoBranchCertificate"
    )
    for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]:
        if item["evidence_digest"] not in manifest["linked_lifecycle_digests"]:
            continue
        if item["operation"] != operation:
            continue
        if lifecycle_kind == "certificate" and item["certificate_kind"] != certificate_kind:
            continue
        return item
    raise AssertionError(f"missing linked tetra lifecycle manifest: {lifecycle_kind}")


def _append_unrelated_raw_lifecycle(branch, *, manifest):
    linked = _linked_tetra_lifecycle_manifest(
        branch=branch,
        manifest=manifest,
        lifecycle_kind="raw",
        certificate_kind="",
    )
    unrelated = deepcopy(linked)
    unrelated["evidence_digest"] = f"unrelated:{linked['evidence_digest']}"
    unrelated["linked_residual_work_digests"] = []
    unrelated["residual_work_digests"] = []
    unrelated["residual_work_operations"] = []
    branch["payload"]["obligation_manifests"]["stereo_lifecycle"].append(unrelated)
    return unrelated


def _first_graph_ring_delta_event(branch, kind: str):
    for item in branch["payload"]["graph_ring_delta"]["manifest"]["event_manifests"]:
        if item["kind"] == kind:
            return item
    raise AssertionError(f"missing graph/ring event kind: {kind}")


def _first_directional_bond_delta_branch(artifact):
    for item in artifact["objects"]:
        if item["kind"] != "branch_support":
            continue
        delta = item["payload"]["graph_ring_delta"]
        if delta["kind"] != "bond_advance":
            continue
        event = _first_graph_ring_delta_event(item, "bond_emitted")
        direction_mark = event["direction_mark"]
        if direction_mark["value"] != 0:
            return item
    raise AssertionError("missing directional bond delta branch")


def _first_terminal_projection_object(artifact):
    support = _first_support_string_object(artifact)
    return _object(artifact, support["payload"]["terminal_projection_ref"])


def _first_terminal_support_object(artifact):
    support = _first_support_string_object(artifact)
    return _object(artifact, support["payload"]["terminal_support_refs"][0])


def _first_local_evidence(artifact, kind: str):
    for item in artifact["objects"]:
        if item["kind"] != "branch_support":
            continue
        evidence = item["payload"]["local_evidence"]
        if evidence["kind"] == kind:
            return evidence
    raise AssertionError(f"missing local evidence kind: {kind}")


def _refresh_local_evidence_digest(evidence) -> None:
    evidence["digest"] = _identity_digest(
        {"kind": evidence["kind"], "manifest": evidence["manifest"]},
    )


def _refresh_graph_ring_delta_digest(delta) -> None:
    delta["digest"] = _identity_digest(
        {"kind": delta["kind"], "manifest": delta["manifest"]},
    )


def _term_field(term, name: str):
    for field_name, value in term["fields"]:
        if field_name == name:
            return value
    raise AssertionError(f"missing term field: {name}")


def _set_term_field(term, name: str, value) -> None:
    for field in term["fields"]:
        if field[0] == name:
            field[1] = value
            return
    raise AssertionError(f"missing term field: {name}")


def _closed_term_digest(term) -> str:
    return _digest_terms_bounded(
        term,
        budget=WriterEnvelopeWorkBudget(),
        operation="test.closed_term.digest",
    )


def _refresh_transition_manifest_digest(manifest) -> None:
    manifest["transition_digest"] = _closed_term_digest(manifest["transition_term"])


def _refresh_cursor_digest(cursor) -> None:
    cursor["digest"] = _closed_term_digest(cursor["terms"])


def _single_cursor_state(cursor):
    weighted_states = _term_field(cursor["terms"], "weighted_states")
    if len(weighted_states) != 1:
        raise AssertionError("expected single-state cursor")
    return weighted_states[0][0]


def _cursor_state_by_digest(cursor, digest: str):
    matches = [
        state
        for state, _weight in _term_field(cursor["terms"], "weighted_states")
        if _closed_term_digest(state) == digest
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one cursor state for digest {digest}")
    return matches[0]


def _text_projection_identity_digest(payload) -> str:
    return _identity_digest(
        {
            "source_cursor_digest": payload["source_cursor"]["digest"],
            "emitted_text": payload["emitted_text"],
            "successor_cursor_digest": payload["successor_cursor"]["digest"],
            "immediate_multiplicity": payload["immediate_multiplicity"],
            "support_count": payload["support_count"],
            "completion_count": payload["completion_count"],
            "branch_certificate_digests": payload["branch_certificate_digests"],
        },
    )


def _propagate_text_projection_cursor_change(
    artifact,
    *,
    old_cursor_digest: str,
    new_cursor,
) -> None:
    for item in artifact["objects"]:
        if item["kind"] != "text_projection":
            continue
        payload = item["payload"]
        if payload["source_cursor"]["digest"] == old_cursor_digest:
            payload["source_cursor"] = new_cursor
            payload["digest"] = _text_projection_identity_digest(payload)
            for branch_ref in payload["branch_support_refs"]:
                branch = _object(artifact, branch_ref)
                branch["payload"]["source_cursor_digest"] = new_cursor["digest"]
                branch["payload"]["graph_ring_delta"]["manifest"][
                    "source_cursor_digest"
                ] = new_cursor["digest"]
                _refresh_graph_ring_delta_digest(branch["payload"]["graph_ring_delta"])
    for item in artifact["objects"]:
        if item["kind"] != "replay_path":
            continue
        if item["payload"]["final_cursor_digest"] == old_cursor_digest:
            item["payload"]["final_cursor_digest"] = new_cursor["digest"]
    for item in artifact["objects"]:
        if item["kind"] != "terminal_projection":
            continue
        payload = item["payload"]
        if payload["source_cursor"]["digest"] == old_cursor_digest:
            payload["source_cursor"] = new_cursor


def _refresh_object_and_artifact_digest(artifact, obj) -> None:
    del obj
    changed = True
    while changed:
        changed = False
        for item in artifact["objects"]:
            digest = _identity_digest(
                support_artifact_object_identity_term(item["kind"], item["payload"]),
            )
            object_id = f"obj:{digest}"
            if item["digest"] == digest and item["object_id"] == object_id:
                continue
            old_id = item["object_id"]
            item["digest"] = digest
            item["object_id"] = object_id
            _replace_artifact_ref(artifact, old_id=old_id, new_id=object_id)
            changed = True
    artifact["metrics"] = artifact_metrics(
        artifact["objects"],
        roots=artifact["roots"],
    )
    _refresh_artifact_digest(artifact)


def _replace_artifact_ref(value, *, old_id: str, new_id: str) -> None:
    if isinstance(value, dict):
        for key, item in list(value.items()):
            if item == old_id:
                value[key] = new_id
            else:
                _replace_artifact_ref(item, old_id=old_id, new_id=new_id)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            if item == old_id:
                value[index] = new_id
            else:
                _replace_artifact_ref(item, old_id=old_id, new_id=new_id)


def _assert_structural_checker_accepts(test_case, artifact) -> None:
    checked = verify_writer_support_artifact_consistency(artifact)
    test_case.assertTrue(checked.accepted, checked.reason)


def _refresh_linked_raw_lifecycle_residual_digest(
    branch,
    *,
    manifest,
    field: str,
    digest: str,
) -> None:
    lifecycle = _linked_tetra_lifecycle_manifest(
        branch=branch,
        manifest=manifest,
        lifecycle_kind="raw",
        certificate_kind="",
    )
    lifecycle[field] = digest


def _first_closure_evidence_item(artifact):
    evidence = _first_local_evidence(artifact, "closure_bond_text")
    return evidence["manifest"]["items"][0]


def _different_local_order_digest(artifact, *, branch, cursor_name: str, atom: int) -> str:
    projection = _text_projection_for_branch(artifact, branch)
    cursor = projection["payload"][cursor_name]
    state = cursor["terms"]["fields"][0][1][0][0]
    stereo = _term_field(state, "stereo_state")
    for record in _term_field(stereo, "local_orders"):
        if _term_field(record, "atom") != atom:
            return _closed_term_digest(record)
    raise AssertionError("missing alternate local-order record")


def _text_projection_for_branch(artifact, branch):
    for item in artifact["objects"]:
        if item["kind"] != "text_projection":
            continue
        if branch["object_id"] in item["payload"]["branch_support_refs"]:
            return item
    raise AssertionError("missing text projection for branch")


def _refresh_local_order_event_identity_digest(event) -> None:
    identity = {
        "site": event["site"],
        "atom": event["atom"],
        "local_order": event["local_order"],
        "reference_order": event["reference_order"],
        "source_local_order_record_digest": event[
            "source_local_order_record_digest"
        ],
        "successor_local_order_record_digest": event[
            "successor_local_order_record_digest"
        ],
    }
    event["local_order_identity_digest"] = _identity_digest(identity)


def _initial_snapshot(prepared, options):
    return capture_writer_frontier_snapshot(
        prepared=prepared,
        runtime_options=options,
        cursor=initial_writer_frontier_cursor(prepared, options),
    )


def _tetra_facts_with_implicit_h_only_outside_specified_site(facts):
    site = facts.stereo.tetrahedral[0]
    outside_occurrence = LigandOccurrence(
        id=OccurrenceId(99),
        site=SiteId(99),
        kind=LigandKind.IMPLICIT_H,
        atom=site.center,
        bond=None,
    )
    return replace(
        facts,
        stereo=replace(
            facts.stereo,
            tetrahedral=(
                replace(
                    site,
                    ligand_occurrences=site.ligand_occurrences[:-1],
                    reference_order=site.reference_order[:-1],
                ),
            ),
        ),
        ligand_occurrences=facts.ligand_occurrences[:-1] + (outside_occurrence,),
    )


def _writer_options(rooted_at_atom=-1):
    return SouthStarRuntimeOptions(
        rooted_at_atom=rooted_at_atom,
        canonical=False,
        do_random=True,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


def _prepare(facts):
    return prepare_south_star_mol_from_facts(
        facts,
        writer_surface=SouthStarWriterSurface(),
    )


def _object(artifact, object_id):
    return next(item for item in artifact["objects"] if item["object_id"] == object_id)


def _support_strings(artifact):
    root = _object(artifact, artifact["roots"]["support_image_root"])
    return root["payload"]["support_strings"]


def _refresh_artifact_digest(artifact):
    artifact["digest"] = _digest_terms_bounded(
        artifact_manifest(artifact),
        budget=WriterEnvelopeWorkBudget(),
        operation="test.artifact_manifest.digest",
    )


if __name__ == "__main__":
    unittest.main()
