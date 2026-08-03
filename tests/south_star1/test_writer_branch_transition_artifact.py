"""Count-free writer branch transition artifact tests."""

from __future__ import annotations

from functools import lru_cache
from copy import deepcopy
import unittest
from unittest.mock import patch

from grimace._south_star1.ids import BondId
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.writer_branch_transition_artifact import verify_writer_branch_transition_artifact_envelope
from grimace._south_star1.writer_branch_transition_artifact import branch_transition_artifact_manifest
from grimace._south_star1.writer_branch_transition_artifact import writer_branch_transition_artifact_for_support
from grimace._south_star1.writer_branch_transition_artifact_checker import verify_writer_branch_transition_artifact_consistency
from grimace._south_star1.writer_branch_transition_artifact_fact_verifier import verify_writer_branch_transition_artifact_for_facts
from grimace._south_star1.writer_envelope_terms import _digest_terms_bounded
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_support_artifact_checker import artifact_metrics
from grimace._south_star1.writer_support_artifact_checker import support_artifact_object_identity_term
from grimace._south_star1.writer_support_artifact_envelope import _ObjectTable
from grimace._south_star1.writer_support_artifact_envelope import _add_text_projection
from grimace._south_star1.writer_events import WriterRingEndpointEmitted
from grimace._south_star1.writer_events import WriterRingEndpointPaired
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_snapshot import WriterDecoderBoundary
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from tests.south_star1.writer_test_fixtures import shared_directional_ring_carrier_facts
from tests.south_star1.writer_test_fixtures import directional_non_single_ring_carrier_facts
from tests.south_star1.writer_test_fixtures import directional_ring_carrier_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import shared_acyclic_directional_facts
from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options
from tests.south_star1.writer_proof_sources import shared_ring_branch_source


class WriterBranchTransitionArtifactTest(unittest.TestCase):
    def test_aromatic_plain_atom_evidence_forgeries_reject_facts_bound(self) -> None:
        facts = ordinary_molecule_facts_from_smiles("c1ccccc1")
        options = writer_runtime_options(rooted_at_atom=0)
        _prepared, source = _branch_artifact_for_event_kind(
            facts, options, "atom_emitted"
        )
        mutations = (
            ("atom_id", lambda manifest: manifest.__setitem__("atom_id", 1)),
            ("element", lambda manifest: manifest.__setitem__("element", "N")),
            ("aromatic", lambda manifest: manifest.__setitem__("aromatic", False)),
        )
        for name, mutate in mutations:
            with self.subTest(field=name):
                forged = deepcopy(source)
                branch = next(
                    item for item in forged["objects"] if item["kind"] == "branch_support"
                )["payload"]
                evidence = branch["local_evidence"]
                mutate(evidence["manifest"])
                evidence["digest"] = _identity_digest(
                    {"kind": evidence["kind"], "manifest": evidence["manifest"]}
                )
                delta = branch["graph_ring_delta"]
                delta["manifest"]["local_evidence_digest"] = evidence["digest"]
                delta["digest"] = _identity_digest(
                    {"kind": delta["kind"], "manifest": delta["manifest"]}
                )
                _redigest_branch_artifact(forged)

                structural = verify_writer_branch_transition_artifact_consistency(forged)
                replay = verify_writer_branch_transition_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=forged,
                )
                self.assertTrue(structural.accepted, structural.reason)
                self.assertFalse(replay.accepted)
                self.assertTrue(
                    "aromatic_atom" in replay.reason
                    or "local_plain_atom_text" in replay.reason
                )

    def test_aromatic_atom_and_bond_forgeries_reject_facts_bound(self) -> None:
        cases = (
            ("c1ccccc1", "atom_emitted", "n", "aromatic_policy_domain_mismatch"),
            (
                "c1ccccc1-c1ccccc1",
                "bond_emitted",
                "",
                "aromatic_single_bridge_text_mismatch",
            ),
        )
        for smiles, event_kind, forged_text, reason in cases:
            with self.subTest(smiles=smiles, event=event_kind):
                facts = ordinary_molecule_facts_from_smiles(smiles)
                options = writer_runtime_options(rooted_at_atom=0)
                prepared, artifact = _branch_artifact_for_event_kind(
                    facts,
                    options,
                    event_kind,
                    require_text="-" if event_kind == "bond_emitted" else None,
                )
                forged = deepcopy(artifact)
                by_kind = {item["kind"]: item for item in forged["objects"]}
                branch = by_kind["branch_support"]["payload"]
                event = next(
                    item
                    for item in branch["graph_ring_delta"]["manifest"]["event_manifests"]
                    if item["kind"] == event_kind
                )
                event["text"] = forged_text
                branch["emitted_text"] = forged_text
                branch["graph_ring_delta"]["manifest"]["emitted_text"] = forged_text
                if event_kind == "atom_emitted":
                    branch["local_evidence"]["manifest"]["rendered_text"] = forged_text
                    branch["local_evidence"]["digest"] = _identity_digest(
                        {
                            "kind": branch["local_evidence"]["kind"],
                            "manifest": branch["local_evidence"]["manifest"],
                        }
                    )
                    branch["graph_ring_delta"]["manifest"][
                        "local_evidence_digest"
                    ] = branch["local_evidence"]["digest"]
                branch["graph_ring_delta"]["digest"] = _identity_digest(
                    {
                        "kind": branch["graph_ring_delta"]["kind"],
                        "manifest": branch["graph_ring_delta"]["manifest"],
                    }
                )
                by_kind["text_projection"]["payload"]["emitted_text"] = forged_text
                _redigest_branch_artifact(forged)

                structural = verify_writer_branch_transition_artifact_consistency(forged)
                replay = verify_writer_branch_transition_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=forged,
                )
                self.assertTrue(structural.accepted, structural.reason)
                self.assertFalse(replay.accepted)
                self.assertIn(reason, replay.reason)

    def test_supported_transition_matrix_replays_facts_bound(self) -> None:
        cases = (
            (tetrahedral_facts(), writer_runtime_options(), "tetrahedral atom-token restriction"),
            (tetrahedral_facts(), writer_runtime_options(), "tetrahedral local-order factor closure"),
            (directional_facts(), writer_runtime_options(rooted_at_atom=2), "directional carrier-mark restriction"),
            (shared_acyclic_directional_facts(), writer_runtime_options(rooted_at_atom=0), "directional carrier-mark restriction"),
            (directional_ring_carrier_facts(), writer_runtime_options(rooted_at_atom=0), "directional ring endpoint projection"),
            (directional_ring_carrier_facts(), writer_runtime_options(rooted_at_atom=0), "directional ring pair restriction"),
        )
        for facts, options, operation in cases:
            with self.subTest(operation=operation, facts=type(facts).__name__):
                prepared, artifact = _branch_artifact_for_operation(facts, options, operation)
                live = verify_writer_branch_transition_artifact_envelope(prepared=prepared, artifact=artifact)
                facts_bound = verify_writer_branch_transition_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=artifact,
                )
                self.assertTrue(live.accepted, live.reason)
                self.assertTrue(facts_bound.accepted, facts_bound.reason)
                self.assertEqual(facts_bound.unchecked_obligation_families, ())
                self.assertIn(operation, facts_bound.semantically_replayed_operations)
                self.assertEqual(
                    facts_bound.semantically_replayed_operations.count(operation),
                    1,
                )

    def test_non_single_ring_opening_and_pair_replay_semantically(self) -> None:
        facts = directional_non_single_ring_carrier_facts()
        options = writer_runtime_options(rooted_at_atom=0)
        for operation in (
            "directional ring endpoint projection",
            "directional ring pair restriction",
        ):
            with self.subTest(operation=operation):
                prepared, artifact = _branch_artifact_for_operation(
                    facts,
                    options,
                    operation,
                )
                live = verify_writer_branch_transition_artifact_envelope(
                    prepared=prepared,
                    artifact=artifact,
                )
                facts_bound = verify_writer_branch_transition_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=artifact,
                )
                self.assertTrue(live.accepted, live.reason)
                self.assertTrue(facts_bound.accepted, facts_bound.reason)
                self.assertEqual(facts_bound.unchecked_obligation_families, ())
                self.assertEqual(
                    facts_bound.semantically_replayed_operations,
                    (operation,),
                )
                if operation == "directional ring pair restriction":
                    branch = next(
                        item["payload"]
                        for item in artifact["objects"]
                        if item["kind"] == "branch_support"
                    )
                    coupling = branch["obligation_manifests"][
                        "directional_ring_closure_lifecycle"
                    ][0]["coupling_term"]
                    raw_lifecycle = next(
                        manifest
                        for manifest in branch["obligation_manifests"][
                            "stereo_lifecycle"
                        ]
                        if manifest["operation"] == "WriterStereoLifecycleEvidence"
                    )
                    residual = branch["obligation_manifests"]["residual_work"][0]
                    self.assertEqual(
                        _closed_field(coupling, "stereo_lifecycle_digest"),
                        raw_lifecycle["evidence_digest"],
                    )
                    self.assertEqual(
                        _closed_field(coupling, "residual_work_digests"),
                        [residual["evidence_digest"]],
                    )
                    self.assertIn(
                        "directional_ring_closure_lifecycle",
                        facts_bound.checked_obligation_families,
                    )

    def test_non_single_ring_pair_coupling_forgeries_reject_facts_bound(self) -> None:
        facts = directional_non_single_ring_carrier_facts()
        options = writer_runtime_options(rooted_at_atom=0)
        _prepared, source = _branch_artifact_for_operation(
            facts,
            options,
            "directional ring pair restriction",
        )
        _opening_prepared, opening = _branch_artifact_for_operation(
            facts,
            options,
            "directional ring endpoint projection",
        )
        opening_branch = next(
            item["payload"]
            for item in opening["objects"]
            if item["kind"] == "branch_support"
        )
        opening_lifecycle_digest = next(
            manifest["evidence_digest"]
            for manifest in opening_branch["obligation_manifests"]["stereo_lifecycle"]
            if manifest["operation"] == "WriterStereoLifecycleEvidence"
        )
        opening_residual_digest = opening_branch["obligation_manifests"][
            "residual_work"
        ][0]["evidence_digest"]

        def mutate_marker_side(branch, item, term):
            closure = branch["local_evidence"]["manifest"]["closure_bond_text"][0]
            closure["marker_side"] = "closing"
            _set_closed_field(term, "marker_side", "closing")
            _set_closed_field(term, "closure_manifest_digest", _identity_digest(closure))

        def mutate_lifecycle(branch, _item, term):
            certificate = next(
                manifest
                for manifest in branch["obligation_manifests"]["stereo_lifecycle"]
                if manifest["operation"] == "WriterStereoBranchCertificate"
            )
            _set_closed_field(term, "stereo_lifecycle_digest", certificate["evidence_digest"])

        def mutate_residual(_branch, _item, term):
            _set_closed_field(term, "residual_work_digests", ["0" * 64])

        def mutate_closed_record(_branch, _item, term):
            _set_closed_field(term, "closed_closure_record_digest", "0" * 64)

        def mutate_both_markers(branch, _item, term):
            closure = branch["local_evidence"]["manifest"]["closure_bond_text"][0]
            closure["opening_marker"] = "="
            closure["closing_marker"] = "="
            _set_closed_field(term, "opening_marker", "=")
            _set_closed_field(term, "closing_marker", "=")
            _set_closed_field(term, "closure_manifest_digest", _identity_digest(closure))

        def mutate_no_markers(branch, _item, term):
            closure = branch["local_evidence"]["manifest"]["closure_bond_text"][0]
            closure["opening_marker"] = ""
            closure["closing_marker"] = ""
            _set_closed_field(term, "opening_marker", "")
            _set_closed_field(term, "closing_marker", "")
            _set_closed_field(term, "closure_manifest_digest", _identity_digest(closure))

        def mutate_direction_mark(branch, _item, _term):
            event = next(
                event
                for event in branch["graph_ring_delta"]["manifest"]["event_manifests"]
                if event["kind"] == "ring_endpoint_paired"
            )
            event["direction_mark"]["value"] = 1

        def mutate_cross_branch_lifecycle(_branch, _item, term):
            _set_closed_field(term, "stereo_lifecycle_digest", opening_lifecycle_digest)

        def mutate_cross_branch_residual(_branch, _item, term):
            _set_closed_field(term, "residual_work_digests", [opening_residual_digest])

        cases = (
            ("marker_side", mutate_marker_side, "local_closure_marker_side_mismatch"),
            ("wrong_lifecycle", mutate_lifecycle, "directional_ring_coupling_lifecycle_branch_mismatch"),
            ("wrong_residual", mutate_residual, "directional_ring_coupling_residual_state_mismatch"),
            ("stale_closed_record", mutate_closed_record, "directional_ring_coupling_closed_record_mismatch"),
            ("both_markers", mutate_both_markers, "local_closure_marker_duplicate"),
            ("no_markers", mutate_no_markers, "local_closure_marker_missing"),
            ("direction_mark", mutate_direction_mark, "graph_ring_bond_marker_mismatch"),
            ("cross_branch_lifecycle", mutate_cross_branch_lifecycle, "directional_ring_coupling_lifecycle_branch_mismatch"),
            ("cross_branch_residual", mutate_cross_branch_residual, "directional_ring_coupling_residual_state_mismatch"),
        )
        for name, mutate, reason in cases:
            with self.subTest(name=name):
                forged = deepcopy(source)
                branch = next(
                    item["payload"]
                    for item in forged["objects"]
                    if item["kind"] == "branch_support"
                )
                item = branch["obligation_manifests"][
                    "directional_ring_closure_lifecycle"
                ][0]
                term = item["coupling_term"]
                mutate(branch, item, term)
                _refresh_directional_ring_coupling(branch, item)
                _redigest_branch_artifact(forged)

                structural = verify_writer_branch_transition_artifact_consistency(forged)
                replay = verify_writer_branch_transition_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=forged,
                )
                self.assertTrue(structural.accepted, structural.reason)
                self.assertFalse(replay.accepted)
                self.assertIn(reason, replay.reason)

    def test_shared_ring_opening_and_pair_branches_replay_semantically(self) -> None:
        for phase in ("opening", "pair"):
            for mark in (DirectionMark.ABSENT, DirectionMark.FWD, DirectionMark.REV):
                with self.subTest(phase=phase, mark=mark):
                    facts, options, prepared, artifact = _shared_ring_branch_artifact(phase, mark)

                    structural = verify_writer_branch_transition_artifact_consistency(artifact)
                    live = verify_writer_branch_transition_artifact_envelope(
                        prepared=prepared,
                        artifact=artifact,
                    )
                    facts_bound = verify_writer_branch_transition_artifact_for_facts(
                        facts=facts,
                        runtime_options=options,
                        artifact=artifact,
                    )

                    self.assertTrue(structural.accepted, structural.reason)
                    self.assertTrue(live.accepted, live.reason)
                    self.assertTrue(facts_bound.accepted, facts_bound.reason)
                    self.assertEqual(len(artifact["objects"]), 3)
                    self.assertLess(
                        artifact["metrics"]["largest_object_identity_input_bytes"],
                        WriterEnvelopeWorkBudget().max_digest_term_bytes,
                    )
                    self.assertEqual(facts_bound.unchecked_obligation_families, ())
                    operation = (
                        "directional ring endpoint projection"
                        if phase == "opening"
                        else "directional ring pair restriction"
                    )
                    self.assertEqual(
                        facts_bound.semantically_replayed_operations,
                        (operation,),
                    )
                    branch = next(
                        item["payload"]
                        for item in artifact["objects"]
                        if item["kind"] == "branch_support"
                    )
                    self.assertEqual(
                        branch["obligation_manifests"][
                            "directional_ring_closure_lifecycle"
                        ],
                        [],
                    )
                    manifest = next(
                        item
                        for item in branch["obligation_manifests"]["residual_work"]
                        if item["operation"] == operation
                    )
                    raw_lifecycle = next(
                        item
                        for item in branch["obligation_manifests"][
                            "stereo_lifecycle"
                        ]
                        if item["operation"] == "WriterStereoLifecycleEvidence"
                    )
                    expected_capabilities = (
                        [
                            "directional_ring_pair_compatibility",
                            "residual_propagation",
                            "shared_directional_carrier_restriction",
                        ]
                        if phase == "opening"
                        else [
                            "directional_carrier_restriction",
                            "directional_ring_pair_compatibility",
                            "directional_site_compatibility",
                            "residual_factor_discharge",
                            "residual_propagation",
                            "shared_directional_carrier_restriction",
                        ]
                    )
                    self.assertEqual(
                        raw_lifecycle["lifecycle_capabilities"],
                        expected_capabilities,
                    )
                    term = manifest["transition_term"]
                    models = _closed_field(term, "carrier_models")
                    self.assertEqual(len(models), 2)
                    if phase == "opening":
                        self.assertTrue(
                            term["__dataclass__"].endswith(
                                "SharedDirectionalRingEndpointProjectionTransitionTerm"
                            )
                        )
                        self.assertEqual(
                            len(_closed_field(term, "domain_intersections")),
                            2,
                        )
                        self.assertEqual(_closed_field(term, "projected_variables"), [])
                        self.assertEqual(
                            _closed_field(term, "discharged_factor_keys"),
                            [],
                        )
                        values = tuple(
                            tuple(value["value"] for value in intersection[1])
                            for intersection in _closed_field(
                                term,
                                "domain_intersections",
                            )
                        )
                        expected_values = {
                            DirectionMark.ABSENT: (
                                ("absent", "positive", "negative"),
                                ("absent", "positive", "negative"),
                            ),
                            DirectionMark.FWD: (("positive",), ("negative",)),
                            DirectionMark.REV: (("negative",), ("positive",)),
                        }
                        self.assertEqual(values, expected_values[mark])
                    else:
                        self.assertEqual(len(_closed_field(term, "restrictions")), 2)

    def test_shared_ring_transition_term_forgeries_reject_semantically(self) -> None:
        opening_cases = (
            ("missing", _forge_transition_missing, "directional_ring_projection_transition_missing"),
            ("singular_term", _convert_shared_opening_to_singular, "shared_directional_ring_model_mismatch"),
            ("remove_model", lambda term: _closed_field(term, "carrier_models").pop(), "shared_directional_ring_model_mismatch"),
            ("duplicate_model", _duplicate_first_shared_model, "shared_directional_ring_model_mismatch"),
            ("reverse_models", lambda term: _closed_field(term, "carrier_models").reverse(), "shared_directional_ring_model_mismatch"),
            ("change_model_side", _change_first_model_side, "shared_directional_ring_model_mismatch"),
            ("remove_choice", lambda term: _closed_field(term, "compatible_second_endpoint_choices").pop(), "shared_directional_ring_choice_relation_mismatch"),
            ("add_incompatible_choice", _add_incompatible_shared_choice, "shared_directional_ring_choice_relation_mismatch"),
            ("change_intersection", _remove_first_intersection_value, "shared_directional_ring_intersection_mismatch"),
            ("forged_correlation", _change_shared_factor_correlation, "shared_directional_ring_transition_successor_lifecycle_mismatch"),
            ("detached_successor", _change_successor_snapshot, "shared_directional_ring_transition_successor_lifecycle_mismatch"),
        )
        pair_cases = (
            ("missing", _forge_transition_missing, "directional_ring_pair_transition_missing"),
            ("remove_model", lambda term: _closed_field(term, "carrier_models").pop(), "shared_directional_ring_model_mismatch"),
            ("remove_restriction", lambda term: _closed_field(term, "restrictions").pop(), "shared_directional_ring_restriction_mismatch"),
            ("swap_restrictions", lambda term: _closed_field(term, "restrictions").reverse(), "shared_directional_ring_restriction_mismatch"),
            ("remove_choice", lambda term: _closed_field(term, "compatible_second_endpoint_choices").pop(), "shared_directional_ring_choice_relation_mismatch"),
            ("wrong_orientation", _change_pair_orientation, "directional_ring_pair_canonical_orientation_mismatch"),
            ("wrong_occurrence_mark", _change_occurrence_mark, "directional_ring_pair_bond_occurrence_mismatch"),
            ("missing_discharge", lambda term: _closed_field(term, "discharged_factor_keys").pop(0), "directional_ring_pair_discharge_factor_mismatch"),
            ("forged_projection", _forge_projected_variable, "directional_ring_pair_projected_variables_mismatch"),
            ("detached_successor", _change_successor_snapshot, "shared_directional_ring_transition_successor_lifecycle_mismatch"),
        )
        for phase, cases in (("opening", opening_cases), ("pair", pair_cases)):
            for name, mutate, reason in cases:
                with self.subTest(phase=phase, name=name):
                    facts, options, _prepared, source = _shared_ring_branch_artifact(
                        phase,
                        DirectionMark.FWD,
                    )
                    forged = deepcopy(source)
                    manifest = _branch_residual_manifest(forged)
                    if mutate is _forge_transition_missing:
                        mutate(manifest)
                    else:
                        mutate(manifest["transition_term"])
                        _refresh_transition_digest(manifest)
                    _redigest_branch_artifact(forged)
                    structural = verify_writer_branch_transition_artifact_consistency(
                        forged
                    )
                    replay = verify_writer_branch_transition_artifact_for_facts(
                        facts=facts,
                        runtime_options=options,
                        artifact=forged,
                    )
                    self.assertTrue(structural.accepted, structural.reason)
                    self.assertFalse(replay.accepted)
                    self.assertIn(reason, replay.reason)

    def test_shared_ring_term_borrowed_from_another_mark_rejects(self) -> None:
        for phase in ("opening", "pair"):
            with self.subTest(phase=phase):
                facts, options, _prepared, source = _shared_ring_branch_artifact(
                    phase,
                    DirectionMark.FWD,
                )
                _facts, _options, _prepared, donor = _shared_ring_branch_artifact(
                    phase,
                    DirectionMark.REV,
                )
                forged = deepcopy(source)
                target_manifest = _branch_residual_manifest(forged)
                donor_manifest = _branch_residual_manifest(donor)
                target_manifest["transition_term"] = deepcopy(
                    donor_manifest["transition_term"]
                )
                target_manifest["transition_digest"] = donor_manifest[
                    "transition_digest"
                ]
                _redigest_branch_artifact(forged)
                structural = verify_writer_branch_transition_artifact_consistency(
                    forged
                )
                replay = verify_writer_branch_transition_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=forged,
                )
                self.assertTrue(structural.accepted, structural.reason)
                self.assertFalse(replay.accepted)
                self.assertIn("shared_directional_ring_transition", replay.reason)

    def test_shared_ring_lifecycle_capability_requires_semantic_replay(self) -> None:
        facts, options, _prepared, source = _shared_ring_branch_artifact(
            "pair",
            DirectionMark.FWD,
        )
        forged = deepcopy(source)
        branch = next(
            item for item in forged["objects"] if item["kind"] == "branch_support"
        )
        raw = next(
            item
            for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]
            if item["operation"] == "WriterStereoLifecycleEvidence"
        )
        raw["lifecycle_capabilities"].remove(
            "shared_directional_carrier_restriction"
        )
        _redigest_branch_artifact(forged)
        structural = verify_writer_branch_transition_artifact_consistency(forged)
        replay = verify_writer_branch_transition_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=forged,
        )
        self.assertTrue(structural.accepted, structural.reason)
        self.assertFalse(replay.accepted)
        self.assertIn(
            "directional_ring_pair_lifecycle_capabilities_mismatch",
            replay.reason,
        )

    def test_build_and_live_verification_do_not_enter_count_or_support_paths(self) -> None:
        source = shared_ring_branch_source("opening", DirectionMark.FWD)
        facts = source.context.prepared.facts
        options = source.context.runtime_options
        prepared, snapshot, support = source.context.prepared, source.snapshot, source.support
        del facts, options
        blockers = (
            patch(
                "grimace._south_star1.writer_frontier_count_envelope."
                "writer_frontier_count_envelope_for_snapshot",
                side_effect=AssertionError("count envelope invoked"),
            ),
            patch(
                "grimace._south_star1.writer_count_dag_envelope."
                "writer_count_certificate_dag_envelope_for_product",
                side_effect=AssertionError("count DAG invoked"),
            ),
            patch(
                "grimace._south_star1.writer_snapshot."
                "_iter_writer_snapshot_certified_support_strings",
                side_effect=AssertionError("support enumeration invoked"),
                create=True,
            ),
        )
        with blockers[0], blockers[1], blockers[2]:
            artifact = writer_branch_transition_artifact_for_support(
                prepared=prepared,
                snapshot=snapshot,
                support=support,
            )
            live = verify_writer_branch_transition_artifact_envelope(
                prepared=prepared,
                artifact=artifact,
            )
        self.assertTrue(live.accepted, live.reason)

    def test_coherently_redigested_emitted_text_forgery_is_live_rejected(self) -> None:
        _facts, _options, prepared, artifact = _shared_ring_branch_artifact(
            "opening",
            DirectionMark.FWD,
        )
        forged = deepcopy(artifact)
        objects = {item["kind"]: item for item in forged["objects"]}
        objects["branch_support"]["payload"]["emitted_text"] += "X"
        objects["text_projection"]["payload"]["emitted_text"] += "X"
        _redigest_branch_artifact(forged)

        structural = verify_writer_branch_transition_artifact_consistency(forged)
        live = verify_writer_branch_transition_artifact_envelope(
            prepared=prepared,
            artifact=forged,
        )

        self.assertTrue(structural.accepted, structural.reason)
        self.assertFalse(live.accepted)
        self.assertIn("live_branch_artifact_mismatch", live.reason)

    def test_coherent_graph_delta_and_obligation_substitutions_reject(self) -> None:
        facts, options, prepared, artifact = _shared_ring_branch_artifact(
            "opening", DirectionMark.FWD,
        )
        forged = deepcopy(artifact)
        branch = next(item for item in forged["objects"] if item["kind"] == "branch_support")
        delta = branch["payload"]["graph_ring_delta"]
        ring_event = next(
            event
            for event in delta["manifest"]["event_manifests"]
            if event["kind"] == "ring_endpoint_emitted"
        )
        ring_event["bond"] = 999
        delta["digest"] = _identity_digest(
            {"kind": delta["kind"], "manifest": delta["manifest"]}
        )
        _redigest_branch_artifact(forged)
        self.assertTrue(
            verify_writer_branch_transition_artifact_consistency(forged).accepted
        )
        rejected = verify_writer_branch_transition_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=forged,
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("local_closure_bond_missing", rejected.reason)

        _other_facts, _other_options, _other_prepared, other = _shared_ring_branch_artifact(
            "opening", DirectionMark.REV,
        )
        forged = deepcopy(artifact)
        branch = next(item for item in forged["objects"] if item["kind"] == "branch_support")
        other_branch = next(item for item in other["objects"] if item["kind"] == "branch_support")
        branch["payload"]["obligation_manifests"] = deepcopy(
            other_branch["payload"]["obligation_manifests"]
        )
        branch["payload"]["obligation_summary"] = deepcopy(
            other_branch["payload"]["obligation_summary"]
        )
        _redigest_branch_artifact(forged)
        self.assertTrue(
            verify_writer_branch_transition_artifact_consistency(forged).accepted
        )
        rejected = verify_writer_branch_transition_artifact_envelope(
            prepared=prepared,
            artifact=forged,
        )
        self.assertFalse(rejected.accepted)
        self.assertIn("live_branch_artifact_mismatch", rejected.reason)

    def test_prepared_identity_detached_from_snapshot_is_facts_rejected(self) -> None:
        facts, options, _prepared, artifact = _shared_ring_branch_artifact(
            "opening", DirectionMark.ABSENT,
        )
        forged = deepcopy(artifact)
        forged["prepared_identity"]["digest"] = "0" * 64
        forged["source_snapshot"]["prepared_identity_digest"] = "0" * 64
        _redigest_branch_artifact(forged)
        structural = verify_writer_branch_transition_artifact_consistency(forged)
        rejected = verify_writer_branch_transition_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=forged,
        )
        self.assertTrue(structural.accepted, structural.reason)
        self.assertFalse(rejected.accepted)
        self.assertIn("prepared_identity", rejected.reason)

    def test_count_object_and_duplicate_object_are_structurally_rejected(self) -> None:
        _facts, _options, _prepared, artifact = _shared_ring_branch_artifact(
            "opening",
            DirectionMark.ABSENT,
        )
        self.assertEqual(artifact["schema_version"], 3)
        duplicate = deepcopy(artifact)
        duplicate["objects"].append(deepcopy(duplicate["objects"][0]))
        self.assertFalse(
            verify_writer_branch_transition_artifact_consistency(duplicate).accepted
        )

        count_object = deepcopy(artifact)
        count_object["objects"][0]["kind"] = "count_dag"
        self.assertFalse(
            verify_writer_branch_transition_artifact_consistency(count_object).accepted
        )

        old_schema = deepcopy(artifact)
        old_schema["schema_version"] = 2
        rejected = verify_writer_branch_transition_artifact_consistency(old_schema)
        self.assertFalse(rejected.accepted)
        self.assertIn("unknown_schema_version", rejected.reason)

    def test_snapshot_decoder_rejects_nonclosed_terms(self) -> None:
        _facts, _options, _prepared, artifact = _shared_ring_branch_artifact(
            "opening", DirectionMark.ABSENT,
        )
        cases = (
            ("unknown_class", lambda term: term.__setitem__("__dataclass__", "unknown.Cursor")),
            (
                "unapproved_class",
                lambda term: term.__setitem__(
                    "__dataclass__",
                    "grimace._south_star1.writer_frontier.WriterFrontierState",
                ),
            ),
            ("extra_field", lambda term: term["fields"].append(["extra", 0])),
            ("missing_field", lambda term: term["fields"].pop()),
            ("duplicate_field", lambda term: term["fields"].append(deepcopy(term["fields"][0]))),
        )
        for name, mutate in cases:
            with self.subTest(name=name):
                forged = deepcopy(artifact)
                term = forged["source_snapshot"]["cursor"]["terms"]
                mutate(term)
                _redigest_branch_artifact(forged)
                checked = verify_writer_branch_transition_artifact_consistency(forged)
                self.assertFalse(checked.accepted)
                self.assertIn("writer snapshot closed term violation", checked.reason)

        forged = deepcopy(artifact)
        enum_term = _find_closed_term(forged["source_snapshot"]["cursor"]["terms"], "__enum__")
        enum_term["__enum__"] = "grimace._south_star1.policy.TetraToken"
        _redigest_branch_artifact(forged)
        checked = verify_writer_branch_transition_artifact_consistency(forged)
        self.assertFalse(checked.accepted)
        self.assertIn("enum_value_mismatch", checked.reason)

        forged = deepcopy(artifact)
        cursor_fields = forged["source_snapshot"]["cursor"]["terms"]["fields"]
        next(item for item in cursor_fields if item[0] == "weighted_states")[1] = {
            "not": "a closed collection"
        }
        _redigest_branch_artifact(forged)
        checked = verify_writer_branch_transition_artifact_consistency(forged)
        self.assertFalse(checked.accepted)
        self.assertIn("dataclass_shape_mismatch", checked.reason)

    def test_support_projection_default_and_explicit_all_branches_are_identical(self) -> None:
        source = shared_ring_branch_source("opening", DirectionMark.FWD)
        facts = source.context.prepared.facts
        _options = source.context.runtime_options
        prepared, snapshot, support = source.context.prepared, source.snapshot, source.support
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            snapshot.cursor,
            include_counts=False,
            include_frontier_certificate=True,
            include_count_certificate=False,
        )
        projection = next(
            item
            for item in batch.text_choice_projection_certificates
            if support.checked_branch_certificate in item.branch_certificates
        )
        budget = WriterEnvelopeWorkBudget()
        default_table = _ObjectTable(budget)
        explicit_table = _ObjectTable(budget)
        default_ref = _add_text_projection(
            default_table,
            projection=projection,
            facts=facts,
            budget=budget,
        )
        explicit_ref = _add_text_projection(
            explicit_table,
            projection=projection,
            facts=facts,
            budget=budget,
            branch_certificates=projection.branch_certificates,
        )
        self.assertEqual(default_ref, explicit_ref)
        self.assertEqual(default_table.objects(), explicit_table.objects())


@lru_cache(maxsize=6)
def _shared_ring_branch_artifact(phase: str, mark: DirectionMark):
    source = shared_ring_branch_source(phase, mark)
    facts = source.context.prepared.facts
    options = source.context.runtime_options
    prepared, snapshot, support = source.context.prepared, source.snapshot, source.support
    artifact = writer_branch_transition_artifact_for_support(
        prepared=prepared,
        snapshot=snapshot,
        support=support,
    )
    return facts, options, prepared, artifact


@lru_cache(maxsize=1)


def _branch_artifact_for_operation(facts, options, operation):
    prepared = prepare_writer_facts(facts)
    initial = initial_writer_snapshot(prepared, options)
    pending = [(initial.cursor, 0)]
    seen = set()
    while pending:
        cursor, depth = pending.pop()
        key = repr(cursor)
        if key in seen:
            continue
        seen.add(key)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=True,
            include_count_certificate=False,
        )
        for support in batch.supports:
            if any(item.operation == operation for item in support.residual_work_evidence):
                snapshot = capture_writer_frontier_snapshot(
                    prepared=prepared,
                    runtime_options=options,
                    cursor=cursor,
                    decoder_boundary=WriterDecoderBoundary(consumed_token_count=depth),
                )
                return prepared, writer_branch_transition_artifact_for_support(
                    prepared=prepared,
                    snapshot=snapshot,
                    support=support,
                )
            pending.append((support.successor_cursor, depth + 1))
    raise AssertionError(f"missing branch operation {operation!r}")


def _branch_artifact_for_event_kind(
    facts,
    options,
    event_kind: str,
    *,
    require_text: str | None = None,
):
    prepared = prepare_writer_facts(facts)
    initial = initial_writer_snapshot(prepared, options)
    pending = [(initial.cursor, 0)]
    seen = set()
    while pending:
        cursor, depth = pending.pop()
        key = repr(cursor)
        if key in seen:
            continue
        seen.add(key)
        batch = _checked_writer_frontier_branch_supports(
            prepared,
            cursor,
            include_counts=False,
            include_frontier_certificate=True,
            include_count_certificate=False,
        )
        for support in batch.supports:
            matching = tuple(
                event
                for event in support.events
                if event.__class__.__name__ == {
                    "atom_emitted": "WriterAtomEmitted",
                    "bond_emitted": "WriterBondEmitted",
                }[event_kind]
                and (require_text is None or getattr(event, "text", None) == require_text)
            )
            if matching:
                snapshot = capture_writer_frontier_snapshot(
                    prepared=prepared,
                    runtime_options=options,
                    cursor=cursor,
                    decoder_boundary=WriterDecoderBoundary(consumed_token_count=depth),
                )
                return prepared, writer_branch_transition_artifact_for_support(
                    prepared=prepared,
                    snapshot=snapshot,
                    support=support,
                )
            pending.append((support.successor_cursor, depth + 1))
    raise AssertionError(f"missing branch event {event_kind!r}")


def _redigest_branch_artifact(artifact) -> None:
    budget = WriterEnvelopeWorkBudget()
    by_kind = {item["kind"]: item for item in artifact["objects"]}
    source = by_kind["source_snapshot"]
    source_digest = _identity_digest(
        support_artifact_object_identity_term(source["kind"], source["payload"]),
        budget=budget,
        operation="test.branch_transition.source_object",
    )
    source["digest"] = source_digest
    source["object_id"] = f"obj:{source_digest}"
    branch = by_kind["branch_support"]
    branch_digest = _identity_digest(
        support_artifact_object_identity_term(branch["kind"], branch["payload"]),
        budget=budget,
        operation="test.branch_transition.branch_object",
    )
    branch["digest"] = branch_digest
    branch["object_id"] = f"obj:{branch_digest}"
    projection = by_kind["text_projection"]
    projection["payload"]["branch_support_refs"] = [branch["object_id"]]
    identity = {
        key: value
        for key, value in projection["payload"].items()
        if key not in ("digest", "branch_support_refs")
    }
    projection["payload"]["digest"] = _identity_digest(
        identity,
        budget=budget,
        operation="test.branch_transition.projection_identity",
    )
    projection_digest = _identity_digest(
        support_artifact_object_identity_term(projection["kind"], projection["payload"]),
        budget=budget,
        operation="test.branch_transition.projection_object",
    )
    projection["digest"] = projection_digest
    projection["object_id"] = f"obj:{projection_digest}"
    artifact["roots"]["branch_support_ref"] = branch["object_id"]
    artifact["roots"]["text_projection_ref"] = projection["object_id"]
    artifact["roots"]["source_ref"] = source["object_id"]
    artifact["objects"] = sorted(artifact["objects"], key=lambda item: item["object_id"])
    metrics = artifact_metrics(artifact["objects"])
    artifact["metrics"] = {**metrics, "reachable_object_count": 3, "unreferenced_object_count": 0}
    artifact["digest"] = _digest_terms_bounded(
        branch_transition_artifact_manifest(artifact),
        budget=budget,
        operation="test.branch_transition.artifact",
    )


def _refresh_directional_ring_coupling(branch, item) -> None:
    budget = WriterEnvelopeWorkBudget()
    digest = _digest_terms_bounded(
        item["coupling_term"],
        budget=budget,
        operation="test.branch_transition.coupling_term",
    )
    item["coupling_term_digest"] = digest
    local = branch["local_evidence"]
    local["manifest"]["directional_coupled_digests"] = [digest]
    local["digest"] = _identity_digest(
        {"kind": local["kind"], "manifest": local["manifest"]}
    )
    delta = branch["graph_ring_delta"]
    delta["manifest"]["local_evidence_digest"] = local["digest"]
    delta["digest"] = _identity_digest(
        {"kind": delta["kind"], "manifest": delta["manifest"]}
    )


def _find_closed_term(value, marker: str):
    if isinstance(value, dict):
        if marker in value:
            return value
        for child in value.values():
            found = _find_closed_term(child, marker)
            if found is not None:
                return found
    elif isinstance(value, list):
        for child in value:
            found = _find_closed_term(child, marker)
            if found is not None:
                return found
    return None


def _closed_field(term, name: str):
    for field_name, value in term["fields"]:
        if field_name == name:
            return value
    raise AssertionError(f"missing closed field {name!r}")


def _set_closed_field(term, name: str, value) -> None:
    for field in term["fields"]:
        if field[0] == name:
            field[1] = value
            return
    raise AssertionError(f"missing closed field {name!r}")


def _branch_residual_manifest(artifact):
    branch = next(
        item["payload"]
        for item in artifact["objects"]
        if item["kind"] == "branch_support"
    )
    matches = [
        item
        for item in branch["obligation_manifests"]["residual_work"]
        if item["operation"] in (
            "directional ring endpoint projection",
            "directional ring pair restriction",
        )
    ]
    if len(matches) != 1:
        raise AssertionError("expected one directional ring residual manifest")
    return matches[0]


def _refresh_transition_digest(manifest) -> None:
    manifest["transition_digest"] = _digest_terms_bounded(
        manifest["transition_term"],
        budget=WriterEnvelopeWorkBudget(),
        operation="test.branch_transition.transition",
    )


def _forge_transition_missing(manifest) -> None:
    manifest["transition_term"] = None
    manifest["transition_digest"] = None


def _duplicate_first_shared_model(term) -> None:
    models = _closed_field(term, "carrier_models")
    models[1] = deepcopy(models[0])


def _convert_shared_opening_to_singular(term) -> None:
    term["__dataclass__"] = (
        "grimace._south_star1.writer_residual_transition_terms."
        "DirectionalRingEndpointProjectionTransitionTerm"
    )
    for field in term["fields"]:
        if field[0] == "carrier_models":
            field[0] = "carrier_model"
            field[1] = field[1][0]
            return
    raise AssertionError("missing shared carrier models")


def _change_first_model_side(term) -> None:
    model = _closed_field(term, "carrier_models")[0]
    side = _closed_field(model, "side")
    _set_closed_field(model, "side", "right" if side == "left" else "left")


def _remove_first_intersection_value(term) -> None:
    values = _closed_field(term, "domain_intersections")[0][1]
    values.pop()


def _add_incompatible_shared_choice(term) -> None:
    choices = _closed_field(term, "compatible_second_endpoint_choices")
    choices.append([
        "",
        {
            "__enum__": "grimace._south_star1.policy.DirectionMark",
            "value": DirectionMark.FWD.value,
        },
    ])


def _change_pair_orientation(term) -> None:
    _set_closed_field(
        term,
        "second_canonical_orientation",
        _closed_field(term, "first_canonical_orientation"),
    )


def _change_occurrence_mark(term) -> None:
    mark = _closed_field(term, "bond_occurrence_mark")
    mark["value"] = -mark["value"] if mark["value"] else 1


def _forge_projected_variable(term) -> None:
    var = deepcopy(_closed_field(term, "restrictions")[0][0])
    _closed_field(term, "projected_variables").append(var)


def _change_successor_snapshot(term) -> None:
    snapshot = _closed_field(term, "successor_snapshot")
    domains = _closed_field(snapshot, "domains")
    domains.reverse()
    if len(domains) < 2:
        domains.append(deepcopy(domains[0]))
    _set_closed_field(
        term,
        "successor_snapshot_digest",
        _digest_terms_bounded(
            snapshot,
            budget=WriterEnvelopeWorkBudget(),
            operation="test.branch_transition.successor_snapshot",
        ),
    )


def _change_shared_factor_correlation(term) -> None:
    snapshot = _closed_field(term, "successor_snapshot")
    shared_factor = next(
        factor
        for factor in _closed_field(snapshot, "factors")
        if factor["__dataclass__"].endswith(
            "DirectionalBondEmissionFactorValueSnapshot"
        )
        and len(_closed_field(factor, "models")) == 2
    )
    _closed_field(shared_factor, "models").reverse()
    _set_closed_field(
        term,
        "successor_snapshot_digest",
        _digest_terms_bounded(
            snapshot,
            budget=WriterEnvelopeWorkBudget(),
            operation="test.branch_transition.successor_snapshot",
        ),
    )


if __name__ == "__main__":
    unittest.main()
