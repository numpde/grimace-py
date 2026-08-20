"""Count-free writer terminalization artifact regressions."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
import unittest
from unittest.mock import patch

from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot import WriterDecoderBoundary
from grimace._south_star1.writer_terminalization_artifact import verify_writer_terminalization_artifact_envelope
from grimace._south_star1.writer_terminalization_artifact import writer_terminalization_artifact_for_support
from grimace._south_star1.writer_terminalization_artifact import _source_snapshot_from_terminalization_artifact
from grimace._south_star1.writer_terminalization_artifact_checker import verify_writer_terminalization_artifact_consistency
from grimace._south_star1.writer_terminalization_artifact_fact_verifier import verify_writer_terminalization_artifact_for_facts
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_envelope_terms import _digest_terms_bounded
from grimace._south_star1.writer_envelope_terms import _snapshot_identity_envelope
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_snapshot_closed_terms import writer_frontier_cursor_from_closed_terms
from tests.south_star1.helpers import cco_facts
from tests.south_star1.writer_test_fixtures import directional_non_single_ring_carrier_facts
from tests.south_star1.writer_test_fixtures import directional_ring_carrier_facts
from tests.south_star1.writer_test_fixtures import shared_directional_ring_carrier_facts
from tests.south_star1.writer_test_fixtures import terminal_tetra_center_facts
from tests.south_star1.writer_test_fixtures import terminal_tetra_center_policy
from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import writer_runtime_options
from tests.south_star1.writer_proof_sources import first_terminal_proof_source
from tests.south_star1.writer_artifact_resealing import reseal_terminalization_artifact
from tests.south_star1.writer_artifact_test_support import closed_term_field
from tests.south_star1.writer_artifact_test_support import set_closed_term_field
from tests.south_star1.writer_artifact_test_support import set_nested_closed_term_field
from tests.south_star1.writer_artifact_test_support import unique_artifact_object_by_kind


class WriterTerminalizationArtifactTest(unittest.TestCase):
    def test_positive_terminalization_matrix(self) -> None:
        cases = (
            ("ordinary", cco_facts(), writer_runtime_options(), None, "noop", ()),
            (
                "tetra",
                terminal_tetra_center_facts(),
                writer_runtime_options(rooted_at_atom=0),
                terminal_tetra_center_policy(),
                "tetra_local_order_factor_closure",
                ("tetrahedral local-order factor closure",),
            ),
            ("simple_ring", directional_ring_carrier_facts(), writer_runtime_options(rooted_at_atom=0), None, "noop", ()),
            ("shared_ring", shared_directional_ring_carrier_facts(), writer_runtime_options(rooted_at_atom=1), None, "noop", ()),
            ("non_single_ring", directional_non_single_ring_carrier_facts(), writer_runtime_options(rooted_at_atom=0), None, "noop", ()),
        )
        for name, facts, options, policy, mode, operations in cases:
            with self.subTest(name=name):
                prepared, artifact = _terminal_artifact(facts, options, policy)
                structural = verify_writer_terminalization_artifact_consistency(artifact)
                live = verify_writer_terminalization_artifact_envelope(prepared=prepared, artifact=artifact)
                checked = verify_writer_terminalization_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=artifact,
                    policy=policy,
                )
                self.assertTrue(structural.accepted, structural.reason)
                self.assertTrue(live.accepted, live.reason)
                self.assertTrue(checked.accepted, checked.reason)
                self.assertEqual(len(artifact["objects"]), 3)
                self.assertEqual(checked.terminalization_mode, mode)
                self.assertEqual(checked.semantically_replayed_operations, operations)
                self.assertEqual(checked.unchecked_obligation_families, ())

    def test_build_and_live_check_do_not_materialize_counts_or_support(self) -> None:
        facts = cco_facts()
        options = writer_runtime_options()
        source = first_terminal_proof_source(facts, options)
        prepared, snapshot, support = source.context.prepared, source.snapshot, source.support
        patches = (
            patch("grimace._south_star1.writer_frontier_count_envelope.writer_frontier_count_envelope_for_snapshot", side_effect=AssertionError("count path")),
            patch("grimace._south_star1.writer_count_dag_envelope.writer_count_certificate_dag_envelope_for_product", side_effect=AssertionError("dag path")),
            patch("grimace._south_star1.writer_support_string_envelope._iter_writer_snapshot_certified_support_strings", side_effect=AssertionError("support path")),
        )
        with patches[0], patches[1], patches[2]:
            artifact = writer_terminalization_artifact_for_support(
                prepared=prepared, snapshot=snapshot, support=support
            )
            checked = verify_writer_terminalization_artifact_envelope(
                prepared=prepared, artifact=artifact
            )
        self.assertTrue(checked.accepted, checked.reason)

    def test_structural_object_and_schema_regressions(self) -> None:
        _prepared, artifact = _terminal_artifact(cco_facts(), writer_runtime_options(), None)
        old = deepcopy(artifact)
        old["schema_version"] = 0
        self.assertIn(
            "unknown_schema_version",
            verify_writer_terminalization_artifact_consistency(old).reason,
        )
        extra = deepcopy(artifact)
        extra["objects"].append({
            "object_id": "obj:count",
            "kind": "count_dag",
            "payload": {},
            "digest": "count",
        })
        self.assertIn(
            "object_count_mismatch",
            verify_writer_terminalization_artifact_consistency(extra).reason,
        )
        duplicate = deepcopy(artifact)
        duplicate["objects"][2] = deepcopy(duplicate["objects"][1])
        self.assertIn(
            "duplicate_object_id",
            verify_writer_terminalization_artifact_consistency(duplicate).reason,
        )
        unknown = deepcopy(artifact)
        unknown["objects"][0]["kind"] = "invented_terminal_object"
        self.assertIn(
            "count_or_unknown_object_kind",
            verify_writer_terminalization_artifact_consistency(unknown).reason,
        )
        unreachable = deepcopy(artifact)
        unreachable["roots"]["terminal_support_ref"] = unreachable["roots"][
            "source_ref"
        ]
        self.assertIn(
            "terminal_support_root_mismatch",
            verify_writer_terminalization_artifact_consistency(unreachable).reason,
        )

    def test_coherent_terminal_semantic_forgery_matrix(self) -> None:
        facts = cco_facts()
        options = writer_runtime_options()
        cases = (
            ("active_atom", lambda support: set_closed_term_field(support["terminalization_term"], "active_atom", 999), "terminalization_state_identity_mismatch"),
            ("graph_status", lambda support: set_nested_closed_term_field(support["terminalization_term"], "graph_completion_status", "complete", value=False), "terminal_graph_completion_status_mismatch"),
            ("graph_digest", lambda support: set_closed_term_field(support["terminalization_term"], "graph_obligation_work_digests", ["0" * 64]), "terminal_graph_work_digest_mismatch"),
            ("graph_operation", lambda support: support["obligation_manifests"]["terminal_graph_obligation_work"][0].__setitem__("operation", "forged graph work"), "terminal_graph_manifest_mismatch"),
            ("lifecycle_event", lambda support: support["obligation_manifests"]["terminal_stereo_lifecycle"][0].__setitem__("lifecycle_event_kind", "ring_endpoint_paired"), "terminal_lifecycle_provenance_mismatch"),
            ("lifecycle_outcome", lambda support: support["obligation_manifests"]["terminal_stereo_lifecycle"][0].__setitem__("lifecycle_outcome_kind", "residual_restricted"), "terminal_lifecycle_identity_mismatch"),
            ("residual_changed", lambda support: support["obligation_manifests"]["terminal_stereo_lifecycle"][0].__setitem__("residual_snapshot_changed", True), "terminal_lifecycle_identity_mismatch"),
            ("local_orders", lambda support: support["obligation_manifests"]["terminal_stereo_lifecycle"][0].__setitem__("local_orders_changed", False), "terminal_lifecycle_identity_mismatch"),
            ("support_key", lambda support: support.__setitem__("terminal_support_key_digest", "0" * 64), "terminal_support_terminal_support_key_digest_mismatch"),
            ("ordinal", lambda support: support.__setitem__("terminal_ordinal", support["terminal_ordinal"] + 1), "terminal_support_terminal_support_key_digest_mismatch"),
            ("capabilities_digest", lambda support: support.__setitem__("terminal_execution_capabilities_digest", "0" * 64), "terminal_support_terminal_execution_capabilities_digest_mismatch"),
            ("residual_tuple_digest", lambda support: support.__setitem__("terminal_residual_work_evidence_digest", "0" * 64), "terminal_false_noop_residual_work"),
            ("lifecycle_tuple_digest", lambda support: support.__setitem__("terminal_stereo_lifecycle_evidence_digest", "0" * 64), "terminal_support_terminal_stereo_lifecycle_evidence_digest_mismatch"),
            ("graph_tuple_digest", lambda support: support.__setitem__("graph_obligation_work_evidence_digest", "0" * 64), "terminal_support_graph_obligation_work_evidence_digest_mismatch"),
            ("certificate_order", lambda support: support["terminal_certificate_digests"].reverse(), "terminal_support_terminal_certificate_digests_mismatch"),
            ("certificate_substitution", lambda support: support["terminal_certificate_digests"].__setitem__(0, support["terminal_certificate_digests"][1]), "terminal_support_terminal_certificate_digests_mismatch"),
            ("lifecycle_omitted", lambda support: (support["obligation_manifests"]["terminal_stereo_lifecycle"].clear(), support["obligation_summary"].__setitem__("terminal_stereo_lifecycle_count", 0)), "terminal_lifecycle_digest_mismatch"),
        )
        for name, mutate, reason in cases:
            with self.subTest(name=name):
                _prepared, original = _terminal_artifact(facts, options, None)
                artifact = deepcopy(original)
                support = unique_artifact_object_by_kind(artifact, "terminal_support")["payload"]
                mutate(support)
                reseal_terminalization_artifact(artifact)
                structural = verify_writer_terminalization_artifact_consistency(artifact)
                checked = verify_writer_terminalization_artifact_for_facts(
                    facts=facts, runtime_options=options, artifact=artifact
                )
                self.assertTrue(structural.accepted, structural.reason)
                self.assertFalse(checked.accepted)
                self.assertIn(reason, checked.reason)

    def test_coherent_tetra_transition_forgery_matrix(self) -> None:
        facts = terminal_tetra_center_facts()
        policy = terminal_tetra_center_policy()
        options = writer_runtime_options(rooted_at_atom=0)
        cases = (
            ("wrong_site", lambda term: set_closed_term_field(term, "site", 999), "terminal_tetra"),
            ("wrong_atom", lambda term: set_closed_term_field(term, "atom", 0), "terminal_transition_state_anchor_mismatch"),
            ("wrong_reference", lambda term: closed_term_field(term, "reference_order").reverse(), "terminal_tetra_reference_order_mismatch"),
            ("wrong_local_order", lambda term: closed_term_field(term, "local_order").reverse(), "terminal_tetra_local_order_mismatch"),
            ("wrong_constraint", lambda term: set_closed_term_field(term, "constraint_value", {"__enum__": "grimace._south_star1.residual_constraints.TetraLocalParity", "value": "odd"}), "terminal_tetra_restriction_mismatch"),
            ("wrong_discharge", lambda term: closed_term_field(term, "discharged_factor_keys").clear(), "terminal_tetra_restriction_mismatch"),
            ("wrong_projection", lambda term: closed_term_field(term, "projected_variables").clear(), "terminal_tetra_restriction_mismatch"),
            ("missing_link", lambda term: None, "terminal_tetra_residual_manifest_mismatch"),
            ("extra_capability", lambda term: None, "terminal_lifecycle_provenance_mismatch"),
        )
        for name, mutate, reason in cases:
            with self.subTest(name=name):
                _prepared, original = _terminal_artifact(facts, options, policy)
                artifact = deepcopy(original)
                support = unique_artifact_object_by_kind(artifact, "terminal_support")["payload"]
                manifest = support["obligation_manifests"]["terminal_residual_work"][0]
                if name == "missing_link":
                    manifest["linked_lifecycle_digests"].clear()
                elif name == "extra_capability":
                    closed_term_field(
                        support["terminalization_term"],
                        "terminal_execution_capabilities",
                    ).append("tree_child_entry")
                else:
                    mutate(manifest["transition_term"])
                manifest["transition_digest"] = _digest_terms_bounded(
                    manifest["transition_term"],
                    budget=WriterEnvelopeWorkBudget(),
                    operation="test.terminalization.transition",
                )
                reseal_terminalization_artifact(artifact)
                structural = verify_writer_terminalization_artifact_consistency(artifact)
                checked = verify_writer_terminalization_artifact_for_facts(
                    facts=facts,
                    runtime_options=options,
                    artifact=artifact,
                    policy=policy,
                )
                self.assertTrue(structural.accepted, structural.reason)
                self.assertFalse(checked.accepted)
                self.assertIn(reason, checked.reason)

    def test_terminalization_term_shape_regressions(self) -> None:
        _prepared, original = _terminal_artifact(cco_facts(), writer_runtime_options(), None)
        cases = (
            ("extra", lambda term: term["fields"].append(["extra", 1]), "terminalization_term_fields_mismatch"),
            ("missing", lambda term: term["fields"].pop(), "terminalization_term_fields_mismatch"),
            ("mode", lambda term: set_closed_term_field(term, "stereo_mode", "invented"), "terminalization_term_stereo_mode_mismatch"),
        )
        for name, mutate, reason in cases:
            with self.subTest(name=name):
                artifact = deepcopy(original)
                support = unique_artifact_object_by_kind(artifact, "terminal_support")["payload"]
                mutate(support["terminalization_term"])
                support["terminalization_term_digest"] = _identity_digest(
                    support["terminalization_term"]
                )
                reseal_terminalization_artifact(artifact)
                checked = verify_writer_terminalization_artifact_consistency(artifact)
                self.assertFalse(checked.accepted)
                self.assertIn(reason, checked.reason)

    def test_coherent_writer_state_forgery_matrix(self) -> None:
        facts = cco_facts()
        options = writer_runtime_options()
        cases = (
            ("component_index", _forge_component_index, "terminal_graph_completion_mismatch"),
            ("component_roots", _forge_component_roots, "terminal_component_roots_mismatch"),
            ("active_atom", _forge_active_atom, "terminal_graph_completion_mismatch"),
            ("pending_entry", _forge_pending_entry, "terminal_graph_completion_mismatch"),
            ("branch_frame", _forge_branch_frame, "terminal_graph_completion_mismatch"),
            ("final_policy", _forge_final_policy, "terminal_finalized_non_stereo_state_mismatch"),
        )
        for name, mutate, reason in cases:
            with self.subTest(name=name):
                prepared, original = _terminal_artifact(facts, options, None)
                artifact = deepcopy(original)
                mutate(artifact)
                _refresh_state_anchors(
                    artifact=artifact,
                    prepared=prepared,
                    options=options,
                )
                structural = verify_writer_terminalization_artifact_consistency(artifact)
                checked = verify_writer_terminalization_artifact_for_facts(
                    facts=facts, runtime_options=options, artifact=artifact
                )
                self.assertTrue(structural.accepted, structural.reason)
                self.assertFalse(checked.accepted)
                self.assertIn(reason, checked.reason)

    def test_coherent_closed_ring_partition_forgery_matrix(self) -> None:
        facts = directional_ring_carrier_facts()
        options = writer_runtime_options(rooted_at_atom=0)
        cases = (
            ("duplicate", _forge_duplicate_closure, "terminal_graph_completion_mismatch"),
            ("wrong_endpoint", _forge_closure_endpoint, "terminal_closed_closure_endpoint_mismatch"),
            ("unknown_bond", _forge_closure_bond, "terminal_closed_closure_unknown_bond"),
            ("written_and_closed", _forge_written_closed_overlap, "terminal_graph_completion_mismatch"),
            ("wrong_label", _forge_closure_label, "terminal_closed_closure_label_mismatch"),
        )
        for name, mutate, reason in cases:
            with self.subTest(name=name):
                prepared, original = _terminal_artifact(facts, options, None)
                artifact = deepcopy(original)
                mutate(artifact)
                _refresh_state_anchors(
                    artifact=artifact, prepared=prepared, options=options
                )
                structural = verify_writer_terminalization_artifact_consistency(artifact)
                checked = verify_writer_terminalization_artifact_for_facts(
                    facts=facts, runtime_options=options, artifact=artifact
                )
                self.assertTrue(structural.accepted, structural.reason)
                self.assertFalse(checked.accepted)
                self.assertIn(reason, checked.reason)

    def test_coherent_false_noop_residual_forgeries(self) -> None:
        facts = cco_facts()
        options = writer_runtime_options()
        tetra_facts = terminal_tetra_center_facts()
        _prepared, tetra_artifact = _terminal_artifact(
            tetra_facts,
            writer_runtime_options(rooted_at_atom=0),
            terminal_tetra_center_policy(),
        )
        tetra_support = unique_artifact_object_by_kind(tetra_artifact, "terminal_support")["payload"]
        residual = deepcopy(
            closed_term_field(
                tetra_support["obligation_manifests"]["terminal_residual_work"][0][
                    "transition_term"
                ],
                "source_snapshot",
            )
        )
        for role in ("source", "finalized"):
            with self.subTest(role=role):
                prepared, original = _terminal_artifact(facts, options, None)
                artifact = deepcopy(original)
                state = _projection_state_term(artifact, role)
                set_nested_closed_term_field(
                    state, "stereo_state", "residual_snapshot", value=deepcopy(residual)
                )
                _refresh_state_anchors(
                    artifact=artifact, prepared=prepared, options=options
                )
                structural = verify_writer_terminalization_artifact_consistency(artifact)
                checked = verify_writer_terminalization_artifact_for_facts(
                    facts=facts, runtime_options=options, artifact=artifact
                )
                self.assertTrue(structural.accepted, structural.reason)
                self.assertFalse(checked.accepted)
                self.assertIn("terminal_false_noop", checked.reason)


@lru_cache(maxsize=None)
def _terminal_artifact(facts, options, policy):
    source = first_terminal_proof_source(facts, options, policy=policy)
    prepared, snapshot, support = source.context.prepared, source.snapshot, source.support
    return prepared, writer_terminalization_artifact_for_support(
        prepared=prepared, snapshot=snapshot, support=support
    )




def _projection_state_term(artifact, role):
    projection = unique_artifact_object_by_kind(artifact, "terminal_projection")["payload"]
    cursor = projection[f"{role}_cursor"]["terms"]
    return closed_term_field(cursor, "weighted_states")[0][0]


def _forge_component_index(artifact):
    for role in ("source", "finalized"):
        state = _projection_state_term(artifact, role)
        set_nested_closed_term_field(state, "component_cursor", "component_index", value=-1)


def _forge_component_roots(artifact):
    for role in ("source", "finalized"):
        state = _projection_state_term(artifact, role)
        set_nested_closed_term_field(state, "component_cursor", "component_roots", value=[])


def _forge_active_atom(artifact):
    for role in ("source", "finalized"):
        state = _projection_state_term(artifact, role)
        set_nested_closed_term_field(state, "active", "atom", value=999)
    support = unique_artifact_object_by_kind(artifact, "terminal_support")["payload"]
    set_closed_term_field(support["terminalization_term"], "active_atom", 999)


def _forge_pending_entry(artifact):
    entry = {
        "__dataclass__": "grimace._south_star1.writer_state.PendingWriterEntry",
        "fields": [
            ["parent", 0], ["child", 1], ["bond", 0], ["branch", False],
            ["phase", {"__enum__": "grimace._south_star1.writer_state.PendingEntryPhase", "value": "needs_bond_or_atom"}],
        ],
    }
    for role in ("source", "finalized"):
        state = _projection_state_term(artifact, role)
        set_nested_closed_term_field(state, "obligations", "pending_entry", value=deepcopy(entry))


def _forge_branch_frame(artifact):
    for role in ("source", "finalized"):
        state = _projection_state_term(artifact, role)
        active = deepcopy(closed_term_field(state, "active"))
        set_closed_term_field(
            state,
            "branch_stack",
            [{"__dataclass__": "grimace._south_star1.writer_state.WriterBranchFrame", "fields": [["return_atom", active]]}],
        )


def _forge_final_policy(artifact):
    state = _projection_state_term(artifact, "finalized")
    policy = closed_term_field(state, "policy_state")
    set_closed_term_field(policy, "atom_text", [[999, "X"]])


def _closed_closures(artifact, role):
    state = _projection_state_term(artifact, role)
    return closed_term_field(closed_term_field(state, "ring_state"), "closed_closures")


def _forge_duplicate_closure(artifact):
    for role in ("source", "finalized"):
        closures = _closed_closures(artifact, role)
        closures.append(deepcopy(closures[0]))


def _forge_closure_endpoint(artifact):
    for role in ("source", "finalized"):
        set_closed_term_field(_closed_closures(artifact, role)[0], "first_atom", 999)


def _forge_closure_bond(artifact):
    for role in ("source", "finalized"):
        set_closed_term_field(_closed_closures(artifact, role)[0], "bond", 999)


def _forge_written_closed_overlap(artifact):
    for role in ("source", "finalized"):
        state = _projection_state_term(artifact, role)
        written = closed_term_field(state, "written_bonds")
        written.append(closed_term_field(_closed_closures(artifact, role)[0], "bond"))
        written.sort()


def _forge_closure_label(artifact):
    for role in ("source", "finalized"):
        closure = _closed_closures(artifact, role)[0]
        set_nested_closed_term_field(closure, "label", "text", value="%1")


def _refresh_state_anchors(*, artifact, prepared, options) -> None:
    del options
    original_snapshot = _source_snapshot_from_terminalization_artifact(
        prepared=prepared,
        artifact=artifact,
        budget=WriterEnvelopeWorkBudget(),
    )
    projection = unique_artifact_object_by_kind(artifact, "terminal_projection")["payload"]
    source_cursor = writer_frontier_cursor_from_closed_terms(
        projection["source_cursor"]["terms"]
    )
    snapshot = replace(original_snapshot, cursor=source_cursor)
    source_payload = _snapshot_identity_envelope(snapshot)
    artifact["source_snapshot"] = source_payload
    unique_artifact_object_by_kind(artifact, "source_snapshot")["payload"] = deepcopy(source_payload)
    projection["source_cursor"] = deepcopy(source_payload["cursor"])
    finalized_cursor = projection["finalized_cursor"]
    finalized_cursor["digest"] = _digest_terms_bounded(
        finalized_cursor["terms"],
        budget=WriterEnvelopeWorkBudget(),
        operation="test.terminalization.finalized_cursor",
    )
    source_state = _projection_state_term(artifact, "source")
    finalized_state = _projection_state_term(artifact, "finalized")
    source_digest = _digest_terms_bounded(
        source_state,
        budget=WriterEnvelopeWorkBudget(),
        operation="test.terminalization.source_state",
    )
    finalized_digest = _digest_terms_bounded(
        finalized_state,
        budget=WriterEnvelopeWorkBudget(),
        operation="test.terminalization.finalized_state",
    )
    support = unique_artifact_object_by_kind(artifact, "terminal_support")["payload"]
    support["source_state_digest"] = source_digest
    support["finalized_state_digest"] = finalized_digest
    set_closed_term_field(support["terminalization_term"], "source_state_digest", source_digest)
    set_closed_term_field(support["terminalization_term"], "finalized_state_digest", finalized_digest)
    source_residual = closed_term_field(
        closed_term_field(source_state, "stereo_state"), "residual_snapshot"
    )
    finalized_residual = closed_term_field(
        closed_term_field(finalized_state, "stereo_state"), "residual_snapshot"
    )
    source_residual_digest = _digest_terms_bounded(
        source_residual,
        budget=WriterEnvelopeWorkBudget(),
        operation="test.terminalization.source_residual",
    )
    finalized_residual_digest = _digest_terms_bounded(
        finalized_residual,
        budget=WriterEnvelopeWorkBudget(),
        operation="test.terminalization.finalized_residual",
    )
    set_closed_term_field(
        support["terminalization_term"],
        "source_residual_snapshot_digest",
        source_residual_digest,
    )
    set_closed_term_field(
        support["terminalization_term"],
        "finalized_residual_snapshot_digest",
        finalized_residual_digest,
    )
    for items in support["obligation_manifests"].values():
        for item in items:
            item["source_digest"] = source_digest
            item["successor_digest"] = finalized_digest
            item["is_noop"] = source_digest == finalized_digest
    reseal_terminalization_artifact(artifact)


if __name__ == "__main__":
    unittest.main()
