from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_branch_transition_artifact import branch_transition_artifact_manifest
from grimace._south_star1.writer_branch_transition_artifact import (
    verify_writer_branch_transition_artifact_consistency,
    writer_branch_transition_artifact_for_support,
)
from grimace._south_star1.writer_support_artifact_checker import (
    artifact_manifest,
    verify_writer_support_artifact_consistency,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_snapshot,
)
from grimace._south_star1.writer_terminalization_artifact import (
    terminalization_artifact_manifest,
    verify_writer_terminalization_artifact_consistency,
    writer_terminalization_artifact_for_support,
)
from grimace._south_star1.writer_envelope_terms import _digest_terms_bounded
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from tests.south_star1.default_writer_capability_ledger import default_writer_capability_case
from tests.south_star1.helpers import cco_facts
from tests.south_star1.qualification_support import facts_for_case
from tests.south_star1.writer_artifact_resealing import reseal_branch_transition_artifact
from tests.south_star1.writer_artifact_resealing import reseal_support_artifact
from tests.south_star1.writer_artifact_resealing import reseal_terminalization_artifact
from tests.south_star1.writer_proof_sources import first_terminal_proof_source
from tests.south_star1.writer_proof_sources import shared_ring_branch_source
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options
from tests.south_star1.writer_test_context import writer_test_context


class WriterArtifactResealingTest(unittest.TestCase):
    def test_migrated_modules_have_no_generic_resealing_helpers(self):
        forbidden = {
            "_object",
            "_object_by_kind",
            "_closed_field",
            "_term_field",
            "_set_closed_field",
            "_set_term_field",
            "_set_nested_closed_field",
            "_redigest_branch_artifact",
            "_redigest_terminal_artifact",
            "_refresh_object_and_artifact_digest",
            "_replace_artifact_ref",
            "_refresh_artifact_digest",
        }
        root = Path(__file__).parent
        modules = (
            "test_writer_branch_transition_artifact.py",
            "test_writer_terminalization_artifact.py",
            "test_writer_support_artifact_fact_verifier.py",
            "test_writer_disconnected_composition.py",
        )
        for name in modules:
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            definitions = {
                node.name
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            }
            self.assertTrue(forbidden.isdisjoint(definitions), name)
            imported_names = set()
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module:
                    imported_names.update(alias.asname or alias.name for alias in node.names)
            forbidden_calls = {
                "artifact_metrics",
                "branch_transition_artifact_manifest",
                "terminalization_artifact_manifest",
                "artifact_manifest",
            }
            for node in ast.walk(tree):
                if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                    self.assertFalse(
                        node.func.id in forbidden_calls and node.func.id in imported_names,
                        f"{name}: {node.func.id}",
                    )
                if isinstance(node, ast.Assign):
                    for target in node.targets:
                        self.assertFalse(
                            isinstance(target, ast.Attribute) and target.attr == "object_id",
                            f"{name}: direct object_id assignment",
                        )

    def _disconnected_branch(self):
        case = default_writer_capability_case("disconnected_cc_oxygen")
        facts = ordinary_molecule_facts_from_smiles(case.smiles, case.extraction_options)
        prepared = prepare_writer_facts(facts)
        options = writer_runtime_options(rooted_at_atom=case.rooted_at_atom)
        pending = [initial_writer_frontier_cursor(prepared, options)]
        seen = set()
        while pending:
            cursor = pending.pop(0)
            if cursor in seen:
                continue
            seen.add(cursor)
            batch = _checked_writer_frontier_branch_supports(
                prepared,
                cursor,
                include_counts=False,
                include_frontier_certificate=True,
                include_count_certificate=False,
            )
            for support in batch.supports:
                if support.emitted_text == ".":
                    from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
                    return writer_branch_transition_artifact_for_support(
                        prepared=prepared,
                        snapshot=capture_writer_frontier_snapshot(
                            prepared=prepared,
                            runtime_options=options,
                            cursor=cursor,
                        ),
                        support=support,
                    )
            pending.extend(
                item.successor_cursor
                for item in batch.text_choice_projection_certificates
            )
        raise AssertionError("missing disconnected component-boundary branch")

    def _branch(self):
        source = shared_ring_branch_source("opening", DirectionMark.FWD)
        return deepcopy(
            writer_branch_transition_artifact_for_support(
                prepared=source.context.prepared,
                snapshot=source.snapshot,
                support=source.support,
            )
        )

    def _terminal(self):
        source = first_terminal_proof_source(cco_facts(), writer_runtime_options())
        return deepcopy(
            writer_terminalization_artifact_for_support(
                prepared=source.context.prepared,
                snapshot=source.snapshot,
                support=source.support,
            )
        )

    def _support(self):
        case = default_writer_capability_case("ethanol")
        context = writer_test_context(
            facts_for_case(case), rooted_at_atom=case.rooted_at_atom
        )
        return deepcopy(
            writer_support_artifact_envelope_for_snapshot(
                prepared=context.prepared,
                snapshot=context.initial_snapshot,
            )
        )

    def _assert_permuted_reseals_to_expected(self, artifact, resealer):
        expected = deepcopy(artifact)
        artifact["objects"] = list(reversed(artifact["objects"]))
        budget = WriterEnvelopeWorkBudget()
        if artifact["schema_name"] == "writer_support_artifact":
            artifact["digest"] = _digest_terms_bounded(
                artifact_manifest(artifact), budget=budget, operation="test.permuted"
            )
        elif artifact["schema_name"] == "writer_terminalization_artifact":
            artifact["digest"] = _digest_terms_bounded(
                terminalization_artifact_manifest(artifact), budget=budget, operation="test.permuted"
            )
        else:
            artifact["digest"] = _digest_terms_bounded(
                branch_transition_artifact_manifest(artifact), budget=budget, operation="test.permuted"
            )
        resealer(artifact)
        self.assertEqual(artifact, expected)

    def test_resealing_restores_production_canonical_object_order(self):
        self._assert_permuted_reseals_to_expected(self._branch(), reseal_branch_transition_artifact)
        self._assert_permuted_reseals_to_expected(self._terminal(), reseal_terminalization_artifact)
        self._assert_permuted_reseals_to_expected(self._support(), reseal_support_artifact)
        self._assert_permuted_reseals_to_expected(self._disconnected_branch(), reseal_branch_transition_artifact)

    def test_resealing_cluster_has_no_test_module_imports(self):
        root = Path(__file__).parent
        names = (
            "writer_artifact_test_support.py",
            "writer_artifact_resealing.py",
            "test_writer_artifact_test_support.py",
            "test_writer_artifact_resealing.py",
        )
        for name in names:
            tree = ast.parse((root / name).read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom):
                    self.assertFalse(
                        node.module and node.module.startswith("tests.south_star1.test_"),
                        name,
                    )
                elif isinstance(node, ast.Import):
                    self.assertFalse(
                        any(alias.name.startswith("tests.south_star1.test_") for alias in node.names),
                        name,
                    )

    def test_untouched_artifacts_are_byte_identical_after_resealing(self):
        branch = self._branch(); expected = deepcopy(branch)
        reseal_branch_transition_artifact(branch)
        self.assertEqual(branch, expected)
        terminal = self._terminal(); expected = deepcopy(terminal)
        reseal_terminalization_artifact(terminal)
        self.assertEqual(terminal, expected)
        support = self._support(); expected = deepcopy(support)
        reseal_support_artifact(support)
        self.assertEqual(support, expected)

    def test_branch_and_terminal_resealing_repair_coherent_references(self):
        branch = self._branch()
        obj = next(item for item in branch["objects"] if item["kind"] == "branch_support")
        obj["object_id"] = "obj:forged"
        reseal_branch_transition_artifact(branch)
        self.assertTrue(verify_writer_branch_transition_artifact_consistency(branch).accepted)
        terminal = self._terminal()
        support = next(item for item in terminal["objects"] if item["kind"] == "terminal_support")
        support["object_id"] = "obj:forged"
        reseal_terminalization_artifact(terminal)
        self.assertTrue(verify_writer_terminalization_artifact_consistency(terminal).accepted)

    def test_support_resealing_propagates_ids_and_is_idempotent(self):
        artifact = self._support()
        artifact["objects"][0]["object_id"] = "obj:forged"
        reseal_support_artifact(artifact)
        self.assertTrue(verify_writer_support_artifact_consistency(artifact).accepted)
        once = deepcopy(artifact)
        reseal_support_artifact(artifact)
        self.assertEqual(artifact, once)

    def test_wrong_schema_and_duplicate_required_kinds_reject(self):
        branch = self._branch(); branch["schema_name"] = "wrong"
        with self.assertRaises(AssertionError):
            reseal_branch_transition_artifact(branch)
        branch = self._branch(); branch["objects"].append(deepcopy(branch["objects"][0]))
        with self.assertRaises(AssertionError):
            reseal_branch_transition_artifact(branch)

    def test_resealing_does_not_repair_payload_internal_digest(self):
        artifact = self._branch()
        branch = next(item for item in artifact["objects"] if item["kind"] == "branch_support")
        evidence = branch["payload"]["local_evidence"]
        original_digest = evidence["digest"]
        evidence["manifest"]["rendered_text"] = "forged rendered text"
        reseal_branch_transition_artifact(artifact)
        self.assertEqual(evidence["digest"], original_digest)

    def test_support_resealing_detects_nonconvergence(self):
        artifact = self._support()
        calls = [0]

        def oscillating_identity(obj, **kwargs):
            calls[0] += 1
            obj["object_id"] = f"obj:{calls[0]}"
            return obj["object_id"]

        with patch(
            "tests.south_star1.writer_artifact_resealing._object_identity",
            side_effect=oscillating_identity,
        ):
            with self.assertRaisesRegex(AssertionError, "did not converge"):
                reseal_support_artifact(artifact)
