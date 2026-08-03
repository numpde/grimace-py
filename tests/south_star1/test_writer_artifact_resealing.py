from __future__ import annotations

import ast
from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.writer_branch_transition_artifact import (
    verify_writer_branch_transition_artifact_consistency,
    writer_branch_transition_artifact_for_support,
)
from grimace._south_star1.writer_support_artifact_checker import (
    verify_writer_support_artifact_consistency,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_snapshot,
)
from grimace._south_star1.writer_terminalization_artifact import (
    verify_writer_terminalization_artifact_consistency,
)
from tests.south_star1.helpers import cco_facts
from tests.south_star1.test_writer_terminalization_artifact import _terminal_artifact
from tests.south_star1.test_writer_support_artifact_fact_verifier import _rdkit_facts
from tests.south_star1.writer_artifact_resealing import reseal_branch_transition_artifact
from tests.south_star1.writer_artifact_resealing import reseal_support_artifact
from tests.south_star1.writer_artifact_resealing import reseal_terminalization_artifact
from tests.south_star1.writer_proof_sources import shared_ring_branch_source
from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options


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
        return deepcopy(_terminal_artifact(cco_facts(), writer_runtime_options(), None)[1])

    def _support(self):
        facts = _rdkit_facts("CCO")
        prepared = prepare_writer_facts(facts)
        options = writer_runtime_options()
        return deepcopy(
            writer_support_artifact_envelope_for_snapshot(
                prepared=prepared,
                snapshot=initial_writer_snapshot(prepared, options),
            )
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
        manifest = branch["payload"]["obligation_manifests"]["graph_obligation_work"][0]
        original_digest = manifest["digest"]
        manifest["operation"] = "forged graph obligation context"
        reseal_branch_transition_artifact(artifact)
        self.assertEqual(manifest["digest"], original_digest)

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
