"""Behavioral checks for the typed default writer capability ledger."""

from __future__ import annotations

import ast
from pathlib import Path
import unittest

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_snapshot,
)
from grimace._south_star1.writer_support_artifact_fact_verifier import (
    verify_writer_support_artifact_for_facts,
)
from tests.south_star1.default_writer_capability_contracts import (
    SUPPORT_SURFACE_DEFINITIONS,
    validate_default_writer_capability_contracts,
)
from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
    BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES,
    DEFAULT_WRITER_CAPABILITY_CASES,
    default_writer_cases_for_rdkit_audit,
    validate_default_writer_capability_ledger,
)


class WriterDefaultCapabilityLedgerTest(unittest.TestCase):
    def test_typed_registries_and_ledger_validate(self) -> None:
        validate_default_writer_capability_contracts()
        validate_default_writer_capability_ledger()
        self.assertEqual(
            {case.support_surface for case in DEFAULT_WRITER_CAPABILITY_CASES},
            {surface.name for surface in SUPPORT_SURFACE_DEFINITIONS},
        )

    def test_audit_selectors_are_deterministic_and_ledger_ordered(self) -> None:
        for family in ("aromatic", "bracket", "disconnected", "stereo"):
            selected = default_writer_cases_for_rdkit_audit(family)
            self.assertEqual(
                selected,
                tuple(case for case in DEFAULT_WRITER_CAPABILITY_CASES if family in case.rdkit_audit_families),
            )
            self.assertEqual(selected, default_writer_cases_for_rdkit_audit(family))

    def test_accepted_and_blocked_cases_expose_only_valid_contracts(self) -> None:
        self.assertTrue(ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES)
        self.assertTrue(BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES)
        for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES:
            self.assertEqual(case.expected, "accepted")
            self.assertIsNotNone(case.expected_support_count)
            self.assertIsNone(case.blocker_phase)
            self.assertTrue(case.expected_structural_artifact)
        for case in BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES:
            self.assertEqual(case.expected, "blocked")
            self.assertIsNone(case.qualification_authority)
            self.assertIsNotNone(case.blocker_phase)
            self.assertIsNone(case.expected_support_count)
            self.assertFalse(case.expected_structural_artifact)

    def test_case_declarations_use_only_constrained_builders(self) -> None:
        tree = ast.parse(
            (Path(__file__).parent / "default_writer_capability_ledger.py").read_text()
        )
        assignment = next(
            node for node in tree.body
            if isinstance(node, ast.Assign)
            and any(isinstance(target, ast.Name) and target.id == "DEFAULT_WRITER_CAPABILITY_CASES" for target in node.targets)
        )
        self.assertIsInstance(assignment.value, ast.Tuple)
        allowed = {
            "accepted_writer_case",
            "preparation_blocked_writer_case",
            "frontier_blocked_writer_case",
        }
        for item in assignment.value.elts:
            self.assertIsInstance(item, ast.Call)
            self.assertIsInstance(item.func, ast.Name)
            self.assertIn(item.func.id, allowed)
            self.assertNotIn("extraction_options", {keyword.arg for keyword in item.keywords})

    def test_audit_pinning_is_derived_from_audit_families(self) -> None:
        for case in DEFAULT_WRITER_CAPABILITY_CASES:
            self.assertEqual(case.expected_rdkit_audit_version_pinned, bool(case.rdkit_audit_families))

        remote_a = next(case for case in DEFAULT_WRITER_CAPABILITY_CASES if case.name == "remote_coupled_tetrahedral_a")
        remote_b = next(case for case in DEFAULT_WRITER_CAPABILITY_CASES if case.name == "remote_coupled_tetrahedral_b")
        self.assertIs(remote_a.expectation.continuation, remote_b.expectation.continuation)

    def test_cyclopropene_default_artifact_binds_default_joint_policy(self) -> None:
        case = next(item for item in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES if item.name == "cyclopropene_double_closure")
        facts = ordinary_molecule_facts_from_smiles(case.smiles, case.extraction_options)
        prepared = prepare_south_star_mol_from_facts(facts, writer_surface=SouthStarWriterSurface())
        options = SouthStarRuntimeOptions(
            rooted_at_atom=0,
            serialization_language=SerializationLanguageMode.WRITER_SHAPED,
        )
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=initial_writer_frontier_cursor(prepared, options),
        )
        artifact = writer_support_artifact_envelope_for_snapshot(prepared=prepared, snapshot=snapshot)
        verification = verify_writer_support_artifact_for_facts(
            facts=facts, runtime_options=options, artifact=artifact
        )
        self.assertTrue(verification.accepted, verification.reason)
        with self.assertRaisesRegex(SouthStarError, "non-single ring closures"):
            ordinary_policy_for_facts(
                facts,
                OrdinaryPolicyOptions(non_single_ring_closures="unsupported"),
            )


if __name__ == "__main__":
    unittest.main()
