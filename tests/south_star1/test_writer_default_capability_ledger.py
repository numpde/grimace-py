"""Machine-checked contract for the default ordinary writer capability ledger."""

from __future__ import annotations

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
from tests.south_star1.default_writer_capability_ledger import (
    ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.default_writer_capability_ledger import (
    BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.default_writer_capability_ledger import (
    DEFAULT_OFFLINE_OBJECT_KINDS,
)
from tests.south_star1.default_writer_capability_ledger import (
    DEFAULT_OFFLINE_RELATION_FAMILIES,
)
from tests.south_star1.default_writer_capability_ledger import (
    DEFAULT_OFFLINE_UNCHECKED_OBJECT_KINDS,
)
from tests.south_star1.default_writer_capability_ledger import (
    DEFAULT_OFFLINE_UNCHECKED_OBLIGATION_FAMILIES,
)
from tests.south_star1.default_writer_capability_ledger import (
    DEFAULT_WRITER_CAPABILITY_CASES,
)


class WriterDefaultCapabilityLedgerTest(unittest.TestCase):
    def test_ledger_names_and_support_surfaces_are_unique(self) -> None:
        names = [item.name for item in DEFAULT_WRITER_CAPABILITY_CASES]
        surfaces = [item.support_surface for item in DEFAULT_WRITER_CAPABILITY_CASES]

        self.assertEqual(len(names), len(set(names)))
        self.assertEqual(
            set(surfaces),
            {
                "acyclic_graph",
                "branched_graph",
                "single_ring_closure",
                "non_single_ring_closure_double",
                "non_single_ring_closure_triple",
                "branched_ring",
                "simple_bracket_charge",
                "simple_isotope_bracket_atom",
                "unsupported_charged_isotope",
                "unsupported_charged_oxygen_isotope",
                "unsupported_negative_nitrogen_charge",
                "unsupported_positive_oxygen_charge",
                "unsupported_potential_directional_non_neighbor",
            },
        )

    def test_accepted_entries_declare_counts_and_graph_extraction(self) -> None:
        self.assertTrue(ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES)
        for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES:
            with self.subTest(case=case.name):
                self.assertEqual(case.expected, "accepted")
                self.assertEqual(case.extraction_profile, "graph_no_potential_sites")
                self.assertIsNotNone(case.expected_support_count)
                self.assertIsNotNone(case.expected_completion_count)
                self.assertIsNone(case.blocker_phase)
                self.assertIsNone(case.blocker_kind)
                self.assertTrue(case.expected_structural_artifact)
                self.assertTrue(case.expected_live_artifact_verifier)
                self.assertTrue(case.expected_facts_bound_verifier)
                self.assertTrue(case.expected_offline_replay_complete)
                self.assertEqual(
                    case.expected_offline_object_kinds,
                    DEFAULT_OFFLINE_OBJECT_KINDS,
                )
                self.assertEqual(
                    case.expected_offline_unchecked_object_kinds,
                    DEFAULT_OFFLINE_UNCHECKED_OBJECT_KINDS,
                )
                self.assertEqual(
                    case.expected_offline_relation_families,
                    DEFAULT_OFFLINE_RELATION_FAMILIES,
                )
                self.assertEqual(
                    case.expected_offline_unchecked_obligation_families,
                    DEFAULT_OFFLINE_UNCHECKED_OBLIGATION_FAMILIES,
                )

    def test_blocked_entries_declare_typed_blockers(self) -> None:
        self.assertTrue(BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES)
        for case in BLOCKED_DEFAULT_WRITER_CAPABILITY_CASES:
            with self.subTest(case=case.name):
                self.assertEqual(case.expected, "blocked")
                self.assertIsNotNone(case.blocker_phase)
                self.assertIsNotNone(case.blocker_kind)
                self.assertFalse(case.expected_structural_artifact)
                self.assertFalse(case.expected_live_artifact_verifier)
                self.assertFalse(case.expected_facts_bound_verifier)
                self.assertFalse(case.expected_offline_replay_complete)
                self.assertEqual(case.expected_offline_object_kinds, ())
                self.assertEqual(case.expected_offline_unchecked_object_kinds, ())
                self.assertEqual(case.expected_offline_relation_families, ())
                self.assertEqual(
                    case.expected_offline_unchecked_obligation_families,
                    (),
                )
                if case.blocker_phase == "frontier":
                    self.assertIsNotNone(case.blocker_operation)
                    self.assertIsNone(case.blocker_message_contains)
                if case.blocker_phase == "preparation":
                    self.assertIsNotNone(case.blocker_error_kind)
                    self.assertIsNotNone(case.blocker_message_contains)

    def test_cyclopropene_default_artifact_binds_default_joint_policy(
        self,
    ) -> None:
        case = next(
            item
            for item in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
            if item.name == "cyclopropene_double_closure"
        )
        facts = ordinary_molecule_facts_from_smiles(
            case.smiles,
            case.extraction_options,
        )
        prepared = prepare_south_star_mol_from_facts(
            facts,
            writer_surface=SouthStarWriterSurface(),
        )
        options = _writer_options()
        snapshot = capture_writer_frontier_snapshot(
            prepared=prepared,
            runtime_options=options,
            cursor=initial_writer_frontier_cursor(prepared, options),
        )
        artifact = writer_support_artifact_envelope_for_snapshot(
            prepared=prepared,
            snapshot=snapshot,
        )

        verification = verify_writer_support_artifact_for_facts(
            facts=facts,
            runtime_options=options,
            artifact=artifact,
        )

        self.assertTrue(verification.accepted, verification.reason)
        with self.assertRaisesRegex(SouthStarError, "non-single ring closures"):
            ordinary_policy_for_facts(
                facts,
                OrdinaryPolicyOptions(non_single_ring_closures="unsupported"),
            )


def _writer_options() -> SouthStarRuntimeOptions:
    return SouthStarRuntimeOptions(
        rooted_at_atom=0,
        serialization_language=SerializationLanguageMode.WRITER_SHAPED,
    )


if __name__ == "__main__":
    unittest.main()
