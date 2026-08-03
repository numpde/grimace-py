from __future__ import annotations

import unittest

from grimace._south_star1.errors import SouthStarErrorKind
from tests.south_star1.default_writer_capability_contracts import (
    ACCEPTED_EVIDENCE_PROFILES,
    EXTRACTION_CONTRACT_DEFINITIONS,
    SUPPORT_SURFACE_DEFINITIONS,
    accepted_writer_case,
    frontier_blocked_writer_case,
    preparation_blocked_writer_case,
    validate_default_writer_capability_contracts,
)


class DefaultWriterCapabilityContractsTest(unittest.TestCase):
    def test_registries_validate(self) -> None:
        validate_default_writer_capability_contracts()

    def test_specified_profile_has_two_distinct_contracts(self) -> None:
        specified = [
            item for item in EXTRACTION_CONTRACT_DEFINITIONS
            if item.public_profile == "specified_stereo_closure"
        ]
        self.assertEqual({item.contract_id for item in specified}, {
            "ordinary_specified_stereo", "ordinary_coupled_tetrahedral_stereo",
        })
        self.assertNotEqual(specified[0].options, specified[1].options)

    def test_evidence_profiles_are_typed_and_complete(self) -> None:
        self.assertEqual(set(ACCEPTED_EVIDENCE_PROFILES), {"default", "disconnected"})
        self.assertEqual(
            ACCEPTED_EVIDENCE_PROFILES["disconnected"].offline_relation_families[-1],
            "component_boundary_transition",
        )

    def test_builders_reject_mixed_states(self) -> None:
        with self.assertRaises(ValueError):
            accepted_writer_case(
                name="bad", smiles="CC", extraction_contract_id="ordinary_graph",
                support_surface="acyclic_graph", support_count=1, completion_count=1,
                support_digest="0" * 64,
                qualification_authority="continuation_proof_complete",
            )
        with self.assertRaises(ValueError):
            preparation_blocked_writer_case(
                name="bad", smiles="CC", extraction_contract_id="ordinary_graph",
                support_surface="acyclic_graph", kind="x",
                error_kind=SouthStarErrorKind.INTERNAL_INVARIANT,
                message_contains="x",
            )
        with self.assertRaises(ValueError):
            frontier_blocked_writer_case(
                name="bad", smiles="CC", extraction_contract_id="ordinary_graph",
                support_surface="acyclic_graph", kind="x", operation="y",
            )

    def test_surface_registry_is_unique(self) -> None:
        self.assertEqual(len(SUPPORT_SURFACE_DEFINITIONS), len({x.name for x in SUPPORT_SURFACE_DEFINITIONS}))


if __name__ == "__main__":
    unittest.main()
