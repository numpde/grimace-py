"""Version-keyed RDKit audit fixture checks for durable bracket ledger cases."""

from __future__ import annotations

import unittest

from rdkit import rdBase

from grimace._south_star1.fact_isomorphism import facts_are_isomorphic
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from tests.helpers.rdkit_south_star_bracket_audit import (
    load_pinned_south_star_bracket_audit_cases,
)
from tests.south_star1.default_writer_capability_ledger import (
    DEFAULT_WRITER_CAPABILITY_CASES,
)
from tests.south_star1.qualification_support import accepted_case_result
from tests.south_star1.qualification_support import blocked_case_result
from tests.south_star1.qualification_support import support_image_for_case


DURABLE_BRACKET_SUPPORT_SURFACES = frozenset(
    {
        "simple_bracket_charge",
        "simple_isotope_bracket_atom",
        "unsupported_charged_isotope",
        "unsupported_charged_oxygen_isotope",
        "unsupported_negative_nitrogen_charge",
        "unsupported_positive_oxygen_charge",
    }
)


class WriterDefaultBracketAuditFixtureTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_cases = load_pinned_south_star_bracket_audit_cases(
            rdBase.rdkitVersion,
        )
        cls.ledger_by_name = {
            case.name: case
            for case in DEFAULT_WRITER_CAPABILITY_CASES
            if case.support_surface in DURABLE_BRACKET_SUPPORT_SURFACES
        }

    def test_fixture_entries_cover_durable_bracket_ledger_cases(self) -> None:
        fixture_by_name = {case.name: case for case in self.fixture_cases}

        self.assertEqual(
            {
                case.name
                for case in self.ledger_by_name.values()
                if case.expected == "accepted"
            },
            {
                name
                for name, case in fixture_by_name.items()
                if case.expected == "accepted"
            },
        )
        self.assertEqual(
            {
                case.name
                for case in self.ledger_by_name.values()
                if case.expected == "blocked"
            },
            {
                name
                for name, case in fixture_by_name.items()
                if case.expected == "blocked"
            },
        )

    def test_ledger_and_fixture_case_names_do_not_drift(self) -> None:
        fixture_names = {case.name for case in self.fixture_cases}
        self.assertEqual(fixture_names, set(self.ledger_by_name))

        for fixture_case in self.fixture_cases:
            with self.subTest(case=fixture_case.name):
                ledger_case = self.ledger_by_name[fixture_case.name]
                self.assertEqual(fixture_case.smiles, ledger_case.smiles)
                self.assertEqual(
                    fixture_case.extraction_profile,
                    ledger_case.extraction_profile,
                )
                self.assertEqual(fixture_case.expected, ledger_case.expected)
                self.assertEqual(
                    fixture_case.support_surface,
                    ledger_case.support_surface,
                )
                self.assertTrue(ledger_case.expected_rdkit_audit_version_pinned)

    def test_accepted_fixture_support_matches_generated_support_and_ledger(
        self,
    ) -> None:
        accepted_fixture_cases = [
            case for case in self.fixture_cases if case.expected == "accepted"
        ]
        self.assertEqual(
            len(accepted_fixture_cases),
            sum(
                1
                for case in self.ledger_by_name.values()
                if case.expected == "accepted"
            ),
        )

        for fixture_case in accepted_fixture_cases:
            with self.subTest(case=fixture_case.name):
                ledger_case = self.ledger_by_name[fixture_case.name]
                result = accepted_case_result(ledger_case)
                image = support_image_for_case(ledger_case)

                self.assertEqual(
                    tuple(sorted(image.strings)),
                    fixture_case.expected_support,
                )
                self.assertEqual(
                    fixture_case.expected_support_count,
                    ledger_case.expected_support_count,
                )
                self.assertEqual(
                    fixture_case.expected_completion_count,
                    ledger_case.expected_completion_count,
                )
                self.assertEqual(
                    result.support_count,
                    fixture_case.expected_support_count,
                )
                self.assertEqual(
                    result.completion_count,
                    fixture_case.expected_completion_count,
                )

    def test_accepted_fixture_support_reparses_to_source_facts(self) -> None:
        for fixture_case in self.fixture_cases:
            if fixture_case.expected != "accepted":
                continue
            with self.subTest(case=fixture_case.name):
                ledger_case = self.ledger_by_name[fixture_case.name]
                source_facts = ordinary_molecule_facts_from_smiles(
                    ledger_case.smiles,
                    ledger_case.extraction_options,
                )

                for text in fixture_case.expected_support:
                    with self.subTest(case=fixture_case.name, text=text):
                        reparsed = ordinary_molecule_facts_from_smiles(
                            text,
                            ledger_case.extraction_options,
                        )
                        self.assertTrue(
                            facts_are_isomorphic(source_facts, reparsed).isomorphic,
                            text,
                        )

    def test_blocked_fixture_blocker_matches_ledger_driven_result(self) -> None:
        blocked_fixture_cases = [
            case for case in self.fixture_cases if case.expected == "blocked"
        ]
        self.assertEqual(
            len(blocked_fixture_cases),
            sum(
                1
                for case in self.ledger_by_name.values()
                if case.expected == "blocked"
            ),
        )

        for fixture_case in blocked_fixture_cases:
            with self.subTest(case=fixture_case.name):
                ledger_case = self.ledger_by_name[fixture_case.name]
                blocked = blocked_case_result(ledger_case)

                self.assertEqual(fixture_case.blocker_phase, ledger_case.blocker_phase)
                self.assertEqual(fixture_case.blocker_kind, ledger_case.blocker_kind)
                self.assertEqual(blocked["stage"], "prepare")
                self.assertIsNotNone(ledger_case.blocker_error_kind)
                self.assertEqual(
                    fixture_case.blocker_error_kind,
                    ledger_case.blocker_error_kind.name,
                )
                self.assertEqual(
                    blocked["error_kind"].name,
                    fixture_case.blocker_error_kind,
                )
                self.assertIsNotNone(fixture_case.blocker_message_contains)
                self.assertIn(
                    fixture_case.blocker_message_contains,
                    blocked["message"],
                )
                self.assertEqual(
                    fixture_case.blocker_message_contains,
                    ledger_case.blocker_message_contains,
                )


if __name__ == "__main__":
    unittest.main()
