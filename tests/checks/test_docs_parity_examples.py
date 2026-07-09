from __future__ import annotations

import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]
PARITY_EXAMPLES = ROOT / "docs" / "parity-examples.md"
RDKIT_VERSION = "2026.03.1"


def _json(path: str) -> dict[str, object]:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def _case(data: dict[str, object], case_id: str) -> dict[str, object]:
    cases = data["cases"]
    if not isinstance(cases, list):
        raise AssertionError("fixture cases must be a list")
    for case in cases:
        if isinstance(case, dict) and case.get("id") == case_id:
            return case
    raise AssertionError(f"missing fixture case: {case_id}")


class ParityExamplesDocsTests(unittest.TestCase):
    def setUp(self) -> None:
        self.text = PARITY_EXAMPLES.read_text(encoding="utf-8")

    def test_exact_support_example_matches_fixture(self) -> None:
        case = _case(
            _json("tests/fixtures/rdkit_exact_small_support/2026.03.1.json"),
            "cco_root1_nonstereo",
        )

        self.assertIn("`cco_root1_nonstereo`", self.text)
        self.assertEqual("CCO", case["smiles"])
        self.assertEqual(1, case["rooted_at_atom"])
        self.assertIs(case["isomeric_smiles"], False)
        self.assertEqual(("C(C)O", "C(O)C"), tuple(case["expected"]))
        for expected in case["expected"]:
            self.assertIn(f"`{expected}`", self.text)

    def test_writer_membership_example_matches_fixture(self) -> None:
        case = _case(
            _json(
                "tests/fixtures/rdkit_writer_membership/"
                "2026.03.1/20_writer_flags.json",
            ),
            "writer_flags_01_propane_all_bonds_explicit",
        )

        self.assertIn("`writer_flags_01_propane_all_bonds_explicit`", self.text)
        self.assertEqual("CCC", case["smiles"])
        self.assertIs(case["all_bonds_explicit"], True)
        self.assertEqual("C-C-C", case["expected"])
        self.assertIn("RDKit writer call", self.text)
        self.assertIn("Grimace support surface", self.text)
        self.assertIn("`C-C-C`", self.text)

    def test_semantic_boundary_example_uses_exact_support_fixture(self) -> None:
        case = _case(
            _json("tests/fixtures/rdkit_exact_small_support/2026.03.1.json"),
            "cco_root1_nonstereo",
        )

        self.assertNotIn("CCO", case["expected"])
        self.assertIn("string `CCO` parses", self.text)
        self.assertIn("pinned rooted writer support", self.text)
        self.assertIn("specific rooted writer language", self.text)

    def test_known_gap_example_matches_fixture(self) -> None:
        case = _case(
            _json("tests/fixtures/rdkit_known_stereo_gaps/2026.03.1.json"),
            "github3967_part2_directional_ring_closure_canonical",
        )

        self.assertIn(
            "`github3967_part2_directional_ring_closure_canonical`",
            self.text,
        )
        self.assertIs(case["isomeric_smiles"], True)
        self.assertIs(case["rdkit_canonical"], True)
        self.assertIn(f"`{case['smiles']}`", self.text)
        self.assertIn(f"`{case['expected']}`", self.text)

    def test_examples_state_pinned_rdkit_version(self) -> None:
        self.assertGreaterEqual(self.text.count(f"`{RDKIT_VERSION}`"), 4)


if __name__ == "__main__":
    unittest.main()
