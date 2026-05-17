from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import json
import unittest

from rdkit import Chem, rdBase

from grimace import _core, _runtime
from tests.helpers.fixture_paths import checked_in_fixture_path
from tests.helpers.public_runtime import make_determinized_decoder
from tests.helpers.rdkit_writer_membership import (
    load_pinned_writer_membership_cases,
)
from tests.rdkit_serialization._support import (
    grimace_support,
    mol_from_pinned_source,
    rdkit_mol_to_smiles_kwargs_from_options,
    supported_public_kwargs_from_rdkit_options,
)


FIXTURE_ROOT = checked_in_fixture_path("rdkit_known_stereo_gaps")
SUPPORTED_STEREO_DIAGNOSTIC_FLAGS = _runtime.MolToSmilesFlags(
    isomeric_smiles=True,
    kekule_smiles=False,
    rooted_at_atom=-1,
    canonical=False,
    all_bonds_explicit=False,
    all_hs_explicit=False,
    do_random=True,
    ignore_atom_map_numbers=False,
)
ALLOWED_ACCEPTANCE_ROLES = {
    "decoder_path_acceptance",
    "red_support_acceptance",
    "support_present_family_guard",
}
ALLOWED_GAP_CLASSES = {
    "missing_rdkit_writer_policy",
    "missing_semantic_constraint",
    "unsupported_representation",
}
ALLOWED_CURRENT_RESULTS = {
    "decoder_path_only",
    "support_enumeration_error",
    "support_missing",
    "support_present",
}


@dataclass(frozen=True, slots=True)
class KnownStereoGapCase:
    case_id: str
    source: str
    acceptance_role: str
    gap_class: str | None
    gap_detail: str
    expected_current_result: str
    smiles: str | None
    molblock: str | None
    writer_membership_case_id: str | None
    expected: str
    rooted_at_atom: int | None
    isomeric_smiles: bool
    rdkit_canonical: bool
    rdkit_random_vector_seed: int | None
    rdkit_random_vector_index: int | None
    check_grimace_decoder_path: bool
    check_grimace_support: bool


def _load_known_stereo_gap_cases(rdkit_version: str) -> tuple[KnownStereoGapCase, ...]:
    fixture_path = FIXTURE_ROOT / f"{rdkit_version}.json"
    if not fixture_path.is_file():
        raise FileNotFoundError(
            f"no pinned known stereo-gap corpus for RDKit {rdkit_version}"
        )

    payload = json.loads(fixture_path.read_text())
    if payload["rdkit_version"] != rdkit_version:
        raise ValueError(f"fixture {fixture_path} RDKit version mismatch")

    cases = []
    for raw_case in payload["cases"]:
        smiles = raw_case.get("smiles")
        molblock = raw_case.get("molblock")
        writer_membership_case_id = raw_case.get("writer_membership_case_id")
        molecule_source_count = sum(
            value is not None
            for value in (smiles, molblock, writer_membership_case_id)
        )
        if molecule_source_count != 1:
            raise ValueError(
                f"known stereo-gap case {raw_case['id']!r} must define exactly "
                "one of 'smiles', 'molblock', or 'writer_membership_case_id'"
            )
        acceptance_role = raw_case["acceptance_role"]
        gap_class = raw_case["gap_class"]
        expected_current_result = raw_case["expected_current_result"]
        if acceptance_role not in ALLOWED_ACCEPTANCE_ROLES:
            raise ValueError(
                f"known stereo-gap case {raw_case['id']!r} has unsupported "
                f"acceptance_role {acceptance_role!r}"
            )
        if gap_class is not None and gap_class not in ALLOWED_GAP_CLASSES:
            raise ValueError(
                f"known stereo-gap case {raw_case['id']!r} has unsupported "
                f"gap_class {gap_class!r}"
            )
        if expected_current_result not in ALLOWED_CURRENT_RESULTS:
            raise ValueError(
                f"known stereo-gap case {raw_case['id']!r} has unsupported "
                f"expected_current_result {expected_current_result!r}"
            )
        cases.append(
            KnownStereoGapCase(
                case_id=raw_case["id"],
                source=raw_case["source"],
                acceptance_role=acceptance_role,
                gap_class=gap_class,
                gap_detail=raw_case["gap_detail"],
                expected_current_result=expected_current_result,
                smiles=smiles,
                molblock=molblock,
                writer_membership_case_id=writer_membership_case_id,
                expected=raw_case["expected"],
                rooted_at_atom=raw_case["rooted_at_atom"],
                isomeric_smiles=raw_case["isomeric_smiles"],
                rdkit_canonical=raw_case["rdkit_canonical"],
                rdkit_random_vector_seed=raw_case.get("rdkit_random_vector_seed"),
                rdkit_random_vector_index=raw_case.get("rdkit_random_vector_index"),
                check_grimace_decoder_path=raw_case.get(
                    "check_grimace_decoder_path", False
                ),
                check_grimace_support=raw_case.get("check_grimace_support", True),
            )
        )
    return tuple(cases)


class KnownStereoGapTests(unittest.TestCase):
    """Red tests for coupled directional-stereo parity gaps.

    These cases are not part of the passing writer-membership corpus yet. They
    should move there once Grimace resolves coupled double-bond direction tokens
    with RDKit-equivalent traversal-order state.
    """

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.cases = _load_known_stereo_gap_cases(rdBase.rdkitVersion)
            writer_cases = load_pinned_writer_membership_cases(rdBase.rdkitVersion)
        except FileNotFoundError:
            raise unittest.SkipTest(
                f"no pinned known stereo-gap corpus for RDKit {rdBase.rdkitVersion}"
            )
        cls.writer_cases_by_id = {case.case_id: case for case in writer_cases}

    def test_pinned_known_stereo_gap_cases_are_classified(self) -> None:
        status_counts = Counter(case.expected_current_result for case in self.cases)
        self.assertEqual(
            Counter(
                {
                    "decoder_path_only": 1,
                    "support_enumeration_error": 1,
                    "support_missing": 8,
                    "support_present": 6,
                }
            ),
            status_counts,
        )
        for case in self.cases:
            with self.subTest(case_id=case.case_id):
                self.assertTrue(case.gap_detail.strip(), case.case_id)
                if case.acceptance_role == "red_support_acceptance":
                    self.assertIn(case.gap_class, ALLOWED_GAP_CLASSES)
                    self.assertIn(
                        case.expected_current_result,
                        {"support_enumeration_error", "support_missing"},
                    )
                    self.assertTrue(case.check_grimace_support)
                else:
                    self.assertIsNone(case.gap_class)
                if case.acceptance_role == "decoder_path_acceptance":
                    self.assertTrue(case.check_grimace_decoder_path)
                    self.assertFalse(case.check_grimace_support)
                    self.assertEqual("decoder_path_only", case.expected_current_result)
                if case.acceptance_role == "support_present_family_guard":
                    self.assertTrue(case.check_grimace_support)
                    self.assertEqual("support_present", case.expected_current_result)

    def _mol_from_case(self, case: KnownStereoGapCase) -> Chem.Mol:
        if case.writer_membership_case_id is not None:
            writer_case = self.writer_cases_by_id[case.writer_membership_case_id]
            return mol_from_pinned_source(writer_case)
        return mol_from_pinned_source(case)

    def _molecule_source_key(self, case: KnownStereoGapCase) -> str:
        if case.writer_membership_case_id is not None:
            return f"writer-membership:{case.writer_membership_case_id}"
        if case.smiles is not None:
            return f"smiles:{case.smiles}"
        return f"molblock:{case.case_id}"

    def _rdkit_output(self, mol: Chem.Mol, case: KnownStereoGapCase) -> str:
        if case.rdkit_random_vector_seed is not None:
            if case.rdkit_random_vector_index is None:
                raise ValueError(
                    f"{case.case_id}: rdkit_random_vector_seed requires "
                    "rdkit_random_vector_index"
                )
            random_outputs = Chem.MolToRandomSmilesVect(
                Chem.Mol(mol),
                case.rdkit_random_vector_index + 1,
                randomSeed=case.rdkit_random_vector_seed,
                isomericSmiles=case.isomeric_smiles,
            )
            return random_outputs[case.rdkit_random_vector_index]
        if case.rdkit_random_vector_index is not None:
            raise ValueError(
                f"{case.case_id}: rdkit_random_vector_index requires "
                "rdkit_random_vector_seed"
            )
        return Chem.MolToSmiles(
            Chem.Mol(mol),
            **rdkit_mol_to_smiles_kwargs_from_options(
                rooted_at_atom=case.rooted_at_atom,
                isomeric_smiles=case.isomeric_smiles,
                canonical=case.rdkit_canonical,
                do_random=False,
            ),
        )

    def _assert_determinized_decoder_accepts(
        self,
        mol: Chem.Mol,
        case: KnownStereoGapCase,
    ) -> None:
        decoder = make_determinized_decoder(
            mol,
            **supported_public_kwargs_from_rdkit_options(
                rooted_at_atom=case.rooted_at_atom,
                isomeric_smiles=case.isomeric_smiles,
            ),
        )
        pos = 0
        while pos < len(case.expected):
            choice = next(
                (
                    choice
                    for choice in decoder.next_choices
                    if case.expected.startswith(choice.text, pos)
                ),
                None,
            )
            if choice is None:
                choices = tuple(choice.text for choice in decoder.next_choices)
                self.fail(
                    f"{case.case_id}: decoder rejected RDKit output at offset "
                    f"{pos}; prefix={case.expected[:pos]!r}; "
                    f"next={case.expected[pos:pos + 24]!r}; choices={choices!r}"
                )
            decoder = choice.next_state
            pos += len(choice.text)
        self.assertTrue(decoder.is_terminal, case.case_id)

    def test_pinned_rdkit_stereo_gap_outputs_are_in_grimace_support(self) -> None:
        support_cache: dict[tuple[str, int | None, bool], set[str]] = {}
        for case in self.cases:
            with self.subTest(case_id=case.case_id, source=case.source):
                mol = self._mol_from_case(case)
                rdkit_output = self._rdkit_output(mol, case)
                self.assertEqual(case.expected, rdkit_output)
                if case.check_grimace_decoder_path:
                    self._assert_determinized_decoder_accepts(mol, case)
                if not case.check_grimace_support:
                    continue

                support_key = (
                    self._molecule_source_key(case),
                    case.rooted_at_atom,
                    case.isomeric_smiles,
                )
                support = support_cache.get(support_key)
                if support is None:
                    try:
                        support = grimace_support(
                            mol,
                            rooted_at_atom=case.rooted_at_atom,
                            isomeric_smiles=case.isomeric_smiles,
                        )
                    except Exception as exc:
                        self.fail(
                            f"{case.case_id}: Grimace support enumeration failed "
                            f"with {type(exc).__name__}: {exc}"
                        )
                    support_cache[support_key] = support
                self.assertTrue(
                    case.expected in support,
                    f"{case.case_id}: missing RDKit output {case.expected!r} "
                    f"from Grimace support of size {len(support)}",
                )

    def test_manual_difficult_cases_keep_nonempty_marker_row_state(self) -> None:
        cases = tuple(
            case
            for case in self.cases
            if case.case_id.startswith("manual_bond_stereo_difficult_")
        )
        self.assertEqual(4, len(cases))

        for case in cases:
            with self.subTest(case_id=case.case_id, source=case.source):
                mol = self._mol_from_case(case)
                prepared = _runtime.prepare_smiles_graph(
                    mol,
                    flags=SUPPORTED_STEREO_DIAGNOSTIC_FLAGS,
                )
                rows = _core._stereo_constraint_output_facts(prepared)
                self.assertTrue(rows)

                for row in rows:
                    marker_event_counts = Counter(
                        event["component_idx"] for event in row["marker_event_facts"]
                    )
                    self.assertTrue(marker_event_counts)
                    for component in row["marker_placement_state"]["semantic"]:
                        self.assertEqual(
                            marker_event_counts[component["component_idx"]],
                            component["marker_event_count"],
                        )
                        self.assertGreater(
                            component["row_count_after_marker_events"],
                            0,
                        )
                        self.assertEqual(
                            component["row_count_after_marker_events"],
                            len(component["rows_after_marker_events"]),
                        )

    def test_manual_difficult_cases_identify_row_routing_membership_gap(self) -> None:
        cases = tuple(
            case
            for case in self.cases
            if case.case_id.startswith("manual_bond_stereo_difficult_")
        )
        self.assertEqual(4, len(cases))

        missing_case_ids = set[str]()
        present_case_ids = set[str]()
        for case in cases:
            with self.subTest(case_id=case.case_id, source=case.source):
                mol = self._mol_from_case(case)
                support = grimace_support(
                    mol,
                    rooted_at_atom=case.rooted_at_atom,
                    isomeric_smiles=case.isomeric_smiles,
                )
                if case.expected in support:
                    present_case_ids.add(case.case_id)
                else:
                    missing_case_ids.add(case.case_id)

        self.assertEqual(set(), missing_case_ids)
        self.assertEqual(
            {
                "manual_bond_stereo_difficult_cis_cis",
                "manual_bond_stereo_difficult_cis_trans",
                "manual_bond_stereo_difficult_trans_cis",
                "manual_bond_stereo_difficult_trans_trans",
            },
            present_case_ids,
        )


if __name__ == "__main__":
    unittest.main()
