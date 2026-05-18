from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from itertools import combinations, product
import json
import unittest

from rdkit import Chem, RDLogger, rdBase

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
MarkerSlots = tuple[tuple[int, str], ...]
MarkerSlotSet = tuple[MarkerSlots, ...]


@dataclass(frozen=True, slots=True)
class KnownStereoGapCase:
    case_id: str
    source: str
    acceptance_role: str
    gap_class: str | None
    gap_detail: str
    expected_current_result: str
    expected_current_same_skeleton_support_count: int | None
    expected_current_same_skeleton_marker_slots: MarkerSlotSet | None
    semantic_minimal_marker_slots: MarkerSlotSet | None
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


def _marker_slot_set_from_json(
    raw_case: dict[str, object],
    field_name: str,
) -> MarkerSlotSet | None:
    raw_marker_slot_set = raw_case.get(field_name)
    if raw_marker_slot_set is None:
        return None

    marker_slot_set = tuple(
        tuple((slot, marker) for slot, marker in marker_slots)
        for marker_slots in raw_marker_slot_set
    )
    for marker_slots in marker_slot_set:
        if marker_slots != tuple(sorted(marker_slots)):
            raise ValueError(
                f"known stereo-gap case {raw_case['id']!r} has unsorted "
                f"{field_name}"
            )
        if len({slot for slot, _marker in marker_slots}) != len(marker_slots):
            raise ValueError(
                f"known stereo-gap case {raw_case['id']!r} has duplicate "
                f"{field_name} slots"
            )
        for slot, marker in marker_slots:
            if type(slot) is not int or slot < 0 or marker not in {"/", "\\"}:
                raise ValueError(
                    f"known stereo-gap case {raw_case['id']!r} has invalid "
                    f"{field_name} slot"
                )
    return marker_slot_set


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
        expected_same_skeleton_support_count = raw_case.get(
            "expected_current_same_skeleton_support_count"
        )
        same_skeleton_marker_slots = _marker_slot_set_from_json(
            raw_case,
            "expected_current_same_skeleton_marker_slots",
        )
        semantic_minimal_marker_slots = _marker_slot_set_from_json(
            raw_case,
            "semantic_minimal_marker_slots",
        )
        if same_skeleton_marker_slots is not None:
            if expected_current_result != "support_missing":
                raise ValueError(
                    f"known stereo-gap case {raw_case['id']!r} may only define "
                    "expected_current_same_skeleton_marker_slots for "
                    "support_missing"
                )
            if (
                expected_same_skeleton_support_count is not None
                and len(same_skeleton_marker_slots)
                != expected_same_skeleton_support_count
            ):
                raise ValueError(
                    f"known stereo-gap case {raw_case['id']!r} marker-slot "
                    "fixtures must match "
                    "expected_current_same_skeleton_support_count"
                )
        if expected_current_result == "support_missing":
            if (
                type(expected_same_skeleton_support_count) is not int
                or expected_same_skeleton_support_count < 0
            ):
                raise ValueError(
                    f"known stereo-gap case {raw_case['id']!r} must pin a "
                    "nonnegative expected_current_same_skeleton_support_count"
                )
        elif expected_same_skeleton_support_count is not None:
            raise ValueError(
                f"known stereo-gap case {raw_case['id']!r} may only define "
                "expected_current_same_skeleton_support_count for support_missing"
            )
        cases.append(
            KnownStereoGapCase(
                case_id=raw_case["id"],
                source=raw_case["source"],
                acceptance_role=acceptance_role,
                gap_class=gap_class,
                gap_detail=raw_case["gap_detail"],
                expected_current_result=expected_current_result,
                expected_current_same_skeleton_support_count=(
                    expected_same_skeleton_support_count
                ),
                expected_current_same_skeleton_marker_slots=(
                    same_skeleton_marker_slots
                ),
                semantic_minimal_marker_slots=semantic_minimal_marker_slots,
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


def _direction_erased_skeleton(smiles: str) -> str:
    return "".join(char for char in smiles if char not in {"/", "\\"})


def _direction_marker_slots(smiles: str) -> MarkerSlots:
    slots = []
    skeleton_slot = 0
    for char in smiles:
        if char in {"/", "\\"}:
            slots.append((skeleton_slot, char))
        else:
            skeleton_slot += 1
    return tuple(slots)


def _smiles_from_direction_marker_slots(
    skeleton: str,
    marker_slots: MarkerSlots,
) -> str:
    markers_by_slot = dict(marker_slots)
    parts = []
    for slot, char in enumerate(skeleton):
        if slot in markers_by_slot:
            parts.append(markers_by_slot[slot])
        parts.append(char)
    if len(skeleton) in markers_by_slot:
        parts.append(markers_by_slot[len(skeleton)])
    return "".join(parts)


def _directional_bond_signature(mol: Chem.Mol) -> tuple[tuple[int, int, str], ...]:
    return tuple(
        sorted(
            (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                str(bond.GetBondDir()),
            )
            for bond in mol.GetBonds()
            if bond.GetBondDir() != Chem.BondDir.NONE
        )
    )


def _double_bond_stereo_signature(
    mol: Chem.Mol,
) -> tuple[tuple[int, int, str, tuple[int, ...]], ...]:
    return tuple(
        sorted(
            (
                bond.GetBeginAtomIdx(),
                bond.GetEndAtomIdx(),
                str(bond.GetStereo()),
                tuple(bond.GetStereoAtoms()),
            )
            for bond in mol.GetBonds()
            if bond.GetStereo() != Chem.BondStereo.STEREONONE
        )
    )


def _single_marker_candidate_slots(skeleton: str) -> tuple[int, ...]:
    candidate_slots = []
    RDLogger.DisableLog("rdApp.*")
    try:
        for slot in range(len(skeleton) + 1):
            if any(
                Chem.MolFromSmiles(
                    _smiles_from_direction_marker_slots(skeleton, ((slot, marker),))
                )
                is not None
                for marker in ("/", "\\")
            ):
                candidate_slots.append(slot)
    finally:
        RDLogger.EnableLog("rdApp.*")
    return tuple(candidate_slots)


def _semantic_minimal_marker_slot_sets(
    skeleton: str,
    reference_stereo_signature: tuple[tuple[int, int, str, tuple[int, ...]], ...],
) -> MarkerSlotSet:
    # A minimal directional-bond basis needs at most two marker-bearing
    # neighboring bonds per double bond; shared markers can reduce that count.
    max_marker_count = 2 * len(reference_stereo_signature)
    candidate_slots = _single_marker_candidate_slots(skeleton)
    valid_marker_slot_sets = []

    RDLogger.DisableLog("rdApp.*")
    try:
        for marker_count in range(max_marker_count + 1):
            for slots in combinations(candidate_slots, marker_count):
                for markers in product(("/", "\\"), repeat=marker_count):
                    marker_slots = tuple(zip(slots, markers))
                    mol = Chem.MolFromSmiles(
                        _smiles_from_direction_marker_slots(skeleton, marker_slots)
                    )
                    if (
                        mol is not None
                        and _double_bond_stereo_signature(mol)
                        == reference_stereo_signature
                    ):
                        valid_marker_slot_sets.append(marker_slots)

        minimal_marker_slot_sets = []
        for marker_slots in valid_marker_slot_sets:
            is_minimal = True
            for marker_idx in range(len(marker_slots)):
                reduced_marker_slots = tuple(
                    marker_slot
                    for idx, marker_slot in enumerate(marker_slots)
                    if idx != marker_idx
                )
                reduced_mol = Chem.MolFromSmiles(
                    _smiles_from_direction_marker_slots(
                        skeleton,
                        reduced_marker_slots,
                    )
                )
                if (
                    reduced_mol is not None
                    and _double_bond_stereo_signature(reduced_mol)
                    == reference_stereo_signature
                ):
                    is_minimal = False
                    break
            if is_minimal:
                minimal_marker_slot_sets.append(marker_slots)
    finally:
        RDLogger.EnableLog("rdApp.*")

    return tuple(sorted(minimal_marker_slot_sets))


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
                    "support_missing": 9,
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

    def test_smallest_gap_separates_stereo_assignment_from_marker_basis(self) -> None:
        case = next(
            case
            for case in self.cases
            if case.case_id == "github3967_part2_directional_ring_closure_canonical"
        )
        source_smiles = case.smiles
        self.assertIsNotNone(source_smiles)
        assert source_smiles is not None

        source_mol = self._mol_from_case(case)
        expected_mol = Chem.MolFromSmiles(case.expected)
        self.assertIsNotNone(expected_mol)
        assert expected_mol is not None

        self.assertEqual(case.expected, self._rdkit_output(source_mol, case))
        self.assertEqual(
            _double_bond_stereo_signature(source_mol),
            _double_bond_stereo_signature(expected_mol),
        )
        self.assertNotEqual(
            _directional_bond_signature(source_mol),
            _directional_bond_signature(expected_mol),
        )

        source_marker_slots = _direction_marker_slots(source_smiles)
        rdkit_marker_slots = _direction_marker_slots(case.expected)
        self.assertEqual(4, len(source_marker_slots))
        self.assertEqual(3, len(rdkit_marker_slots))
        self.assertNotEqual(source_marker_slots, rdkit_marker_slots)

    def test_smallest_gap_enumerates_semantic_marker_basis_space(self) -> None:
        case = next(
            case
            for case in self.cases
            if case.case_id == "github3967_part2_directional_ring_closure_canonical"
        )
        source_smiles = case.smiles
        self.assertIsNotNone(source_smiles)
        self.assertIsNotNone(case.semantic_minimal_marker_slots)
        self.assertIsNotNone(case.expected_current_same_skeleton_marker_slots)
        assert source_smiles is not None
        assert case.semantic_minimal_marker_slots is not None
        assert case.expected_current_same_skeleton_marker_slots is not None

        skeleton = _direction_erased_skeleton(case.expected)
        source_mol = self._mol_from_case(case)
        reference_signature = _double_bond_stereo_signature(source_mol)
        semantic_marker_slots = _semantic_minimal_marker_slot_sets(
            skeleton,
            reference_signature,
        )

        source_marker_slots = _direction_marker_slots(source_smiles)
        rdkit_marker_slots = _direction_marker_slots(case.expected)
        self.assertEqual(case.semantic_minimal_marker_slots, semantic_marker_slots)
        self.assertIn(source_marker_slots, semantic_marker_slots)
        self.assertIn(rdkit_marker_slots, semantic_marker_slots)

        current_same_skeleton_marker_slots = set(
            case.expected_current_same_skeleton_marker_slots
        )
        self.assertTrue(
            current_same_skeleton_marker_slots.isdisjoint(semantic_marker_slots)
        )

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

    def test_support_missing_cases_pin_direction_skeleton_profile(self) -> None:
        support_cache: dict[tuple[str, int | None, bool], set[str]] = {}
        for case in self.cases:
            if case.expected_current_result != "support_missing":
                continue

            with self.subTest(case_id=case.case_id, source=case.source):
                mol = self._mol_from_case(case)
                support_key = (
                    self._molecule_source_key(case),
                    case.rooted_at_atom,
                    case.isomeric_smiles,
                )
                support = support_cache.get(support_key)
                if support is None:
                    support = grimace_support(
                        mol,
                        rooted_at_atom=case.rooted_at_atom,
                        isomeric_smiles=case.isomeric_smiles,
                    )
                    support_cache[support_key] = support

                expected_skeleton = _direction_erased_skeleton(case.expected)
                same_skeleton_support = tuple(
                    sorted(
                        smiles
                        for smiles in support
                        if _direction_erased_skeleton(smiles) == expected_skeleton
                    )
                )

                self.assertNotIn(case.expected, support)
                self.assertEqual(
                    case.expected_current_same_skeleton_support_count,
                    len(same_skeleton_support),
                )
                actual_marker_slots = tuple(
                    sorted(
                        _direction_marker_slots(smiles)
                        for smiles in same_skeleton_support
                    )
                )
                if case.expected_current_same_skeleton_marker_slots is not None:
                    self.assertEqual(
                        case.expected_current_same_skeleton_marker_slots,
                        actual_marker_slots,
                    )
                if case.expected_current_same_skeleton_support_count:
                    self.assertTrue(
                        all(
                            marker_slots != _direction_marker_slots(case.expected)
                            for marker_slots in actual_marker_slots
                        ),
                        case.case_id,
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
