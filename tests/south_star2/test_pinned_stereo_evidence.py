"""Replay bounded RDKit stereo witnesses and preserve semantic controls."""

from __future__ import annotations

import hashlib
import json
import math
import unittest
from itertools import permutations

from rdkit import Chem, rdBase

from tests.helpers.fixture_paths import checked_in_fixture_path
from tests.helpers.pinned_rdkit_fixtures import load_pinned_rdkit_fixture_cases
from tests.helpers.pinned_rdkit_fixtures import normalized_unique_sorted_strings


RDKIT_VERSION = rdBase.rdkitVersion
FIXTURE_ROOT = checked_in_fixture_path("rdkit_south_star2_stereo_evidence")
FIXTURE_PATH = FIXTURE_ROOT / f"{RDKIT_VERSION}.json"


def _fixture_payload() -> dict[str, object]:
    payload = json.loads(FIXTURE_PATH.read_text())
    if payload.get("rdkit_version") != RDKIT_VERSION:
        raise AssertionError(f"{FIXTURE_PATH} declares an unexpected RDKit version")
    return payload


def _tree_schedule_bound(atoms: list[dict[str, object]]) -> int:
    degrees = [int(atom["degree"]) for atom in atoms]
    edge_count = sum(degrees) // 2
    root_sum = sum(
        math.factorial(root_degree)
        * math.prod(
            math.factorial(degree - 1)
            for index, degree in enumerate(degrees)
            if index != root
        )
        for root, root_degree in enumerate(degrees)
    )
    simplified_bound = 2 * edge_count * math.prod(
        math.factorial(degree - 1) for degree in degrees
    )
    if root_sum != simplified_bound:
        raise AssertionError("tree schedule root sum disagrees with its degree formula")
    return root_sum


def _rooted_tree_schedules(mol: Chem.Mol) -> set[tuple[int, tuple[int, ...], tuple[int, ...]]]:
    adjacency = {
        atom.GetIdx(): tuple(sorted(neighbor.GetIdx() for neighbor in atom.GetNeighbors()))
        for atom in mol.GetAtoms()
    }

    def expand(atom: int, parent: int | None) -> list[tuple[tuple[int, ...], dict[int, tuple[int, ...]]]]:
        children = tuple(neighbor for neighbor in adjacency[atom] if neighbor != parent)
        outcomes = []
        for child_order in permutations(children):
            partial = [((atom,), {atom: child_order})]
            for child in child_order:
                combined = []
                for prefix, decisions in partial:
                    for suffix, child_decisions in expand(child, atom):
                        combined.append((prefix + suffix, decisions | child_decisions))
                partial = combined
            outcomes.extend(partial)
        return outcomes

    schedules = set()
    for root in adjacency:
        for visit_order, decisions in expand(root, None):
            schedules.add((root, decisions[root], visit_order))
    return schedules


def _expected_unspecified_stereo(case_id: str, fact: dict[str, object]):
    kind = fact["kind"]
    if kind == "alkene":
        enum_type = Chem.BondStereo
    elif kind == "tetrahedral":
        enum_type = Chem.ChiralType
    else:
        raise ValueError(f"fixture case {case_id!r} has unsupported stereo kind {kind!r}")
    return getattr(enum_type, fact["expected"])


def _source_molecule(case: dict[str, object]) -> Chem.Mol:
    mol = Chem.MolFromSmiles(case["input_smiles"])
    if mol is None:
        raise AssertionError(f"invalid fixture input {case['input_smiles']!r}")
    facts = case["graph_facts"]
    if "." in case["input_smiles"] or mol.GetNumBonds() != mol.GetNumAtoms() - 1:
        raise AssertionError(f"{case['id']} is not a connected tree")
    atoms = facts["atoms"]
    if len(atoms) != mol.GetNumAtoms():
        raise AssertionError(f"atom facts do not cover {case['id']}")
    for fact in atoms:
        atom = mol.GetAtomWithIdx(fact["index"])
        if (atom.GetSymbol(), atom.GetDegree(), atom.GetTotalNumHs()) != (
            fact["element"], fact["degree"], fact["non_graph_hydrogens"]
        ):
            raise AssertionError(f"atom facts disagree for {case['id']} atom {fact['index']}")
    for fact in facts["tetrahedral_centers"]:
        atom = mol.GetAtomWithIdx(fact["atom"])
        if atom.GetTotalNumHs() != fact["non_graph_hydrogens"]:
            raise AssertionError(f"tetrahedral H count disagrees for {case['id']}")
        if str(atom.GetChiralTag()) != fact["input_chiral_tag"]:
            raise AssertionError(f"tetrahedral tag disagrees for {case['id']}")
        incident = sorted(bond.GetIdx() for bond in atom.GetBonds())
        if incident != sorted(fact["incident_bonds"]):
            raise AssertionError(f"incident bonds disagree for {case['id']}")
    for fact in facts["bonds"]:
        bond = mol.GetBondWithIdx(fact["index"])
        if (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()) != (fact["begin"], fact["end"]):
            raise AssertionError(f"bond endpoints disagree for {case['id']} bond {fact['index']}")
        if str(bond.GetBondType()) != fact["type"]:
            raise AssertionError(f"bond type disagrees for {case['id']} bond {fact['index']}")
        if "stereo" in fact and str(bond.GetStereo()) != fact["stereo"]:
            raise AssertionError(f"bond stereo disagrees for {case['id']} bond {fact['index']}")
        if "stereo_atoms" in fact and list(bond.GetStereoAtoms()) != fact["stereo_atoms"]:
            raise AssertionError(f"stereo reference atoms disagree for {case['id']} bond {fact['index']}")
        if "direction" in fact and str(bond.GetBondDir()) != fact["direction"]:
            raise AssertionError(f"bond direction disagrees for {case['id']} bond {fact['index']}")
    return mol


class PinnedStereoEvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.fixture_cases = load_pinned_rdkit_fixture_cases(
            fixture_root=FIXTURE_ROOT,
            rdkit_version=RDKIT_VERSION,
            fixture_label="South Star 2 stereo evidence",
        )
        cls.cases = {case.case_id: case.raw for case in cls.fixture_cases}
        cls.payload = _fixture_payload()

    def test_pinned_rdkit_source_manifest_preserves_atom_reference_frame(self) -> None:
        root = checked_in_fixture_path("rdkit_upstream_serializer_sources") / RDKIT_VERSION
        manifest = json.loads((root / "manifest.json").read_text())
        self.assertEqual(self.payload["source_commit"], manifest["source_commit"])
        self.assertEqual(
            f"tests/fixtures/rdkit_upstream_serializer_sources/{RDKIT_VERSION}/manifest.json",
            self.payload["source_manifest"],
        )
        atom_entry = next(entry for entry in manifest["files"] if entry["path"] == "Code/GraphMol/Atom.cpp")
        atom_source = root / "source" / atom_entry["path"]
        content = atom_source.read_bytes()
        self.assertEqual(atom_entry["bytes"], len(content))
        self.assertEqual(atom_entry["lines"], len(content.splitlines()))
        self.assertEqual(atom_entry["sha256"], hashlib.sha256(content).hexdigest())
        license_data = manifest["license"]
        self.assertEqual("BSD-3-Clause", license_data["spdx"])
        self.assertEqual(license_data["sha256"], hashlib.sha256((root / license_data["path"]).read_bytes()).hexdigest())

    def test_tree_schedule_bounds_and_seeded_support_witnesses(self) -> None:
        cases = [case for case in self.cases.values() if case["evidence_kind"] == "complete_bounded_writer_support"]
        self.assertEqual(
            {"stereo_tetra_ccw_tree", "stereo_tetra_cw_tree", "stereo_mixed_alkene_tetra_tree", "stereo_e_butene_tree", "stereo_z_butene_tree"},
            {case["id"] for case in cases},
        )
        support_by_case = {}
        for case in cases:
            with self.subTest(case=case["id"]):
                mol = _source_molecule(case)
                self.assertEqual(case["schedule_upper_bound"], _tree_schedule_bound(case["graph_facts"]["atoms"]))
                if case["id"] in {"stereo_e_butene_tree", "stereo_z_butene_tree"}:
                    rows = case["source_schedule_rows"]
                    schedules = {
                        (row["root"], tuple(row["root_children"]), tuple(row["visit_order"]))
                        for row in rows
                    }
                    self.assertEqual(_rooted_tree_schedules(mol), schedules)
                    self.assertEqual(len(schedules), len(rows))
                    self.assertEqual(6, len(schedules))
                    double_bond = mol.GetBondWithIdx(1)
                    endpoints = (double_bond.GetBeginAtomIdx(), double_bond.GetEndAtomIdx())
                    carriers = {
                        endpoint: [bond.GetIdx() for bond in mol.GetAtomWithIdx(endpoint).GetBonds() if bond.GetIdx() != double_bond.GetIdx()]
                        for endpoint in endpoints
                    }
                    for row in rows:
                        with self.subTest(seed=row["seed"]):
                            visit_order = row["visit_order"]
                            self.assertEqual(visit_order, row["atom_output_order"])
                            first_endpoint = min(endpoints, key=visit_order.index)
                            second_endpoint = next(endpoint for endpoint in endpoints if endpoint != first_endpoint)
                            self.assertEqual(first_endpoint, row["first_endpoint"])
                            positions = {atom: index for index, atom in enumerate(visit_order)}
                            control_endpoints = next(
                                (bond["begin"], bond["end"])
                                for bond in case["graph_facts"]["bonds"]
                                if bond["index"] == row["control_carrier"]["bond_index"]
                            )
                            control_anchor = next(atom for atom in control_endpoints if atom != first_endpoint)
                            other_endpoints = next(
                                (bond["begin"], bond["end"])
                                for bond in case["graph_facts"]["bonds"]
                                if bond["index"] == row["other_carrier"]["bond_index"]
                            )
                            other_anchor = next(atom for atom in other_endpoints if atom != second_endpoint)
                            self.assertEqual(row["control_flipped"], positions[first_endpoint] < positions[control_anchor])
                            self.assertEqual(row["other_flipped"], positions[other_anchor] < positions[second_endpoint])
                            self.assertEqual([carriers[first_endpoint][0], carriers[second_endpoint][0]], [row["control_carrier"]["bond_index"], row["other_carrier"]["bond_index"]])
                            for carrier in (row["control_carrier"], row["other_carrier"]):
                                fact = next(bond for bond in case["graph_facts"]["bonds"] if bond["index"] == carrier["bond_index"])
                                self.assertEqual({fact["begin"], fact["end"]}, {carrier["from_atom"], carrier["to_atom"]})
                            fresh = Chem.Mol(mol)
                            output = Chem.MolToRandomSmilesVect(fresh, 1, row["seed"], True, False, False, False)[0]
                            self.assertEqual(row["rendered_smiles"], output)
                            self.assertEqual(row["atom_output_order"], json.loads(fresh.GetProp("_smilesAtomOutputOrder")))
                            self.assertEqual(row["bond_output_order"], json.loads(fresh.GetProp("_smilesBondOutputOrder")))
                            parsed = Chem.MolFromSmiles(output)
                            self.assertIsNotNone(parsed)
                            assert parsed is not None
                            parsed_atom_for_source = {
                                source_atom: parsed_atom
                                for parsed_atom, source_atom in enumerate(row["atom_output_order"])
                            }
                            for carrier in (row["control_carrier"], row["other_carrier"]):
                                parsed_bond = parsed.GetBondBetweenAtoms(
                                    parsed_atom_for_source[carrier["from_atom"]],
                                    parsed_atom_for_source[carrier["to_atom"]],
                                )
                                self.assertIsNotNone(parsed_bond)
                                assert parsed_bond is not None
                                self.assertEqual(
                                    (parsed_atom_for_source[carrier["from_atom"]], parsed_atom_for_source[carrier["to_atom"]]),
                                    (parsed_bond.GetBeginAtomIdx(), parsed_bond.GetEndAtomIdx()),
                                )
                                glyph = {
                                    Chem.BondDir.ENDUPRIGHT: "/",
                                    Chem.BondDir.ENDDOWNRIGHT: "\\",
                                }[parsed_bond.GetBondDir()]
                                self.assertEqual(carrier["glyph"], glyph)
                    rendered = {row["rendered_smiles"] for row in rows}
                    self.assertEqual(case["expected_distinct_outputs"], len(rendered))
                    self.assertLessEqual(
                        case["expected_distinct_outputs"],
                        case["distinct_output_upper_bound"],
                    )
                    self.assertLessEqual(
                        case["distinct_output_upper_bound"], len(schedules)
                    )
                    support_by_case[case["id"]] = rendered
                    continue

                expected = normalized_unique_sorted_strings(
                    case["expected_support"], field_name="expected_support",
                    fixture_path=FIXTURE_PATH, case_id=case["id"]
                )
                self.assertEqual(case["expected_distinct_outputs"], len(expected))
                self.assertLessEqual(
                    case["expected_distinct_outputs"],
                    case["distinct_output_upper_bound"],
                )
                self.assertLessEqual(
                    case["distinct_output_upper_bound"], case["schedule_upper_bound"]
                )
                support_by_case[case["id"]] = set(expected)
                seeds = case["witness_seeds"]
                self.assertEqual(len(expected), len(seeds))
                for expected_text, seed in zip(expected, seeds, strict=True):
                    with self.subTest(seed=seed):
                        self.assertIs(type(seed), int)
                        self.assertGreater(seed, 0)
                        self.assertEqual(
                            [expected_text],
                            Chem.MolToRandomSmilesVect(Chem.Mol(mol), 1, seed, True, False, False, False),
                        )
        self.assertEqual(
            set(),
            support_by_case["stereo_e_butene_tree"]
            & support_by_case["stereo_z_butene_tree"],
        )

    def test_unspecified_stereo_stays_unspecified_on_seeded_parseback(self) -> None:
        cases = [case for case in self.cases.values() if case["evidence_kind"] == "unspecified_stereo_semantic_control"]
        self.assertEqual(2, len(cases))
        for case in cases:
            with self.subTest(case=case["id"]):
                stereo_controls = tuple(
                    (fact, _expected_unspecified_stereo(case["id"], fact))
                    for fact in case["unspecified_stereo"]
                )
                source = Chem.MolFromSmiles(case["input_smiles"])
                self.assertIsNotNone(source)
                assert source is not None
                for alkene in case.get("alkenes", []):
                    bond = source.GetBondWithIdx(alkene["bond_index"])
                    self.assertEqual((alkene["begin"], alkene["end"]), (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
                    self.assertEqual(alkene["stereo"], str(bond.GetStereo()))
                    self.assertEqual(alkene["stereo_atoms"], list(bond.GetStereoAtoms()))
                for fact, expected_stereo in stereo_controls:
                    if fact["kind"] == "alkene":
                        self.assertEqual(expected_stereo, source.GetBondWithIdx(fact["bond_index"]).GetStereo())
                    else:
                        self.assertEqual(expected_stereo, source.GetAtomWithIdx(fact["atom_index"]).GetChiralTag())
                for row in case["witnesses"]:
                    written = Chem.Mol(source)
                    output = Chem.MolToRandomSmilesVect(
                        written, 1, row["seed"], True, False, False, False
                    )[0]
                    self.assertEqual(row["smiles"], output)
                    output_atom_order = json.loads(written.GetProp("_smilesAtomOutputOrder"))
                    parsed_atom_for_source = {
                        source_atom: parsed_atom
                        for parsed_atom, source_atom in enumerate(output_atom_order)
                    }
                    parsed = Chem.MolFromSmiles(output)
                    self.assertIsNotNone(parsed)
                    assert parsed is not None
                    for fact, expected_stereo in stereo_controls:
                        if fact["kind"] == "alkene":
                            source_bond = source.GetBondWithIdx(fact["bond_index"])
                            parsed_bond = parsed.GetBondBetweenAtoms(
                                parsed_atom_for_source[source_bond.GetBeginAtomIdx()],
                                parsed_atom_for_source[source_bond.GetEndAtomIdx()],
                            )
                            self.assertIsNotNone(parsed_bond)
                            assert parsed_bond is not None
                            self.assertEqual(Chem.BondType.DOUBLE, parsed_bond.GetBondType())
                            self.assertEqual(expected_stereo, parsed_bond.GetStereo())
                        else:
                            source_atom = source.GetAtomWithIdx(fact["atom_index"])
                            parsed_atom = parsed.GetAtomWithIdx(
                                parsed_atom_for_source[fact["atom_index"]]
                            )
                            self.assertEqual(source_atom.GetAtomicNum(), parsed_atom.GetAtomicNum())
                            self.assertEqual(source_atom.GetDegree(), parsed_atom.GetDegree())
                            self.assertEqual(expected_stereo, parsed_atom.GetChiralTag())

    def test_directional_stereo_quirk_is_membership_with_parseback_discrepancy(self) -> None:
        case = next(case for case in self.cases.values() if case["evidence_kind"] == "known_writer_quirk")
        mol = Chem.MolFromSmiles(case["input_smiles"])
        self.assertIsNotNone(mol)
        assert mol is not None
        for fact in case["configured_bonds"]:
            bond = mol.GetBondWithIdx(fact["index"])
            self.assertEqual((fact["begin"], fact["end"]), (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            bond.SetStereo(getattr(Chem.BondStereo, fact["stereo"]))
            if fact["stereo_atoms"]:
                bond.SetStereoAtoms(*fact["stereo_atoms"])
        if case["clear_all_bond_directions_before_write"]:
            for bond in mol.GetBonds():
                bond.SetBondDir(Chem.BondDir.NONE)
        for row in case["witnesses"]:
            output = Chem.MolToRandomSmilesVect(
                Chem.Mol(mol), 1, row["seed"], True, False, False, False
            )[0]
            self.assertEqual(row["smiles"], output)
            parsed = Chem.MolFromSmiles(output)
            self.assertIsNotNone(parsed)
            assert parsed is not None
            doubles = [bond for bond in parsed.GetBonds() if bond.GetBondType() == Chem.BondType.DOUBLE]
            self.assertEqual(row["parsed_middle_stereo"], str(doubles[1].GetStereo()))
            self.assertNotEqual(Chem.BondStereo.STEREONONE, doubles[1].GetStereo())


if __name__ == "__main__":
    unittest.main()
