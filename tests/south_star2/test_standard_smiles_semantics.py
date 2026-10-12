"""Check bounded OpenSMILES examples independently of RDKit writer parity."""

from __future__ import annotations

from collections import Counter
import json
import unittest

from rdkit import Chem, rdBase

from grimace import decoding
from tests.helpers.fixture_paths import checked_in_fixture_path
from tests.helpers.pinned_rdkit_fixtures import load_pinned_rdkit_fixture_cases


FIXTURE_PATH = checked_in_fixture_path("opensmiles_examples", "representational_aliases.json")
CHEMISTRY_FACTS_PATH = checked_in_fixture_path(
    "opensmiles_examples", "native_chemistry_facts.json"
)
PARSER_OBSERVATION_ROOT = checked_in_fixture_path(
    "rdkit_opensmiles_parser_observations"
)


def _fixture() -> dict[str, object]:
    return json.loads(FIXTURE_PATH.read_text())


def _chemistry_fixture() -> dict[str, object]:
    return json.loads(CHEMISTRY_FACTS_PATH.read_text())


def _parse(smiles: str) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise AssertionError(f"RDKit could not parse standard example {smiles!r}")
    return mol


def _connected_component_count(mol: Chem.Mol) -> int:
    unseen = set(range(mol.GetNumAtoms()))
    components = 0
    while unseen:
        components += 1
        pending = [unseen.pop()]
        while pending:
            atom_index = pending.pop()
            for neighbor in mol.GetAtomWithIdx(atom_index).GetNeighbors():
                if neighbor.GetIdx() in unseen:
                    unseen.remove(neighbor.GetIdx())
                    pending.append(neighbor.GetIdx())
    return components


def _assert_graph_facts(
    test: unittest.TestCase, mol: Chem.Mol, facts: dict[str, object]
) -> None:
    atoms = tuple(mol.GetAtoms())
    bonds = tuple(mol.GetBonds())
    component_count = _connected_component_count(mol)
    test.assertEqual(facts["atom_count"], len(atoms))
    test.assertEqual(facts["elements_sorted"], sorted(atom.GetSymbol() for atom in atoms))
    test.assertEqual(facts["degrees_sorted"], sorted(atom.GetDegree() for atom in atoms))
    test.assertEqual(facts["component_count"], component_count)
    test.assertEqual(
        facts["cycle_rank"],
        mol.GetNumBonds() - mol.GetNumAtoms() + component_count,
    )
    test.assertEqual(facts["bond_count"], len(bonds))
    test.assertEqual(
        facts["bond_type_counts"],
        dict(sorted(Counter(str(bond.GetBondType()) for bond in bonds).items())),
    )
    test.assertEqual(
        facts["implicit_hydrogens_sorted"],
        sorted(atom.GetNumImplicitHs() for atom in atoms),
    )
    test.assertEqual(
        facts["explicit_hydrogens_total"],
        sum(atom.GetNumExplicitHs() for atom in atoms),
    )
    test.assertEqual(
        facts["formal_charges_sorted"], sorted(atom.GetFormalCharge() for atom in atoms)
    )
    test.assertEqual(facts["isotopes_sorted"], sorted(atom.GetIsotope() for atom in atoms))
    test.assertEqual(facts["aromatic_atoms"], sum(atom.GetIsAromatic() for atom in atoms))
    test.assertFalse(facts["assigned_atom_stereo"])
    test.assertTrue(
        all(atom.GetChiralTag() == Chem.ChiralType.CHI_UNSPECIFIED for atom in atoms)
    )
    test.assertFalse(facts["assigned_bond_stereo"])
    test.assertTrue(all(bond.GetStereo() == Chem.BondStereo.STEREONONE for bond in bonds))
    test.assertTrue(all(bond.GetBondDir() == Chem.BondDir.NONE for bond in bonds))


def _assert_same_mapped_graph(
    test: unittest.TestCase,
    target: Chem.Mol,
    parsed: Chem.Mol,
) -> None:
    test.assertEqual(target.GetNumAtoms(), parsed.GetNumAtoms())
    test.assertEqual(target.GetNumBonds(), parsed.GetNumBonds())
    matches = parsed.GetSubstructMatches(target, useChirality=True, uniquify=False)
    test.assertTrue(matches, "parsed SMILES has no graph mapping to its intended molecule")
    for mapping in matches:
        atom_mapping_matches = True
        for atom in target.GetAtoms():
            mapped = parsed.GetAtomWithIdx(mapping[atom.GetIdx()])
            target_facts = (
                atom.GetAtomicNum(),
                atom.GetFormalCharge(),
                atom.GetIsotope(),
                atom.GetIsAromatic(),
                atom.GetTotalNumHs(),
                atom.GetAtomMapNum(),
                atom.GetNumRadicalElectrons(),
                atom.GetChiralTag(),
            )
            parsed_facts = (
                mapped.GetAtomicNum(),
                mapped.GetFormalCharge(),
                mapped.GetIsotope(),
                mapped.GetIsAromatic(),
                mapped.GetTotalNumHs(),
                mapped.GetAtomMapNum(),
                mapped.GetNumRadicalElectrons(),
                mapped.GetChiralTag(),
            )
            if parsed_facts != target_facts:
                atom_mapping_matches = False
                break
        if not atom_mapping_matches:
            continue
        bond_mapping_matches = True
        for bond in target.GetBonds():
            mapped = parsed.GetBondBetweenAtoms(
                mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]
            )
            if mapped is None or (mapped.GetBondType(), mapped.GetStereo()) != (
                bond.GetBondType(), bond.GetStereo()
            ):
                bond_mapping_matches = False
                break
        if not bond_mapping_matches:
            continue
        return
    test.fail("no atom mapping preserves graph bonds, hydrogens, and stereo facts")


def _public_prefix_accepts(mol: Chem.Mol, text: str) -> bool:
    pending = [(decoding.start(mol), 0)]
    while pending:
        prefix, offset = pending.pop()
        if offset == len(text):
            if prefix.is_accepted:
                return True
            continue
        for piece, successor in prefix.choices():
            if text.startswith(piece, offset):
                pending.append((successor, offset + len(piece)))
    return False


class OpenSmilesSpecificationExampleTests(unittest.TestCase):
    """Check source-listed examples without claiming full syntax completeness.

    The leading-zero isotope aliases remain source fixtures only: RDKit .1 and
    .4 reject those spellings, so this test parses only the unpadded form.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls.data = _fixture()
        cls.cases = tuple(cls.data["cases"])

    def test_parseable_specification_aliases_preserve_declared_graph_facts(self) -> None:
        self.assertEqual(4, len(self.cases))
        source = self.data["source"]
        self.assertEqual("OpenSMILES Specification", source["name"])
        self.assertEqual("https://opensmiles.org/opensmiles.html", source["url"])
        self.assertEqual("1.0", source["version"])
        self.assertEqual("2016-05-15", source["date"])
        all_aliases = [
            smiles for case in self.cases for smiles in case["example_smiles"]
        ]
        self.assertEqual(len(all_aliases), len(set(all_aliases)))
        for case in self.cases:
            with self.subTest(case=case["id"]):
                self.assertIn(
                    case["source_section"],
                    {
                        "3.4 Rings",
                        "3.7.1 Other Uses of Ring Numbers and Dot Bond",
                        "3.1.4 Isotopes",
                    },
                )
                self.assertTrue(case["source_claim"])
                aliases = tuple(case["example_smiles"])
                self.assertGreaterEqual(len(aliases), 2)
                self.assertEqual(len(aliases), len(set(aliases)))
                if case["id"] == "deuterium_leading_zero_isotope_alias":
                    continue
                target = _parse(aliases[0])
                _assert_graph_facts(self, target, case["graph_facts"])
                for smiles in aliases[1:]:
                    with self.subTest(smiles=smiles):
                        parsed = _parse(smiles)
                        _assert_graph_facts(self, parsed, case["graph_facts"])
                        _assert_same_mapped_graph(self, target, parsed)

    def test_unpadded_deuterium_example_has_declared_isotope_facts(self) -> None:
        case = next(
            case
            for case in self.cases
            if case["id"] == "deuterium_leading_zero_isotope_alias"
        )
        _assert_graph_facts(self, _parse(case["example_smiles"][0]), case["graph_facts"])


class PublicSupportOfOpenSmilesExamplesTests(unittest.TestCase):
    """Check these standard examples in current public support, not completeness."""

    def test_bounded_standard_examples_appear_in_public_support(self) -> None:
        case = next(
            case for case in _fixture()["cases"]
            if case["id"] == "cyclohexene_ring_bond_placement"
        )
        aliases = tuple(case["example_smiles"])
        source = _parse(aliases[0])
        self.assertTrue(all(_public_prefix_accepts(source, alias) for alias in aliases))


class PublicOutputGraphSemanticsTests(unittest.TestCase):
    """Check graph facts separately from whether standard examples are admitted."""

    def test_public_outputs_keep_the_intended_graph_hydrogens_and_stereo(self) -> None:
        case = next(
            case for case in _fixture()["cases"]
            if case["id"] == "cyclohexene_ring_bond_placement"
        )
        aliases = tuple(case["example_smiles"])
        facts = case["graph_facts"]
        target = _parse(aliases[0])
        for smiles in aliases:
            with self.subTest(output=smiles):
                parsed = _parse(smiles)
                _assert_graph_facts(self, parsed, facts)
                _assert_same_mapped_graph(self, target, parsed)


class PublicChemistryPrefixTests(unittest.TestCase):
    """Check one declared complete grammar path per bounded chemistry fact case."""

    def test_declared_paths_are_online_prefixes_and_preserve_mapped_graph_facts(self) -> None:
        cases = tuple(_chemistry_fixture()["cases"])
        for case in cases:
            with self.subTest(case=case["id"]):
                params = Chem.SmilesParserParams()
                params.removeHs = not case.get("preserve_graph_hydrogens", False)
                source = Chem.MolFromSmiles(case["smiles"], params)
                self.assertIsNotNone(source)
                assert source is not None
                path = case["accepted_path"]
                self.assertTrue(_public_prefix_accepts(source, path))

                parsed = Chem.MolFromSmiles(path, params)
                self.assertIsNotNone(parsed)
                assert parsed is not None
                _assert_same_mapped_graph(self, source, parsed)


class RDKitParserObservationTests(unittest.TestCase):
    """Record RDKit parser behavior, separately from OpenSMILES validity."""

    def test_leading_zero_isotope_parser_observation_matches_installed_release(self) -> None:
        fixture_cases = load_pinned_rdkit_fixture_cases(
            fixture_root=PARSER_OBSERVATION_ROOT,
            rdkit_version=rdBase.rdkitVersion,
            fixture_label="OpenSMILES parser observation",
        )
        self.assertEqual(1, len(fixture_cases))
        case = fixture_cases[0]
        self.assertEqual("deuterium_leading_zero_isotope_parser", case.case_id)
        self.assertEqual(
            "deuterium_leading_zero_isotope_alias", case.raw["reference_case_id"]
        )
        self.assertEqual("OpenSMILES 1.0", case.raw["normative_reference"])
        normative_case = next(
            item for item in _fixture()["cases"]
            if item["id"] == case.raw["reference_case_id"]
        )
        aliases = normative_case["example_smiles"]
        self.assertEqual([0], case.raw["accepted_alias_indices"])
        self.assertEqual([1, 2], case.raw["rejected_alias_indices"])
        self.assertEqual(3, len(aliases))

        accepted = case.raw["accepted_alias_indices"]
        rejected = case.raw["rejected_alias_indices"]
        with rdBase.BlockLogs():
            for index in accepted:
                with self.subTest(index=index, result="accepted"):
                    self.assertIsNotNone(Chem.MolFromSmiles(aliases[index]))
            for index in rejected:
                with self.subTest(index=index, result="rejected"):
                    self.assertIsNone(Chem.MolFromSmiles(aliases[index]))


if __name__ == "__main__":
    unittest.main()
