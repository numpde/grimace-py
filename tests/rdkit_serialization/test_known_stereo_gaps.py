from __future__ import annotations

from dataclasses import dataclass
import json
import unittest

from rdkit import Chem, rdBase

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


@dataclass(frozen=True, slots=True)
class KnownStereoGapCase:
    case_id: str
    source: str
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
        cases.append(
            KnownStereoGapCase(
                case_id=raw_case["id"],
                source=raw_case["source"],
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


if __name__ == "__main__":
    unittest.main()
