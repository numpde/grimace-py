from __future__ import annotations

from dataclasses import dataclass
import json
import unittest

from rdkit import Chem, rdBase

from tests.helpers.fixture_paths import checked_in_fixture_path
from tests.helpers.rdkit_writer_membership import (
    PinnedWriterMembershipCase,
    load_pinned_writer_membership_cases,
)
from tests.rdkit_serialization._support import (
    grimace_support,
    mol_from_pinned_source,
    rdkit_mol_to_smiles_kwargs_from_options,
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
    rdkit_random_seed: int | None
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
                rdkit_random_seed=raw_case.get("rdkit_random_seed"),
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

    def test_pinned_rdkit_stereo_gap_outputs_are_in_grimace_support(self) -> None:
        for case in self.cases:
            with self.subTest(case_id=case.case_id, source=case.source):
                mol = self._mol_from_case(case)
                do_random = case.rdkit_random_seed is not None
                if case.rdkit_random_seed is not None:
                    rdBase.SeedRandomNumberGenerator(case.rdkit_random_seed)

                rdkit_output = Chem.MolToSmiles(
                    Chem.Mol(mol),
                    **rdkit_mol_to_smiles_kwargs_from_options(
                        rooted_at_atom=case.rooted_at_atom,
                        isomeric_smiles=case.isomeric_smiles,
                        canonical=case.rdkit_canonical,
                        do_random=do_random,
                    ),
                )
                self.assertEqual(case.expected, rdkit_output)
                if not case.check_grimace_support:
                    continue

                try:
                    support = grimace_support(
                        mol,
                        rooted_at_atom=case.rooted_at_atom,
                        isomeric_smiles=case.isomeric_smiles,
                    )
                except Exception as exc:
                    self.fail(
                        f"{case.case_id}: Grimace support enumeration failed with "
                        f"{type(exc).__name__}: {exc}"
                    )
                self.assertTrue(
                    case.expected in support,
                    f"{case.case_id}: missing RDKit output {case.expected!r} "
                    f"from Grimace support of size {len(support)}",
                )


if __name__ == "__main__":
    unittest.main()
