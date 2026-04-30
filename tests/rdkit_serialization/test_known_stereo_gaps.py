from __future__ import annotations

from dataclasses import dataclass
import json
import unittest

from rdkit import Chem, rdBase

from tests.helpers.fixture_paths import checked_in_fixture_path
from tests.rdkit_serialization._support import (
    grimace_support,
    rdkit_mol_to_smiles_kwargs_from_options,
)


FIXTURE_ROOT = checked_in_fixture_path("rdkit_known_stereo_gaps")


@dataclass(frozen=True, slots=True)
class KnownStereoGapCase:
    case_id: str
    source: str
    smiles: str
    expected: str
    rooted_at_atom: int | None
    isomeric_smiles: bool
    rdkit_canonical: bool


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
        cases.append(
            KnownStereoGapCase(
                case_id=raw_case["id"],
                source=raw_case["source"],
                smiles=raw_case["smiles"],
                expected=raw_case["expected"],
                rooted_at_atom=raw_case["rooted_at_atom"],
                isomeric_smiles=raw_case["isomeric_smiles"],
                rdkit_canonical=raw_case["rdkit_canonical"],
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
        except FileNotFoundError:
            raise unittest.SkipTest(
                f"no pinned known stereo-gap corpus for RDKit {rdBase.rdkitVersion}"
            )

    def test_rdkit_deterministic_coupled_stereo_outputs_are_in_grimace_support(self) -> None:
        for case in self.cases:
            with self.subTest(case_id=case.case_id, source=case.source):
                mol = Chem.MolFromSmiles(case.smiles)
                self.assertIsNotNone(mol, case.smiles)
                assert mol is not None

                rdkit_output = Chem.MolToSmiles(
                    Chem.Mol(mol),
                    **rdkit_mol_to_smiles_kwargs_from_options(
                        rooted_at_atom=case.rooted_at_atom,
                        isomeric_smiles=case.isomeric_smiles,
                        canonical=case.rdkit_canonical,
                        do_random=False,
                    ),
                )
                self.assertEqual(case.expected, rdkit_output)

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
