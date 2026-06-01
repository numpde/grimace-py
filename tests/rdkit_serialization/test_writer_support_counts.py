from __future__ import annotations

import unittest

from rdkit import rdBase

from tests.helpers.mols import parse_smiles
from tests.helpers.public_runtime import public_enum_support
from tests.helpers.rdkit_writer_support_counts import (
    load_pinned_writer_support_count_cases,
)


class RDKITWriterSupportCountTests(unittest.TestCase):
    """Pinned RDKit random-writer support-count evidence."""

    @classmethod
    def setUpClass(cls) -> None:
        try:
            cls.cases = load_pinned_writer_support_count_cases(rdBase.rdkitVersion)
        except FileNotFoundError:
            raise unittest.SkipTest(
                f"no pinned writer-support-count corpus for RDKit {rdBase.rdkitVersion}"
            )

    def test_grimace_support_cardinality_matches_pinned_count(self) -> None:
        for case in self.cases:
            mol = parse_smiles(case.smiles)
            with self.subTest(
                case_id=case.case_id,
                source=case.source,
                smiles=case.smiles,
                rooted_at_atom=case.rooted_at_atom,
                isomeric_smiles=case.isomeric_smiles,
                support_count=case.support_count,
            ):
                self.assertEqual(
                    case.support_count,
                    len(public_enum_support(mol, **case.public_kwargs())),
                )


if __name__ == "__main__":
    unittest.main()
