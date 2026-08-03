from __future__ import annotations

import ast
from pathlib import Path
import unittest

from tests.south_star1.default_writer_capability_ledger import ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
from tests.south_star1.qualification_support import support_image_for_case, support_strings_digest


class QualificationSupportTest(unittest.TestCase):
    def test_pinned_support_digests(self) -> None:
        expected = {
            case.name: case.expected_support_digest
            for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES
            if case.name in {"ethanol", "zero_h_tetrahedral", "remote_coupled_tetrahedral_a"}
        }
        for case in ACCEPTED_DEFAULT_WRITER_CAPABILITY_CASES:
            if case.name not in expected:
                continue
            with self.subTest(case=case.name):
                image = support_image_for_case(case)
                self.assertEqual(support_strings_digest(tuple(sorted(image.strings))), expected[case.name])

    def test_qualification_cluster_has_no_test_to_test_private_imports(self) -> None:
        root = Path(__file__).parent
        names = {
            "test_public_continuation_asset.py",
            "test_public_continuation_asset_verification.py",
            "test_public_continuation_proofs.py",
            "test_writer_default_continuation_corpus.py",
            "test_writer_default_offline_complete.py",
            "test_writer_default_parity_corpus.py",
            "test_writer_count_dag_envelope.py",
            "test_slow_support_artifact_qualification.py",
            "test_writer_default_stereo_audit_fixture.py",
            "test_slow_qualification_assets.py",
            "test_continuation_qualification_contract.py",
        }
        for name in names:
            tree = ast.parse((root / name).read_text(), filename=name)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tests.south_star1.test_"):
                    self.fail(f"cross-test import remains in {name}: {node.module}")


if __name__ == "__main__":
    unittest.main()
