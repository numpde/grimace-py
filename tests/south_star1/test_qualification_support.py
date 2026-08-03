from __future__ import annotations

import ast
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rdkit import rdBase

from tests.helpers.rdkit_south_star_stereo_audit import load_pinned_south_star_stereo_audit_cases
from tests.south_star1.default_writer_capability_ledger import (
    default_writer_capability_case,
    default_writer_cases_for_rdkit_audit,
)
from tests.south_star1.qualification_plan import PUBLIC_PROOF_SHARD_COUNT
from tests.south_star1.qualification_support import (
    PublicProofCursorTargets,
    partition_public_proof_targets,
    support_image_for_case,
    support_strings_digest,
)


class QualificationSupportTest(unittest.TestCase):
    def test_public_proof_partition_is_deterministic_and_disjoint(self) -> None:
        groups = tuple(
            PublicProofCursorTargets(
                source_raw_cursor_digest=f"cursor-{index}",
                state=None,
                branch_locators=tuple(
                    SimpleNamespace(
                        source_raw_cursor_digest=f"cursor-{index}",
                        emitted_text=f"text-{offset}",
                        branch_certificate_digest=f"branch-{index}-{offset}",
                    )
                    for offset in range(index + 1)
                ),
                terminal_locators=(),
            )
            for index in range(PUBLIC_PROOF_SHARD_COUNT + 3)
        )
        first = partition_public_proof_targets(groups)
        second = partition_public_proof_targets(tuple(reversed(groups)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), PUBLIC_PROOF_SHARD_COUNT)
        flattened = [
            group.source_raw_cursor_digest for shard in first for group in shard
        ]
        self.assertEqual(len(flattened), len(set(flattened)))
        self.assertEqual(
            set(flattened), {group.source_raw_cursor_digest for group in groups}
        )

    def test_duplicate_public_proof_locator_rejects(self) -> None:
        locator = SimpleNamespace(
            source_raw_cursor_digest="cursor",
            emitted_text="C",
            branch_certificate_digest="branch",
        )
        group = PublicProofCursorTargets(
            source_raw_cursor_digest="cursor",
            state=None,
            branch_locators=(locator, locator),
            terminal_locators=(),
        )
        with self.assertRaisesRegex(AssertionError, "duplicate"):
            partition_public_proof_targets((group,))

    def test_pinned_support_digests(self) -> None:
        ethanol = default_writer_capability_case("ethanol")
        self.assertEqual(
            support_strings_digest(tuple(sorted(support_image_for_case(ethanol).strings))),
            ethanol.expected_support_digest,
        )
        fixtures = {
            item.name: item
            for item in load_pinned_south_star_stereo_audit_cases(rdBase.rdkitVersion)
        }
        with patch(
            "tests.south_star1.qualification_support.support_image_for_case",
            side_effect=AssertionError("continuation fixture must not materialize support"),
        ):
            for case in default_writer_cases_for_rdkit_audit("stereo"):
                if case.name not in fixtures:
                    continue
                with self.subTest(case=case.name):
                    self.assertEqual(
                        support_strings_digest(fixtures[case.name].expected_support),
                        case.expected_support_digest,
                    )

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
        }
        for name in names:
            tree = ast.parse((root / name).read_text(), filename=name)
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("tests.south_star1.test_"):
                    self.fail(f"cross-test import remains in {name}: {node.module}")


if __name__ == "__main__":
    unittest.main()
