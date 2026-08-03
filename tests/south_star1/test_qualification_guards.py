from __future__ import annotations

import ast
from pathlib import Path
import unittest
from unittest.mock import patch

from tests.south_star1 import qualification_guards as guards


class QualificationGuardsTest(unittest.TestCase):
    def test_profiles_are_named_and_nonempty(self) -> None:
        self.assertTrue(guards.QUALIFICATION_GUARD_PROFILES)
        self.assertTrue(all(profile for profile in guards.QUALIFICATION_GUARD_PROFILES.values()))
        guards.validate_qualification_guard_registry()

    def test_guard_reports_zero_calls_when_paths_are_unused(self) -> None:
        with guards.forbid_qualification_profile("public-runtime") as report:
            pass
        report.assert_unused(self)
        self.assertTrue(all(count == 0 for count in report.call_counts().values()))

    def test_guard_blocks_real_owner_lookup(self) -> None:
        with guards.forbid_qualification_paths(guards.QualificationPath.COUNT_DAG_BUILD):
            with self.assertRaisesRegex(AssertionError, "count_dag_build"):
                guards.writer_count_dag_envelope.writer_count_certificate_dag_envelope_for_product(
                    prepared=None, product=None
                )

    def test_guard_targets_are_real_attributes(self) -> None:
        for path, targets in guards._TARGETS.items():
            with self.subTest(path=path):
                for owner, name in targets:
                    self.assertTrue(hasattr(owner, name), (path, owner, name))

    def test_profiles_have_stable_order_and_are_exercised(self) -> None:
        self.assertEqual(
            tuple(guards.QUALIFICATION_GUARD_PROFILES),
            (
                "public-build-without-legacy-materialization",
                "continuation-asset-build-without-legacy-materialization",
                "public-runtime",
                "public-recertification",
                "public-proofs",
                "cached-continuation-verification",
                "slow-rich-artifact-build",
                "slow-rich-artifact-live",
                "slow-stereo-audit",
                "stereo-audit-fast",
            ),
        )

    def test_consumers_do_not_patch_registry_targets_directly(self) -> None:
        root = Path(__file__).parent
        consumers = (
            "test_public_continuation_asset.py",
            "test_public_continuation_asset_verification.py",
            "test_public_continuation_proofs.py",
            "test_writer_default_continuation_corpus.py",
            "test_writer_default_stereo_audit_fixture.py",
            "test_slow_support_artifact_qualification.py",
        )
        target_names = {
            name
            for targets in guards._TARGETS.values()
            for _owner, name in targets
        }
        for filename in consumers:
            tree = ast.parse((root / filename).read_text(), filename)
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                target = None
                if isinstance(node.func, ast.Name) and node.func.id == "patch":
                    if node.args and isinstance(node.args[0], ast.Constant):
                        target = node.args[0].value.rsplit(".", 1)[-1]
                elif (
                    isinstance(node.func, ast.Attribute)
                    and node.func.attr == "object"
                    and isinstance(node.func.value, ast.Name)
                    and node.func.value.id == "patch"
                    and len(node.args) >= 2
                    and isinstance(node.args[1], ast.Constant)
                ):
                    target = node.args[1].value
                if target is not None:
                    self.assertNotIn(target, target_names, (filename, node.lineno))


if __name__ == "__main__":
    unittest.main()
