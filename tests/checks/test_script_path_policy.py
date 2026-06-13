from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "_path_policy.py"


def load_path_policy_module():
    spec = importlib.util.spec_from_file_location("_path_policy", SCRIPT)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"failed to load {SCRIPT}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


PATH_POLICY = load_path_policy_module()


class ScriptPathPolicyTests(unittest.TestCase):
    def test_accepts_output_under_approved_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            approved_root = Path(tmpdir) / "approved"
            output = approved_root / "nested" / "report.json"

            self.assertEqual(
                output.resolve(strict=False),
                PATH_POLICY.checked_output_path(
                    output,
                    approved_roots=(approved_root,),
                    allow_outside_repo=False,
                    force=False,
                ),
            )

    def test_rejects_parent_traversal_out_of_approved_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            approved_root = Path(tmpdir) / "approved"
            output = approved_root / ".." / "report.json"

            with self.assertRaisesRegex(ValueError, "output path must be under"):
                PATH_POLICY.checked_output_path(
                    output,
                    approved_roots=(approved_root,),
                    allow_outside_repo=False,
                    force=False,
                )

    def test_rejects_symlink_escape_from_approved_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            approved_root = Path(tmpdir) / "approved"
            outside_root = Path(tmpdir) / "outside"
            approved_root.mkdir()
            outside_root.mkdir()
            symlink = approved_root / "link"
            symlink.symlink_to(outside_root, target_is_directory=True)

            with self.assertRaisesRegex(ValueError, "output path must be under"):
                PATH_POLICY.checked_output_path(
                    symlink / "report.json",
                    approved_roots=(approved_root,),
                    allow_outside_repo=False,
                    force=False,
                )

    def test_allows_outside_output_only_when_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            approved_root = Path(tmpdir) / "approved"
            output = Path(tmpdir) / "scratch" / "report.json"

            self.assertEqual(
                output.resolve(strict=False),
                PATH_POLICY.checked_output_path(
                    output,
                    approved_roots=(approved_root,),
                    allow_outside_repo=True,
                    force=False,
                ),
            )

    def test_requires_force_for_existing_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            approved_root = Path(tmpdir) / "approved"
            approved_root.mkdir()
            output = approved_root / "report.json"
            output.write_text("{}\n", encoding="utf-8")

            with self.assertRaisesRegex(FileExistsError, "already exists"):
                PATH_POLICY.checked_output_path(
                    output,
                    approved_roots=(approved_root,),
                    allow_outside_repo=False,
                    force=False,
                )
            self.assertEqual(
                output.resolve(strict=False),
                PATH_POLICY.checked_output_path(
                    output,
                    approved_roots=(approved_root,),
                    allow_outside_repo=False,
                    force=True,
                ),
            )


if __name__ == "__main__":
    unittest.main()
