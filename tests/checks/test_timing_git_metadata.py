from __future__ import annotations

import subprocess
import sys
import unittest

from scripts import timing_git_metadata


TIMING_OUTPUT_IGNORES = (
    "docs/timings-enum.tsv",
    "docs/timings-enum.md",
    "notes/004_perf_history.jsonl",
    "docs/timings-enum-plots",
    "docs/timings-prepared-mol-zstd*.tsv",
    "docs/timings-prepared-mol-zstd.md",
    "docs/timings-prepared-mol-zstd-plots",
)


class TimingGitMetadataTests(unittest.TestCase):
    def test_generated_timing_outputs_do_not_make_metadata_dirty(self) -> None:
        records = timing_git_metadata.parse_status_records(
            b" M docs/timings-enum.tsv\0"
            b" M docs/timings-enum-plots/non-stereo-01.png\0"
            b" M docs/timings-prepared-mol-zstd-20260531_ebdfcd5d.tsv\0"
            b"?? docs/timings-prepared-mol-zstd-plots/20260531_40762836/new.png\0",
        )

        self.assertFalse(
            timing_git_metadata.has_unignored_status(
                records,
                ignore_patterns=TIMING_OUTPUT_IGNORES,
            ),
        )

    def test_source_changes_still_make_metadata_dirty(self) -> None:
        records = timing_git_metadata.parse_status_records(
            b" M docs/timings-enum.tsv\0"
            b" M scripts/timings_enum_measure.py\0",
        )

        self.assertTrue(
            timing_git_metadata.has_unignored_status(
                records,
                ignore_patterns=TIMING_OUTPUT_IGNORES,
            ),
        )

    def test_renames_are_dirty_if_either_side_is_not_generated_output(self) -> None:
        records = timing_git_metadata.parse_status_records(
            b"R  docs/timings-enum.tsv\0scripts/timing_input.py\0",
        )

        self.assertTrue(
            timing_git_metadata.has_unignored_status(
                records,
                ignore_patterns=TIMING_OUTPUT_IGNORES,
            ),
        )

    def test_help_loads_without_dependencies(self) -> None:
        proc = subprocess.run(
            [sys.executable, "-I", "scripts/timing_git_metadata.py", "--help"],
            capture_output=True,
            text=True,
            check=False,
        )

        self.assertEqual(proc.returncode, 0, msg=proc.stderr or proc.stdout)
        self.assertIn("Emit shell exports for timing metadata", proc.stdout)


if __name__ == "__main__":
    unittest.main()
