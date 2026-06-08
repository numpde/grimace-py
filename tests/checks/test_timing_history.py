from __future__ import annotations

import contextlib
from collections.abc import Iterator
from pathlib import Path
import tempfile
import unittest

from scripts import timing_history


@contextlib.contextmanager
def use_history_path(path: Path) -> Iterator[None]:
    original_path = timing_history.HISTORY_PATH
    timing_history.HISTORY_PATH = path
    try:
        yield
    finally:
        timing_history.HISTORY_PATH = original_path


class TimingHistoryTests(unittest.TestCase):
    def test_latest_history_record_returns_newest_matching_kind(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.jsonl"
            history_path.write_text(
                "\n".join(
                    (
                        '{"kind": "enum", "value": 1}',
                        '{"kind": "other", "value": 2}',
                        '{"kind": "enum", "value": 3}',
                    )
                )
                + "\n",
                encoding="utf-8",
            )
            with use_history_path(history_path):
                self.assertEqual(
                    {"kind": "enum", "value": 3},
                    timing_history.latest_history_record("enum"),
                )
                self.assertIsNone(timing_history.latest_history_record("missing"))

    def test_latest_history_record_rejects_malformed_jsonl(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.jsonl"
            history_path.write_text('{"kind": "enum"}\n{\n', encoding="utf-8")
            with use_history_path(history_path):
                with self.assertRaisesRegex(ValueError, r"history\.jsonl:2"):
                    timing_history.latest_history_record("enum")

    def test_latest_history_record_rejects_non_object_record(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            history_path = Path(tmpdir) / "history.jsonl"
            history_path.write_text('{"kind": "enum"}\n[]\n', encoding="utf-8")
            with use_history_path(history_path):
                with self.assertRaisesRegex(ValueError, r"history\.jsonl:2"):
                    timing_history.latest_history_record("enum")


if __name__ == "__main__":
    unittest.main()
