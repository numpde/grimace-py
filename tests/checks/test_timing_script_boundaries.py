from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType
import unittest


ROOT = Path(__file__).resolve().parents[2]
PREPARED_MOL_ZSTD_TIMING_SCRIPT = (
    ROOT / "scripts" / "timings_prepared_mol_zstd_measure.py"
)


def load_prepared_mol_zstd_timing_module() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "timings_prepared_mol_zstd_measure",
        PREPARED_MOL_ZSTD_TIMING_SCRIPT,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("Could not load PreparedMol zstd timing module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


class TimingScriptBoundaryTests(unittest.TestCase):
    def test_timing_scripts_do_not_import_tests(self) -> None:
        scripts = (
            ROOT / "scripts" / "timings_enum_measure.py",
            ROOT / "scripts" / "timings_prepared_mol_zstd_measure.py",
            ROOT / "scripts" / "timings_prepared_mol_zstd_plot.py",
            ROOT / "scripts" / "timing_git_metadata.py",
            ROOT / "scripts" / "record_perf_hotspots.py",
        )
        offenders: list[str] = []
        for script in scripts:
            text = script.read_text(encoding="utf-8")
            if "from tests." in text or "import tests." in text:
                offenders.append(str(script.relative_to(ROOT)))

        self.assertEqual([], offenders)

    def test_prepared_mol_zstd_timing_manifest_rejects_malformed_inputs(self) -> None:
        timing = load_prepared_mol_zstd_timing_module()
        cases = (
            ("{", "not readable JSON"),
            ("[]", "not a JSON object"),
            ("{}", "lacks timing metadata"),
            (
                json.dumps(
                    {
                        "files": {"dictionary": "../default_v1.zstdict"},
                        "zstd_dictionary_id": 123_456,
                        "zstd_dictionary_sha256": "a" * 64,
                    },
                ),
                "unexpected dictionary file",
            ),
            (
                json.dumps(
                    {
                        "files": {"dictionary": "default_v1.zstdict"},
                        "zstd_dictionary_id": True,
                        "zstd_dictionary_sha256": "a" * 64,
                    },
                ),
                "invalid dictionary id",
            ),
            (
                json.dumps(
                    {
                        "files": {"dictionary": "default_v1.zstdict"},
                        "zstd_dictionary_id": 123_456,
                        "zstd_dictionary_sha256": "abc",
                    },
                ),
                "invalid dictionary SHA-256",
            ),
        )
        for payload, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmpdir:
                    artifact_dir = Path(tmpdir)
                    artifact_dir.joinpath("default_v1.json").write_text(
                        payload,
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(RuntimeError, message):
                        timing._dictionary_manifest(artifact_dir)

    def test_prepared_mol_zstd_timing_manifest_reads_expected_metadata(self) -> None:
        timing = load_prepared_mol_zstd_timing_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            artifact_dir.joinpath("default_v1.json").write_text(
                json.dumps(
                    {
                        "files": {"dictionary": "default_v1.zstdict"},
                        "zstd_dictionary_id": 123_456,
                        "zstd_dictionary_sha256": "a" * 64,
                    },
                ),
                encoding="utf-8",
            )

            manifest = timing._dictionary_manifest(artifact_dir)

        self.assertEqual("default_v1.zstdict", manifest.dictionary_file)
        self.assertEqual(123_456, manifest.dictionary_id)
        self.assertEqual("a" * 64, manifest.dictionary_sha256)


if __name__ == "__main__":
    unittest.main()
