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
PREPARED_MOL_ZSTD_PLOT_SCRIPT = (
    ROOT / "scripts" / "timings_prepared_mol_zstd_plot.py"
)


def load_script_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise AssertionError(f"Could not load {name} module spec")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_prepared_mol_zstd_timing_module() -> ModuleType:
    return load_script_module(
        "timings_prepared_mol_zstd_measure",
        PREPARED_MOL_ZSTD_TIMING_SCRIPT,
    )


def load_prepared_mol_zstd_plot_module() -> ModuleType:
    return load_script_module(
        "timings_prepared_mol_zstd_plot",
        PREPARED_MOL_ZSTD_PLOT_SCRIPT,
    )


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
                        "artifact_dir": "other",
                        "files": {
                            "dictionary": "default_v1.zstdict",
                            "manifest": "default_v1.json",
                        },
                        "zstd_dictionary_id": 123_456,
                        "zstd_dictionary_sha256": "a" * 64,
                    },
                ),
                "unexpected artifact",
            ),
            (
                json.dumps(
                    {
                        "artifact_dir": "__ARTIFACT__",
                        "files": {
                            "dictionary": "../default_v1.zstdict",
                            "manifest": "default_v1.json",
                        },
                        "zstd_dictionary_id": 123_456,
                        "zstd_dictionary_sha256": "a" * 64,
                    },
                ),
                "unexpected dictionary file",
            ),
            (
                json.dumps(
                    {
                        "artifact_dir": "__ARTIFACT__",
                        "files": {
                            "dictionary": "default_v1.zstdict",
                            "manifest": "../default_v1.json",
                        },
                        "zstd_dictionary_id": 123_456,
                        "zstd_dictionary_sha256": "a" * 64,
                    },
                ),
                "unexpected manifest file",
            ),
            (
                json.dumps(
                    {
                        "artifact_dir": "__ARTIFACT__",
                        "files": {
                            "dictionary": "default_v1.zstdict",
                            "manifest": "default_v1.json",
                        },
                        "zstd_dictionary_id": True,
                        "zstd_dictionary_sha256": "a" * 64,
                    },
                ),
                "invalid dictionary id",
            ),
            (
                json.dumps(
                    {
                        "artifact_dir": "__ARTIFACT__",
                        "files": {
                            "dictionary": "default_v1.zstdict",
                            "manifest": "default_v1.json",
                        },
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
                        payload.replace("__ARTIFACT__", artifact_dir.name),
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
                        "artifact_dir": artifact_dir.name,
                        "files": {
                            "dictionary": "default_v1.zstdict",
                            "manifest": "default_v1.json",
                        },
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

    def test_prepared_mol_zstd_timing_dictionary_bytes_match_manifest_hash(
        self,
    ) -> None:
        timing = load_prepared_mol_zstd_timing_module()
        with tempfile.TemporaryDirectory() as tmpdir:
            artifact_dir = Path(tmpdir)
            artifact_dir.joinpath("default_v1.zstdict").write_bytes(b"dictionary")
            manifest = timing.DictionaryManifest(
                dictionary_file="default_v1.zstdict",
                dictionary_id=123_456,
                dictionary_sha256=timing.generator.sha256_hex(b"dictionary"),
            )

            self.assertEqual(
                b"dictionary",
                timing._dictionary_bytes(artifact_dir, manifest),
            )

            bad_manifest = timing.DictionaryManifest(
                dictionary_file="default_v1.zstdict",
                dictionary_id=123_456,
                dictionary_sha256="0" * 64,
            )
            with self.assertRaisesRegex(RuntimeError, "SHA-256"):
                timing._dictionary_bytes(artifact_dir, bad_manifest)

    def test_prepared_mol_zstd_plot_input_rejects_malformed_tsv(self) -> None:
        plot = load_prepared_mol_zstd_plot_module()
        header = tuple(plot.REQUIRED_FIELDS)
        valid_row = {field: "1" for field in header}
        valid_row["mode"] = "dictionary"
        cases = (
            (
                ("mode",),
                ("dictionary",),
                "lacks required field",
            ),
            (
                header,
                tuple(
                    "not-a-mode" if field == "mode" else valid_row[field]
                    for field in header
                ),
                "unknown mode",
            ),
            (
                header,
                tuple(valid_row[field] for field in header) + ("extra",),
                "too many columns",
            ),
        )
        for fields, values, message in cases:
            with self.subTest(message=message):
                with tempfile.TemporaryDirectory() as tmpdir:
                    input_path = Path(tmpdir) / "timings.tsv"
                    input_path.write_text(
                        "\t".join(fields)
                        + "\n"
                        + "\t".join(values)
                        + "\n",
                        encoding="utf-8",
                    )

                    with self.assertRaisesRegex(SystemExit, message):
                        plot._read_rows(input_path)


if __name__ == "__main__":
    unittest.main()
