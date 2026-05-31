from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[2]


class TimingScriptBoundaryTests(unittest.TestCase):
    def test_timing_scripts_do_not_import_tests(self) -> None:
        scripts = (
            ROOT / "scripts" / "timings_enum_measure.py",
            ROOT / "scripts" / "timings_prepared_mol_zstd_measure.py",
            ROOT / "scripts" / "timings_prepared_mol_zstd_plot.py",
            ROOT / "scripts" / "record_perf_hotspots.py",
        )
        offenders: list[str] = []
        for script in scripts:
            text = script.read_text(encoding="utf-8")
            if "from tests." in text or "import tests." in text:
                offenders.append(str(script.relative_to(ROOT)))

        self.assertEqual([], offenders)


if __name__ == "__main__":
    unittest.main()
