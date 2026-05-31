from __future__ import annotations

import os
import unittest

from scripts.timings_enum_measure import EnumTimingBenchmark


@unittest.skipUnless(
    os.environ.get("RUN_PERF_TESTS") == "1",
    "set RUN_PERF_TESTS=1 to run performance checks",
)
class ReadmeTimingPerfTests(unittest.TestCase):
    def test_generate_readme_timing_table(self) -> None:
        benchmark = EnumTimingBenchmark()
        benchmark.run()

        self.assertTrue(benchmark.OUTPUT_TSV_PATH.is_file())
        self.assertTrue(benchmark.OUTPUT_MD_PATH.is_file())
        self.assertTrue(benchmark.OUTPUT_PLOT_DIR.is_dir())


if __name__ == "__main__":
    unittest.main()
