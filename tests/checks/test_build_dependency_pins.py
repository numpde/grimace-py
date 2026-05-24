from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
MATURIN_VERSION = "1.13.1"
MATURIN_ACTION_VERSION = f"v{MATURIN_VERSION}"


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class BuildDependencyPinTests(unittest.TestCase):
    def test_pyproject_build_backend_uses_pinned_maturin(self) -> None:
        pyproject = tomllib.loads(read_text("pyproject.toml"))
        self.assertEqual(
            [f"maturin=={MATURIN_VERSION}"],
            pyproject["build-system"]["requires"],
        )

    def test_container_and_release_lanes_use_same_maturin_pin(self) -> None:
        checked_files = (
            ".github/workflows/release.yml",
            "containers/package/Dockerfile",
            "containers/perf/Dockerfile",
            "containers/test/Dockerfile",
        )
        for relative_path in checked_files:
            text = read_text(relative_path)
            with self.subTest(path=relative_path):
                if relative_path.endswith(".yml"):
                    self.assertIn(
                        f'MATURIN_PIP_VERSION: "{MATURIN_VERSION}"',
                        text,
                    )
                    self.assertIn(
                        f'MATURIN_ACTION_VERSION: "{MATURIN_ACTION_VERSION}"',
                        text,
                    )
                else:
                    self.assertRegex(text, rf"\bmaturin=={re.escape(MATURIN_VERSION)}\b")


if __name__ == "__main__":
    unittest.main()
