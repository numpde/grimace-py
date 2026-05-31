from pathlib import Path
import re
import tomllib
import unittest


ROOT = Path(__file__).resolve().parents[2]
MATURIN_VERSION = "1.13.1"
MATURIN_ACTION_VERSION = f"v{MATURIN_VERSION}"
RDKIT_VERSION = "2026.3.1"
TWINE_VERSION = "6.2.0"
ZSTANDARD_VERSION = "0.25.0"


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def pinned_constraints() -> dict[str, str]:
    constraints: dict[str, str] = {}
    for line in constraint_lines():
        name, version = line.split("==", 1)
        constraints[name.lower()] = version
    return constraints


def constraint_lines() -> tuple[str, ...]:
    lines: list[str] = []
    for line in read_text("requirements/container-build-constraints.txt").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        lines.append(stripped)
    return tuple(lines)


class BuildDependencyPinTests(unittest.TestCase):
    def test_pyproject_build_backend_uses_pinned_maturin(self) -> None:
        pyproject = tomllib.loads(read_text("pyproject.toml"))
        self.assertEqual(
            [f"maturin=={MATURIN_VERSION}"],
            pyproject["build-system"]["requires"],
        )

    def test_pyproject_is_authoritative_for_locked_maturin_builds(self) -> None:
        pyproject = tomllib.loads(read_text("pyproject.toml"))
        self.assertIs(True, pyproject["tool"]["maturin"]["locked"])

        checked_files = (
            ".github/workflows/release.yml",
            "compose/package.yml",
            "containers/perf/Dockerfile",
            "containers/test/Dockerfile",
        )
        for relative_path in checked_files:
            with self.subTest(path=relative_path):
                self.assertNotRegex(read_text(relative_path), r"maturin build\b[^\n]*--locked")

    def test_container_and_release_lanes_use_same_direct_pins(self) -> None:
        constraints = pinned_constraints()
        self.assertEqual(MATURIN_VERSION, constraints["maturin"])
        self.assertEqual(RDKIT_VERSION, constraints["rdkit"])
        self.assertEqual(TWINE_VERSION, constraints["twine"])
        self.assertEqual(ZSTANDARD_VERSION, constraints["zstandard"])

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
                    self.assertIn(
                        f'RDKIT_FIXTURE_PIP_VERSION: "{RDKIT_VERSION}"',
                        text,
                    )
                else:
                    self.assertRegex(text, rf"\bmaturin=={re.escape(MATURIN_VERSION)}\b")
                    self.assertRegex(text, rf"\brdkit=={re.escape(RDKIT_VERSION)}\b")
                    if relative_path == "containers/package/Dockerfile":
                        self.assertRegex(text, rf"\btwine=={re.escape(TWINE_VERSION)}\b")

    def test_container_constraints_pin_direct_fixture_tools(self) -> None:
        constraints = pinned_constraints()
        self.assertEqual(RDKIT_VERSION, constraints["rdkit"])
        self.assertEqual(TWINE_VERSION, constraints["twine"])
        self.assertEqual(ZSTANDARD_VERSION, constraints["zstandard"])

    def test_dev_dependencies_include_dictionary_generator_tool(self) -> None:
        pyproject = tomllib.loads(read_text("pyproject.toml"))
        self.assertIn(
            f"zstandard=={ZSTANDARD_VERSION}",
            pyproject["dependency-groups"]["dev"],
        )

    def test_container_constraints_are_exact_and_sorted(self) -> None:
        lines = constraint_lines()
        names = tuple(line.split("==", 1)[0] for line in lines)
        self.assertTrue(lines)
        self.assertEqual(tuple(sorted(names)), names)
        for line in lines:
            with self.subTest(line=line):
                self.assertRegex(line, r"^[a-z0-9_.-]+==[A-Za-z0-9_.!+-]+$")


if __name__ == "__main__":
    unittest.main()
