from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


class ContainerPostureTests(unittest.TestCase):
    def test_compose_files_do_not_set_project_name(self) -> None:
        compose_files = sorted((ROOT / "compose").glob("*.yml"))
        self.assertTrue(compose_files)
        for compose_file in compose_files:
            with self.subTest(compose_file=compose_file.name):
                self.assertNotRegex(
                    compose_file.read_text(encoding="utf-8"),
                    r"(?m)^name\s*:",
                )

    def test_checks_compose_has_strict_runtime_posture(self) -> None:
        compose = read_text("compose/checks.yml")
        self.assertRegex(compose, r'(?m)^\s+user:\s+"65532:65532"\s*$')
        self.assertRegex(compose, r'(?m)^\s+network_mode:\s+"none"\s*$')
        self.assertRegex(compose, r"(?m)^\s+read_only:\s+true\s*$")
        self.assertRegex(compose, r"(?ms)^\s+cap_drop:\n\s+- ALL\s*$")
        self.assertRegex(
            compose,
            r"(?ms)^\s+security_opt:\n\s+- no-new-privileges:true\s*$",
        )
        self.assertRegex(compose, r"(?m)^\s+pids_limit:\s+64\s*$")
        self.assertRegex(compose, r"(?m)^\s+mem_limit:\s+128m\s*$")
        self.assertRegex(compose, r"(?ms)^\s+tmpfs:\n\s+- /tmp:")
        self.assertRegex(compose, r"(?ms)^\s+volumes:\n\s+- type: bind")
        self.assertRegex(compose, r"(?m)^\s+target:\s+/work\s*$")
        self.assertGreaterEqual(compose.count("read_only: true"), 2)
        self.assertNotIn("docker.sock", compose)
        self.assertNotIn("privileged: true", compose)

    def test_test_compose_has_strict_copied_context_posture(self) -> None:
        compose = read_text("compose/test.yml")
        for service in ("rust", "test", "parity", "exact-public-invariants"):
            with self.subTest(service=service):
                self.assertRegex(compose, rf"(?m)^  {service}:$")
        self.assertIn("dockerfile: containers/test/Dockerfile", compose)
        self.assertRegex(compose, r'(?m)^  user:\s+"65532:65532"\s*$')
        self.assertRegex(compose, r'(?m)^  network_mode:\s+"none"\s*$')
        self.assertRegex(compose, r"(?m)^  read_only:\s+true\s*$")
        self.assertRegex(compose, r"(?ms)^  cap_drop:\n    - ALL\s*$")
        self.assertRegex(
            compose,
            r"(?ms)^  security_opt:\n    - no-new-privileges:true\s*$",
        )
        self.assertNotIn("volumes:", compose)
        self.assertNotIn(".venv", compose)
        self.assertNotIn("docker.sock", compose)
        self.assertNotIn("privileged: true", compose)

    def test_checks_dockerfile_is_pinned_and_does_not_embed_repo(self) -> None:
        dockerfile = read_text("containers/checks/Dockerfile")
        self.assertNotRegex(dockerfile, r"(?m)^(COPY|ADD|RUN)\b")
        self.assertIn("USER 65532:65532", dockerfile)
        self.assertIn('ENTRYPOINT ["python"]', dockerfile)

    def test_all_container_base_images_are_digest_pinned(self) -> None:
        dockerfiles = sorted((ROOT / "containers").glob("*/Dockerfile"))
        self.assertTrue(dockerfiles)
        for dockerfile_path in dockerfiles:
            with self.subTest(dockerfile=dockerfile_path):
                dockerfile = dockerfile_path.read_text(encoding="utf-8")
                from_lines = re.findall(r"(?m)^FROM\s+(.+)$", dockerfile)
                self.assertTrue(from_lines)
                for from_line in from_lines:
                    image = from_line.split(" AS ", 1)[0]
                    self.assertRegex(image, r"@sha256:[0-9a-f]{64}$")

    def test_test_dockerfile_builds_installed_package_image(self) -> None:
        dockerfile = read_text("containers/test/Dockerfile")
        self.assertIn("rust:1.83.0-slim-bookworm@", dockerfile)
        self.assertIn("python:3.12.13-slim-bookworm@", dockerfile)
        self.assertIn("maturin==1.13.1", dockerfile)
        self.assertIn("rdkit==2026.3.1", dockerfile)
        self.assertIn("COPY . /src", dockerfile)
        self.assertNotIn("apt-get", dockerfile)
        self.assertIn("python -m maturin build --release", dockerfile)
        self.assertIn(
            "python -m pip install --no-cache-dir /tmp/grimace-dist/*.whl",
            dockerfile,
        )
        self.assertIn("tests.run_installed_package_correctness", dockerfile)
        self.assertIn("USER 65532:65532", dockerfile)

    def test_makefile_exposes_guarded_checks_lane(self) -> None:
        makefile = read_text("Makefile")
        self.assertIn("SHELL := bash", makefile)
        self.assertIn(".SHELLFLAGS := -eu -o pipefail -c", makefile)
        self.assertIn("DOCKER_COMPOSE ?= docker compose", makefile)
        self.assertIn("ACTUAL_UID := $(shell id -u)", makefile)
        self.assertIn("Refusing to run Docker lanes as root", makefile)
        expected_targets = {
            "checks": "checks.yml,checks",
            "rust": "test.yml,rust",
            "test": "test.yml,test",
            "parity": "test.yml,parity",
            "exact-public-invariants": "test.yml,exact-public-invariants",
        }
        for target, compose_call in expected_targets.items():
            with self.subTest(target=target):
                self.assertRegex(
                    makefile,
                    rf"(?m)^{target}:\n\t\$\(call compose_run,{compose_call}\)",
                )
        self.assertRegex(
            makefile,
            r"(?m)^ci: checks rust test parity exact-public-invariants$",
        )

    def test_dockerignore_excludes_local_and_generated_paths(self) -> None:
        patterns = {
            line.strip()
            for line in read_text(".dockerignore").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        required_patterns = {
            ".git",
            ".codex",
            ".agents",
            ".idea",
            ".vscode",
            ".venv",
            ".env",
            ".env.*",
            "dist",
            "build",
            "target",
            "pip-wheel-metadata",
            "*.egg-info",
            "**/*.egg-info",
            "__pycache__",
            "**/__pycache__",
            "*.py[cod]",
            "**/*.py[cod]",
            ".pytest_cache",
            "**/.pytest_cache",
            ".ruff_cache",
            "**/.ruff_cache",
            ".mypy_cache",
            "**/.mypy_cache",
            ".coverage",
            ".coverage.*",
            "htmlcov",
            "**/htmlcov",
            "python/grimace/_core*.so",
            "python/grimace/_core*.dylib",
            "python/grimace/_core*.dll",
            "python/grimace/_core*.pyd",
        }
        self.assertFalse(required_patterns - patterns)
        self.assertNotIn("Cargo.lock", patterns)
        self.assertNotIn("pyproject.toml", patterns)
        self.assertNotIn("Cargo.toml", patterns)
        self.assertNotIn("rust-toolchain.toml", patterns)


if __name__ == "__main__":
    unittest.main()
