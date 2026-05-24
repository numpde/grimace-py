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

    def test_compose_files_avoid_host_escape_hatches(self) -> None:
        compose_files = sorted((ROOT / "compose").glob("*.yml"))
        self.assertTrue(compose_files)
        forbidden_patterns = {
            "privileged mode": r"(?m)^\s*privileged:\s*true\s*$",
            "docker socket": r"docker\.sock",
            "host network": r"(?m)^\s*network_mode:\s*[\"']?host[\"']?\s*$",
            "host pid namespace": r"(?m)^\s*pid:\s*[\"']?host[\"']?\s*$",
            "host ipc namespace": r"(?m)^\s*ipc:\s*[\"']?host[\"']?\s*$",
            "device mounts": r"(?m)^\s*devices:\s*$",
            "added capabilities": r"(?m)^\s*cap_add:\s*$",
            "extra groups": r"(?m)^\s*group_add:\s*$",
        }
        for compose_file in compose_files:
            compose = compose_file.read_text(encoding="utf-8")
            for label, pattern in forbidden_patterns.items():
                with self.subTest(compose_file=compose_file.name, forbidden=label):
                    self.assertNotRegex(compose, pattern)

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
        self.assertIn("- --offline\n      - --locked", compose)
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

    def test_package_compose_writes_only_dist(self) -> None:
        compose = read_text("compose/package.yml")
        self.assertRegex(compose, r"(?m)^  package:$")
        self.assertIn("dockerfile: containers/package/Dockerfile", compose)
        self.assertIn('user: "${LOCAL_UID:-65532}:${LOCAL_GID:-65532}"', compose)
        self.assertRegex(compose, r'(?m)^\s+network_mode:\s+"none"\s*$')
        self.assertRegex(compose, r"(?m)^\s+read_only:\s+true\s*$")
        self.assertRegex(compose, r"(?ms)^\s+cap_drop:\n\s+- ALL\s*$")
        self.assertRegex(
            compose,
            r"(?ms)^\s+security_opt:\n\s+- no-new-privileges:true\s*$",
        )
        self.assertIn("source: ../dist", compose)
        self.assertIn("target: /dist", compose)
        self.assertIn("python -m maturin build --release --out /dist", compose)
        self.assertIn(
            "python -m maturin build --release --sdist --out /dist",
            compose,
        )
        self.assertIn(
            "python scripts/validate_release_artifacts.py /dist/*.tar.gz --sdist-only",
            compose,
        )
        self.assertNotIn("source: ..\n", compose)
        self.assertNotIn(".venv", compose)
        self.assertNotIn("docker.sock", compose)
        self.assertNotIn("privileged: true", compose)

    def test_perf_compose_is_explicit_write_enabled_lane(self) -> None:
        compose = read_text("compose/perf.yml")
        self.assertRegex(compose, r"(?m)^  perf:$")
        self.assertIn("dockerfile: containers/perf/Dockerfile", compose)
        self.assertIn('user: "${LOCAL_UID:-65532}:${LOCAL_GID:-65532}"', compose)
        self.assertRegex(compose, r'(?m)^\s+network_mode:\s+"none"\s*$')
        self.assertRegex(compose, r"(?m)^\s+read_only:\s+true\s*$")
        self.assertRegex(compose, r"(?ms)^\s+cap_drop:\n\s+- ALL\s*$")
        self.assertRegex(
            compose,
            r"(?ms)^\s+security_opt:\n\s+- no-new-privileges:true\s*$",
        )
        self.assertIn('RUN_PERF_TESTS: "1"', compose)
        self.assertIn("source: ../docs/timings.tsv", compose)
        self.assertIn("target: /build-src/docs/timings.tsv", compose)
        self.assertIn("source: ../docs/timings.md", compose)
        self.assertIn("target: /build-src/docs/timings.md", compose)
        self.assertIn("source: ../notes/004_perf_history.jsonl", compose)
        self.assertIn("target: /build-src/notes/004_perf_history.jsonl", compose)
        self.assertNotIn("source: ..\n", compose)
        self.assertNotIn("target: /src", compose)
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

    def test_build_dockerfiles_disable_pip_network_notice_and_root_warning(self) -> None:
        for relative_path in (
            "containers/package/Dockerfile",
            "containers/perf/Dockerfile",
            "containers/test/Dockerfile",
        ):
            dockerfile = read_text(relative_path)
            with self.subTest(path=relative_path):
                self.assertIn("PIP_DISABLE_PIP_VERSION_CHECK=1", dockerfile)
                self.assertIn("PIP_NO_CACHE_DIR=1", dockerfile)
                self.assertIn("PIP_ROOT_USER_ACTION=ignore", dockerfile)

    def test_build_dockerfiles_use_pip_constraints(self) -> None:
        for relative_path in (
            "containers/package/Dockerfile",
            "containers/perf/Dockerfile",
            "containers/test/Dockerfile",
        ):
            dockerfile = read_text(relative_path)
            with self.subTest(path=relative_path):
                self.assertIn(
                    "COPY requirements/container-build-constraints.txt /tmp/container-build-constraints.txt",
                    dockerfile,
                )
                self.assertIn(
                    "python -m pip install --constraint /tmp/container-build-constraints.txt",
                    dockerfile,
                )

    def test_test_dockerfile_builds_installed_package_image(self) -> None:
        dockerfile = read_text("containers/test/Dockerfile")
        self.assertIn("rust:1.83.0-slim-bookworm@", dockerfile)
        self.assertIn("python:3.12.13-slim-bookworm@", dockerfile)
        self.assertIn("maturin==1.13.1", dockerfile)
        self.assertIn("rdkit==2026.3.1", dockerfile)
        self.assertIn("COPY . /src", dockerfile)
        self.assertNotIn("apt-get", dockerfile)
        self.assertIn("python -m maturin build --release --out", dockerfile)
        self.assertIn(
            "python -m pip install --no-deps /tmp/grimace-dist/*.whl",
            dockerfile,
        )
        self.assertIn("tests.run_installed_package_correctness", dockerfile)
        self.assertIn("USER 65532:65532", dockerfile)

    def test_package_dockerfile_builds_release_artifact_image(self) -> None:
        dockerfile = read_text("containers/package/Dockerfile")
        self.assertIn("rust:1.83.0-slim-bookworm@", dockerfile)
        self.assertIn("python:3.12.13-slim-bookworm@", dockerfile)
        self.assertIn("maturin==1.13.1", dockerfile)
        self.assertIn("rdkit==2026.3.1", dockerfile)
        self.assertIn("twine==6.2.0", dockerfile)
        self.assertIn("COPY . /src", dockerfile)
        self.assertIn("cargo fetch --locked", dockerfile)
        self.assertNotIn("apt-get", dockerfile)
        self.assertIn("USER 65532:65532", dockerfile)

    def test_perf_dockerfile_builds_installed_package_image(self) -> None:
        dockerfile = read_text("containers/perf/Dockerfile")
        self.assertIn("rust:1.83.0-slim-bookworm@", dockerfile)
        self.assertIn("python:3.12.13-slim-bookworm@", dockerfile)
        self.assertIn("maturin==1.13.1", dockerfile)
        self.assertIn("rdkit==2026.3.1", dockerfile)
        self.assertNotIn("apt-get", dockerfile)
        self.assertNotIn("git", dockerfile)
        self.assertIn("COPY . /build-src", dockerfile)
        self.assertIn("WORKDIR /build-src", dockerfile)
        self.assertNotIn("WORKDIR /src", dockerfile)
        self.assertIn("python -m maturin build --release --out", dockerfile)
        self.assertIn(
            "python -m pip install --no-deps /tmp/grimace-dist/*.whl",
            dockerfile,
        )
        self.assertIn('"discover", "-s", "tests/perf"', dockerfile)
        self.assertIn("USER 65532:65532", dockerfile)

    def test_makefile_exposes_guarded_checks_lane(self) -> None:
        makefile = read_text("Makefile")
        self.assertIn("SHELL := bash", makefile)
        self.assertIn(".SHELLFLAGS := -eu -o pipefail -c", makefile)
        self.assertIn("DOCKER_COMPOSE ?= docker compose", makefile)
        self.assertIn("override ACTUAL_UID := $(shell id -u)", makefile)
        self.assertIn("LOCAL_UID ?= $(shell id -u)", makefile)
        self.assertIn("LOCAL_GID ?= $(shell id -g)", makefile)
        self.assertIn("override REPO_ROOT := $(shell pwd -P)", makefile)
        self.assertIn(
            "override PERF_ARTIFACTS := docs/timings.tsv docs/timings.md notes/004_perf_history.jsonl",
            makefile,
        )
        self.assertIn("COMPOSE_ENV := LOCAL_UID=$(LOCAL_UID) LOCAL_GID=$(LOCAL_GID)", makefile)
        self.assertIn('"$(ACTUAL_UID)" == "0"', makefile)
        self.assertIn('! "$(LOCAL_UID)" =~ ^[1-9][0-9]*$$', makefile)
        self.assertIn('! "$(LOCAL_GID)" =~ ^[1-9][0-9]*$$', makefile)
        self.assertIn("positive numeric LOCAL_UID and LOCAL_GID", makefile)
        self.assertIn("DIST_GUARD := if [[ -L dist ]]", makefile)
        self.assertIn('PERF_ARTIFACTS_GUARD := repo_root="$(REPO_ROOT)"', makefile)
        self.assertIn('resolved="$$(realpath -e -- "$$path"', makefile)
        self.assertIn('expected="$$repo_root/$$path"', makefile)
        self.assertIn("missing, a symlink, or outside the repository", makefile)
        self.assertLess(
            makefile.index("package:\n\t@$(NON_ROOT_GUARD)"),
            makefile.index("\t@find dist"),
        )
        self.assertLess(
            makefile.index("\t@$(DIST_GUARD)"),
            makefile.index("\t@find dist"),
        )
        self.assertIn("GRIMACE_PERF_GIT_COMMIT", makefile)
        self.assertIn("GRIMACE_PERF_GIT_CHANGE", makefile)
        self.assertIn("GRIMACE_PERF_GIT_DIRTY", makefile)
        expected_targets = {
            "checks": "checks.yml,checks",
            "rust": "test.yml,rust",
            "test": "test.yml,test",
            "parity": "test.yml,parity",
            "exact-public-invariants": "test.yml,exact-public-invariants",
            "package": "package.yml,package",
        }
        for target, compose_call in expected_targets.items():
            with self.subTest(target=target):
                self.assertRegex(makefile, rf"(?m)^{target}:")
                self.assertIn(f"$(call compose_run,{compose_call})", makefile)
        self.assertRegex(makefile, r"(?m)^perf:")
        self.assertIn("$(DOCKER_COMPOSE) -f $(COMPOSE_DIR)/perf.yml", makefile)
        self.assertIn("run --build --rm perf", makefile)
        self.assertRegex(
            makefile,
            r"(?m)^ci: checks rust test parity exact-public-invariants$",
        )

    def test_ci_workflow_uses_container_make_lanes(self) -> None:
        workflow = read_text(".github/workflows/ci.yml")
        self.assertIn("run: make ci", workflow)
        self.assertIn("run: make package", workflow)
        self.assertNotIn("actions/setup-python", workflow)
        self.assertNotIn("dtolnay/rust-toolchain", workflow)
        self.assertNotIn("maturin", workflow)
        self.assertNotIn("RUN_PERF_TESTS", workflow)
        self.assertNotIn("make perf", workflow)

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
            "tmp",
            ".venv",
            ".env",
            ".env.*",
            ".envrc",
            ".netrc",
            ".npmrc",
            ".pypirc",
            ".aws",
            ".azure",
            ".cargo/credentials",
            ".cargo/credentials.toml",
            ".config/gcloud",
            ".gcloud",
            ".gnupg",
            ".kube",
            ".ssh",
            "pip.conf",
            "pip.ini",
            "*.pem",
            "*.key",
            "*.p12",
            "*.pfx",
            "*.crt",
            "*.secret",
            "*.token",
            "id_rsa",
            "id_ed25519",
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

    def test_rust_lockfile_is_part_of_the_container_source_boundary(self) -> None:
        gitignore_patterns = {
            line.strip()
            for line in read_text(".gitignore").splitlines()
            if line.strip() and not line.strip().startswith("#")
        }
        self.assertTrue((ROOT / "Cargo.lock").is_file())
        self.assertNotIn("Cargo.lock", gitignore_patterns)


if __name__ == "__main__":
    unittest.main()
