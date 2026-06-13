from pathlib import Path
import importlib.util
import re
import unittest

from tests.checks.posture_helpers import assert_before, line_count


ROOT = Path(__file__).resolve().parents[2]
RELEASE_VALIDATOR = ROOT / "scripts" / "validate_release_artifacts.py"
PINNED_ACTION_REF = re.compile(
    r"(?m)^\s*-\s+uses:\s+[^@\s]+@[0-9a-f]{40}(?:\s+#\s+\S+)?\s*$"
)


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def job_section(workflow: str, job_name: str) -> str:
    pattern = rf"(?ms)^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)"
    match = re.search(pattern, workflow)
    if match is None:
        raise AssertionError(f"missing job {job_name!r}")
    return match.group("body")


def load_release_validator():
    spec = importlib.util.spec_from_file_location(
        "validate_release_artifacts",
        RELEASE_VALIDATOR,
    )
    if spec is None or spec.loader is None:
        raise AssertionError("could not load release artifact validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def matrix_values(job: str, key: str) -> tuple[str, ...]:
    pattern = rf"(?m)^\s{{8}}{re.escape(key)}:\n(?P<body>(?:^\s{{10}}- [^\n]+\n)+)"
    match = re.search(pattern, job)
    if match is None:
        raise AssertionError(f"missing matrix axis {key!r}")
    return tuple(
        item.strip().strip('"')
        for item in re.findall(r"(?m)^\s{10}- ([^\n]+)$", match.group("body"))
    )


class WorkflowPostureTests(unittest.TestCase):
    def test_workflows_pin_github_hosted_runner_image(self) -> None:
        for workflow_path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
            workflow = workflow_path.read_text(encoding="utf-8")
            with self.subTest(workflow=workflow_path.name):
                self.assertNotIn("ubuntu-latest", workflow)
                self.assertIn("ubuntu-24.04", workflow)

    def test_workflows_use_explicit_bash_shell(self) -> None:
        for workflow_path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
            workflow = workflow_path.read_text(encoding="utf-8")
            with self.subTest(workflow=workflow_path.name):
                self.assertRegex(
                    workflow,
                    r"(?m)^defaults:\n  run:\n    shell: bash$",
                )

    def test_workflow_actions_are_pinned_to_commit_sha(self) -> None:
        for workflow_path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
            workflow = workflow_path.read_text(encoding="utf-8")
            uses_count = len(re.findall(r"(?m)^\s*-\s+uses:\s+", workflow))
            pinned_count = len(PINNED_ACTION_REF.findall(workflow))
            with self.subTest(workflow=workflow_path.name):
                self.assertEqual(uses_count, pinned_count)

    def test_ci_workflow_uses_read_only_token_and_non_persistent_checkout(self) -> None:
        workflow = read_text(".github/workflows/ci.yml")
        self.assertRegex(workflow, r"(?m)^permissions:\n  contents: read$")
        self.assertEqual(
            line_count(workflow, r"\s+persist-credentials:\s+false"),
            line_count(workflow, r"\s*-\s+uses:\s+actions/checkout@[0-9a-f]{40}.*"),
        )
        self.assertNotIn("contents: write", workflow)
        self.assertNotIn("id-token: write", workflow)

    def test_docs_workflow_uses_read_only_token_and_non_persistent_checkout(self) -> None:
        workflow = read_text(".github/workflows/docs.yml")
        self.assertRegex(workflow, r"(?m)^permissions:\n  contents: read$")
        self.assertEqual(
            line_count(workflow, r"\s+persist-credentials:\s+false"),
            line_count(workflow, r"\s*-\s+uses:\s+actions/checkout@[0-9a-f]{40}.*"),
        )
        self.assertNotIn("contents: write", workflow)
        self.assertNotIn("id-token: write", workflow)

    def test_docs_workflow_covers_docs_only_pushes_lightly(self) -> None:
        workflow = read_text(".github/workflows/docs.yml")
        self.assertRegex(workflow, r"(?m)^  push:\n    branches:\n      - main$")
        self.assertIn('- "README.md"', workflow)
        self.assertIn('- "docs/**"', workflow)
        self.assertIn('- "compose/docs.yml"', workflow)
        self.assertIn('- ".github/workflows/docs.yml"', workflow)
        self.assertIn('- "Makefile"', workflow)
        self.assertNotIn("containers/checks", workflow)
        self.assertNotIn("containers/docs", workflow)
        self.assertNotIn("tests/checks", workflow)
        self.assertIn("fetch-depth: 0", workflow)
        self.assertIn("run: make docs", workflow)
        self.assertIn("run: make checks", workflow)
        self.assertNotIn("run: make ci", workflow)
        self.assertNotIn("run: make test-package", workflow)

    def test_ci_checks_job_fetches_tags_for_release_note_checks(self) -> None:
        workflow = read_text(".github/workflows/ci.yml")
        checks_job = job_section(workflow, "container-ci")
        package_job = job_section(workflow, "test-package")
        self.assertIn("fetch-depth: 0", checks_job)
        self.assertNotIn("fetch-depth: 0", package_job)

    def test_release_workflow_scopes_token_permissions_by_job(self) -> None:
        workflow = read_text(".github/workflows/release.yml")
        self.assertRegex(workflow, r"(?m)^permissions:\n  contents: read$")
        self.assertEqual(
            line_count(workflow, r"\s+persist-credentials:\s+false"),
            line_count(workflow, r"\s*-\s+uses:\s+actions/checkout@[0-9a-f]{40}.*"),
        )

        wheel = job_section(workflow, "wheel")
        sdist = job_section(workflow, "sdist")
        release = job_section(workflow, "release")
        publish = job_section(workflow, "publish-pypi")

        self.assertNotIn("permissions:", wheel)
        self.assertNotIn("permissions:", sdist)
        self.assertRegex(
            release,
            r"(?m)^    permissions:\n      actions: read\n      contents: write$",
        )
        self.assertRegex(
            publish,
            r"(?m)^    permissions:\n      actions: read\n      contents: read\n      id-token: write$",
        )
        self.assertNotIn("contents: write", publish)
        self.assertNotIn("id-token: write", release)

    def test_release_workflow_validates_release_metadata_before_building(self) -> None:
        workflow = read_text(".github/workflows/release.yml")
        self.assertRegex(workflow, r"(?m)^  validate-release:")
        self.assertIn('"$GITHUB_REF_NAME" =~ ^v[0-9]+\\.[0-9]+\\.[0-9]+$', workflow)
        validate = job_section(workflow, "validate-release")
        self.assertIn('test -f "notes/releases/${GITHUB_REF_NAME}.md"', validate)

        wheel = job_section(workflow, "wheel")
        sdist = job_section(workflow, "sdist")
        self.assertRegex(wheel, r"(?m)^    needs:\n      - validate-release$")
        self.assertRegex(sdist, r"(?m)^    needs:\n      - validate-release$")

    def test_release_workflow_allowlists_uploaded_and_published_artifacts(self) -> None:
        workflow = read_text(".github/workflows/release.yml")
        self.assertIn('MATURIN_PIP_VERSION: "1.13.1"', workflow)
        self.assertIn('MATURIN_ACTION_VERSION: "v1.13.1"', workflow)
        self.assertIn('ZSTANDARD_FIXTURE_PIP_VERSION: "0.25.0"', workflow)
        self.assertIn('TWINE_PIP_VERSION: "6.2.0"', workflow)
        self.assertIn('PIP_DISABLE_PIP_VERSION_CHECK: "1"', workflow)
        self.assertIn('PIP_NO_CACHE_DIR: "1"', workflow)
        self.assertRegex(
            workflow,
            r'MANYLINUX_2_28_X86_64_IMAGE: "quay\.io/pypa/manylinux_2_28_x86_64@sha256:[0-9a-f]{64}"',
        )
        self.assertEqual(
            line_count(
                workflow,
                r"\s+maturin-version:\s+\$\{\{ env\.MATURIN_ACTION_VERSION \}\}",
            ),
            2,
        )
        self.assertEqual(
            line_count(
                workflow,
                r"\s+container:\s+\$\{\{ env\.MANYLINUX_2_28_X86_64_IMAGE \}\}",
            ),
            1,
        )
        self.assertIn(
            "args: --release --compatibility pypi --out dist -i python${{ matrix.python-version }}",
            workflow,
        )
        self.assertEqual(
            line_count(
                workflow,
                r"\s+run: python -m pip install --constraint requirements/container-build-constraints\.txt .*",
            ),
            4,
        )
        self.assertIn('"maturin==$MATURIN_PIP_VERSION"', workflow)
        self.assertEqual(
            line_count(workflow, r"\s+run: .*\"twine==\$TWINE_PIP_VERSION\".*"),
            4,
        )
        self.assertEqual(
            line_count(
                workflow,
                r"\s+run: .*\"zstandard==\$ZSTANDARD_FIXTURE_PIP_VERSION\".*",
            ),
            2,
        )
        self.assertIn("python -m twine check dist/*.whl", workflow)
        self.assertIn("python -m twine check dist/*.tar.gz", workflow)
        self.assertEqual(
            len(re.findall(r"(?m)^\s+run: python -m twine check dist/\*$", workflow)),
            2,
        )
        self.assertIn("python -m pip install --no-deps dist/*.whl", workflow)
        self.assertIn("python -m pip install --no-deps --no-build-isolation dist/*.tar.gz", workflow)
        self.assertEqual(
            line_count(
                workflow,
                r"\s+run: python scripts/validate_release_artifacts\.py dist/\*\.whl --wheel-only",
            ),
            1,
        )
        self.assertEqual(
            line_count(
                workflow,
                r"\s+run: python scripts/validate_release_artifacts\.py dist/\*\.tar\.gz --sdist-only",
            ),
            1,
        )
        self.assertIn("path: dist/*.whl", workflow)
        self.assertIn("path: dist/*.tar.gz", workflow)
        self.assertEqual(line_count(workflow, r"\s+if-no-files-found:\s+error"), 2)
        self.assertEqual(
            line_count(
                workflow,
                r'\s+run: python scripts/validate_release_artifacts\.py dist --tag "\$GITHUB_REF_NAME"',
            ),
            2,
        )

    def test_release_workflow_validates_archives_before_installing_them(self) -> None:
        workflow = read_text(".github/workflows/release.yml")
        wheel = job_section(workflow, "wheel")
        sdist = job_section(workflow, "sdist")
        release = job_section(workflow, "release")
        publish = job_section(workflow, "publish-pypi")

        assert_before(
            self,
            wheel,
            "python scripts/validate_release_artifacts.py dist/*.whl --wheel-only",
            "python -m twine check dist/*.whl",
        )
        assert_before(
            self,
            wheel,
            "python -m twine check dist/*.whl",
            "python -m pip install --no-deps dist/*.whl",
        )
        assert_before(
            self,
            sdist,
            "python scripts/validate_release_artifacts.py dist/*.tar.gz --sdist-only",
            "python -m twine check dist/*.tar.gz",
        )
        assert_before(
            self,
            sdist,
            "python -m twine check dist/*.tar.gz",
            "python -m pip install --no-deps --no-build-isolation dist/*.tar.gz",
        )
        for job_name, job in (("release", release), ("publish-pypi", publish)):
            with self.subTest(job=job_name):
                assert_before(
                    self,
                    job,
                    'python scripts/validate_release_artifacts.py dist --tag "$GITHUB_REF_NAME"',
                    "python -m twine check dist/*",
                )
        assert_before(
            self,
            release,
            "python -m twine check dist/*",
            "softprops/action-gh-release@",
        )
        assert_before(
            self,
            publish,
            "python -m twine check dist/*",
            "pypa/gh-action-pypi-publish@",
        )

    def test_release_workflow_wheel_matrix_matches_artifact_validator(self) -> None:
        workflow = read_text(".github/workflows/release.yml")
        wheel = job_section(workflow, "wheel")
        validator = load_release_validator()

        python_versions = matrix_values(wheel, "python-version")
        python_tags = tuple(
            f"cp{python_version.replace('.', '')}"
            for python_version in python_versions
        )
        platform_tags = tuple(
            f"manylinux_{manylinux}_{target}"
            for manylinux in matrix_values(wheel, "manylinux")
            for target in matrix_values(wheel, "target")
        )

        self.assertEqual(validator.PYTHON_TAGS, python_tags)
        self.assertEqual((validator.PLATFORM_TAG,), platform_tags)
        self.assertEqual(("linux-x86_64",), matrix_values(wheel, "name"))
        self.assertEqual(("ubuntu-24.04",), matrix_values(wheel, "os"))
        self.assertEqual(
            validator.expected_artifact_names("0.0.0"),
            tuple(
                sorted(
                    (
                        *(
                            f"grimace_py-0.0.0-{tag}-{tag}-{validator.PLATFORM_TAG}.whl"
                            for tag in python_tags
                        ),
                        "grimace_py-0.0.0.tar.gz",
                    )
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
