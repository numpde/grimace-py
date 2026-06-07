from pathlib import Path
import importlib.util
import re
import unittest


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
            workflow.count("persist-credentials: false"),
            workflow.count("uses: actions/checkout@"),
        )
        self.assertNotIn("contents: write", workflow)
        self.assertNotIn("id-token: write", workflow)

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
            workflow.count("persist-credentials: false"),
            workflow.count("uses: actions/checkout@"),
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
        self.assertIn('PIP_DISABLE_PIP_VERSION_CHECK: "1"', workflow)
        self.assertIn('PIP_NO_CACHE_DIR: "1"', workflow)
        self.assertRegex(
            workflow,
            r'MANYLINUX_2_28_X86_64_IMAGE: "quay\.io/pypa/manylinux_2_28_x86_64@sha256:[0-9a-f]{64}"',
        )
        self.assertEqual(workflow.count("maturin-version: ${{ env.MATURIN_ACTION_VERSION }}"), 2)
        self.assertEqual(workflow.count("container: ${{ env.MANYLINUX_2_28_X86_64_IMAGE }}"), 1)
        self.assertIn(
            "args: --release --compatibility pypi --out dist -i python${{ matrix.python-version }}",
            workflow,
        )
        self.assertEqual(
            workflow.count("--constraint requirements/container-build-constraints.txt"),
            2,
        )
        self.assertIn('"maturin==$MATURIN_PIP_VERSION"', workflow)
        self.assertEqual(workflow.count('"zstandard==$ZSTANDARD_FIXTURE_PIP_VERSION"'), 2)
        self.assertIn("python -m pip install --no-deps dist/*.whl", workflow)
        self.assertIn("python -m pip install --no-deps --no-build-isolation dist/*.tar.gz", workflow)
        self.assertEqual(
            workflow.count(
                "python scripts/validate_release_artifacts.py dist/*.whl --wheel-only"
            ),
            1,
        )
        self.assertEqual(
            workflow.count(
                "python scripts/validate_release_artifacts.py dist/*.tar.gz --sdist-only"
            ),
            1,
        )
        self.assertIn("path: dist/*.whl", workflow)
        self.assertIn("path: dist/*.tar.gz", workflow)
        self.assertEqual(workflow.count("if-no-files-found: error"), 2)
        self.assertEqual(
            workflow.count("python scripts/validate_release_artifacts.py dist --tag"),
            2,
        )

    def test_release_workflow_wheel_matrix_matches_artifact_validator(self) -> None:
        workflow = read_text(".github/workflows/release.yml")
        wheel = job_section(workflow, "wheel")
        validator = load_release_validator()

        matrix_entries = re.findall(
            r"(?ms)^\s{10}- name:.*?(?=^\s{10}- name:|^    steps:)",
            wheel,
        )
        self.assertTrue(matrix_entries)
        python_tags: list[str] = []
        platforms: set[str] = set()
        for entry in matrix_entries:
            target = re.search(r"(?m)^\s+target: (\S+)$", entry)
            manylinux = re.search(r'(?m)^\s+manylinux: "([0-9]+_[0-9]+)"$', entry)
            python_version = re.search(
                r'(?m)^\s+python-version: "([0-9]+\.[0-9]+)"$',
                entry,
            )
            self.assertIsNotNone(target)
            self.assertIsNotNone(manylinux)
            self.assertIsNotNone(python_version)
            python_tags.append(f"cp{python_version.group(1).replace('.', '')}")
            platforms.add(f"manylinux_{manylinux.group(1)}_{target.group(1)}")

        self.assertEqual(validator.PYTHON_TAGS, tuple(python_tags))
        self.assertEqual((validator.PLATFORM_TAG,), tuple(sorted(set(platforms))))
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
