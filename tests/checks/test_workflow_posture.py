from pathlib import Path
import re
import unittest

from scripts.validate_release_artifacts import (
    PLATFORM_TAG,
    PYTHON_TAGS,
    expected_artifact_names,
)
from tests.checks.posture_helpers import (
    assert_before,
    full_line_count,
    yaml_scalar_count,
    yaml_scalar_line,
)


ROOT = Path(__file__).resolve().parents[2]
ACTION_USES_LINE = re.compile(r"(?m)^(?:uses:| {8}uses:)[ \t]+[^\n]+$")
PINNED_ACTION_USES_LINE = re.compile(
    r"(?m)^(?:uses:| {8}uses:)[ \t]+[^@\s]+@[0-9a-f]{40}(?:[ \t]+#[ \t]*\S+)?[ \t]*$"
)
JOB_USES_LINE = re.compile(r"(?m)^    uses:[ \t]+[^\n]+$")
LOCAL_JOB_USES_LINE = re.compile(
    r"(?m)^    uses:[ \t]+\./\.github/workflows/[^\n]+\.ya?ml[ \t]*$"
)
PINNED_JOB_USES_LINE = re.compile(
    r"(?m)^    uses:[ \t]+[^@\s]+@[0-9a-f]{40}(?:[ \t]+#[ \t]*\S+)?[ \t]*$"
)
PINNED_CHECKOUT_USES_LINE = re.compile(
    r"(?m)^(?:uses:| {8}uses:)[ \t]+actions/checkout@[0-9a-f]{40}"
    r"(?:[ \t]+#[ \t]*\S+)?[ \t]*$"
)
WORKFLOW_STEP_BLOCKS = re.compile(r"(?ms)^      - (?P<body>.*?)(?=^      - |\Z)")


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def job_section(workflow: str, job_name: str) -> str:
    pattern = rf"(?ms)^  {re.escape(job_name)}:\n(?P<body>.*?)(?=^  [A-Za-z0-9_-]+:\n|\Z)"
    match = re.search(pattern, workflow)
    if match is None:
        raise AssertionError(f"missing job {job_name!r}")
    return match.group("body")


def checkout_step(job: str) -> str:
    steps = checkout_steps(job)
    if len(steps) != 1:
        raise AssertionError(f"expected exactly one pinned checkout step, got {len(steps)}")
    return steps[0]


def checkout_steps(text: str) -> tuple[str, ...]:
    # Select whole step blocks before checking checkout options. That keeps the
    # security assertion tied to the checkout step without depending on key
    # order inside the YAML step.
    return tuple(
        match.group("body")
        for match in WORKFLOW_STEP_BLOCKS.finditer(text)
        if PINNED_CHECKOUT_USES_LINE.search(match.group("body"))
    )


def workflow_uses_lines(workflow: str, pattern: re.Pattern[str]) -> tuple[str, ...]:
    return tuple(
        line
        for step in WORKFLOW_STEP_BLOCKS.finditer(workflow)
        for line in pattern.findall(step.group("body"))
    )


def unpinned_workflow_uses_lines(workflow: str) -> tuple[str, ...]:
    step_uses = workflow_uses_lines(workflow, ACTION_USES_LINE)
    pinned_step_uses = workflow_uses_lines(workflow, PINNED_ACTION_USES_LINE)
    # Reusable workflow jobs are `jobs.<id>.uses`, outside any step block.
    job_uses = tuple(
        line
        for line in JOB_USES_LINE.findall(workflow)
        if not LOCAL_JOB_USES_LINE.fullmatch(line)
    )
    pinned_job_uses = PINNED_JOB_USES_LINE.findall(workflow)
    pinned_uses = {*pinned_step_uses, *pinned_job_uses}
    return tuple(
        line
        for line in (*step_uses, *job_uses)
        if line not in pinned_uses
    )


def assert_checkouts_do_not_persist_credentials(
    test: unittest.TestCase,
    workflow: str,
) -> None:
    steps = checkout_steps(workflow)
    test.assertTrue(steps)
    for step in steps:
        test.assertRegex(step, yaml_scalar_line("persist-credentials", "false"))
        test.assertNotRegex(step, yaml_scalar_line("persist-credentials", "true"))


def matrix_values(job: str, key: str) -> tuple[str, ...]:
    pattern = rf"(?m)^ {{8}}{re.escape(key)}:\n(?P<body>(?:^ {{10}}- [^\n]+\n)+)"
    match = re.search(pattern, job)
    if match is None:
        raise AssertionError(f"missing matrix axis {key!r}")
    return tuple(
        item.strip().strip('"')
        for item in re.findall(r"(?m)^ {10}- ([^\n]+)$", match.group("body"))
    )


class WorkflowPostureTests(unittest.TestCase):
    def test_workflow_posture_helpers_cover_uses_ref_shapes(self) -> None:
        workflow = """\
jobs:
  example:
    steps:
      - name: Checkout source
        id: checkout
        uses: actions/checkout@de0fac2e4500dabe0009e67214ff5f5447ce83dd # v6
        with:
          persist-credentials: false
      - name: Publish
        uses: example/action@1111111111111111111111111111111111111111
      - name: Script with action-like text
        run: |
          uses: not/an/action@2222222222222222222222222222222222222222
          - uses: not/a/step@3333333333333333333333333333333333333333
      - if: always()
        run: true
  reusable:
    uses: example/workflow/.github/workflows/build.yml@1111111111111111111111111111111111111111
  local-reusable:
    uses: ./.github/workflows/local.yml
"""

        self.assertEqual((), unpinned_workflow_uses_lines(workflow))
        assert_checkouts_do_not_persist_credentials(self, workflow)
        self.assertNotIn("if: always()", checkout_step(workflow))

        unpinned_reusable = (
            workflow
            + "  unpinned:\n"
            + "    uses: example/workflow/.github/workflows/build.yml@v1\n"
        )
        self.assertEqual(
            ("    uses: example/workflow/.github/workflows/build.yml@v1",),
            unpinned_workflow_uses_lines(unpinned_reusable),
        )

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

    def test_workflow_uses_refs_are_pinned_to_commit_sha(self) -> None:
        for workflow_path in sorted((ROOT / ".github" / "workflows").glob("*.yml")):
            workflow = workflow_path.read_text(encoding="utf-8")
            with self.subTest(workflow=workflow_path.name):
                self.assertEqual((), unpinned_workflow_uses_lines(workflow))

    def test_ci_workflow_uses_read_only_token_and_non_persistent_checkout(self) -> None:
        workflow = read_text(".github/workflows/ci.yml")
        self.assertRegex(workflow, r"(?m)^permissions:\n  contents: read$")
        assert_checkouts_do_not_persist_credentials(self, workflow)
        self.assertNotIn("contents: write", workflow)
        self.assertNotIn("id-token: write", workflow)

    def test_docs_workflow_uses_read_only_token_and_non_persistent_checkout(self) -> None:
        workflow = read_text(".github/workflows/docs.yml")
        self.assertRegex(workflow, r"(?m)^permissions:\n  contents: read$")
        assert_checkouts_do_not_persist_credentials(self, workflow)
        self.assertNotIn("contents: write", workflow)
        self.assertNotIn("id-token: write", workflow)

    def test_docs_workflow_covers_docs_only_pushes_lightly(self) -> None:
        workflow = read_text(".github/workflows/docs.yml")
        docs_job = job_section(workflow, "docs")
        docs_checkout = checkout_step(docs_job)
        self.assertRegex(workflow, r"(?m)^  push:\n    branches:\n      - main$")
        self.assertIn('- "README.md"', workflow)
        self.assertIn('- "docs/**"', workflow)
        self.assertIn('- "compose/docs.yml"', workflow)
        self.assertIn('- ".github/workflows/docs.yml"', workflow)
        self.assertIn('- "Makefile"', workflow)
        self.assertNotIn("containers/checks", workflow)
        self.assertNotIn("containers/docs", workflow)
        self.assertNotIn("tests/checks", workflow)
        self.assertRegex(docs_checkout, yaml_scalar_line("fetch-depth", "0"))
        self.assertIn("run: make docs", docs_job)
        self.assertIn("run: make checks", docs_job)
        self.assertNotIn("run: make ci", workflow)
        self.assertNotIn("run: make test-package", workflow)

    def test_ci_checks_job_fetches_tags_for_release_note_checks(self) -> None:
        workflow = read_text(".github/workflows/ci.yml")
        checks_job = job_section(workflow, "container-ci")
        package_job = job_section(workflow, "test-package")
        fetch_depth_tags = yaml_scalar_line("fetch-depth", "0")
        self.assertRegex(checkout_step(checks_job), fetch_depth_tags)
        self.assertNotRegex(checkout_step(package_job), fetch_depth_tags)

    def test_release_workflow_scopes_token_permissions_by_job(self) -> None:
        workflow = read_text(".github/workflows/release.yml")
        self.assertRegex(workflow, r"(?m)^permissions:\n  contents: read$")
        assert_checkouts_do_not_persist_credentials(self, workflow)

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
            full_line_count(
                workflow,
                r"[ \t]+maturin-version:[ \t]+\$\{\{ env\.MATURIN_ACTION_VERSION \}\}",
            ),
            2,
        )
        self.assertEqual(
            full_line_count(
                workflow,
                r"[ \t]+container:[ \t]+\$\{\{ env\.MANYLINUX_2_28_X86_64_IMAGE \}\}",
            ),
            1,
        )
        self.assertIn(
            "args: --release --compatibility pypi --out dist -i python${{ matrix.python-version }}",
            workflow,
        )
        self.assertEqual(
            full_line_count(
                workflow,
                r"[ \t]+run: python -m pip install --constraint requirements/container-build-constraints\.txt .*",
            ),
            4,
        )
        self.assertIn('"maturin==$MATURIN_PIP_VERSION"', workflow)
        self.assertEqual(
            full_line_count(workflow, r'[ \t]+run: .*"twine==\$TWINE_PIP_VERSION".*'),
            4,
        )
        self.assertEqual(
            full_line_count(
                workflow,
                r'[ \t]+run: .*"zstandard==\$ZSTANDARD_FIXTURE_PIP_VERSION".*',
            ),
            2,
        )
        self.assertIn("python -m twine check dist/*.whl", workflow)
        self.assertIn("python -m twine check dist/*.tar.gz", workflow)
        self.assertEqual(
            full_line_count(workflow, r"[ \t]+run: python -m twine check dist/\*"),
            2,
        )
        self.assertIn("python -m pip install --no-deps dist/*.whl", workflow)
        self.assertIn("python -m pip install --no-deps --no-build-isolation dist/*.tar.gz", workflow)
        self.assertEqual(
            full_line_count(
                workflow,
                r"[ \t]+run: python scripts/validate_release_artifacts\.py dist/\*\.whl --wheel-only",
            ),
            1,
        )
        self.assertEqual(
            full_line_count(
                workflow,
                r"[ \t]+run: python scripts/validate_release_artifacts\.py dist/\*\.tar\.gz --sdist-only",
            ),
            1,
        )
        self.assertIn("path: dist/*.whl", workflow)
        self.assertIn("path: dist/*.tar.gz", workflow)
        self.assertEqual(yaml_scalar_count(workflow, "if-no-files-found", "error"), 2)
        self.assertEqual(
            full_line_count(
                workflow,
                r'[ \t]+run: python scripts/validate_release_artifacts\.py dist --tag "\$GITHUB_REF_NAME"',
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

        self.assertEqual(PYTHON_TAGS, python_tags)
        self.assertEqual((PLATFORM_TAG,), platform_tags)
        self.assertEqual(("linux-x86_64",), matrix_values(wheel, "name"))
        self.assertEqual(("ubuntu-24.04",), matrix_values(wheel, "os"))
        self.assertEqual(
            expected_artifact_names("0.0.0"),
            tuple(
                sorted(
                    (
                        *(
                            f"grimace_py-0.0.0-{tag}-{tag}-{PLATFORM_TAG}.whl"
                            for tag in python_tags
                        ),
                        "grimace_py-0.0.0.tar.gz",
                    )
                )
            ),
        )


if __name__ == "__main__":
    unittest.main()
