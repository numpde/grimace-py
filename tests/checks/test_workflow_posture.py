from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[2]
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


class WorkflowPostureTests(unittest.TestCase):
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

    def test_release_workflow_allowlists_uploaded_and_published_artifacts(self) -> None:
        workflow = read_text(".github/workflows/release.yml")
        self.assertIn(
            "args: --release --locked --out dist -i python${{ matrix.python-version }}",
            workflow,
        )
        self.assertIn("path: dist/*.whl", workflow)
        self.assertIn("path: dist/*.tar.gz", workflow)
        self.assertEqual(workflow.count("if-no-files-found: error"), 2)
        self.assertEqual(
            workflow.count("python scripts/validate_release_artifacts.py dist --tag"),
            2,
        )


if __name__ == "__main__":
    unittest.main()
