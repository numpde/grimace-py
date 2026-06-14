from pathlib import Path
import unittest

from scripts.validate_release_artifacts import TAG_PATTERN


ROOT = Path(__file__).resolve().parents[2]


def read_text(relative_path: str) -> str:
    return (ROOT / relative_path).read_text(encoding="utf-8")


def release_tags() -> tuple[str, ...]:
    tags: set[str] = set()

    refs_tags = ROOT / ".git" / "refs" / "tags"
    if refs_tags.is_dir():
        for path in refs_tags.iterdir():
            if TAG_PATTERN.fullmatch(path.name):
                tags.add(path.name)

    packed_refs = ROOT / ".git" / "packed-refs"
    if packed_refs.is_file():
        for line in packed_refs.read_text(encoding="utf-8").splitlines():
            if not line or line.startswith("#") or line.startswith("^"):
                continue
            parts = line.split()
            if len(parts) != 2:
                continue
            ref = parts[1]
            if not ref.startswith("refs/tags/"):
                continue
            tag = ref.rsplit("/", 1)[-1]
            if TAG_PATTERN.fullmatch(tag):
                tags.add(tag)

    return tuple(sorted(tags))


class ReleaseNotesTests(unittest.TestCase):
    def test_every_release_tag_has_checked_in_note(self) -> None:
        tags = release_tags()
        self.assertTrue(tags, "expected local git metadata with vX.Y.Z tags")
        missing = [
            tag
            for tag in tags
            if not (ROOT / "notes" / "releases" / f"{tag}.md").is_file()
        ]
        self.assertEqual([], missing)

    def test_release_workflow_uses_checked_in_note_for_tag(self) -> None:
        workflow = read_text(".github/workflows/release.yml")
        self.assertIn('test -f "notes/releases/${GITHUB_REF_NAME}.md"', workflow)
        self.assertIn(
            "body_path: notes/releases/${{ github.ref_name }}.md",
            workflow,
        )


if __name__ == "__main__":
    unittest.main()
