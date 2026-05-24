from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[2]
LOCAL_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")


def markdown_files() -> tuple[Path, ...]:
    return (
        ROOT / "README.md",
        *(
            path
            for path in sorted((ROOT / "docs").rglob("*.md"))
            if path.relative_to(ROOT) != Path("docs/timings.md")
        ),
    )


def linked_path(source: Path, raw_target: str) -> Path | None:
    target = raw_target.strip()
    if not target or target.startswith("#"):
        return None
    parsed = urlparse(target)
    if parsed.scheme or parsed.netloc:
        return None

    path_text = unquote(parsed.path)
    if not path_text:
        return None
    return (source.parent / path_text).resolve()


class DocsPagesTests(unittest.TestCase):
    def test_pages_entrypoint_exists(self) -> None:
        self.assertTrue((ROOT / "docs" / "_config.yml").is_file())
        self.assertTrue((ROOT / "docs" / "index.md").is_file())
        config = (ROOT / "docs" / "_config.yml").read_text(encoding="utf-8")
        self.assertIn("theme: minima", config)
        self.assertIn("header_pages:", config)

    def test_readme_points_to_pages_index(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn("[documentation index](docs/index.md)", readme)

    def test_markdown_local_links_resolve(self) -> None:
        missing: list[str] = []
        for markdown in markdown_files():
            text = markdown.read_text(encoding="utf-8")
            for match in LOCAL_LINK.finditer(text):
                target = linked_path(markdown, match.group(1))
                if target is None:
                    continue
                try:
                    target.relative_to(ROOT)
                except ValueError:
                    missing.append(f"{markdown.relative_to(ROOT)} -> {match.group(1)}")
                    continue
                if not target.exists():
                    missing.append(f"{markdown.relative_to(ROOT)} -> {match.group(1)}")

        self.assertEqual([], missing)


if __name__ == "__main__":
    unittest.main()
