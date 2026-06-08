from pathlib import Path
import re
import unittest
from urllib.parse import unquote, urlparse

from scripts import timings_enum_measure
from scripts import timings_prepared_mol_zstd_plot


ROOT = Path(__file__).resolve().parents[2]
LOCAL_LINK = re.compile(r"!?\[[^\]]*\]\(([^)]+)\)")
HEADING = re.compile(r"^(#{1,6}) ")
HTML_IMAGE_SRC = re.compile(r'<img\s+[^>]*src="([^"]+)"')


def markdown_files() -> tuple[Path, ...]:
    return (
        ROOT / "README.md",
        *(
            path
            for path in sorted((ROOT / "docs").rglob("*.md"))
            if path.relative_to(ROOT) != Path("docs/timings-enum.md")
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
    if source.is_relative_to(ROOT / "docs") and path_text.endswith(".html"):
        path_text = f"{path_text[:-5]}.md"
    return (source.parent / path_text).resolve()


def document_headings(markdown: Path) -> tuple[int, ...]:
    headings: list[int] = []
    in_fence = False
    in_front_matter = False

    for line_number, line in enumerate(markdown.read_text(encoding="utf-8").splitlines()):
        if line_number == 0 and line == "---":
            in_front_matter = True
            continue
        if in_front_matter:
            if line == "---":
                in_front_matter = False
            continue
        if line.startswith("```"):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        match = HEADING.match(line)
        if match is not None:
            headings.append(len(match.group(1)))

    return tuple(headings)


class DocsPagesTests(unittest.TestCase):
    def test_pages_entrypoint_exists(self) -> None:
        self.assertTrue((ROOT / "docs" / "_config.yml").is_file())
        self.assertTrue((ROOT / "docs" / "index.md").is_file())
        config = (ROOT / "docs" / "_config.yml").read_text(encoding="utf-8")
        self.assertIn("theme: minima", config)
        self.assertIn("url: https://numpde.github.io", config)
        self.assertIn("baseurl: /grimace-py", config)
        self.assertIn("header_pages:", config)
        local_config = (ROOT / "docs" / "_config_local.yml").read_text(encoding="utf-8")
        self.assertIn('baseurl: ""', local_config)
        self.assertNotRegex(local_config, r"(?m)^url:")
        index = (ROOT / "docs" / "index.md").read_text(encoding="utf-8")
        self.assertIn("layout: page", index)

    def test_pages_disable_minima_default_footer(self) -> None:
        footer = ROOT / "docs" / "_includes" / "footer.html"
        self.assertTrue(footer.is_file())
        self.assertEqual("", footer.read_text(encoding="utf-8").strip())

    def test_readme_points_to_pages_site(self) -> None:
        readme = (ROOT / "README.md").read_text(encoding="utf-8")
        self.assertIn(
            "[numpde.github.io/grimace-py](https://numpde.github.io/grimace-py/)",
            readme,
        )
        self.assertNotIn("[documentation index](docs/index.md)", readme)

    def test_pages_use_system_theme_stylesheet(self) -> None:
        stylesheet = ROOT / "docs" / "assets" / "main.scss"
        self.assertTrue(stylesheet.is_file())
        text = stylesheet.read_text(encoding="utf-8")
        self.assertIn('@import "minima";', text)
        self.assertIn("color-scheme: light dark;", text)
        self.assertIn("@media (prefers-color-scheme: dark)", text)

    def test_timings_tables_are_scrollable(self) -> None:
        timings = (ROOT / "docs" / "timings-enum.md").read_text(encoding="utf-8")
        self.assertIn("table.timings-table", timings)
        self.assertIn("overflow-x: auto;", timings)
        self.assertIn("white-space: nowrap;", timings)
        self.assertEqual(2, timings.count("{: .timings-table}"))

    def test_timings_environment_is_present_and_public(self) -> None:
        timings = (ROOT / "docs" / "timings-enum.md").read_text(encoding="utf-8")
        self.assertIn("## Benchmark environment", timings)
        for field in (
            "| Recorded |",
            "| Runtime |",
            "| Platform |",
            "| CPU |",
            "| Memory limit |",
            "| Container |",
        ):
            self.assertIn(field, timings)
        self.assertNotIn("hostname", timings.lower())

    def test_timing_tsv_metadata_is_self_contained(self) -> None:
        paths = (
            ROOT / "docs" / "timings-enum.tsv",
            *sorted((ROOT / "docs").glob("timings-prepared-mol-zstd*.tsv")),
        )
        for path in paths:
            with self.subTest(path=path.name):
                if path.name == "timings-enum.tsv":
                    rows = timings_enum_measure._read_timing_rows_from_tsv(path)
                else:
                    rows = tuple(timings_prepared_mol_zstd_plot._read_rows(path))
                self.assertTrue(rows)
                for field in (
                    "recorded_at_utc",
                    "git_commit",
                    "git_change",
                    "git_dirty",
                    "platform",
                    "python",
                    "rdkit",
                    "cpu_model",
                    "visible_cpus",
                    "cgroup_memory_limit_bytes",
                    "container",
                ):
                    values = {row[field] for row in rows}
                    self.assertEqual(1, len(values), field)
                    self.assertNotEqual("", next(iter(values)), field)

    def test_timings_overview_links_benchmark_pages(self) -> None:
        timings = (ROOT / "docs" / "timings.md").read_text(encoding="utf-8")
        self.assertIn("[Enum/support timings](timings-enum.html)", timings)
        self.assertIn(
            "[PreparedMol zstd timings](timings-prepared-mol-zstd.html)",
            timings,
        )
        self.assertNotIn("## Benchmark environment", timings)

    def test_timings_plots_are_captioned_and_present(self) -> None:
        timings = (ROOT / "docs" / "timings-enum.md").read_text(encoding="utf-8")
        self.assertEqual(18, timings.count('<figure class="timing-plot">'))
        self.assertEqual(18, timings.count("</code>:</figcaption>"))

        image_paths = HTML_IMAGE_SRC.findall(timings)
        self.assertEqual(18, len(image_paths))
        for image_path in image_paths:
            self.assertTrue((ROOT / "docs" / image_path).is_file(), image_path)

        for figure in timings.split('<figure class="timing-plot">')[1:]:
            self.assertLess(figure.index("<figcaption>"), figure.index("<img "))

    def test_prepared_mol_zstd_plots_are_under_dictionary_artifact(self) -> None:
        dictionary_artifacts: set[str] = set()
        for path in sorted((ROOT / "docs").glob("timings-prepared-mol-zstd*.tsv")):
            rows = timings_prepared_mol_zstd_plot._read_rows(path)
            artifacts = {row["dictionary_artifact"] for row in rows}
            self.assertEqual(1, len(artifacts), path.name)
            dictionary_artifacts.update(artifacts)

        timings = (ROOT / "docs" / "timings-prepared-mol-zstd.md").read_text(
            encoding="utf-8",
        )
        image_paths = HTML_IMAGE_SRC.findall(timings)
        self.assertEqual(2 * len(dictionary_artifacts), len(image_paths))
        for image_path in image_paths:
            self.assertTrue(
                any(
                    f"timings-prepared-mol-zstd-plots/{artifact}/" in image_path
                    for artifact in dictionary_artifacts
                ),
                image_path,
            )
            self.assertTrue((ROOT / "docs" / image_path).is_file(), image_path)

    def test_pages_use_front_matter_title_without_body_h1(self) -> None:
        offenders: list[str] = []
        for markdown in sorted((ROOT / "docs").rglob("*.md")):
            relative = markdown.relative_to(ROOT)
            text = markdown.read_text(encoding="utf-8")
            if not text.startswith("---\n"):
                offenders.append(f"{relative}: missing front matter")
                continue

            in_fence = False
            for line in text.splitlines():
                if line.startswith("```"):
                    in_fence = not in_fence
                elif not in_fence and line.startswith("# "):
                    offenders.append(f"{relative}: body h1")
                    break

        self.assertEqual([], offenders)

    def test_markdown_heading_levels_are_consistent(self) -> None:
        readme_headings = document_headings(ROOT / "README.md")
        self.assertTrue(readme_headings)
        self.assertEqual(1, readme_headings[0])
        self.assertNotIn(1, readme_headings[1:])

        offenders: list[str] = []
        for markdown in (ROOT / "README.md", *sorted((ROOT / "docs").rglob("*.md"))):
            relative = markdown.relative_to(ROOT)
            headings = document_headings(markdown)
            if not headings:
                continue

            if markdown.parent == ROOT:
                expected_first = 1
            else:
                expected_first = 2
                if 1 in headings:
                    offenders.append(f"{relative}: body h1")

            if headings[0] != expected_first:
                offenders.append(
                    f"{relative}: starts at h{headings[0]}, expected h{expected_first}"
                )

            previous = headings[0]
            for current in headings[1:]:
                if current > previous + 1:
                    offenders.append(
                        f"{relative}: jumps from h{previous} to h{current}"
                    )
                    break
                previous = current

        self.assertEqual([], offenders)

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
