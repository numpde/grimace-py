from pathlib import Path
from contextlib import redirect_stderr
import io
import importlib.util
import tarfile
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate_release_artifacts.py"


def write_sdist(path: Path, names: tuple[str, ...]) -> None:
    root = path.name.removesuffix(".tar.gz")
    with tarfile.open(path, "w:gz") as archive:
        for name in names:
            full_name = f"{root}/{name}"
            with tempfile.NamedTemporaryFile() as tmp:
                Path(tmp.name).write_text("", encoding="utf-8")
                archive.add(tmp.name, arcname=full_name)


def write_expected_artifacts(validator, dist: Path, version: str) -> None:
    for name in validator.expected_artifact_names(version):
        if name.endswith(".tar.gz"):
            write_sdist(dist / name, ("pyproject.toml", "Cargo.toml"))
        else:
            (dist / name).write_text("", encoding="utf-8")


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_release_artifacts", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load release artifact validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseArtifactValidationTests(unittest.TestCase):
    def test_accepts_exact_release_artifact_set_for_tag(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            write_expected_artifacts(validator, dist, "0.1.12")
            validator.validate_artifacts(dist, "v0.1.12")

    def test_rejects_extra_artifact(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            write_expected_artifacts(validator, dist, "0.1.12")
            (dist / "unexpected.txt").write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "unexpected release artifacts"):
                validator.validate_artifacts(dist, "v0.1.12")

    def test_rejects_non_file_artifact_entry(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            write_expected_artifacts(validator, dist, "0.1.12")
            (dist / "nested").mkdir()
            with self.assertRaisesRegex(ValueError, "unexpected non-file"):
                validator.validate_artifacts(dist, "v0.1.12")

    def test_rejects_forbidden_sdist_content(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            write_expected_artifacts(validator, dist, "0.1.12")
            write_sdist(
                dist / "grimace_py-0.1.12.tar.gz",
                ("pyproject.toml", "notes/perf_reports/local.perf.txt"),
            )
            with self.assertRaisesRegex(ValueError, "forbidden file in sdist"):
                validator.validate_artifacts(dist, "v0.1.12")

    def test_rejects_secret_shaped_sdist_content(self) -> None:
        validator = load_validator()
        secret_paths = (
            ".env.local",
            ".npmrc",
            ".docker/config.json",
            ".config/gh/hosts.yml",
            "nested/.ssh/config",
            "nested/id_ed25519",
            "nested/private.pem",
        )
        for secret_path in secret_paths:
            with self.subTest(secret_path=secret_path):
                with tempfile.TemporaryDirectory() as tmp:
                    sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
                    write_sdist(sdist, ("pyproject.toml", secret_path))
                    with self.assertRaisesRegex(ValueError, "forbidden file in sdist"):
                        validator.validate_sdist(sdist)

    def test_rejects_unsafe_sdist_path(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            with tarfile.open(sdist, "w:gz") as archive:
                with tempfile.NamedTemporaryFile() as payload:
                    Path(payload.name).write_text("", encoding="utf-8")
                    archive.add(payload.name, arcname="grimace_py-0.1.12/../escape")
            with self.assertRaisesRegex(ValueError, "unsafe sdist path"):
                validator.validate_sdist(sdist)

    def test_rejects_non_tar_sdist(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            sdist.write_text("not a tarball", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "could not read source distribution"):
                validator.validate_sdist(sdist)

    def test_rejects_tag_that_does_not_match_release_version_shape(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "release tag must look like"):
                validator.validate_artifacts(Path(tmp), "not-a-version")

    def test_sdist_only_cli_validates_sdist_content_without_release_wheels(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            write_sdist(sdist, ("pyproject.toml", "Cargo.toml"))
            self.assertEqual(0, validator.main([str(sdist), "--sdist-only"]))

    def test_release_cli_requires_tag(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(2, validator.main([str(tmp)]))
            self.assertIn("--tag is required", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
