from pathlib import Path
from contextlib import redirect_stderr
from io import BytesIO
import hashlib
import io
import importlib.util
import json
import tarfile
import tempfile
import tomllib
import unittest
import warnings
import zipfile


ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / "scripts" / "validate_release_artifacts.py"
TEST_DICTIONARY_ARTIFACT = "20000102_1234abcd"
TEST_SECOND_DICTIONARY_ARTIFACT = "20000103_5678abcd"
TEST_GENERATOR_SCRIPT = "scripts/generate_prepared_mol_zstd_dictionary.py"
TEST_SECOND_GENERATOR_SCRIPT = "scripts/prepared_mol_zstd_dictionary_generate.py"
TEST_DICTIONARY_PAYLOAD = b"test prepared-mol zstd dictionary"
TEST_DICTIONARY_SHA256 = hashlib.sha256(TEST_DICTIONARY_PAYLOAD).hexdigest()
SDIST_DICTIONARY_NAMES = (
    f"python/grimace/data/prepared_mol_zstd/{TEST_DICTIONARY_ARTIFACT}/default_v1.json",
    f"python/grimace/data/prepared_mol_zstd/{TEST_DICTIONARY_ARTIFACT}/default_v1.zstdict",
)
WHEEL_DICTIONARY_NAMES = (
    f"grimace/data/prepared_mol_zstd/{TEST_DICTIONARY_ARTIFACT}/default_v1.json",
    f"grimace/data/prepared_mol_zstd/{TEST_DICTIONARY_ARTIFACT}/default_v1.zstdict",
)


def sdist_dictionary_names(artifact: str) -> tuple[str, str]:
    return (
        f"python/grimace/data/prepared_mol_zstd/{artifact}/default_v1.json",
        f"python/grimace/data/prepared_mol_zstd/{artifact}/default_v1.zstdict",
    )


def wheel_dictionary_names(artifact: str) -> tuple[str, str]:
    return (
        f"grimace/data/prepared_mol_zstd/{artifact}/default_v1.json",
        f"grimace/data/prepared_mol_zstd/{artifact}/default_v1.zstdict",
    )


def project_metadata() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]


def dictionary_manifest(
    script: str = TEST_GENERATOR_SCRIPT,
    *,
    artifact: str = TEST_DICTIONARY_ARTIFACT,
    dictionary_id: object = 123_456,
    dictionary_sha256: str = TEST_DICTIONARY_SHA256,
    dictionary_size_bytes: int = len(TEST_DICTIONARY_PAYLOAD),
    dictionary_file: str = "default_v1.zstdict",
    training_level: object = 3,
) -> str:
    return json.dumps(
        {
            "artifact_dir": artifact,
            "files": {
                "dictionary": dictionary_file,
                "manifest": "default_v1.json",
            },
            "zstd_dictionary_id": dictionary_id,
            "zstd_dictionary_sha256": dictionary_sha256,
            "zstd_dictionary_size_bytes": dictionary_size_bytes,
            "training_identity": {
                "generator": {
                    "script": script,
                },
                "training_parameters": {
                    "level": training_level,
                },
            },
        },
    )


def dictionary_manifest_for_payload(
    payload: bytes,
    script: str = TEST_GENERATOR_SCRIPT,
    *,
    artifact: str = TEST_DICTIONARY_ARTIFACT,
) -> str:
    return dictionary_manifest(
        script,
        artifact=artifact,
        dictionary_sha256=hashlib.sha256(payload).hexdigest(),
        dictionary_size_bytes=len(payload),
    )


def wheel_metadata(
    *,
    version: str = "0.1.12",
    name: str | None = None,
    source_url: str | None = None,
) -> str:
    metadata = project_metadata()
    project_name = metadata["name"] if name is None else name
    project_source_url = (
        metadata["urls"]["Source"] if source_url is None else source_url
    )
    return "\n".join(
        (
            "Metadata-Version: 2.4",
            f"Name: {project_name}",
            f"Version: {version}",
            f"Project-URL: Source, {project_source_url}",
            "",
        )
    )


def wheel_metadata_with_project_urls(*project_urls: str, version: str = "0.1.12") -> str:
    metadata = project_metadata()
    return "\n".join(
        (
            "Metadata-Version: 2.4",
            f"Name: {metadata['name']}",
            f"Version: {version}",
            *(f"Project-URL: {project_url}" for project_url in project_urls),
            "",
        )
    )


def wheel_metadata_with_headers(*headers: str) -> str:
    return "\n".join((*headers, ""))


def wheel_metadata_name(path: Path) -> str:
    version = path.name.split("-", 2)[1]
    return f"grimace_py-{version}.dist-info/METADATA"


def wheel_archive_metadata_name(path: Path) -> str:
    version = path.name.split("-", 2)[1]
    return f"grimace_py-{version}.dist-info/WHEEL"


def wheel_record_name(path: Path) -> str:
    version = path.name.split("-", 2)[1]
    return f"grimace_py-{version}.dist-info/RECORD"


def wheel_tag(path: Path) -> str:
    parts = path.name.removesuffix(".whl").split("-")
    if len(parts) < 5:
        raise ValueError(f"cannot infer wheel tag from filename: {path.name!r}")
    return "-".join(parts[-3:])


def native_extension_name(path: Path) -> str:
    python_tag = wheel_tag(path).split("-", 1)[0]
    return f"grimace/_core.cpython-{python_tag.removeprefix('cp')}-x86_64-linux-gnu.so"


def wheel_names_with_extra_dictionaries(path: Path) -> tuple[str, ...]:
    return (
        "grimace/__init__.py",
        native_extension_name(path),
        *wheel_dictionary_names(TEST_DICTIONARY_ARTIFACT),
        *wheel_dictionary_names(TEST_SECOND_DICTIONARY_ARTIFACT),
    )


def wheel_archive_metadata(*, tag: str) -> str:
    return wheel_archive_metadata_with_headers(
        "Wheel-Version: 1.0",
        "Root-Is-Purelib: false",
        f"Tag: {tag}",
    )


def wheel_archive_metadata_with_headers(*headers: str) -> str:
    return "\n".join((*headers, ""))


def pyproject_toml(
    *,
    version: str = "0.1.12",
    name: str | None = None,
    source_url: str | None = None,
) -> str:
    metadata = project_metadata()
    project_name = metadata["name"] if name is None else name
    project_source_url = (
        metadata["urls"]["Source"] if source_url is None else source_url
    )
    return "\n".join(
        (
            "[project]",
            f"name = {json.dumps(project_name)}",
            f"version = {json.dumps(version)}",
            "",
            "[project.urls]",
            f"Source = {json.dumps(project_source_url)}",
            "",
        )
    )


def archive_payload(
    name: str,
    *,
    manifest_script: str = TEST_GENERATOR_SCRIPT,
    wheel_path: Path | None = None,
    version: str = "0.1.12",
) -> bytes:
    if name.endswith("/default_v1.json"):
        artifact = name.rsplit("/", 2)[-2]
        return dictionary_manifest(manifest_script, artifact=artifact).encode("utf-8")
    if name.endswith("/default_v1.zstdict"):
        return TEST_DICTIONARY_PAYLOAD
    if name.endswith(".dist-info/METADATA"):
        version = name.split("-", 1)[1].split(".dist-info/", 1)[0]
        return wheel_metadata(version=version).encode("utf-8")
    if name.endswith(".dist-info/WHEEL"):
        if wheel_path is None:
            raise AssertionError("wheel_path is required for WHEEL payloads")
        return wheel_archive_metadata(tag=wheel_tag(wheel_path)).encode("utf-8")
    if name.endswith(".dist-info/RECORD"):
        return b""
    if name == "pyproject.toml":
        return pyproject_toml(version=version).encode("utf-8")
    return b""


def write_sdist(
    path: Path,
    names: tuple[str, ...],
    *,
    manifest_script: str = TEST_GENERATOR_SCRIPT,
    include_generator_script: bool = True,
    directory_names: tuple[str, ...] = (),
    include_root_directory: bool = False,
    include_root_directory_marker: bool = False,
    payload_overrides: dict[str, bytes] | None = None,
) -> None:
    root = path.name.removesuffix(".tar.gz")
    version = path.name.removesuffix(".tar.gz").split("-", 1)[1]
    if (
        include_generator_script
        and any(name.endswith("/default_v1.json") for name in names)
        and manifest_script not in names
    ):
        names = (*names, manifest_script)
    payload_overrides = payload_overrides or {}
    with tarfile.open(path, "w:gz") as archive:
        if include_root_directory:
            info = tarfile.TarInfo(root)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        if include_root_directory_marker:
            info = tarfile.TarInfo(f"{root}/")
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        for name in directory_names:
            full_name = f"{root}/{name}"
            info = tarfile.TarInfo(full_name)
            info.type = tarfile.DIRTYPE
            archive.addfile(info)
        for name in names:
            full_name = f"{root}/{name}"
            info = tarfile.TarInfo(full_name)
            payload = payload_overrides.get(
                name,
                archive_payload(
                    name,
                    manifest_script=manifest_script,
                    version=version,
                ),
            )
            info.size = len(payload)
            archive.addfile(info, BytesIO(payload))


def write_wheel(
    path: Path,
    names: tuple[str, ...] | None = None,
    *,
    manifest_script: str = TEST_GENERATOR_SCRIPT,
    include_dist_info_metadata: bool = True,
    payload_overrides: dict[str, bytes] | None = None,
) -> None:
    if names is None:
        names = (
            "grimace/__init__.py",
            native_extension_name(path),
            *WHEEL_DICTIONARY_NAMES,
        )
    if include_dist_info_metadata:
        for metadata_name in (
            wheel_metadata_name(path),
            wheel_archive_metadata_name(path),
            wheel_record_name(path),
        ):
            if metadata_name not in names:
                names = (*names, metadata_name)
    payload_overrides = payload_overrides or {}
    with zipfile.ZipFile(path, "w") as archive:
        for name in names:
            archive.writestr(
                name,
                payload_overrides.get(
                    name,
                    archive_payload(
                        name,
                        manifest_script=manifest_script,
                        wheel_path=path,
                    ),
                ),
            )


def write_wheel_with_archive_metadata(path: Path, *headers: str) -> None:
    write_wheel(
        path,
        payload_overrides={
            wheel_archive_metadata_name(path): (
                wheel_archive_metadata_with_headers(*headers).encode("utf-8")
            ),
        },
    )


def write_expected_artifacts(validator, dist: Path, version: str) -> None:
    for name in validator.expected_artifact_names(version):
        if name.endswith(".tar.gz"):
            write_sdist(
                dist / name,
                ("pyproject.toml", "Cargo.toml", *SDIST_DICTIONARY_NAMES),
            )
        else:
            write_wheel(dist / name)


def load_validator():
    spec = importlib.util.spec_from_file_location("validate_release_artifacts", SCRIPT)
    if spec is None or spec.loader is None:
        raise AssertionError("could not load release artifact validator")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class ReleaseArtifactValidationTests(unittest.TestCase):
    def test_official_source_url_policy_matches_pyproject(self) -> None:
        validator = load_validator()
        self.assertEqual(
            project_metadata()["urls"]["Source"],
            validator.EXPECTED_PROJECT_SOURCE_URL,
        )

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

    def test_accepts_explicit_sdist_root_directory_member(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            write_sdist(
                sdist,
                ("pyproject.toml", "Cargo.toml", *SDIST_DICTIONARY_NAMES),
                include_root_directory=True,
            )
            validator.validate_sdist(sdist)

    def test_accepts_explicit_sdist_root_directory_marker(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            write_sdist(
                sdist,
                ("pyproject.toml", "Cargo.toml", *SDIST_DICTIONARY_NAMES),
                include_root_directory_marker=True,
            )
            validator.validate_sdist(sdist)

    def test_rejects_normalized_sdist_path_aliases(self) -> None:
        validator = load_validator()
        unsafe_names = (
            "grimace_py-0.1.12//pyproject.toml",
            "grimace_py-0.1.12/./pyproject.toml",
            "grimace_py-0.1.12//",
        )
        for unsafe_name in unsafe_names:
            with self.subTest(unsafe_name=unsafe_name):
                with tempfile.TemporaryDirectory() as tmp:
                    sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
                    with tarfile.open(sdist, "w:gz") as archive:
                        payload = b""
                        info = tarfile.TarInfo(unsafe_name)
                        info.size = len(payload)
                        archive.addfile(info, BytesIO(payload))
                    with self.assertRaisesRegex(ValueError, "unsafe sdist path"):
                        validator.validate_sdist(sdist)

    def test_rejects_platform_specific_unsafe_sdist_paths(self) -> None:
        validator = load_validator()
        unsafe_names = (
            "grimace_py-0.1.12/nested\\escape",
            "grimace_py-0.1.12/C:/escape",
        )
        for unsafe_name in unsafe_names:
            with self.subTest(unsafe_name=unsafe_name):
                with tempfile.TemporaryDirectory() as tmp:
                    sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
                    with tarfile.open(sdist, "w:gz") as archive:
                        with tempfile.NamedTemporaryFile() as payload:
                            Path(payload.name).write_text("", encoding="utf-8")
                            archive.add(payload.name, arcname=unsafe_name)
                    with self.assertRaisesRegex(ValueError, "unsafe sdist path"):
                        validator.validate_sdist(sdist)

    def test_rejects_non_tar_sdist(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            sdist.write_text("not a tarball", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "could not read source distribution"):
                validator.validate_sdist(sdist)

    def test_rejects_sdist_special_file(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            with tarfile.open(sdist, "w:gz") as archive:
                info = tarfile.TarInfo("grimace_py-0.1.12/fifo")
                info.type = tarfile.FIFOTYPE
                archive.addfile(info)
            with self.assertRaisesRegex(ValueError, "unexpected special file in sdist"):
                validator.validate_sdist(sdist)

    def test_rejects_duplicate_sdist_member(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            root = sdist.name.removesuffix(".tar.gz")
            with tarfile.open(sdist, "w:gz") as archive:
                for _ in range(2):
                    payload = b""
                    info = tarfile.TarInfo(f"{root}/pyproject.toml")
                    info.size = len(payload)
                    archive.addfile(info, BytesIO(payload))
            with self.assertRaisesRegex(ValueError, "duplicate archive member"):
                validator.validate_sdist(sdist)

    def test_rejects_duplicate_sdist_member_after_directory_marker_normalization(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            write_sdist(
                sdist,
                ("pyproject.toml", "Cargo.toml", *SDIST_DICTIONARY_NAMES),
                include_root_directory=True,
                include_root_directory_marker=True,
            )
            with self.assertRaisesRegex(ValueError, "duplicate archive member"):
                validator.validate_sdist(sdist)

    def test_rejects_unsafe_wheel_path(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(wheel, ("../escape",))
            with self.assertRaisesRegex(ValueError, "unsafe wheel path"):
                validator.validate_wheel(wheel)

    def test_rejects_platform_specific_unsafe_wheel_paths(self) -> None:
        validator = load_validator()
        unsafe_names = ("nested\\escape", "C:/escape")
        for unsafe_name in unsafe_names:
            with self.subTest(unsafe_name=unsafe_name):
                with tempfile.TemporaryDirectory() as tmp:
                    wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
                    write_wheel(wheel, (unsafe_name,))
                    with self.assertRaisesRegex(ValueError, "unsafe wheel path"):
                        validator.validate_wheel(wheel)

    def test_rejects_normalized_wheel_path_aliases(self) -> None:
        validator = load_validator()
        unsafe_names = ("grimace//__init__.py", "grimace/./__init__.py")
        for unsafe_name in unsafe_names:
            with self.subTest(unsafe_name=unsafe_name):
                with tempfile.TemporaryDirectory() as tmp:
                    wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
                    write_wheel(wheel, (unsafe_name,))
                    with self.assertRaisesRegex(ValueError, "unsafe wheel path"):
                        validator.validate_wheel(wheel)

    def test_rejects_wheel_link(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                info = zipfile.ZipInfo("grimace/link")
                info.create_system = 3
                info.external_attr = 0o120777 << 16
                archive.writestr(info, "target")
            with self.assertRaisesRegex(ValueError, "unexpected link in wheel"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_special_file(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            with zipfile.ZipFile(wheel, "w") as archive:
                info = zipfile.ZipInfo("grimace/fifo")
                info.create_system = 3
                info.external_attr = 0o010644 << 16
                archive.writestr(info, "")
            with self.assertRaisesRegex(ValueError, "unexpected special file in wheel"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_directory_mode_without_directory_marker(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            dictionary_name = WHEEL_DICTIONARY_NAMES[1]
            with zipfile.ZipFile(wheel, "w") as archive:
                archive.writestr("grimace/__init__.py", b"")
                archive.writestr(
                    WHEEL_DICTIONARY_NAMES[0],
                    archive_payload(WHEEL_DICTIONARY_NAMES[0]),
                )
                info = zipfile.ZipInfo(dictionary_name)
                info.create_system = 3
                info.external_attr = 0o040755 << 16
                archive.writestr(info, "")
                archive.writestr(
                    wheel_metadata_name(wheel),
                    archive_payload(wheel_metadata_name(wheel)),
                )
            with self.assertRaisesRegex(ValueError, "non-canonical directory"):
                validator.validate_wheel(wheel)

    def test_rejects_duplicate_wheel_member(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            metadata_name = wheel_metadata_name(wheel)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", UserWarning)
                write_wheel(
                    wheel,
                    (
                        "grimace/__init__.py",
                        *WHEEL_DICTIONARY_NAMES,
                        metadata_name,
                        metadata_name,
                    ),
                    include_dist_info_metadata=False,
                )
            with self.assertRaisesRegex(ValueError, "duplicate archive member"):
                validator.validate_wheel(wheel)

    def test_rejects_duplicate_wheel_member_after_directory_marker_normalization(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(wheel, ("grimace/data", "grimace/data/"))
            with self.assertRaisesRegex(ValueError, "path normalization"):
                validator.validate_wheel(wheel)

    def test_rejects_secret_shaped_wheel_content(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(wheel, ("grimace/__init__.py", ".config/gh/hosts.yml"))
            with self.assertRaisesRegex(ValueError, "forbidden file in wheel"):
                validator.validate_wheel(wheel)

    def test_rejects_unexpected_top_level_wheel_content(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(wheel, ("grimace/__init__.py", "other_package/__init__.py"))
            with self.assertRaisesRegex(ValueError, "unexpected top-level wheel member"):
                validator.validate_wheel(wheel)

    def test_rejects_bare_top_level_wheel_root_files(self) -> None:
        validator = load_validator()
        root_names = (
            "grimace",
            "grimace_py-0.1.12.dist-info",
        )
        for root_name in root_names:
            with self.subTest(root_name=root_name):
                with tempfile.TemporaryDirectory() as tmp:
                    wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
                    write_wheel(
                        wheel,
                        (
                            root_name,
                            "grimace/__init__.py",
                            *WHEEL_DICTIONARY_NAMES,
                        ),
                    )
                    with self.assertRaisesRegex(ValueError, "ambiguous top-level"):
                        validator.validate_wheel(wheel)

    def test_rejects_wrong_project_wheel_filename(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "other_project-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(wheel)
            with self.assertRaisesRegex(ValueError, "wheel filename does not match grimace_py"):
                validator.validate_wheel(wheel)

    def test_rejects_malformed_wheel_filename(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-notawheel.whl"
            write_wheel(
                wheel,
                ("grimace/__init__.py",),
                include_dist_info_metadata=False,
            )
            with self.assertRaisesRegex(ValueError, "wheel filename does not match grimace_py"):
                validator.validate_wheel(wheel)

    def test_accepts_wheel_filename_with_build_tag(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-1local-cp312-cp312-linux_x86_64.whl"
            write_wheel(wheel)
            validator.validate_wheel(wheel)

    def test_accepts_wheel_core_type_stub_alongside_native_extension(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                (
                    "grimace/__init__.py",
                    "grimace/_core.pyi",
                    native_extension_name(wheel),
                    *WHEEL_DICTIONARY_NAMES,
                ),
            )
            validator.validate_wheel(wheel)

    def test_rejects_wheel_without_prepared_mol_zstd_dictionary_data(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(wheel, ("grimace/__init__.py",))
            with self.assertRaisesRegex(ValueError, "missing PreparedMol zstd"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_without_native_extension(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(wheel, ("grimace/__init__.py", *WHEEL_DICTIONARY_NAMES))
            with self.assertRaisesRegex(ValueError, "native grimace._core"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_native_extension_that_does_not_match_python_tag(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp313-cp313-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                (
                    "grimace/__init__.py",
                    "grimace/_core.cpython-312-x86_64-linux-gnu.so",
                    *WHEEL_DICTIONARY_NAMES,
                ),
            )
            with self.assertRaisesRegex(ValueError, "native extension tag"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_native_extension_that_does_not_match_platform_tag(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                (
                    "grimace/__init__.py",
                    "grimace/_core.cpython-312-aarch64-linux-gnu.so",
                    *WHEEL_DICTIONARY_NAMES,
                ),
            )
            with self.assertRaisesRegex(ValueError, "native extension platform"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_with_unsupported_platform_tag(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-macosx_14_0_x86_64.whl"
            write_wheel(wheel)
            with self.assertRaisesRegex(ValueError, "platform tag is not supported"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_with_extra_native_extension_artifact(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                (
                    "grimace/__init__.py",
                    native_extension_name(wheel),
                    "grimace/_core.pyd",
                    *WHEEL_DICTIONARY_NAMES,
                ),
            )
            with self.assertRaisesRegex(ValueError, "exactly one native"):
                validator.validate_wheel(wheel)

    def test_rejects_incomplete_wheel_prepared_mol_zstd_dictionary_data(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(wheel, ("grimace/__init__.py", WHEEL_DICTIONARY_NAMES[0]))
            with self.assertRaisesRegex(ValueError, "incomplete PreparedMol zstd"):
                validator.validate_wheel(wheel)

    def test_rejects_sdist_without_prepared_mol_zstd_dictionary_data(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            write_sdist(sdist, ("pyproject.toml", "Cargo.toml"))
            with self.assertRaisesRegex(ValueError, "missing PreparedMol zstd"):
                validator.validate_sdist(sdist)

    def test_rejects_incomplete_sdist_prepared_mol_zstd_dictionary_data(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            write_sdist(
                sdist,
                ("pyproject.toml", "Cargo.toml", SDIST_DICTIONARY_NAMES[0]),
            )
            with self.assertRaisesRegex(ValueError, "incomplete PreparedMol zstd"):
                validator.validate_sdist(sdist)

    def test_rejects_manifest_artifact_dir_that_does_not_match_archive_path(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            write_sdist(
                sdist,
                ("pyproject.toml", "Cargo.toml", *SDIST_DICTIONARY_NAMES),
                payload_overrides={
                    SDIST_DICTIONARY_NAMES[0]: dictionary_manifest(
                        artifact=TEST_SECOND_DICTIONARY_ARTIFACT,
                    ).encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "artifact_dir does not match"):
                validator.validate_sdist(sdist)

    def test_rejects_manifest_dictionary_hash_that_does_not_match_payload(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={
                    WHEEL_DICTIONARY_NAMES[0]: dictionary_manifest(
                        dictionary_sha256="0" * 64,
                    ).encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "dictionary SHA-256"):
                validator.validate_wheel(wheel)

    def test_rejects_manifest_dictionary_size_that_does_not_match_payload(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={
                    WHEEL_DICTIONARY_NAMES[0]: dictionary_manifest(
                        dictionary_size_bytes=1,
                    ).encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "dictionary size"):
                validator.validate_wheel(wheel)

    def test_rejects_zero_size_prepared_mol_zstd_dictionary(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={
                    WHEEL_DICTIONARY_NAMES[0]: dictionary_manifest_for_payload(
                        b"",
                    ).encode("utf-8"),
                    WHEEL_DICTIONARY_NAMES[1]: b"",
                },
            )
            with self.assertRaisesRegex(ValueError, "dictionary size"):
                validator.validate_wheel(wheel)

    def test_rejects_manifest_dictionary_file_that_does_not_match_layout(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={
                    WHEEL_DICTIONARY_NAMES[0]: dictionary_manifest(
                        dictionary_file="other.zstdict",
                    ).encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "dictionary file"):
                validator.validate_wheel(wheel)

    def test_rejects_manifest_dictionary_id_that_is_not_runtime_usable(self) -> None:
        validator = load_validator()
        invalid_dictionary_ids = ("123456", 0, True)
        for dictionary_id in invalid_dictionary_ids:
            with self.subTest(dictionary_id=dictionary_id):
                with tempfile.TemporaryDirectory() as tmp:
                    wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
                    write_wheel(
                        wheel,
                        payload_overrides={
                            WHEEL_DICTIONARY_NAMES[0]: dictionary_manifest(
                                dictionary_id=dictionary_id,
                            ).encode("utf-8"),
                        },
                    )
                    with self.assertRaisesRegex(ValueError, "dictionary id"):
                        validator.validate_wheel(wheel)

    def test_rejects_manifest_training_level_that_is_not_runtime_usable(self) -> None:
        validator = load_validator()
        invalid_training_levels = ("3", True)
        for training_level in invalid_training_levels:
            with self.subTest(training_level=training_level):
                with tempfile.TemporaryDirectory() as tmp:
                    wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
                    write_wheel(
                        wheel,
                        payload_overrides={
                            WHEEL_DICTIONARY_NAMES[0]: dictionary_manifest(
                                training_level=training_level,
                            ).encode("utf-8"),
                        },
                    )
                    with self.assertRaisesRegex(ValueError, "training level"):
                        validator.validate_wheel(wheel)

    def test_rejects_duplicate_manifest_dictionary_ids(self) -> None:
        validator = load_validator()
        artifact_names = (
            *wheel_dictionary_names(TEST_DICTIONARY_ARTIFACT),
            *wheel_dictionary_names(TEST_SECOND_DICTIONARY_ARTIFACT),
        )
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                ("grimace/__init__.py", *artifact_names),
                payload_overrides={
                    wheel_dictionary_names(TEST_DICTIONARY_ARTIFACT)[0]: dictionary_manifest(
                        artifact=TEST_DICTIONARY_ARTIFACT,
                        dictionary_id=123_456,
                        training_level=3,
                    ).encode("utf-8"),
                    wheel_dictionary_names(TEST_SECOND_DICTIONARY_ARTIFACT)[0]: dictionary_manifest(
                        artifact=TEST_SECOND_DICTIONARY_ARTIFACT,
                        dictionary_id=123_456,
                        training_level=10,
                    ).encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "duplicate PreparedMol zstd dictionary id"):
                validator.validate_wheel(wheel)

    def test_rejects_duplicate_manifest_training_levels(self) -> None:
        validator = load_validator()
        artifact_names = (
            *wheel_dictionary_names(TEST_DICTIONARY_ARTIFACT),
            *wheel_dictionary_names(TEST_SECOND_DICTIONARY_ARTIFACT),
        )
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                ("grimace/__init__.py", *artifact_names),
                payload_overrides={
                    wheel_dictionary_names(TEST_DICTIONARY_ARTIFACT)[0]: dictionary_manifest(
                        artifact=TEST_DICTIONARY_ARTIFACT,
                        dictionary_id=123_456,
                        training_level=3,
                    ).encode("utf-8"),
                    wheel_dictionary_names(TEST_SECOND_DICTIONARY_ARTIFACT)[0]: dictionary_manifest(
                        artifact=TEST_SECOND_DICTIONARY_ARTIFACT,
                        dictionary_id=654_321,
                        training_level=3,
                    ).encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "duplicate PreparedMol zstd training level"):
                validator.validate_wheel(wheel)

    def test_rejects_sdist_manifest_without_recorded_generator_script(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            write_sdist(
                sdist,
                ("pyproject.toml", "Cargo.toml", *SDIST_DICTIONARY_NAMES),
                include_generator_script=False,
            )
            with self.assertRaisesRegex(ValueError, "absent from source distribution"):
                validator.validate_sdist(sdist)

    def test_rejects_sdist_manifest_generator_script_directory(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            write_sdist(
                sdist,
                ("pyproject.toml", "Cargo.toml", *SDIST_DICTIONARY_NAMES),
                include_generator_script=False,
                directory_names=(TEST_GENERATOR_SCRIPT,),
            )
            with self.assertRaisesRegex(ValueError, "absent from source distribution"):
                validator.validate_sdist(sdist)

    def test_rejects_unsafe_manifest_generator_script_path(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            write_sdist(
                sdist,
                ("pyproject.toml", "Cargo.toml", *SDIST_DICTIONARY_NAMES),
                manifest_script="../escape.py",
                include_generator_script=False,
            )
            with self.assertRaisesRegex(ValueError, "unsafe PreparedMol zstd"):
                validator.validate_sdist(sdist)

    def test_rejects_invalid_utf8_sdist_manifest(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            write_sdist(
                sdist,
                ("pyproject.toml", "Cargo.toml", *SDIST_DICTIONARY_NAMES),
                payload_overrides={SDIST_DICTIONARY_NAMES[0]: b"\xff"},
            )
            with self.assertRaisesRegex(ValueError, "not valid UTF-8"):
                validator.validate_sdist(sdist)

    def test_rejects_sdist_pyproject_version_mismatch(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            write_sdist(
                sdist,
                ("pyproject.toml", "Cargo.toml", *SDIST_DICTIONARY_NAMES),
                payload_overrides={
                    "pyproject.toml": pyproject_toml(version="0.1.99").encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "project version"):
                validator.validate_sdist(sdist)

    def test_rejects_sdist_source_url_that_is_not_official_repository(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            write_sdist(
                sdist,
                ("pyproject.toml", "Cargo.toml", *SDIST_DICTIONARY_NAMES),
                payload_overrides={
                    "pyproject.toml": pyproject_toml(
                        source_url="https://example.invalid/source",
                    ).encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "official repository URL"):
                validator.validate_sdist(sdist)

    def test_rejects_empty_sdist_source_url(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            write_sdist(
                sdist,
                ("pyproject.toml", "Cargo.toml", *SDIST_DICTIONARY_NAMES),
                payload_overrides={
                    "pyproject.toml": pyproject_toml(source_url="").encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "Source must be non-empty"):
                validator.validate_sdist(sdist)

    def test_rejects_invalid_utf8_wheel_manifest(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={WHEEL_DICTIONARY_NAMES[0]: b"\xff"},
            )
            with self.assertRaisesRegex(ValueError, "not valid UTF-8"):
                validator.validate_wheel(wheel)

    def test_wheel_only_cli_reports_invalid_utf8_metadata(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={wheel_metadata_name(wheel): b"\xff"},
            )
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(2, validator.main([str(wheel), "--wheel-only"]))
            self.assertIn("not valid UTF-8", stderr.getvalue())

    def test_rejects_wheel_manifest_without_source_project_url(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(wheel, include_dist_info_metadata=False)
            with self.assertRaisesRegex(ValueError, "METADATA"):
                validator.validate_wheel(wheel)

    def test_accepts_wheel_source_project_url_label_with_spacing_and_case(self) -> None:
        validator = load_validator()
        source_url = project_metadata()["urls"]["Source"]
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={
                    wheel_metadata_name(wheel): wheel_metadata_with_project_urls(
                        f" source , {source_url} ",
                    ).encode("utf-8"),
                },
            )
            validator.validate_wheel(wheel)

    def test_rejects_duplicate_wheel_source_project_url_labels(self) -> None:
        validator = load_validator()
        source_url = project_metadata()["urls"]["Source"]
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={
                    wheel_metadata_name(wheel): wheel_metadata_with_project_urls(
                        f"Source, {source_url}",
                        "Source, https://example.invalid/source",
                    ).encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "Project-URL"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_metadata_name_mismatch(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={
                    wheel_metadata_name(wheel): wheel_metadata(name="other-project").encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "project name"):
                validator.validate_wheel(wheel)

    def test_rejects_duplicate_wheel_metadata_name_header(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={
                    wheel_metadata_name(wheel): wheel_metadata_with_headers(
                        "Metadata-Version: 2.4",
                        "Name: grimace-py",
                        "Name: other-project",
                        "Version: 0.1.12",
                        f"Project-URL: Source, {project_metadata()['urls']['Source']}",
                    ).encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "exactly one Name"):
                validator.validate_wheel(wheel)

    def test_rejects_missing_wheel_metadata_version_dialect_header(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={
                    wheel_metadata_name(wheel): wheel_metadata_with_headers(
                        "Name: grimace-py",
                        "Version: 0.1.12",
                        f"Project-URL: Source, {project_metadata()['urls']['Source']}",
                    ).encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "exactly one Metadata-Version"):
                validator.validate_wheel(wheel)

    def test_rejects_duplicate_wheel_metadata_version_dialect_header(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={
                    wheel_metadata_name(wheel): wheel_metadata_with_headers(
                        "Metadata-Version: 2.4",
                        "Metadata-Version: 2.1",
                        "Name: grimace-py",
                        "Version: 0.1.12",
                        f"Project-URL: Source, {project_metadata()['urls']['Source']}",
                    ).encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "exactly one Metadata-Version"):
                validator.validate_wheel(wheel)

    def test_rejects_unsupported_wheel_metadata_version_dialect(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={
                    wheel_metadata_name(wheel): wheel_metadata_with_headers(
                        "Metadata-Version: 2.1",
                        "Name: grimace-py",
                        "Version: 0.1.12",
                        f"Project-URL: Source, {project_metadata()['urls']['Source']}",
                    ).encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "dialect is not supported"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_metadata_version_mismatch(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={
                    wheel_metadata_name(wheel): wheel_metadata(version="0.1.99").encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "METADATA version"):
                validator.validate_wheel(wheel)

    def test_rejects_duplicate_wheel_metadata_version_header(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={
                    wheel_metadata_name(wheel): wheel_metadata_with_headers(
                        "Metadata-Version: 2.4",
                        "Name: grimace-py",
                        "Version: 0.1.12",
                        "Version: 0.1.99",
                        f"Project-URL: Source, {project_metadata()['urls']['Source']}",
                    ).encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "exactly one Version"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_source_url_that_is_not_official_repository(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={
                    wheel_metadata_name(wheel): wheel_metadata(
                        source_url="https://example.invalid/source",
                    ).encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "official repository URL"):
                validator.validate_wheel(wheel)

    def test_rejects_empty_wheel_source_url(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                payload_overrides={
                    wheel_metadata_name(wheel): wheel_metadata(
                        source_url="",
                    ).encode("utf-8"),
                },
            )
            with self.assertRaisesRegex(ValueError, "Project-URL"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_without_canonical_metadata_member(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                (
                    "grimace/__init__.py",
                    *WHEEL_DICTIONARY_NAMES,
                    "grimace/not-package.dist-info/METADATA",
                ),
                include_dist_info_metadata=False,
            )
            with self.assertRaisesRegex(ValueError, "canonical METADATA"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_without_canonical_wheel_member(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                (
                    "grimace/__init__.py",
                    *WHEEL_DICTIONARY_NAMES,
                    wheel_metadata_name(wheel),
                    wheel_record_name(wheel),
                ),
                include_dist_info_metadata=False,
            )
            with self.assertRaisesRegex(ValueError, "canonical WHEEL"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_without_canonical_record_member(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel(
                wheel,
                (
                    "grimace/__init__.py",
                    *WHEEL_DICTIONARY_NAMES,
                    wheel_metadata_name(wheel),
                    wheel_archive_metadata_name(wheel),
                ),
                include_dist_info_metadata=False,
            )
            with self.assertRaisesRegex(ValueError, "canonical RECORD"):
                validator.validate_wheel(wheel)

    def test_rejects_unsupported_wheel_archive_metadata_version(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel_with_archive_metadata(
                wheel,
                "Wheel-Version: 2.0",
                "Root-Is-Purelib: false",
                "Tag: cp312-cp312-manylinux_2_28_x86_64",
            )
            with self.assertRaisesRegex(ValueError, "WHEEL metadata version"):
                validator.validate_wheel(wheel)

    def test_rejects_duplicate_wheel_archive_metadata_version(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel_with_archive_metadata(
                wheel,
                "Wheel-Version: 1.0",
                "Wheel-Version: 1.0",
                "Root-Is-Purelib: false",
                "Tag: cp312-cp312-manylinux_2_28_x86_64",
            )
            with self.assertRaisesRegex(ValueError, "exactly one Wheel-Version"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_archive_metadata_without_tags(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel_with_archive_metadata(
                wheel,
                "Wheel-Version: 1.0",
                "Root-Is-Purelib: false",
            )
            with self.assertRaisesRegex(ValueError, "lacks compatibility tags"):
                validator.validate_wheel(wheel)

    def test_rejects_empty_wheel_archive_metadata_tag(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel_with_archive_metadata(
                wheel,
                "Wheel-Version: 1.0",
                "Root-Is-Purelib: false",
                "Tag:",
            )
            with self.assertRaisesRegex(ValueError, "lacks compatibility tags"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_archive_metadata_without_root_is_purelib(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel_with_archive_metadata(
                wheel,
                "Wheel-Version: 1.0",
                "Tag: cp312-cp312-manylinux_2_28_x86_64",
            )
            with self.assertRaisesRegex(ValueError, "Root-Is-Purelib"):
                validator.validate_wheel(wheel)

    def test_rejects_duplicate_wheel_archive_root_is_purelib(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel_with_archive_metadata(
                wheel,
                "Wheel-Version: 1.0",
                "Root-Is-Purelib: false",
                "Root-Is-Purelib: false",
                "Tag: cp312-cp312-manylinux_2_28_x86_64",
            )
            with self.assertRaisesRegex(ValueError, "exactly one Root-Is-Purelib"):
                validator.validate_wheel(wheel)

    def test_rejects_invalid_wheel_archive_root_is_purelib(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel_with_archive_metadata(
                wheel,
                "Wheel-Version: 1.0",
                "Root-Is-Purelib: maybe",
                "Tag: cp312-cp312-manylinux_2_28_x86_64",
            )
            with self.assertRaisesRegex(ValueError, "Root-Is-Purelib is invalid"):
                validator.validate_wheel(wheel)

    def test_rejects_purelib_wheel_archive_metadata(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel_with_archive_metadata(
                wheel,
                "Wheel-Version: 1.0",
                "Root-Is-Purelib: true",
                "Tag: cp312-cp312-manylinux_2_28_x86_64",
            )
            with self.assertRaisesRegex(ValueError, "must be false"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_archive_metadata_tag_mismatch(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel_with_archive_metadata(
                wheel,
                "Wheel-Version: 1.0",
                "Root-Is-Purelib: false",
                "Tag: cp313-cp313-manylinux_2_28_x86_64",
            )
            with self.assertRaisesRegex(ValueError, "tags do not match filename"):
                validator.validate_wheel(wheel)

    def test_rejects_wheel_archive_metadata_extra_tag(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl"
            write_wheel_with_archive_metadata(
                wheel,
                "Wheel-Version: 1.0",
                "Root-Is-Purelib: false",
                "Tag: cp312-cp312-manylinux_2_28_x86_64",
                "Tag: cp312-cp312-linux_x86_64",
            )
            with self.assertRaisesRegex(ValueError, "tags do not match filename"):
                validator.validate_wheel(wheel)

    def test_rejects_release_when_wheel_generator_is_absent_from_sdist(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            write_expected_artifacts(validator, dist, "0.1.12")
            write_wheel(
                dist / "grimace_py-0.1.12-cp312-cp312-manylinux_2_28_x86_64.whl",
                manifest_script="scripts/missing_generator.py",
            )

            with self.assertRaisesRegex(ValueError, "companion source distribution"):
                validator.validate_artifacts(dist, "v0.1.12")

    def test_rejects_release_when_wheel_artifact_generator_differs_from_sdist(self) -> None:
        validator = load_validator()
        sdist_names = (
            "pyproject.toml",
            "Cargo.toml",
            *sdist_dictionary_names(TEST_DICTIONARY_ARTIFACT),
            *sdist_dictionary_names(TEST_SECOND_DICTIONARY_ARTIFACT),
            TEST_GENERATOR_SCRIPT,
            TEST_SECOND_GENERATOR_SCRIPT,
        )
        sdist_payloads = {
            sdist_dictionary_names(TEST_DICTIONARY_ARTIFACT)[0]: dictionary_manifest(
                TEST_GENERATOR_SCRIPT,
                artifact=TEST_DICTIONARY_ARTIFACT,
                dictionary_id=123_456,
                training_level=3,
            ).encode("utf-8"),
            sdist_dictionary_names(TEST_SECOND_DICTIONARY_ARTIFACT)[0]: dictionary_manifest(
                TEST_SECOND_GENERATOR_SCRIPT,
                artifact=TEST_SECOND_DICTIONARY_ARTIFACT,
                dictionary_id=654_321,
                training_level=10,
            ).encode("utf-8"),
        }
        wheel_payloads = {
            wheel_dictionary_names(TEST_DICTIONARY_ARTIFACT)[0]: dictionary_manifest(
                TEST_SECOND_GENERATOR_SCRIPT,
                artifact=TEST_DICTIONARY_ARTIFACT,
                dictionary_id=123_456,
                training_level=3,
            ).encode("utf-8"),
            wheel_dictionary_names(TEST_SECOND_DICTIONARY_ARTIFACT)[0]: dictionary_manifest(
                TEST_SECOND_GENERATOR_SCRIPT,
                artifact=TEST_SECOND_DICTIONARY_ARTIFACT,
                dictionary_id=654_321,
                training_level=10,
            ).encode("utf-8"),
        }
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            for name in validator.expected_artifact_names("0.1.12"):
                if name.endswith(".tar.gz"):
                    write_sdist(
                        dist / name,
                        sdist_names,
                        include_generator_script=False,
                        payload_overrides=sdist_payloads,
                    )
                else:
                    wheel = dist / name
                    write_wheel(
                        wheel,
                        wheel_names_with_extra_dictionaries(wheel),
                        include_dist_info_metadata=True,
                        payload_overrides=wheel_payloads,
                    )
            with self.assertRaisesRegex(ValueError, "generator provenance"):
                validator.validate_artifacts(dist, "v0.1.12")

    def test_rejects_release_when_wheel_dictionary_payload_differs_from_sdist(self) -> None:
        validator = load_validator()
        wheel_dictionary_payload = b"different dictionary"
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            write_expected_artifacts(validator, dist, "0.1.12")
            for wheel in dist.glob("*.whl"):
                write_wheel(
                    wheel,
                    payload_overrides={
                        WHEEL_DICTIONARY_NAMES[0]: dictionary_manifest_for_payload(
                            wheel_dictionary_payload,
                        ).encode("utf-8"),
                        WHEEL_DICTIONARY_NAMES[1]: wheel_dictionary_payload,
                    },
                )
            with self.assertRaisesRegex(ValueError, "package data bytes"):
                validator.validate_artifacts(dist, "v0.1.12")

    def test_rejects_release_when_wheel_manifest_payload_differs_from_sdist(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            dist = Path(tmp)
            write_expected_artifacts(validator, dist, "0.1.12")
            for wheel in dist.glob("*.whl"):
                write_wheel(
                    wheel,
                    payload_overrides={
                        WHEEL_DICTIONARY_NAMES[0]: dictionary_manifest(
                            TEST_GENERATOR_SCRIPT,
                            artifact=TEST_DICTIONARY_ARTIFACT,
                        )
                        .replace("}", ', "extra": true}', 1)
                        .encode("utf-8"),
                    },
                )
            with self.assertRaisesRegex(ValueError, "package data bytes"):
                validator.validate_artifacts(dist, "v0.1.12")

    def test_rejects_tag_that_does_not_match_release_version_shape(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaisesRegex(ValueError, "release tag must look like"):
                validator.validate_artifacts(Path(tmp), "not-a-version")

    def test_sdist_only_cli_validates_sdist_content_without_release_wheels(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            sdist = Path(tmp) / "grimace_py-0.1.12.tar.gz"
            write_sdist(
                sdist,
                ("pyproject.toml", "Cargo.toml", *SDIST_DICTIONARY_NAMES),
            )
            self.assertEqual(0, validator.main([str(sdist), "--sdist-only"]))

    def test_wheel_only_cli_validates_wheel_content_without_release_set(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-linux_x86_64.whl"
            write_wheel(wheel)
            self.assertEqual(0, validator.main([str(wheel), "--wheel-only"]))

    def test_release_cli_requires_tag(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(2, validator.main([str(tmp)]))
            self.assertIn("--tag is required unless an artifact-only mode is used", stderr.getvalue())

    def test_artifact_only_cli_modes_are_mutually_exclusive(self) -> None:
        validator = load_validator()
        with tempfile.TemporaryDirectory() as tmp:
            wheel = Path(tmp) / "grimace_py-0.1.12-cp312-cp312-linux_x86_64.whl"
            write_wheel(wheel)
            stderr = io.StringIO()
            with redirect_stderr(stderr):
                self.assertEqual(2, validator.main([str(wheel), "--wheel-only", "--sdist-only"]))
            self.assertIn("--sdist-only and --wheel-only cannot be used together", stderr.getvalue())


if __name__ == "__main__":
    unittest.main()
