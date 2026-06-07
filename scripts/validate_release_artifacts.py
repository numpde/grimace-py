#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections.abc import Iterable
from email.message import Message
from email.parser import Parser
import hashlib
import json
from pathlib import Path
import re
import sys
import tarfile
import tomllib
import zipfile


PACKAGE_STEM = "grimace_py"
PROJECT_NAME = "grimace-py"
EXPECTED_PROJECT_SOURCE_URL = "https://github.com/numpde/grimace-py"
EXPECTED_WHEEL_METADATA_VERSION = "2.4"
PYTHON_TAGS = ("cp312", "cp313")
PLATFORM_TAG = "manylinux_2_28_x86_64"
TAG_PATTERN = re.compile(r"^v(?P<version>[0-9]+\.[0-9]+\.[0-9]+)$")
WHEEL_NAME_PATTERN = re.compile(
    rf"^{re.escape(PACKAGE_STEM)}-"
    rf"(?P<version>[0-9]+\.[0-9]+\.[0-9]+)"
    rf"(?:-[0-9][A-Za-z0-9_.]*)?"
    rf"-[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*"
    rf"-[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*"
    rf"-[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*"
    rf"\.whl$"
)
SDIST_NAME_PATTERN = re.compile(
    rf"^{re.escape(PACKAGE_STEM)}-(?P<version>[0-9]+\.[0-9]+\.[0-9]+)\.tar\.gz$"
)
PREPARED_MOL_ZSTD_ARTIFACT_PATTERN = re.compile(r"^[0-9]{8}_[0-9a-f]{8}$")
PREPARED_MOL_ZSTD_FILENAMES = frozenset(
    {"default_v1.json", "default_v1.zstdict"}
)
SDIST_PREPARED_MOL_ZSTD_PREFIX = "python/grimace/data/prepared_mol_zstd/"
WHEEL_PREPARED_MOL_ZSTD_PREFIX = "grimace/data/prepared_mol_zstd/"
FORBIDDEN_SDIST_PREFIXES = (
    ".git/",
    ".github/",
    ".venv/",
    "build/",
    "dist/",
    "htmlcov/",
    "notes/perf_reports/",
    "target/",
    "tmp/",
)
FORBIDDEN_SDIST_NAMES = {
    ".env",
    ".envrc",
    ".netrc",
    ".npmrc",
    ".pypirc",
    "id_ed25519",
    "id_rsa",
    "pip.conf",
    "pip.ini",
}
FORBIDDEN_SDIST_SUFFIXES = (
    ".crt",
    ".key",
    ".p12",
    ".pem",
    ".pfx",
    ".secret",
    ".token",
)
FORBIDDEN_SDIST_COMPONENTS = {
    ".aws",
    ".azure",
    ".docker",
    ".gcloud",
    ".gnupg",
    ".kube",
    ".ssh",
}
FORBIDDEN_SDIST_PATH_FRAGMENTS = (
    (".cargo", "credentials"),
    (".cargo", "credentials.toml"),
    (".config", "gcloud"),
    (".config", "gh"),
    (".config", "pip"),
)


class WheelInfo:
    __slots__ = ("version", "prepared_mol_zstd")

    def __init__(
        self,
        *,
        version: str,
        prepared_mol_zstd: "PreparedMolZstdPackageData",
    ) -> None:
        self.version = version
        self.prepared_mol_zstd = prepared_mol_zstd


class SdistInfo:
    __slots__ = ("version", "prepared_mol_zstd")

    def __init__(
        self,
        *,
        version: str,
        prepared_mol_zstd: "PreparedMolZstdPackageData",
    ) -> None:
        self.version = version
        self.prepared_mol_zstd = prepared_mol_zstd


class PreparedMolZstdPackageData:
    __slots__ = ("generator_scripts", "file_sha256")

    def __init__(
        self,
        *,
        generator_scripts: dict[str, str],
        file_sha256: dict[str, dict[str, str]],
    ) -> None:
        self.generator_scripts = generator_scripts
        self.file_sha256 = file_sha256


def expected_artifact_names(version: str) -> tuple[str, ...]:
    wheels = tuple(
        f"{PACKAGE_STEM}-{version}-{python_tag}-{python_tag}-{PLATFORM_TAG}.whl"
        for python_tag in PYTHON_TAGS
    )
    return tuple(sorted((*wheels, f"{PACKAGE_STEM}-{version}.tar.gz")))


def validate_artifacts(dist_dir: Path, tag: str) -> None:
    tag_match = TAG_PATTERN.fullmatch(tag)
    if tag_match is None:
        raise ValueError(f"release tag must look like vX.Y.Z, got {tag!r}")

    if not dist_dir.is_dir():
        raise ValueError(f"artifact directory does not exist: {dist_dir}")

    paths = tuple(sorted(dist_dir.iterdir(), key=lambda path: path.name))
    non_files = tuple(path.name for path in paths if path.is_symlink() or not path.is_file())
    if non_files:
        raise ValueError(f"unexpected non-file release artifacts: {list(non_files)!r}")

    actual = tuple(path.name for path in paths)
    expected = expected_artifact_names(tag_match.group("version"))
    if actual != expected:
        raise ValueError(
            "unexpected release artifacts\n"
            f"expected: {list(expected)!r}\n"
            f"actual:   {list(actual)!r}"
        )

    wheel_infos: list[WheelInfo] = []
    for path in paths:
        if path.name.endswith(".whl"):
            wheel_infos.append(validate_wheel(path))
    sdist_info = validate_sdist(
        dist_dir / f"{PACKAGE_STEM}-{tag_match.group('version')}.tar.gz"
    )
    if sdist_info.version != tag_match.group("version"):
        raise ValueError(
            "source distribution version does not match release tag: "
            f"{sdist_info.version!r}"
        )
    for wheel_info in wheel_infos:
        if wheel_info.version != tag_match.group("version"):
            raise ValueError(
                "wheel version does not match release tag: "
                f"{wheel_info.version!r}"
            )
        if (
            wheel_info.prepared_mol_zstd.generator_scripts
            != sdist_info.prepared_mol_zstd.generator_scripts
        ):
            raise ValueError(
                "wheel PreparedMol zstd generator provenance does not match "
                "the companion source distribution"
            )
        if (
            wheel_info.prepared_mol_zstd.file_sha256
            != sdist_info.prepared_mol_zstd.file_sha256
        ):
            raise ValueError(
                "wheel PreparedMol zstd package data bytes do not match "
                "the companion source distribution"
            )


def is_forbidden_archive_member(relative: str) -> bool:
    path = Path(relative)
    parts = path.parts
    name = path.name
    if name in FORBIDDEN_SDIST_NAMES or name.startswith(".env."):
        return True
    if name.lower().endswith(FORBIDDEN_SDIST_SUFFIXES):
        return True
    if any(part in FORBIDDEN_SDIST_COMPONENTS for part in parts):
        return True
    if any(relative.startswith(prefix) for prefix in FORBIDDEN_SDIST_PREFIXES):
        return True
    return any(
        parts[index : index + 2] == fragment
        for index in range(len(parts) - 1)
        for fragment in FORBIDDEN_SDIST_PATH_FRAGMENTS
    )


def is_unsafe_archive_path(
    name: str,
    *,
    allow_directory_marker: bool = False,
) -> bool:
    if allow_directory_marker and name.endswith("/"):
        name = name[:-1]
    parts = name.split("/")
    return (
        not name
        or name.startswith("/")
        or "\\" in name
        or any(part in ("", ".", "..") for part in parts)
        or any(re.match(r"^[A-Za-z]:", part) is not None for part in parts)
    )


def reject_duplicate_archive_member(name: str, seen_names: set[str]) -> None:
    if name in seen_names:
        raise ValueError(f"duplicate archive member: {name!r}")
    seen_names.add(name)


def decode_archive_text(payload: bytes, member_name: str) -> str:
    try:
        return payload.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ValueError(f"archive member is not valid UTF-8: {member_name!r}") from exc


def sha256_hex(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def canonical_project_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def validate_sdist_project_metadata(
    pyproject_text: str | None,
    *,
    expected_version: str,
) -> None:
    if pyproject_text is None:
        raise ValueError("source distribution lacks pyproject.toml")
    try:
        pyproject = tomllib.loads(pyproject_text)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError("source distribution pyproject.toml is invalid") from exc
    try:
        project = pyproject["project"]
        name = project["name"]
        version = project["version"]
        source_url = project["urls"]["Source"]
    except (KeyError, TypeError) as exc:
        raise ValueError("source distribution pyproject.toml lacks project metadata") from exc
    if not isinstance(name, str) or canonical_project_name(name) != PROJECT_NAME:
        raise ValueError("source distribution project name does not match grimace-py")
    if version != expected_version:
        raise ValueError("source distribution project version does not match filename")
    if not isinstance(source_url, str) or not source_url:
        raise ValueError("source distribution project.urls.Source must be non-empty")
    if source_url != EXPECTED_PROJECT_SOURCE_URL:
        raise ValueError(
            "source distribution project.urls.Source does not match the official "
            "repository URL"
        )


def validate_sdist(sdist_path: Path) -> SdistInfo:
    if sdist_path.is_symlink() or not sdist_path.is_file():
        raise ValueError(f"source distribution does not exist or is not a file: {sdist_path}")
    sdist_match = SDIST_NAME_PATTERN.fullmatch(sdist_path.name)
    if sdist_match is None:
        raise ValueError(
            f"source distribution filename does not match {PACKAGE_STEM}: {sdist_path.name!r}"
        )

    root = f"{sdist_path.name.removesuffix('.tar.gz')}/"
    root_dir = root.rstrip("/")
    names: list[str] = []
    file_names: list[str] = []
    manifest_texts: dict[str, str] = {}
    prepared_mol_zstd_file_sha256: dict[str, str] = {}
    pyproject_text: str | None = None
    seen_names: set[str] = set()
    try:
        with tarfile.open(sdist_path, "r:gz") as archive:
            for member in archive.getmembers():
                name = member.name
                reject_duplicate_archive_member(name, seen_names)
                names.append(name)
                if is_unsafe_archive_path(
                    name,
                    allow_directory_marker=member.isdir(),
                ):
                    raise ValueError(f"unsafe sdist path: {name!r}")
                if member.issym() or member.islnk():
                    raise ValueError(f"unexpected link in sdist: {name!r}")
                if not member.isfile() and not member.isdir():
                    raise ValueError(f"unexpected special file in sdist: {name!r}")
                if name in (root_dir, root) and member.isdir():
                    continue
                if not name.startswith(root):
                    raise ValueError(f"sdist member is outside archive root: {name!r}")

                relative = name[len(root) :]
                if not relative:
                    continue
                if is_forbidden_archive_member(relative):
                    raise ValueError(f"forbidden file in sdist: {relative!r}")
                if member.isfile():
                    file_names.append(relative)
                if relative == "pyproject.toml" and member.isfile():
                    payload = archive.extractfile(member)
                    if payload is None:
                        raise ValueError(f"could not read sdist member: {name!r}")
                    pyproject_text = decode_archive_text(payload.read(), relative)
                if (
                    relative.startswith(SDIST_PREPARED_MOL_ZSTD_PREFIX)
                    and member.isfile()
                ):
                    payload = archive.extractfile(member)
                    if payload is None:
                        raise ValueError(f"could not read sdist member: {name!r}")
                    payload_bytes = payload.read()
                    prepared_mol_zstd_file_sha256[relative] = sha256_hex(payload_bytes)
                    if relative.endswith("/default_v1.json"):
                        manifest_texts[relative] = decode_archive_text(
                            payload_bytes,
                            relative,
                        )
            relative_names = tuple(name[len(root) :] for name in names if name.startswith(root))
            prepared_mol_zstd = validate_prepared_mol_zstd_package_data(
                relative_names,
                prefix=SDIST_PREPARED_MOL_ZSTD_PREFIX,
                manifest_texts=manifest_texts,
                file_sha256_by_name=prepared_mol_zstd_file_sha256,
            )
            validate_prepared_mol_zstd_sdist_provenance(
                prepared_mol_zstd.generator_scripts,
                source_file_names=frozenset(file_names),
            )
            validate_sdist_project_metadata(
                pyproject_text,
                expected_version=sdist_match.group("version"),
            )
            return SdistInfo(
                version=sdist_match.group("version"),
                prepared_mol_zstd=prepared_mol_zstd,
            )
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"could not read source distribution {sdist_path}: {exc}") from exc


def validate_wheel(wheel_path: Path) -> WheelInfo:
    if wheel_path.is_symlink() or not wheel_path.is_file():
        raise ValueError(f"wheel does not exist or is not a file: {wheel_path}")
    if not wheel_path.name.endswith(".whl"):
        raise ValueError(f"wheel must be a .whl file: {wheel_path}")
    wheel_match = WHEEL_NAME_PATTERN.fullmatch(wheel_path.name)
    if wheel_match is None:
        raise ValueError(
            f"wheel filename does not match {PACKAGE_STEM}: {wheel_path.name!r}"
        )
    dist_info_root = f"{PACKAGE_STEM}-{wheel_match.group('version')}.dist-info"
    allowed_roots = {
        "grimace",
        dist_info_root,
    }

    try:
        with zipfile.ZipFile(wheel_path) as archive:
            names = archive.namelist()
            seen_names: set[str] = set()
            for member in archive.infolist():
                name = member.filename
                reject_duplicate_archive_member(name, seen_names)
                if is_unsafe_archive_path(
                    name,
                    allow_directory_marker=member.is_dir(),
                ):
                    raise ValueError(f"unsafe wheel path: {name!r}")
                mode = (member.external_attr >> 16) & 0o170000
                if mode == 0o120000:
                    raise ValueError(f"unexpected link in wheel: {name!r}")
                if mode not in (0, 0o040000, 0o100000):
                    raise ValueError(f"unexpected special file in wheel: {name!r}")
                if mode == 0o040000 and not member.is_dir():
                    raise ValueError(
                        f"non-canonical directory entry in wheel: {name!r}"
                    )
                if is_forbidden_archive_member(name):
                    raise ValueError(f"forbidden file in wheel: {name!r}")
                root = name.rstrip("/").split("/", 1)[0]
                if root not in allowed_roots:
                    raise ValueError(f"unexpected top-level wheel member: {name!r}")
            validate_wheel_source_metadata(
                archive,
                names,
                dist_info_root=dist_info_root,
                expected_version=wheel_match.group("version"),
            )
            prepared_mol_zstd_file_sha256: dict[str, str] = {}
            manifest_texts: dict[str, str] = {}
            for name in names:
                if not name.startswith(WHEEL_PREPARED_MOL_ZSTD_PREFIX):
                    continue
                if name.endswith("/"):
                    continue
                payload = archive.read(name)
                prepared_mol_zstd_file_sha256[name] = sha256_hex(payload)
                if name.endswith("/default_v1.json"):
                    manifest_texts[name] = decode_archive_text(payload, name)
            prepared_mol_zstd = validate_prepared_mol_zstd_package_data(
                names,
                prefix=WHEEL_PREPARED_MOL_ZSTD_PREFIX,
                manifest_texts=manifest_texts,
                file_sha256_by_name=prepared_mol_zstd_file_sha256,
            )
            return WheelInfo(
                version=wheel_match.group("version"),
                prepared_mol_zstd=prepared_mol_zstd,
            )
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"could not read wheel {wheel_path}: {exc}") from exc


def validate_wheel_source_metadata(
    archive: zipfile.ZipFile,
    names: Iterable[str],
    *,
    dist_info_root: str,
    expected_version: str,
) -> None:
    metadata_name = f"{dist_info_root}/METADATA"
    if metadata_name not in names:
        raise ValueError(f"wheel lacks canonical METADATA file: {metadata_name!r}")
    metadata = decode_archive_text(archive.read(metadata_name), metadata_name)
    message = Parser().parsestr(metadata)
    if (
        single_metadata_header(message, "Metadata-Version")
        != EXPECTED_WHEEL_METADATA_VERSION
    ):
        raise ValueError("wheel METADATA version dialect is not supported")
    name = single_metadata_header(message, "Name")
    if canonical_project_name(name) != PROJECT_NAME:
        raise ValueError("wheel METADATA project name does not match grimace-py")
    if single_metadata_header(message, "Version") != expected_version:
        raise ValueError("wheel METADATA version does not match filename")
    source_urls = tuple(
        url
        for value in message.get_all("Project-URL", ())
        for label, url in (project_url_parts(value),)
        if label == "source"
    )
    if len(source_urls) != 1 or not source_urls[0]:
        raise ValueError("wheel METADATA lacks the source repository Project-URL")
    if source_urls[0] != EXPECTED_PROJECT_SOURCE_URL:
        raise ValueError(
            "wheel METADATA source repository URL does not match the official "
            "repository URL"
        )


def single_metadata_header(message: Message, header: str) -> str:
    values = message.get_all(header, ())
    if len(values) != 1 or not values[0]:
        raise ValueError(f"wheel METADATA must contain exactly one {header} header")
    return values[0]


def project_url_parts(value: str) -> tuple[str, str]:
    label, separator, url = value.partition(",")
    if not separator:
        return "", ""
    return label.strip().casefold(), url.strip()


def validate_prepared_mol_zstd_package_data(
    names: Iterable[str],
    *,
    prefix: str,
    manifest_texts: dict[str, str],
    file_sha256_by_name: dict[str, str],
) -> PreparedMolZstdPackageData:
    artifacts: dict[str, set[str]] = {}
    generator_scripts: dict[str, str] = {}
    file_sha256: dict[str, dict[str, str]] = {}
    for name in names:
        if not name.startswith(prefix) or name.endswith("/"):
            continue
        relative = name[len(prefix) :]
        parts = relative.split("/")
        if len(parts) != 2:
            raise ValueError(f"unexpected PreparedMol zstd package data path: {name!r}")
        artifact, filename = parts
        if PREPARED_MOL_ZSTD_ARTIFACT_PATTERN.fullmatch(artifact) is None:
            raise ValueError(
                f"unexpected PreparedMol zstd artifact directory: {name!r}"
            )
        if filename not in PREPARED_MOL_ZSTD_FILENAMES:
            raise ValueError(f"unexpected PreparedMol zstd package data file: {name!r}")
        artifacts.setdefault(artifact, set()).add(filename)
        try:
            digest = file_sha256_by_name[name]
        except KeyError as exc:
            raise ValueError(
                f"missing PreparedMol zstd package data payload: {name!r}"
            ) from exc
        file_sha256.setdefault(artifact, {})[filename] = digest
        if filename == "default_v1.json":
            try:
                manifest_artifact, script = prepared_mol_zstd_manifest_provenance(
                    manifest_texts[name],
                    name,
                )
            except KeyError as exc:
                raise ValueError(
                    f"missing PreparedMol zstd manifest payload: {name!r}"
                ) from exc
            if manifest_artifact != artifact:
                raise ValueError(
                    "PreparedMol zstd manifest artifact_dir does not match "
                    f"archive path: {name!r}"
                )
            generator_scripts[artifact] = script

    if not artifacts:
        raise ValueError("missing PreparedMol zstd dictionary package data")
    incomplete = {
        artifact: sorted(PREPARED_MOL_ZSTD_FILENAMES - filenames)
        for artifact, filenames in artifacts.items()
        if filenames != PREPARED_MOL_ZSTD_FILENAMES
    }
    if incomplete:
        raise ValueError(
            f"incomplete PreparedMol zstd dictionary package data: {incomplete!r}"
        )
    return PreparedMolZstdPackageData(
        generator_scripts=generator_scripts,
        file_sha256=file_sha256,
    )


def validate_prepared_mol_zstd_sdist_provenance(
    artifact_scripts: dict[str, str],
    *,
    source_file_names: frozenset[str],
) -> None:
    for script in artifact_scripts.values():
        if script not in source_file_names:
            raise ValueError(
                "PreparedMol zstd generator script is absent from source "
                f"distribution: {script!r}"
            )


def prepared_mol_zstd_manifest_provenance(
    manifest_text: str,
    member_name: str,
) -> tuple[str, str]:
    try:
        manifest = json.loads(manifest_text)
        artifact = manifest["artifact_dir"]
        script = manifest["training_identity"]["generator"]["script"]
    except (KeyError, TypeError, json.JSONDecodeError) as exc:
        raise ValueError(
            f"PreparedMol zstd manifest lacks generator provenance: {member_name!r}"
        ) from exc
    if not isinstance(artifact, str):
        raise ValueError(
            f"PreparedMol zstd manifest artifact_dir is not a string: {member_name!r}"
        )
    if not isinstance(script, str):
        raise ValueError(
            f"PreparedMol zstd manifest generator script is not a string: {member_name!r}"
        )
    if is_unsafe_archive_path(script):
        raise ValueError(
            f"unsafe PreparedMol zstd generator script path: {script!r}"
        )
    if not script.startswith("scripts/") or not script.endswith(".py"):
        raise ValueError(
            "PreparedMol zstd generator script must be a scripts/*.py source "
            f"path: {script!r}"
        )
    return artifact, script


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_path", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--sdist-only", action="store_true")
    parser.add_argument("--wheel-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.sdist_only and args.wheel_only:
            raise ValueError("--sdist-only and --wheel-only cannot be used together")
        if args.sdist_only:
            if args.tag is not None:
                raise ValueError("--tag cannot be used with --sdist-only")
            validate_sdist(args.artifact_path)
        elif args.wheel_only:
            if args.tag is not None:
                raise ValueError("--tag cannot be used with --wheel-only")
            validate_wheel(args.artifact_path)
        else:
            if args.tag is None:
                raise ValueError("--tag is required unless an artifact-only mode is used")
            validate_artifacts(args.artifact_path, args.tag)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
