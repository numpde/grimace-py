#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys
import tarfile


PACKAGE_STEM = "grimace_py"
PYTHON_TAGS = ("cp312", "cp313")
PLATFORM_TAG = "manylinux_2_28_x86_64"
TAG_PATTERN = re.compile(r"^v(?P<version>[0-9]+\.[0-9]+\.[0-9]+)$")
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

    validate_sdist(dist_dir / f"{PACKAGE_STEM}-{tag_match.group('version')}.tar.gz")


def is_forbidden_sdist_member(relative: str) -> bool:
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


def validate_sdist(sdist_path: Path) -> None:
    if sdist_path.is_symlink() or not sdist_path.is_file():
        raise ValueError(f"source distribution does not exist or is not a file: {sdist_path}")
    if not sdist_path.name.endswith(".tar.gz"):
        raise ValueError(f"source distribution must be a .tar.gz file: {sdist_path}")

    root = f"{sdist_path.name.removesuffix('.tar.gz')}/"
    try:
        with tarfile.open(sdist_path, "r:gz") as archive:
            for member in archive.getmembers():
                name = member.name
                if name.startswith("/") or ".." in Path(name).parts:
                    raise ValueError(f"unsafe sdist path: {name!r}")
                if member.issym() or member.islnk():
                    raise ValueError(f"unexpected link in sdist: {name!r}")
                if not name.startswith(root):
                    raise ValueError(f"sdist member is outside archive root: {name!r}")

                relative = name[len(root) :]
                if not relative:
                    continue
                if is_forbidden_sdist_member(relative):
                    raise ValueError(f"forbidden file in sdist: {relative!r}")
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"could not read source distribution {sdist_path}: {exc}") from exc


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact_path", type=Path)
    parser.add_argument("--tag")
    parser.add_argument("--sdist-only", action="store_true")
    args = parser.parse_args(argv)

    try:
        if args.sdist_only:
            if args.tag is not None:
                raise ValueError("--tag cannot be used with --sdist-only")
            validate_sdist(args.artifact_path)
        else:
            if args.tag is None:
                raise ValueError("--tag is required unless --sdist-only is used")
            validate_artifacts(args.artifact_path, args.tag)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
