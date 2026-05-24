#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import re
import sys


PACKAGE_STEM = "grimace_py"
PYTHON_TAGS = ("cp312", "cp313")
PLATFORM_TAG = "manylinux_2_28_x86_64"
TAG_PATTERN = re.compile(r"^v(?P<version>[0-9]+\.[0-9]+\.[0-9]+)$")


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


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("dist_dir", type=Path)
    parser.add_argument("--tag", required=True)
    args = parser.parse_args(argv)

    try:
        validate_artifacts(args.dist_dir, args.tag)
    except ValueError as exc:
        print(exc, file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
