"""Validate focused release artifact invariants."""

from __future__ import annotations

import json
import re
import tarfile
import zipfile
from pathlib import Path


PACKAGE_STEM = "grimace_py"
WHEEL_PREPARED_MOL_ZSTD_PREFIX = "grimace/data/prepared_mol_zstd/"
_VERSION_TAG_RE = re.compile(r"^v(?P<version>\d+\.\d+\.\d+(?:[A-Za-z0-9_.+-]*)?)$")


def validate_wheel(wheel_path: Path) -> None:
    try:
        with zipfile.ZipFile(wheel_path) as archive:
            for name in archive.namelist():
                parts = name.split("/", 1)
                top_level = parts[0]

                if top_level == "grimace":
                    _validate_wheel_grimace_member(name)
                    continue

                if top_level.startswith(f"{PACKAGE_STEM}-") and top_level.endswith(
                    ".dist-info"
                ):
                    continue

                raise ValueError(f"unexpected top-level wheel member: {name!r}")
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"could not read wheel {wheel_path}: {exc}") from exc


def _validate_wheel_grimace_member(name: str) -> None:
    if not name.startswith(WHEEL_PREPARED_MOL_ZSTD_PREFIX):
        return

    if name.endswith("/"):
        return

    if name.endswith("/default_v1.json") or name.endswith("/default_v1.zstdict"):
        return

    raise ValueError(f"unexpected PreparedMol zstd wheel member: {name!r}")


def validate_sdist(sdist_path: Path) -> None:
    sdist_relative_members(sdist_path)


def sdist_relative_members(sdist_path: Path) -> frozenset[str]:
    if not sdist_path.name.endswith(".tar.gz"):
        raise ValueError(f"source distribution must be a .tar.gz file: {sdist_path}")

    root = f"{sdist_path.name.removesuffix('.tar.gz')}/"
    members: set[str] = set()

    try:
        with tarfile.open(sdist_path, "r:gz") as archive:
            for member in archive.getmembers():
                name = member.name
                if not name.startswith(root):
                    continue
                relative = name[len(root) :]
                if relative:
                    members.add(relative)
    except (OSError, tarfile.TarError) as exc:
        raise ValueError(f"could not read source distribution {sdist_path}: {exc}") from exc

    return frozenset(members)


def wheel_prepared_mol_zstd_manifest_generator_scripts(
    wheel_path: Path,
) -> frozenset[str]:
    scripts: set[str] = set()

    try:
        with zipfile.ZipFile(wheel_path) as archive:
            for name in archive.namelist():
                if not name.startswith(WHEEL_PREPARED_MOL_ZSTD_PREFIX):
                    continue
                if not name.endswith("/default_v1.json"):
                    continue

                try:
                    manifest = json.loads(archive.read(name).decode("utf-8"))
                except (KeyError, UnicodeDecodeError, json.JSONDecodeError) as exc:
                    raise ValueError(
                        f"could not read PreparedMol zstd manifest {name!r} "
                        f"from wheel {wheel_path}"
                    ) from exc

                generator = manifest.get("training_identity", {}).get("generator", {})
                script = generator.get("script")

                if not isinstance(script, str) or not script:
                    raise ValueError(
                        f"PreparedMol zstd manifest lacks generator.script: {name!r}"
                    )

                if script.startswith("/") or "\\" in script or ".." in Path(script).parts:
                    raise ValueError(
                        f"unsafe PreparedMol zstd generator script path: {script!r}"
                    )

                scripts.add(script)
    except (OSError, zipfile.BadZipFile) as exc:
        raise ValueError(f"could not read wheel {wheel_path}: {exc}") from exc

    return frozenset(scripts)


def validate_artifacts(dist_dir: Path, tag: str) -> None:
    tag_match = _VERSION_TAG_RE.match(tag)
    if tag_match is None:
        raise ValueError(f"release tag must be shaped like vX.Y.Z: {tag!r}")

    if not dist_dir.is_dir():
        raise ValueError(f"release artifact directory does not exist: {dist_dir}")

    paths = tuple(sorted(path for path in dist_dir.iterdir() if path.is_file()))
    if not paths:
        raise ValueError(f"release artifact directory is empty: {dist_dir}")

    for path in paths:
        if path.name.endswith(".whl"):
            validate_wheel(path)
        elif path.name.endswith(".tar.gz"):
            validate_sdist(path)
        else:
            raise ValueError(f"unexpected release artifact: {path.name!r}")

    sdist_path = dist_dir / f"{PACKAGE_STEM}-{tag_match.group('version')}.tar.gz"
    if not sdist_path.exists():
        raise ValueError(f"missing companion source distribution: {sdist_path.name}")

    sdist_members = sdist_relative_members(sdist_path)

    for path in paths:
        if not path.name.endswith(".whl"):
            continue

        for script in wheel_prepared_mol_zstd_manifest_generator_scripts(path):
            if script not in sdist_members:
                raise ValueError(
                    "wheel PreparedMol zstd manifest references generator script "
                    f"missing from companion sdist: {path.name}: {script!r}"
                )
