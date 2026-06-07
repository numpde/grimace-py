from __future__ import annotations

import io
import json
import tarfile
import tempfile
import unittest
import zipfile
from pathlib import Path

from scripts.validate_release_artifacts import validate_artifacts


_VERSION = "1.2.3"
_SDIST_ROOT = f"grimace_py-{_VERSION}"
_SCRIPT = "scripts/prepared_mol_zstd_dictionary_generate.py"
_DICT_DIR = "20260531_abcdef12"
_SDIST_MANIFEST = (
    "python/grimace/data/prepared_mol_zstd/"
    f"{_DICT_DIR}/default_v1.json"
)
_SDIST_DICT = (
    "python/grimace/data/prepared_mol_zstd/"
    f"{_DICT_DIR}/default_v1.zstdict"
)
_WHEEL_MANIFEST = (
    "grimace/data/prepared_mol_zstd/"
    f"{_DICT_DIR}/default_v1.json"
)
_WHEEL_DICT = (
    "grimace/data/prepared_mol_zstd/"
    f"{_DICT_DIR}/default_v1.zstdict"
)


class ValidateReleaseArtifactsTest(unittest.TestCase):
    def test_full_release_validation_accepts_wheel_manifest_script_present_in_sdist(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            _write_sdist(
                dist_dir / f"grimace_py-{_VERSION}.tar.gz",
                include_script=True,
            )
            _write_wheel(
                dist_dir
                / f"grimace_py-{_VERSION}-cp312-cp312-manylinux_2_28_x86_64.whl"
            )

            validate_artifacts(dist_dir, f"v{_VERSION}")

    def test_full_release_validation_rejects_wheel_manifest_script_missing_from_sdist(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            dist_dir = Path(tmp)
            _write_sdist(
                dist_dir / f"grimace_py-{_VERSION}.tar.gz",
                include_script=False,
            )
            _write_wheel(
                dist_dir
                / f"grimace_py-{_VERSION}-cp312-cp312-manylinux_2_28_x86_64.whl"
            )

            with self.assertRaises(ValueError) as raised:
                validate_artifacts(dist_dir, f"v{_VERSION}")

        self.assertIn(
            "wheel PreparedMol zstd manifest references generator script "
            "missing from companion sdist",
            str(raised.exception),
        )


def _manifest_bytes() -> bytes:
    return json.dumps(
        {
            "training_identity": {
                "generator": {
                    "script": _SCRIPT,
                },
            },
        },
        sort_keys=True,
    ).encode("utf-8")


def _write_sdist(path: Path, *, include_script: bool) -> None:
    members = {
        _SDIST_MANIFEST: _manifest_bytes(),
        _SDIST_DICT: b"dictionary",
    }

    if include_script:
        members[_SCRIPT] = b"#!/usr/bin/env python3\n"

    with tarfile.open(path, "w:gz") as archive:
        for relative, data in members.items():
            info = tarfile.TarInfo(f"{_SDIST_ROOT}/{relative}")
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))


def _write_wheel(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr(_WHEEL_MANIFEST, _manifest_bytes())
        archive.writestr(_WHEEL_DICT, b"dictionary")
        archive.writestr(
            f"grimace_py-{_VERSION}.dist-info/METADATA",
            "Name: grimace-py\nVersion: 1.2.3\n",
        )


if __name__ == "__main__":
    unittest.main()
