from __future__ import annotations

from pathlib import Path


PACKAGE_REFERENCE_ROOT = Path(__file__).resolve().parent / "_data"
REFERENCE_ARTIFACTS_ROOT = PACKAGE_REFERENCE_ROOT / "reference"
DEFAULT_MOLECULE_SOURCE_PATH = PACKAGE_REFERENCE_ROOT / "top_100000_CIDs.tsv.gz"


def resolve_bundled_reference_path(path: str | Path) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    if raw_path.parts[:2] == ("tests", "fixtures"):
        return PACKAGE_REFERENCE_ROOT.joinpath(*raw_path.parts[2:])
    return PACKAGE_REFERENCE_ROOT / raw_path
