from __future__ import annotations

from pathlib import Path


PACKAGE_REFERENCE_ROOT = Path(__file__).resolve().parent / "_data"
PACKAGE_REFERENCE_LABEL = Path("grimace") / "_reference" / "_data"
DEFAULT_REFERENCE_OUTPUT_ROOT = Path("grimace_reference_artifacts")
REFERENCE_ARTIFACTS_ROOT = PACKAGE_REFERENCE_ROOT / "reference"
DEFAULT_MOLECULE_SOURCE_PATH = PACKAGE_REFERENCE_ROOT / "top_100000_CIDs.tsv.gz"


def resolve_bundled_reference_path(path: str | Path) -> Path:
    raw_path = Path(path)
    if raw_path.is_absolute():
        return raw_path
    if raw_path.parts[:2] == ("tests", "fixtures"):
        return PACKAGE_REFERENCE_ROOT.joinpath(*raw_path.parts[2:])
    return PACKAGE_REFERENCE_ROOT / raw_path


def display_reference_path(path: str | Path) -> str:
    raw_path = Path(path)
    if raw_path.is_absolute():
        try:
            relative_path = raw_path.relative_to(PACKAGE_REFERENCE_ROOT)
        except ValueError:
            try:
                return str(raw_path.relative_to(Path.cwd()))
            except ValueError:
                return str(raw_path)
        return str(PACKAGE_REFERENCE_LABEL / relative_path)
    if raw_path.parts[:2] == ("tests", "fixtures"):
        bundled_path = PACKAGE_REFERENCE_ROOT.joinpath(*raw_path.parts[2:])
        if bundled_path.exists():
            return str(PACKAGE_REFERENCE_LABEL.joinpath(*raw_path.parts[2:]))
        return str(raw_path)
    return str(raw_path)
