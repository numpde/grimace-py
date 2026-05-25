from __future__ import annotations

from pathlib import Path


_DEFAULT_MOLECULE_FIXTURE = Path("tests") / "fixtures" / "top_100000_CIDs.tsv.gz"


def _repo_fixture_path(relative_path: Path) -> Path:
    cwd_path = Path.cwd() / relative_path
    if cwd_path.exists():
        return cwd_path

    for parent in Path(__file__).resolve().parents:
        candidate = parent / relative_path
        if candidate.exists():
            return candidate

    return relative_path


DEFAULT_MOLECULE_SOURCE_PATH = _repo_fixture_path(_DEFAULT_MOLECULE_FIXTURE)
