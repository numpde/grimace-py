# Single owner for pinned tool/runtime versions used by release and container
# posture checks. The checked files still contain the pins; tests import these
# names so version bumps are deliberate one-line policy changes.
MATURIN_VERSION = "1.13.1"
MATURIN_ACTION_VERSION = f"v{MATURIN_VERSION}"
RDKIT_VERSION = "2026.3.1"
TWINE_VERSION = "6.2.0"
ZSTANDARD_VERSION = "0.25.0"
PLOX_VERSION = "0.0.3"


def rdkit_runtime_version() -> str:
    year, month, patch = RDKIT_VERSION.split(".")
    return f"{year}.{int(month):02d}.{patch}"
