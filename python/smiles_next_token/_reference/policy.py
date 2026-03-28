from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[3]
REFERENCE_ARTIFACTS_ROOT = REPO_ROOT / "tests" / "fixtures" / "reference"
DEFAULT_RDKIT_RANDOM_POLICY_PATH = (
    REFERENCE_ARTIFACTS_ROOT
    / "rdkit_random"
    / "branches"
    / "general"
    / "policies"
    / "rdkit_random_v1.json"
)
DEFAULT_RDKIT_RANDOM_CONNECTED_NONSTEREO_POLICY_PATH = (
    REFERENCE_ARTIFACTS_ROOT
    / "rdkit_random"
    / "branches"
    / "connected_nonstereo"
    / "policies"
    / "rdkit_random_connected_nonstereo_v1.json"
)


def _canonicalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ReferencePolicy:
    data: dict[str, Any]
    source_path: Path | None = None

    @classmethod
    def from_path(cls, path: str | Path) -> "ReferencePolicy":
        source_path = Path(path)
        return cls(data=json.loads(source_path.read_text(encoding="utf-8")), source_path=source_path)

    @property
    def policy_name(self) -> str:
        return str(self.data["policy_name"])

    @property
    def policy_kind(self) -> str:
        return str(self.data["policy_kind"])

    @property
    def branch_family(self) -> str:
        return str(self.data["branch_family"])

    def canonical_data(self) -> dict[str, Any]:
        canonical = _canonicalize_json(self.data)
        if not isinstance(canonical, dict):
            raise TypeError("Policy content must be a JSON object")
        return canonical

    def canonical_json(self) -> str:
        return json.dumps(self.canonical_data(), sort_keys=True, separators=(",", ":"))

    def digest(self, *, length: int = 8) -> str:
        if length < 1:
            raise ValueError("length must be positive")
        return hashlib.sha256(self.canonical_json().encode("utf-8")).hexdigest()[:length]

    def artifacts_root(self, base_root: str | Path = REFERENCE_ARTIFACTS_ROOT) -> Path:
        return Path(base_root) / self.policy_kind

    def branches_root(self, base_root: str | Path = REFERENCE_ARTIFACTS_ROOT) -> Path:
        return self.artifacts_root(base_root) / "branches"

    def branch_dir(self, base_root: str | Path = REFERENCE_ARTIFACTS_ROOT) -> Path:
        return self.branches_root(base_root) / self.branch_family

    def policy_dir(self, base_root: str | Path = REFERENCE_ARTIFACTS_ROOT) -> Path:
        return self.branch_dir(base_root) / "policies"

    def snapshot_dir(self, base_root: str | Path = REFERENCE_ARTIFACTS_ROOT) -> Path:
        return self.branch_dir(base_root) / "snapshots" / self.policy_name / self.digest()

    def core_exact_sets_path(self, base_root: str | Path = REFERENCE_ARTIFACTS_ROOT) -> Path:
        return self.snapshot_dir(base_root) / "core_exact_sets.json"

    def metrics_path(
        self,
        selection_tag: str = "full",
        base_root: str | Path = REFERENCE_ARTIFACTS_ROOT,
    ) -> Path:
        return self.snapshot_dir(base_root) / f"{selection_tag}_metrics.json.gz"

    def full_metrics_path(self, base_root: str | Path = REFERENCE_ARTIFACTS_ROOT) -> Path:
        return self.metrics_path("full", base_root)
