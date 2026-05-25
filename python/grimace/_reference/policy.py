from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any


def _canonicalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _canonicalize_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonicalize_json(item) for item in value]
    return value


@dataclass(frozen=True)
class ReferencePolicy:
    data: dict[str, Any]

    @property
    def policy_name(self) -> str:
        return str(self.data["policy_name"])

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
