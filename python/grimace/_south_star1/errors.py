"""Typed South Star boundary errors."""

from __future__ import annotations

from enum import Enum


class SouthStarErrorKind(Enum):
    UNSUPPORTED_ATOM = "unsupported_atom"
    UNSUPPORTED_BOND = "unsupported_bond"
    UNSUPPORTED_STEREO = "unsupported_stereo"
    UNSUPPORTED_POLICY = "unsupported_policy"
    INVALID_FACTS = "invalid_facts"
    SEMANTIC_MISMATCH = "semantic_mismatch"
    INTERNAL_INVARIANT = "internal_invariant"


class SouthStarError(Exception):
    """Machine-checkable error at a South Star boundary."""

    def __init__(self, kind: SouthStarErrorKind, message: str) -> None:
        self.kind = kind
        self.message = message
        super().__init__(message)


__all__ = (
    "SouthStarError",
    "SouthStarErrorKind",
)
