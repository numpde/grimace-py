"""Helpers for optional kernel imports."""

from __future__ import annotations

try:
    import grimace._core as CORE_MODULE
except ImportError:  # pragma: no cover - exercised only when the extension is absent
    CORE_MODULE = None
