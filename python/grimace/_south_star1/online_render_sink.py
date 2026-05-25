"""Render sinks for online South Star enumeration and decoding."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class OnlineRenderSink(Protocol):
    def checkpoint(self) -> int: ...
    def rollback(self, checkpoint: int) -> None: ...
    def append(self, text: str) -> bool: ...
    def complete(self) -> bool: ...
    def value(self) -> str: ...


@dataclass(slots=True)
class OnlineStringBuffer:
    parts: list[str]

    def __init__(self) -> None:
        self.parts = []

    def checkpoint(self) -> int:
        return len(self.parts)

    def rollback(self, checkpoint: int) -> None:
        if checkpoint < 0 or checkpoint > len(self.parts):
            raise ValueError(f"invalid render checkpoint: {checkpoint!r}")
        del self.parts[checkpoint:]

    def append(self, text: str) -> bool:
        if not text:
            return True
        self.parts.append(text)
        return True

    def complete(self) -> bool:
        return True

    def value(self) -> str:
        return "".join(self.parts)


@dataclass(slots=True)
class PrefixConstrainedSink:
    required_prefix: str
    emitted: list[str]

    def __init__(self, required_prefix: str) -> None:
        self.required_prefix = required_prefix
        self.emitted = []

    def checkpoint(self) -> int:
        return len(self.emitted)

    def rollback(self, checkpoint: int) -> None:
        if checkpoint < 0 or checkpoint > len(self.emitted):
            raise ValueError(f"invalid render checkpoint: {checkpoint!r}")
        del self.emitted[checkpoint:]

    def append(self, text: str) -> bool:
        if not text:
            return True
        current = self.value()
        candidate = current + text
        required = self.required_prefix
        if len(current) >= len(required):
            self.emitted.append(text)
            return True
        compare_len = min(len(candidate), len(required))
        if candidate[:compare_len] != required[:compare_len]:
            return False
        self.emitted.append(text)
        return True

    def complete(self) -> bool:
        return self.value().startswith(self.required_prefix)

    def value(self) -> str:
        return "".join(self.emitted)


__all__ = (
    "OnlineRenderSink",
    "OnlineStringBuffer",
    "PrefixConstrainedSink",
)
