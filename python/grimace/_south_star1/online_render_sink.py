"""Render sinks for online South Star enumeration and decoding."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field
from typing import Protocol


class OnlineRenderSink(Protocol):
    def checkpoint(self) -> object: ...
    def rollback(self, checkpoint: object) -> None: ...
    def append(self, text: str, *, token_text: str | None = None) -> bool: ...
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

    def append(self, text: str, *, token_text: str | None = None) -> bool:
        del token_text
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

    def append(self, text: str, *, token_text: str | None = None) -> bool:
        del token_text
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


@dataclass(slots=True)
class PrefixFrontierSink:
    required_prefix: str
    emitted: list[str] = field(default_factory=list)
    frontier_chars: set[str] = field(default_factory=set)
    frontier_token_texts: set[str] = field(default_factory=set)
    pending_char: str | None = None
    pending_token_text: str | None = None
    sink_rejections: int = 0
    completions_seen: int = 0

    def checkpoint(self) -> tuple[int, str | None, str | None]:
        return (len(self.emitted), self.pending_char, self.pending_token_text)

    def rollback(self, checkpoint: object) -> None:
        if not isinstance(checkpoint, tuple) or len(checkpoint) != 3:
            raise ValueError(f"invalid frontier checkpoint: {checkpoint!r}")
        emitted_len, pending_char, pending_token_text = checkpoint
        if not isinstance(emitted_len, int) or emitted_len < 0 or emitted_len > len(self.emitted):
            raise ValueError(f"invalid frontier emitted checkpoint: {checkpoint!r}")
        del self.emitted[emitted_len:]
        self.pending_char = pending_char  # type: ignore[assignment]
        self.pending_token_text = pending_token_text  # type: ignore[assignment]

    def append(self, text: str, *, token_text: str | None = None) -> bool:
        if not text:
            return True
        current = self.value()
        candidate = current + text
        prefix = self.required_prefix
        if len(current) < len(prefix):
            compare_len = min(len(candidate), len(prefix))
            if candidate[:compare_len] != prefix[:compare_len]:
                self.sink_rejections += 1
                return False
            if len(candidate) > len(prefix) and self.pending_char is None:
                self.pending_char = candidate[len(prefix)]
            if len(current) == len(prefix) and token_text is not None:
                self.pending_token_text = token_text
        elif len(current) == len(prefix):
            if self.pending_char is None:
                self.pending_char = text[0]
            if token_text is not None:
                self.pending_token_text = token_text
        self.emitted.append(text)
        return True

    def complete(self) -> bool:
        ok = self.value().startswith(self.required_prefix)
        if ok:
            self.completions_seen += 1
            if self.pending_char is not None:
                self.frontier_chars.add(self.pending_char)
            if self.pending_token_text is not None:
                self.frontier_token_texts.add(self.pending_token_text)
        return ok

    def value(self) -> str:
        return "".join(self.emitted)


__all__ = (
    "OnlineRenderSink",
    "OnlineStringBuffer",
    "PrefixConstrainedSink",
    "PrefixFrontierSink",
)
