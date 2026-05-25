"""Decision paths for lazy online South Star decoder states."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class OnlineDecision:
    kind: str
    value: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class OnlineDecisionPath:
    items: tuple[OnlineDecision, ...]


class OnlineDecisionRecorder:
    def __init__(self) -> None:
        self._items: list[OnlineDecision] = []

    def checkpoint(self) -> int:
        return len(self._items)

    def rollback(self, checkpoint: int) -> None:
        if checkpoint < 0 or checkpoint > len(self._items):
            raise ValueError(f"invalid decision checkpoint: {checkpoint!r}")
        del self._items[checkpoint:]

    def push(self, decision: OnlineDecision) -> None:
        self._items.append(decision)

    def path(self) -> OnlineDecisionPath:
        return OnlineDecisionPath(tuple(self._items))


@dataclass(frozen=True, slots=True)
class DecisionPathFilter:
    allowed_paths: frozenset[OnlineDecisionPath]

    def allows_prefix(self, path: OnlineDecisionPath) -> bool:
        prefix = path.items
        return any(candidate.items[: len(prefix)] == prefix for candidate in self.allowed_paths)

    def allows_complete_prefix(self, path: OnlineDecisionPath) -> bool:
        return self.allows_prefix(path)


__all__ = (
    "DecisionPathFilter",
    "OnlineDecision",
    "OnlineDecisionPath",
    "OnlineDecisionRecorder",
)
