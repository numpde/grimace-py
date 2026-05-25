"""Decision paths for lazy online South Star decoder states."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True)
class OnlineDecision:
    kind: str
    value: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class OnlineDecisionPath:
    items: tuple[OnlineDecision, ...]


@dataclass(frozen=True, slots=True)
class OnlineDecisionFrontier:
    paths: frozenset[OnlineDecisionPath]


class FrontierCompactionMode(Enum):
    TRAVERSAL_ONLY = "traversal_only"
    FULL_DECISION_PREFIX = "full_decision_prefix"


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


@dataclass(slots=True)
class DecisionPathFilter:
    allowed_frontier: OnlineDecisionFrontier
    rejection_count: int = 0

    def allows_prefix(self, path: OnlineDecisionPath) -> bool:
        ok = any(
            _compatible(path.items, frontier.items)
            for frontier in self.allowed_frontier.paths
        )
        if not ok:
            self.rejection_count += 1
        return ok

    def allows_complete_prefix(self, path: OnlineDecisionPath) -> bool:
        return self.allows_prefix(path)


def compact_frontier_path(
    path: OnlineDecisionPath,
    *,
    mode: FrontierCompactionMode = FrontierCompactionMode.TRAVERSAL_ONLY,
) -> OnlineDecisionPath:
    if mode is FrontierCompactionMode.TRAVERSAL_ONLY:
        return OnlineDecisionPath(
            tuple(item for item in path.items if item.kind == "traversal")
        )
    if mode is FrontierCompactionMode.FULL_DECISION_PREFIX:
        return OnlineDecisionPath(path.items)
    raise ValueError(f"unknown frontier compaction mode: {mode!r}")


def _compatible(
    left: tuple[OnlineDecision, ...],
    right: tuple[OnlineDecision, ...],
) -> bool:
    n = min(len(left), len(right))
    return left[:n] == right[:n]


__all__ = (
    "DecisionPathFilter",
    "FrontierCompactionMode",
    "OnlineDecisionFrontier",
    "OnlineDecision",
    "OnlineDecisionPath",
    "OnlineDecisionRecorder",
    "compact_frontier_path",
)
