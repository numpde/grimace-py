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


@dataclass(frozen=True, slots=True)
class OnlineDecisionFrontier:
    paths: frozenset[OnlineDecisionPath]


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
    allowed_frontier: OnlineDecisionFrontier

    def allows_prefix(self, path: OnlineDecisionPath) -> bool:
        return any(
            _compatible(path.items, frontier.items)
            for frontier in self.allowed_frontier.paths
        )

    def allows_complete_prefix(self, path: OnlineDecisionPath) -> bool:
        return self.allows_prefix(path)


def compact_frontier_path(path: OnlineDecisionPath) -> OnlineDecisionPath:
    """Keep the branch prefix needed to resume the current traversal frontier."""

    return OnlineDecisionPath(
        tuple(item for item in path.items if item.kind == "traversal")
    )


def _compatible(
    left: tuple[OnlineDecision, ...],
    right: tuple[OnlineDecision, ...],
) -> bool:
    n = min(len(left), len(right))
    return left[:n] == right[:n]


__all__ = (
    "DecisionPathFilter",
    "OnlineDecisionFrontier",
    "OnlineDecision",
    "OnlineDecisionPath",
    "OnlineDecisionRecorder",
    "compact_frontier_path",
)
