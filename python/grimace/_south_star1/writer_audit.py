"""Optional reachability diagnostics for writer frontiers."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .writer_capabilities import _WriterExecutionCapabilityKind
from .writer_frontier import WriterFrontierCursor
from .writer_snapshot import WriterSearchSnapshot
from . import writer_snapshot


class _WriterFrontierReachabilityAuditKind(Enum):
    READY = "ready"
    BLOCKED = "blocked"
    TRUNCATED = "truncated"


@dataclass(frozen=True, slots=True)
class _WriterFrontierBlockedPrefix:
    emitted_texts: tuple[str, ...]
    choice_snapshot: writer_snapshot._WriterFrontierChoiceSnapshot

    @property
    def graph_policy_blockers(self):
        return self.choice_snapshot.graph_policy_blockers

    @property
    def stereo_policy_blockers(self):
        return self.choice_snapshot.stereo_policy_blockers


@dataclass(frozen=True, slots=True)
class _WriterExecutionCapabilityUse:
    kind: _WriterExecutionCapabilityKind
    emitted_texts: tuple[str, ...]
    source_cursor: WriterFrontierCursor
    successor_cursor: WriterFrontierCursor
    next_emitted_text: str | None = None

    @property
    def terminal(self) -> bool:
        return self.next_emitted_text is None


@dataclass(frozen=True, slots=True)
class _WriterFrontierReachabilityAudit:
    kind: _WriterFrontierReachabilityAuditKind
    visited_prefixes: tuple[tuple[str, ...], ...]
    execution_capability_uses: tuple[
        _WriterExecutionCapabilityUse,
        ...,
    ] = ()
    blocked_prefixes: tuple[
        _WriterFrontierBlockedPrefix,
        ...,
    ] = ()
    truncated_at_prefix: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if self.kind is _WriterFrontierReachabilityAuditKind.READY:
            valid = (
                not self.blocked_prefixes
                and self.truncated_at_prefix is None
            )
        elif self.kind is _WriterFrontierReachabilityAuditKind.BLOCKED:
            valid = (
                bool(self.blocked_prefixes)
                and self.truncated_at_prefix is None
            )
        elif self.kind is _WriterFrontierReachabilityAuditKind.TRUNCATED:
            valid = self.truncated_at_prefix is not None
        else:
            valid = False

        if not valid:
            raise SouthStarError(
                SouthStarErrorKind.INTERNAL_INVARIANT,
                f"invalid writer frontier reachability audit: {self.kind!r}",
            )

    @property
    def ready(self) -> bool:
        return self.kind is _WriterFrontierReachabilityAuditKind.READY

    @property
    def blocked(self) -> bool:
        return self.kind is _WriterFrontierReachabilityAuditKind.BLOCKED

    @property
    def truncated(self) -> bool:
        return self.kind is _WriterFrontierReachabilityAuditKind.TRUNCATED

    @property
    def blocked_emitted_texts(self) -> tuple[tuple[str, ...], ...]:
        return tuple(
            blocked.emitted_texts
            for blocked in self.blocked_prefixes
        )

    @property
    def required_execution_capabilities(self) -> frozenset[
        _WriterExecutionCapabilityKind
    ]:
        return frozenset(
            use.kind
            for use in self.execution_capability_uses
        )


def _audit_writer_frontier_reachability_from_snapshot(
    snapshot: WriterSearchSnapshot,
    *,
    prepared: SouthStarPreparedMol,
    max_depth: int | None = None,
    max_prefixes: int | None = None,
) -> _WriterFrontierReachabilityAudit:
    visited: list[tuple[str, ...]] = []
    blocked: list[_WriterFrontierBlockedPrefix] = []
    execution_capability_uses: list[_WriterExecutionCapabilityUse] = []
    observed_execution_capability_use_signatures: set[
        tuple[
            _WriterExecutionCapabilityKind,
            tuple[str, ...],
            str | None,
            WriterFrontierCursor,
            WriterFrontierCursor,
        ]
    ] = set()
    seen_cursors: set[WriterFrontierCursor] = set()

    def rec(
        current: WriterSearchSnapshot,
        prefix: tuple[str, ...],
    ) -> tuple[bool, tuple[str, ...] | None]:
        if current.cursor in seen_cursors:
            return False, None

        seen_cursors.add(current.cursor)
        visited.append(prefix)

        if max_prefixes is not None and len(visited) > max_prefixes:
            return True, prefix

        if max_depth is not None and len(prefix) > max_depth:
            return True, prefix

        choice_snapshot = (
            writer_snapshot._writer_frontier_choice_snapshot_from_snapshot(
                current,
                prepared=prepared,
                include_counts=False,
                stop_after_first_blocked=True,
            )
        )
        if choice_snapshot.blocked:
            blocked.append(
                _WriterFrontierBlockedPrefix(
                    emitted_texts=prefix,
                    choice_snapshot=choice_snapshot,
                )
            )
            return False, None

        if choice_snapshot.terminal is not None:
            for capability in choice_snapshot.terminal_execution_capabilities:
                use = _WriterExecutionCapabilityUse(
                    kind=capability,
                    emitted_texts=prefix,
                    source_cursor=current.cursor,
                    successor_cursor=choice_snapshot.terminal.finalized_cursor,
                    next_emitted_text=None,
                )
                signature = (
                    use.kind,
                    use.emitted_texts,
                    use.next_emitted_text,
                    use.source_cursor,
                    use.successor_cursor,
                )
                if signature not in observed_execution_capability_use_signatures:
                    observed_execution_capability_use_signatures.add(signature)
                    execution_capability_uses.append(use)

        for choice in choice_snapshot.choices:
            for capability in choice.execution_capabilities:
                use = _WriterExecutionCapabilityUse(
                    kind=capability,
                    emitted_texts=prefix,
                    source_cursor=current.cursor,
                    successor_cursor=choice.successor,
                    next_emitted_text=choice.emitted_text,
                )

                signature = (
                    use.kind,
                    use.emitted_texts,
                    use.next_emitted_text,
                    use.source_cursor,
                    use.successor_cursor,
                )

                if signature in observed_execution_capability_use_signatures:
                    continue

                observed_execution_capability_use_signatures.add(signature)
                execution_capability_uses.append(use)

            successor = (
                writer_snapshot
                ._writer_search_snapshot_after_checked_frontier_cursor_step(
                    current,
                    prepared=prepared,
                    cursor=choice.successor,
                )
            )
            stopped, stopped_prefix = rec(
                successor,
                (*prefix, choice.emitted_text),
            )

            if stopped:
                return True, stopped_prefix

        return False, None

    truncated, truncated_prefix = rec(snapshot, ())

    if truncated:
        return _WriterFrontierReachabilityAudit(
            kind=_WriterFrontierReachabilityAuditKind.TRUNCATED,
            visited_prefixes=tuple(visited),
            blocked_prefixes=tuple(blocked),
            truncated_at_prefix=truncated_prefix,
            execution_capability_uses=tuple(execution_capability_uses),
        )

    if blocked:
        return _WriterFrontierReachabilityAudit(
            kind=_WriterFrontierReachabilityAuditKind.BLOCKED,
            visited_prefixes=tuple(visited),
            blocked_prefixes=tuple(blocked),
            execution_capability_uses=tuple(execution_capability_uses),
        )

    return _WriterFrontierReachabilityAudit(
        kind=_WriterFrontierReachabilityAuditKind.READY,
        visited_prefixes=tuple(visited),
        execution_capability_uses=tuple(execution_capability_uses),
    )


def _audit_writer_frontier_reachability_from_cursor(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions,
    cursor: WriterFrontierCursor,
    max_depth: int | None = None,
    max_prefixes: int | None = None,
) -> _WriterFrontierReachabilityAudit:
    snapshot = writer_snapshot._capture_writer_frontier_snapshot_unchecked(
        prepared=prepared,
        runtime_options=runtime_options,
        cursor=cursor,
    )
    return _audit_writer_frontier_reachability_from_snapshot(
        snapshot,
        prepared=prepared,
        max_depth=max_depth,
        max_prefixes=max_prefixes,
    )
