"""Raw writer-transition lifecycle event installation hooks."""

from __future__ import annotations

from functools import wraps

from .writer_ring_lifecycle import writer_events_with_ring_label_lifecycle


def install() -> None:
    """Install ring-label lifecycle evidence at transition construction time.

    The raw writer transition constructor has the source writer state, so this
    hook can derive fresh-vs-reused label allocation before the transition is
    stored in frontier supports or surfaced through the runtime facade.
    """

    from . import writer_transitions

    current = writer_transitions._transition
    if getattr(current, "_ring_label_lifecycle_installed", False):
        return

    @wraps(current)
    def _transition_with_ring_label_lifecycle(
        prepared,
        state,
        *,
        emitted_text,
        successor,
        kind,
        events,
        evidence,
        finite_relation_work_evidence=(),
    ):
        return current(
            prepared,
            state,
            emitted_text=emitted_text,
            successor=successor,
            kind=kind,
            events=writer_events_with_ring_label_lifecycle(
                source_state=state,
                events=events,
            ),
            evidence=evidence,
            finite_relation_work_evidence=finite_relation_work_evidence,
        )

    setattr(
        _transition_with_ring_label_lifecycle,
        "_ring_label_lifecycle_installed",
        True,
    )
    writer_transitions._transition = _transition_with_ring_label_lifecycle


__all__ = ("install",)
