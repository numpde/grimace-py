"""Raw writer-transition lifecycle event installation hooks."""

from __future__ import annotations

from .writer_ring_lifecycle import writer_events_with_ring_label_lifecycle


_INSTALLED_MARKER = "_ring_label_lifecycle_installed"


def install() -> None:
    """Install ring-label lifecycle evidence at transition construction time."""

    from . import writer_transitions

    current = writer_transitions._transition
    if getattr(current, _INSTALLED_MARKER, False):
        return

    def _transition_with_ring_label_lifecycle(prepared, state, **kwargs):
        kwargs["events"] = writer_events_with_ring_label_lifecycle(
            source_state=state,
            events=kwargs["events"],
        )
        return current(prepared, state, **kwargs)

    setattr(_transition_with_ring_label_lifecycle, _INSTALLED_MARKER, True)
    writer_transitions._transition = _transition_with_ring_label_lifecycle


__all__ = ("install",)
