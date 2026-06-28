"""Online decoder facade for the writer-shaped live runtime.

The generic online factories exist for legacy exhaustive runtimes.  WRITER_SHAPED
has a separate prepared-only route so branch, compaction, and legacy execution
mode knobs cannot become accidental support gates.
"""

from __future__ import annotations

from dataclasses import dataclass

from .policy import SerializationLanguageMode
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import component_root_domains_for_prepared
from .prepared_runtime import require_writer_shaped_runtime_options
from .prepared_runtime import runtime_root_atom_for_prepared
from .writer_runtime import WriterRuntimeState
from .writer_runtime import _advance_writer_runtime_state_by_choice
from .writer_runtime import initial_writer_runtime_state
from .writer_runtime import writer_runtime_choices


EOS = "<EOS>"

_DEFAULT_WRITER_RUNTIME_OPTIONS = SouthStarRuntimeOptions(
    serialization_language=SerializationLanguageMode.WRITER_SHAPED,
)


@dataclass(frozen=True, slots=True)
class WriterRuntimeOnlineStats:
    """Frontier-level stats for writer-shaped online choices.

    These stats are observational summaries of checked writer choices.  They are
    intentionally not a support decision surface.
    """

    support_count: int
    completion_count: int
    choice_count: int
    has_eos: bool


@dataclass(frozen=True, slots=True)
class WriterShapedOnlineChoice:
    text: str
    next_state: "WriterShapedOnlineDecoderState | None"
    is_eos: bool = False
    multiplicity: int = 1
    completion_count: int = 0


@dataclass(frozen=True, slots=True)
class WriterShapedOnlineChoiceResult:
    choices: tuple[WriterShapedOnlineChoice, ...]
    stats: WriterRuntimeOnlineStats


@dataclass(frozen=True, slots=True)
class WriterShapedOnlineDecoderState:
    prefix: str
    raw_state: WriterRuntimeState
    decoder: "WriterShapedOnlineDecoder"

    def choices(self) -> tuple[WriterShapedOnlineChoice, ...]:
        return self.decoder.choices(self)

    def choices_with_stats(self) -> WriterShapedOnlineChoiceResult:
        return self.decoder.choices_with_stats(self)


@dataclass(frozen=True, slots=True)
class WriterShapedOnlineDecoder:
    prepared: SouthStarPreparedMol
    runtime_options: SouthStarRuntimeOptions
    rooted_at_atom: object
    component_root_domains: tuple[tuple[object, ...], ...]
    include_eos: bool = False

    def initial_state(self) -> WriterShapedOnlineDecoderState:
        return WriterShapedOnlineDecoderState(
            prefix="",
            raw_state=initial_writer_runtime_state(
                prepared=self.prepared,
                runtime_options=self.runtime_options,
            ),
            decoder=self,
        )

    def choices(
        self,
        state: WriterShapedOnlineDecoderState,
    ) -> tuple[WriterShapedOnlineChoice, ...]:
        return self.choices_with_stats(state).choices

    def choices_with_stats(
        self,
        state: WriterShapedOnlineDecoderState,
    ) -> WriterShapedOnlineChoiceResult:
        _validate_writer_state_belongs_to_decoder(state, self)
        runtime_choices = writer_runtime_choices(
            prepared=self.prepared,
            state=state.raw_state,
        )
        out = []
        for choice in runtime_choices.choices:
            next_runtime_state = _advance_writer_runtime_state_by_choice(
                prepared=self.prepared,
                state=state.raw_state,
                choice=choice,
            )
            out.append(
                WriterShapedOnlineChoice(
                    text=choice.emitted_text,
                    next_state=WriterShapedOnlineDecoderState(
                        prefix=state.prefix + choice.emitted_text,
                        raw_state=next_runtime_state,
                        decoder=self,
                    ),
                    multiplicity=choice.immediate_multiplicity,
                    completion_count=choice.completion_count or 0,
                )
            )

        terminal = runtime_choices.terminal
        has_eos = terminal is not None
        if self.include_eos and terminal is not None:
            out.append(
                WriterShapedOnlineChoice(
                    text=EOS,
                    next_state=None,
                    is_eos=True,
                    multiplicity=terminal.multiplicity,
                    completion_count=terminal.completion_count,
                )
            )

        support_count = sum(
            choice.support_count or 0
            for choice in runtime_choices.choices
        )
        completion_count = sum(
            choice.completion_count or 0
            for choice in runtime_choices.choices
        )
        if terminal is not None:
            support_count += terminal.support_count
            completion_count += terminal.completion_count

        return WriterShapedOnlineChoiceResult(
            choices=tuple(out),
            stats=WriterRuntimeOnlineStats(
                support_count=support_count,
                completion_count=completion_count,
                choice_count=len(out),
                has_eos=has_eos,
            ),
        )


def make_writer_shaped_online_decoder(
    *,
    prepared: SouthStarPreparedMol,
    runtime_options: SouthStarRuntimeOptions = _DEFAULT_WRITER_RUNTIME_OPTIONS,
    include_eos: bool = False,
) -> WriterShapedOnlineDecoder:
    """Construct the online decoder for the live writer-shaped runtime.

    WRITER_SHAPED is intentionally prepared-only here.  Preparation supplies the
    structural molecule state; support is still enforced later by checked live
    writer frontier operations.
    """

    require_writer_shaped_runtime_options(runtime_options)
    rooted_at_atom = runtime_root_atom_for_prepared(
        runtime_options,
        prepared=prepared,
    )
    root_domains = tuple(
        atoms
        for _, atoms in component_root_domains_for_prepared(
            prepared=prepared,
            rooted_at_atom=rooted_at_atom,
        )
    )
    return WriterShapedOnlineDecoder(
        prepared=prepared,
        runtime_options=runtime_options,
        rooted_at_atom=rooted_at_atom,
        component_root_domains=root_domains,
        include_eos=include_eos,
    )


def _validate_writer_state_belongs_to_decoder(
    state: WriterShapedOnlineDecoderState,
    decoder: WriterShapedOnlineDecoder,
) -> None:
    if state.decoder is not decoder:
        raise ValueError("writer-shaped online decoder state belongs to a different decoder")
    if not isinstance(state.raw_state, WriterRuntimeState):
        raise ValueError("WRITER_SHAPED online decoder received non-writer state")


__all__ = (
    "WriterRuntimeOnlineStats",
    "WriterShapedOnlineChoice",
    "WriterShapedOnlineChoiceResult",
    "WriterShapedOnlineDecoder",
    "WriterShapedOnlineDecoderState",
    "make_writer_shaped_online_decoder",
)
