"""Online decoder facade for the writer-shaped live runtime.

The generic online factories exist for legacy exhaustive runtimes.  WRITER_SHAPED
has a separate prepared-only route so branch, compaction, and legacy execution
mode knobs cannot become accidental support gates.
"""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace

from .policy import SerializationLanguageMode
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .prepared_runtime import require_writer_shaped_runtime_options
from .prepared_runtime import runtime_root_atom_for_prepared
from .writer_online_decoder_certificates import (
    writer_online_choice_result_certificate,
)
from .writer_online_decoder_certificates import writer_online_eos_choice_certificate
from .writer_online_decoder_certificates import (
    writer_online_text_choice_certificate,
)
from .writer_online_stats_certificates import writer_online_stats_certificate
from .writer_runtime import WriterRuntimeState
from .writer_runtime import writer_runtime_choice_transitions
from .writer_runtime import writer_runtime_support_count_certificate
from .writer_runtime import initial_writer_runtime_state


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
    stats_certificate: object | None = None


@dataclass(frozen=True, slots=True)
class WriterShapedOnlineChoice:
    text: str
    next_state: "WriterShapedOnlineDecoderState | None"
    is_eos: bool = False
    multiplicity: int = 1
    completion_count: int = 0
    choice_certificate: object | None = None


@dataclass(frozen=True, slots=True)
class WriterShapedOnlineChoiceResult:
    choices: tuple[WriterShapedOnlineChoice, ...]
    stats: WriterRuntimeOnlineStats
    result_certificate: object | None = None
    checked_frontier_certificate: object | None = None
    count_certificate: object | None = None


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
        runtime_transitions = writer_runtime_choice_transitions(
            prepared=self.prepared,
            state=state.raw_state,
        )
        support_count_certificate = writer_runtime_support_count_certificate(
            prepared=self.prepared,
            state=state.raw_state,
        )
        out = []
        for transition in runtime_transitions.transitions:
            choice = transition.choice
            projection = _single_projection_for_choice(
                runtime_transitions.text_choice_projection_certificates,
                choice,
            )
            out.append(
                WriterShapedOnlineChoice(
                    text=choice.emitted_text,
                    next_state=WriterShapedOnlineDecoderState(
                        prefix=state.prefix + choice.emitted_text,
                        raw_state=transition.next_state,
                        decoder=self,
                    ),
                    multiplicity=choice.immediate_multiplicity,
                    completion_count=choice.completion_count or 0,
                    choice_certificate=writer_online_text_choice_certificate(
                        prefix=state.prefix,
                        choice=choice,
                        next_state=WriterShapedOnlineDecoderState(
                            prefix=state.prefix + choice.emitted_text,
                            raw_state=transition.next_state,
                            decoder=self,
                        ),
                        snapshot_step_certificate=(
                            transition.snapshot_step_certificate
                        ),
                        text_projection_certificate=projection,
                        frontier_projection_certificate=(
                            runtime_transitions.projection_certificate
                        ),
                        checked_frontier_certificate=(
                            runtime_transitions.checked_frontier_certificate
                        ),
                        count_certificate=(
                            runtime_transitions.count_certificate
                        ),
                    ),
                )
            )

        terminal = runtime_transitions.terminal
        if self.include_eos and terminal is not None:
            out.append(
                WriterShapedOnlineChoice(
                    text=EOS,
                    next_state=None,
                    is_eos=True,
                    multiplicity=terminal.multiplicity,
                    completion_count=terminal.completion_count,
                    choice_certificate=writer_online_eos_choice_certificate(
                        prefix=state.prefix,
                        eos_text=EOS,
                        terminal=terminal,
                        terminal_projection_certificate=(
                            runtime_transitions.terminal_projection_certificate
                        ),
                        frontier_projection_certificate=(
                            runtime_transitions.projection_certificate
                        ),
                        checked_frontier_certificate=(
                            runtime_transitions.checked_frontier_certificate
                        ),
                        count_certificate=runtime_transitions.count_certificate,
                    ),
                )
            )

        stats = WriterRuntimeOnlineStats(
            support_count=runtime_transitions.support_count,
            completion_count=runtime_transitions.completion_count,
            choice_count=len(out),
            has_eos=runtime_transitions.has_eos,
        )
        result_certificate = writer_online_choice_result_certificate(
            prefix=state.prefix,
            choices=tuple(out),
            choice_certificates=(
                tuple(item.choice_certificate for item in out)
            ),
            checked_frontier_certificate=(
                runtime_transitions.checked_frontier_certificate
            ),
            count_certificate=runtime_transitions.count_certificate,
        )
        stats_certificate = writer_online_stats_certificate(
            prefix=state.prefix,
            stats=stats,
            choice_result_certificate=result_certificate,
            checked_frontier_certificate=(
                runtime_transitions.checked_frontier_certificate
            ),
            support_count_certificate=support_count_certificate,
            completion_count_certificate=(
                runtime_transitions.count_certificate
            ),
        )
        stats = replace(stats, stats_certificate=stats_certificate)

        return WriterShapedOnlineChoiceResult(
            choices=tuple(out),
            stats=stats,
            result_certificate=result_certificate,
            checked_frontier_certificate=(
                runtime_transitions.checked_frontier_certificate
            ),
            count_certificate=runtime_transitions.count_certificate,
        )


def _single_projection_for_choice(
    projection_certificates: tuple[object, ...],
    choice,
):
    matches = tuple(
        cert
        for cert in projection_certificates
        if cert.emitted_text == choice.emitted_text
    )
    if len(matches) != 1:
        raise ValueError(
            "online decoder observed non-unique choice projection certificate"
        )
    return matches[0]


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
    # Validate the requested root at construction, but do not duplicate derived
    # root-domain state on the decoder.  The live writer runtime remains the
    # single source of transition/support state.
    runtime_root_atom_for_prepared(
        runtime_options,
        prepared=prepared,
    )
    return WriterShapedOnlineDecoder(
        prepared=prepared,
        runtime_options=runtime_options,
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
