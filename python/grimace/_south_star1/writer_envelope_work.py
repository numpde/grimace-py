"""Shared work budgets for durable writer envelope operations."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WriterEnvelopeWorkBudget:
    max_count_nodes: int = 10_000
    max_count_edges: int = 50_000
    max_count_depth: int = 1_000
    max_digest_term_bytes: int = 1_000_000
    max_source_lookup_positions: int = 5_000
    max_nested_envelopes: int = 50_000
    max_support_strings: int = 10_000
    max_total_emitted_text_bytes: int = 1_000_000
    max_support_string_envelopes: int = 10_000
    max_support_search_strings: int = 10_000
    max_replay_steps: int = 10_000
    max_prefix_emitted_texts: int = 10_000
    max_text_projection_chain_length: int = 10_000
    max_terminal_support_identities: int = 10_000
    max_coverage_buckets: int = 20_000
    max_bucket_assignments: int = 20_000
    max_consistency_nodes: int = 300_000
    max_envelope_nodes: int = 50_000


@dataclass(frozen=True, slots=True)
class WriterEnvelopeWorkViolation:
    operation: str
    metric: str
    actual: int
    limit: int


class WriterEnvelopeWorkExceeded(RuntimeError):
    def __init__(self, violation: WriterEnvelopeWorkViolation):
        self.violation = violation
        super().__init__(format_writer_envelope_work_violation(violation))


def format_writer_envelope_work_violation(
    violation: WriterEnvelopeWorkViolation,
) -> str:
    return (
        "WRITER_ENVELOPE_WORK_EXCEEDED: "
        f"operation={violation.operation!r}; "
        f"metric={violation.metric!r}; actual={violation.actual}; "
        f"limit={violation.limit}"
    )


def check_writer_envelope_work(
    *,
    budget: WriterEnvelopeWorkBudget,
    operation: str,
    metric: str,
    actual: int,
    limit: int,
) -> None:
    if actual > limit:
        raise WriterEnvelopeWorkExceeded(
            WriterEnvelopeWorkViolation(
                operation=operation,
                metric=metric,
                actual=actual,
                limit=limit,
            )
        )


def default_writer_envelope_work_budget(
    budget: WriterEnvelopeWorkBudget | None,
) -> WriterEnvelopeWorkBudget:
    return budget or WriterEnvelopeWorkBudget()


def writer_envelope_work_reason(exc: WriterEnvelopeWorkExceeded) -> str:
    return format_writer_envelope_work_violation(exc.violation)


__all__ = (
    "WriterEnvelopeWorkBudget",
    "WriterEnvelopeWorkExceeded",
    "WriterEnvelopeWorkViolation",
    "check_writer_envelope_work",
    "default_writer_envelope_work_budget",
    "format_writer_envelope_work_violation",
    "writer_envelope_work_reason",
)
