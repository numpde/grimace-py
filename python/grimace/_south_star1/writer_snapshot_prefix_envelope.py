"""Durable envelopes for checked writer snapshot prefix reads."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .prepared_runtime import SouthStarPreparedMol
from .writer_envelope_terms import _cursor_envelope
from .writer_envelope_terms import _digest
from .writer_envelope_terms import _identity_digest
from .writer_envelope_terms import _identity_envelope
from .writer_envelope_terms import _runtime_options_from_terms
from .writer_envelope_terms import _snapshot_identity_envelope
from .writer_envelope_terms import _term
from .writer_envelope_work import WriterEnvelopeWorkBudget
from .writer_envelope_work import WriterEnvelopeWorkExceeded
from .writer_envelope_work import check_writer_envelope_work
from .writer_envelope_work import default_writer_envelope_work_budget
from .writer_envelope_work import writer_envelope_work_reason
from .writer_snapshot import _checked_writer_snapshot_prefix_product_after_emitted_texts
from .writer_snapshot import _prepared_identity
from .writer_frontier import _snapshot_advance_writer_frontier_product
from .writer_snapshot_replay_envelope import (
    verify_writer_snapshot_replay_envelope,
)
from .writer_snapshot_replay_envelope import (
    writer_snapshot_replay_envelope_for_emitted_texts,
)


SCHEMA_NAME = "writer_snapshot_prefix_read"
SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset((
    "schema_name",
    "schema_version",
    "prepared_identity",
    "source_snapshot",
    "emitted_texts",
    "include_counts",
    "read_kind",
    "replay_envelope",
    "final_snapshot",
    "final_frontier_product_kind",
    "final_frontier_product",
    "prefix_read_certificate",
    "public_frontier",
    "support_count",
    "completion_count",
    "failure",
))
_READ_KINDS = frozenset((
    "readable",
    "replay_blocked",
    "invalid_emitted_text",
    "final_frontier_blocked",
))


@dataclass(frozen=True, slots=True)
class WriterSnapshotPrefixReadEnvelopeVerification:
    accepted: bool
    read_kind: str
    source_snapshot: object | None
    final_snapshot: object | None
    support_count: int | None = None
    completion_count: int | None = None
    reason: str | None = None


def writer_snapshot_prefix_read_envelope_for_emitted_texts(
    *,
    prepared: SouthStarPreparedMol,
    snapshot,
    emitted_texts: tuple[str, ...],
    include_counts: bool = True,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> dict[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    _check_prefix_text_work(emitted_texts, budget=budget)
    replay_envelope = writer_snapshot_replay_envelope_for_emitted_texts(
        prepared=prepared,
        snapshot=snapshot,
        emitted_texts=emitted_texts,
        budget=budget,
    )
    replay_verification = verify_writer_snapshot_replay_envelope(
        prepared=prepared,
        envelope=replay_envelope,
        budget=budget,
    )
    if not replay_verification.accepted:
        _prefix_envelope_violation("replay_envelope_rejected")
    envelope = _envelope_from_verified_replay(
        prepared=prepared,
        snapshot=snapshot,
        emitted_texts=emitted_texts,
        include_counts=include_counts,
        replay_envelope=replay_envelope,
        replay_verification=replay_verification,
    )

    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope)
    return envelope


def _envelope_from_verified_replay(
    *,
    prepared,
    snapshot,
    emitted_texts: tuple[str, ...],
    include_counts: bool,
    replay_envelope,
    replay_verification,
) -> dict[str, object]:

    if replay_verification.outcome_kind == "blocked":
        return _failed_replay_envelope(
            snapshot=snapshot,
            emitted_texts=emitted_texts,
            include_counts=include_counts,
            replay_envelope=replay_envelope,
            read_kind="replay_blocked",
        )
    if replay_verification.outcome_kind == "invalid_emitted_text":
        return _failed_replay_envelope(
            snapshot=snapshot,
            emitted_texts=emitted_texts,
            include_counts=include_counts,
            replay_envelope=replay_envelope,
            read_kind="invalid_emitted_text",
        )
    final_snapshot = replay_verification.current_snapshot
    if final_snapshot is None:
        _prefix_envelope_violation("advanced_replay_lacks_final_snapshot")
    final_product = _snapshot_advance_writer_frontier_product(
        prepared,
        final_snapshot.cursor,
    )
    if final_product.blocked:
        return _final_frontier_blocked_envelope(
            snapshot=snapshot,
            emitted_texts=emitted_texts,
            include_counts=include_counts,
            replay_envelope=replay_envelope,
            final_snapshot=final_snapshot,
            final_product=final_product,
        )
    prefix = _checked_writer_snapshot_prefix_product_after_emitted_texts(
        snapshot,
        prepared=prepared,
        emitted_texts=emitted_texts,
        include_counts=include_counts,
    )
    return _readable_envelope(
        snapshot=snapshot,
        emitted_texts=emitted_texts,
        include_counts=include_counts,
        replay_envelope=replay_envelope,
        prefix=prefix,
    )


def verify_writer_snapshot_prefix_read_envelope(
    *,
    prepared: SouthStarPreparedMol,
    envelope: object,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> WriterSnapshotPrefixReadEnvelopeVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        _validate_envelope_shape(envelope)
        assert isinstance(envelope, Mapping)
        _check_prefix_text_work(
            tuple(envelope["emitted_texts"]),
            budget=budget,
        )
        _assert_prepared_identity_matches(prepared, envelope)
        replay = verify_writer_snapshot_replay_envelope(
            prepared=prepared,
            envelope=envelope["replay_envelope"],
            budget=budget,
        )
        if not replay.accepted:
            _prefix_envelope_violation("replay_envelope_rejected")
        expected = _envelope_from_verified_replay(
            prepared=prepared,
            snapshot=replay.source_snapshot,
            emitted_texts=tuple(envelope["emitted_texts"]),
            include_counts=bool(envelope["include_counts"]),
            replay_envelope=envelope["replay_envelope"],
            replay_verification=replay,
        )
        if expected != envelope:
            return WriterSnapshotPrefixReadEnvelopeVerification(
                accepted=False,
                read_kind=str(envelope["read_kind"]),
                source_snapshot=replay.source_snapshot,
                final_snapshot=replay.current_snapshot,
                reason="envelope_terms_mismatch",
            )
        return WriterSnapshotPrefixReadEnvelopeVerification(
            accepted=True,
            read_kind=str(envelope["read_kind"]),
            source_snapshot=replay.source_snapshot,
            final_snapshot=(
                replay.current_snapshot
                if envelope["read_kind"] in (
                    "readable",
                    "final_frontier_blocked",
                )
                else None
            ),
            support_count=envelope["support_count"],
            completion_count=envelope["completion_count"],
        )
    except WriterEnvelopeWorkExceeded as exc:
        return WriterSnapshotPrefixReadEnvelopeVerification(
            accepted=False,
            read_kind=(
                envelope.get("read_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            source_snapshot=None,
            final_snapshot=None,
            reason=writer_envelope_work_reason(exc),
        )
    except SouthStarError as exc:
        return WriterSnapshotPrefixReadEnvelopeVerification(
            accepted=False,
            read_kind=(
                envelope.get("read_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            source_snapshot=None,
            final_snapshot=None,
            reason=exc.args[-1] if exc.args else "verification_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterSnapshotPrefixReadEnvelopeVerification(
            accepted=False,
            read_kind=(
                envelope.get("read_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            source_snapshot=None,
            final_snapshot=None,
            reason=f"malformed_envelope:{type(exc).__name__}",
        )


def _failed_replay_envelope(
    *,
    snapshot,
    emitted_texts: tuple[str, ...],
    include_counts: bool,
    replay_envelope,
    read_kind: str,
) -> dict[str, object]:
    return _base_envelope(
        snapshot=snapshot,
        emitted_texts=emitted_texts,
        include_counts=include_counts,
        read_kind=read_kind,
        replay_envelope=replay_envelope,
        final_snapshot=None,
        final_frontier_product_kind=None,
        final_frontier_product=None,
        prefix_read_certificate=None,
        public_frontier=None,
        support_count=None,
        completion_count=None,
        failure={
            "kind": read_kind,
            "failed_advance_envelope": replay_envelope[
                "failed_advance_envelope"
            ],
        },
    )


def _check_prefix_text_work(
    emitted_texts: tuple[str, ...],
    *,
    budget: WriterEnvelopeWorkBudget,
) -> None:
    check_writer_envelope_work(
        budget=budget,
        operation="snapshot_prefix_read_envelope",
        metric="prefix_emitted_text_count",
        actual=len(emitted_texts),
        limit=budget.max_prefix_emitted_texts,
    )
    check_writer_envelope_work(
        budget=budget,
        operation="snapshot_prefix_read_envelope",
        metric="total_emitted_text_bytes",
        actual=sum(len(text.encode("utf-8")) for text in emitted_texts),
        limit=budget.max_total_emitted_text_bytes,
    )


def _final_frontier_blocked_envelope(
    *,
    snapshot,
    emitted_texts: tuple[str, ...],
    include_counts: bool,
    replay_envelope,
    final_snapshot,
    final_product,
) -> dict[str, object]:
    blocked = final_product.blocked_frontier_certificate
    diagnostic = final_product.diagnostic_certificate
    return _base_envelope(
        snapshot=snapshot,
        emitted_texts=emitted_texts,
        include_counts=include_counts,
        read_kind="final_frontier_blocked",
        replay_envelope=replay_envelope,
        final_snapshot=_snapshot_identity_envelope(final_snapshot),
        final_frontier_product_kind="blocked",
        final_frontier_product=_writer_frontier_product_identity_envelope(
            final_product
        ),
        prefix_read_certificate=None,
        public_frontier=None,
        support_count=None,
        completion_count=None,
        failure={
            "kind": "final_frontier_blocked",
            "blocked_frontier_certificate": {
                "cursor": _cursor_envelope(blocked.cursor),
                "blocked": blocked.blocked,
                "diagnostic_certificate_digest": _identity_digest(diagnostic),
                "digest": _identity_digest(blocked),
            },
            "diagnostic_certificate_digest": _identity_digest(diagnostic),
        },
    )


def _readable_envelope(
    *,
    snapshot,
    emitted_texts: tuple[str, ...],
    include_counts: bool,
    replay_envelope,
    prefix,
) -> dict[str, object]:
    certificate = prefix.prefix_read_certificate
    support_count = certificate.support_count if include_counts else None
    completion_count = certificate.completion_count if include_counts else None
    return _base_envelope(
        snapshot=snapshot,
        emitted_texts=emitted_texts,
        include_counts=include_counts,
        read_kind="readable",
        replay_envelope=replay_envelope,
        final_snapshot=_snapshot_identity_envelope(prefix.final_snapshot),
        final_frontier_product_kind="legal",
        final_frontier_product=_writer_frontier_product_identity_envelope(
            prefix.frontier_product
        ),
        prefix_read_certificate=_prefix_read_certificate_envelope(
            certificate
        ),
        public_frontier=_public_frontier_envelope(
            prefix.frontier_product
        ),
        support_count=support_count,
        completion_count=completion_count,
        failure=None,
    )


def _base_envelope(
    *,
    snapshot,
    emitted_texts: tuple[str, ...],
    include_counts: bool,
    read_kind: str,
    replay_envelope,
    final_snapshot,
    final_frontier_product_kind,
    final_frontier_product,
    prefix_read_certificate,
    public_frontier,
    support_count,
    completion_count,
    failure,
) -> dict[str, object]:
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "prepared_identity": _identity_envelope(snapshot.prepared_identity),
        "source_snapshot": _snapshot_identity_envelope(snapshot),
        "emitted_texts": list(emitted_texts),
        "include_counts": include_counts,
        "read_kind": read_kind,
        "replay_envelope": replay_envelope,
        "final_snapshot": final_snapshot,
        "final_frontier_product_kind": final_frontier_product_kind,
        "final_frontier_product": final_frontier_product,
        "prefix_read_certificate": prefix_read_certificate,
        "public_frontier": public_frontier,
        "support_count": support_count,
        "completion_count": completion_count,
        "failure": failure,
    }


def _prefix_read_certificate_envelope(certificate) -> dict[str, object]:
    envelope = {
        "source_snapshot": _snapshot_identity_envelope(
            certificate.source_snapshot
        ),
        "emitted_texts": list(certificate.emitted_texts),
        "replay_certificate_digest": _full_term_digest(
            certificate.replay_certificate
        ),
        "final_snapshot": _snapshot_identity_envelope(certificate.final_snapshot),
        "final_frontier_projection_certificate": (
            _frontier_projection_certificate_identity_envelope(
                certificate.final_frontier_projection_certificate
            )
        ),
        "checked_frontier_certificate": (
            _checked_frontier_certificate_identity_envelope(
                certificate.checked_frontier_certificate
            )
        ),
        "support_count_certificate": _support_count_certificate_envelope(
            certificate.support_count_certificate
        ),
        "completion_count_certificate": _completion_count_certificate_envelope(
            certificate.completion_count_certificate
        ),
        "support_count": certificate.support_count,
        "completion_count": certificate.completion_count,
    }
    envelope["digest"] = _identity_digest(envelope)
    return envelope


def _full_term_digest(certificate) -> str | None:
    if certificate is None:
        return None
    return _identity_digest(certificate)


def _writer_frontier_product_identity_envelope(product) -> dict[str, object]:
    if product is None:
        return None
    if product.blocked:
        blocked = product.blocked_frontier_certificate
        diagnostic = product.diagnostic_certificate
        envelope = {
            "kind": "blocked",
            "cursor": _cursor_envelope(product.cursor),
            "blocked_frontier_certificate": {
                "cursor": _cursor_envelope(blocked.cursor),
                "blocked": blocked.blocked,
                "diagnostic_certificate_digest": _full_term_digest(diagnostic),
                "digest": _full_term_digest(blocked),
            },
            "diagnostic_certificate_digest": _full_term_digest(diagnostic),
        }
    else:
        envelope = {
            "kind": "legal",
            "cursor": _cursor_envelope(product.cursor),
            "frontier_projection_certificate": (
                _frontier_projection_certificate_identity_envelope(
                    product.projection_certificate
                )
            ),
            "text_projection_certificates": [
                _text_projection_certificate_identity_envelope(projection)
                for projection in product.text_choice_projection_certificates
            ],
            "terminal_projection_certificate": (
                _terminal_projection_certificate_identity_envelope(
                    product.terminal_projection_certificate
                )
            ),
            "checked_frontier_certificate": (
                _checked_frontier_certificate_identity_envelope(
                    product.checked_frontier_certificate
                )
            ),
            "support_count_certificate": _support_count_certificate_envelope(
                product.support_count_certificate
            ),
            "completion_count_certificate": (
                _completion_count_certificate_envelope(product.count_certificate)
            ),
            "branch_support_identities": [
                _branch_support_identity_envelope(support)
                for support in product.branch_supports
            ],
            "terminal_support_identities": [
                _terminal_support_identity_envelope(support)
                for support in product.terminal_supports
            ],
        }
    envelope["digest"] = _identity_digest(envelope)
    return envelope


def _frontier_projection_certificate_identity_envelope(certificate):
    if certificate is None:
        return None
    envelope = {
        "cursor": _cursor_envelope(certificate.cursor),
        "text_projection_digests": [
            item["digest"]
            for item in (
                _text_projection_certificate_identity_envelope(projection)
                for projection in certificate.text_choice_projection_certificates
            )
        ],
        "terminal_projection_digest": (
            None
            if certificate.terminal_projection_certificate is None
            else _terminal_projection_certificate_identity_envelope(
                certificate.terminal_projection_certificate
            )["digest"]
        ),
        "branch_certificate_digests": [
            _branch_certificate_identity_envelope(certificate)["digest"]
            for certificate in certificate.branch_certificates
        ],
        "terminal_certificate_digests": [
            _terminal_support_identity_envelope_from_certificate(
                certificate
            )["digest"]
            for certificate in certificate.terminal_certificates
        ],
    }
    envelope["digest"] = _identity_digest(envelope)
    return envelope


def _text_projection_certificate_identity_envelope(certificate):
    if certificate is None:
        return None
    envelope = {
        "source_cursor": _cursor_envelope(certificate.source_cursor),
        "emitted_text": certificate.emitted_text,
        "successor_cursor": _cursor_envelope(certificate.successor_cursor),
        "immediate_multiplicity": certificate.immediate_multiplicity,
        "support_count": certificate.support_count,
        "completion_count": certificate.completion_count,
        "branch_certificate_digests": [
            _branch_certificate_identity_envelope(branch)["digest"]
            for branch in certificate.branch_certificates
        ],
    }
    envelope["digest"] = _identity_digest(envelope)
    return envelope


def _terminal_projection_certificate_identity_envelope(certificate):
    if certificate is None:
        return None
    envelope = {
        "source_cursor": _cursor_envelope(certificate.source_cursor),
        "finalized_cursor": _cursor_envelope(certificate.finalized_cursor),
        "multiplicity": certificate.multiplicity,
        "support_count": certificate.support_count,
        "completion_count": certificate.completion_count,
        "terminal_support_identities": [
            _terminal_support_identity_envelope_from_certificate(terminal)
            for terminal in certificate.terminal_certificates
        ],
        "terminal_certificate_digests": [
            _terminal_support_identity_envelope_from_certificate(
                terminal
            )["digest"]
            for terminal in certificate.terminal_certificates
        ],
    }
    envelope["digest"] = _identity_digest(envelope)
    return envelope


def _checked_frontier_certificate_identity_envelope(certificate):
    if certificate is None:
        return None
    envelope = {
        "cursor": _cursor_envelope(certificate.cursor),
        "projection_certificate_digest": (
            _frontier_projection_certificate_identity_envelope(
                certificate.projection_certificate
            )["digest"]
        ),
        "support_count_certificate_digest": (
            _support_count_certificate_envelope(
                certificate.support_count_certificate
            )["digest"]
            if certificate.support_count_certificate is not None
            else None
        ),
        "completion_count_certificate_digest": (
            _completion_count_certificate_envelope(
                certificate.count_certificate
            )["digest"]
            if certificate.count_certificate is not None
            else None
        ),
        "support_count_term_coverage": _support_count_coverage_envelope(
            certificate.support_count_term_coverage_certificate
        ),
        "frontier_completion_count": _frontier_completion_count_envelope(
            certificate.frontier_completion_count_certificate
        ),
        "choice_count_coverage": _choice_count_coverage_envelope(
            certificate.choice_count_coverage_certificate
        ),
    }
    envelope["digest"] = _identity_digest(envelope)
    return envelope


def _support_count_coverage_envelope(certificate):
    if certificate is None:
        return None
    envelope = {
        "projection_certificate_digest": (
            _frontier_projection_certificate_identity_envelope(
                certificate.projection_certificate
            )["digest"]
        ),
        "support_count_certificate_digest": (
            _support_count_certificate_envelope(
                certificate.support_count_certificate
            )["digest"]
        ),
        "text_terms": [
            {
                "text_projection_digest": (
                    _text_projection_certificate_identity_envelope(
                        term.text_projection_certificate
                    )["digest"]
                ),
                "support_count": term.support_count,
            }
            for term in certificate.text_terms
        ],
        "terminal_term": (
            None
            if certificate.terminal_term is None
            else {
                "terminal_projection_digest": (
                    _terminal_projection_certificate_identity_envelope(
                        certificate
                        .terminal_term
                        .terminal_projection_certificate
                    )["digest"]
                ),
                "terminal_count": certificate.terminal_term.terminal_count,
            }
        ),
        "text_support_count": certificate.text_support_count,
        "terminal_support_count": certificate.terminal_support_count,
        "support_count": certificate.support_count,
    }
    envelope["digest"] = _identity_digest(envelope)
    return envelope


def _frontier_completion_count_envelope(certificate):
    if certificate is None:
        return None
    coverage = certificate.term_coverage_certificate
    envelope = {
        "projection_certificate_digest": (
            _frontier_projection_certificate_identity_envelope(
                certificate.projection_certificate
            )["digest"]
        ),
        "count_certificate_digest": (
            _completion_count_certificate_envelope(
                certificate.count_certificate
            )["digest"]
        ),
        "terminal_completion_count": certificate.terminal_completion_count,
        "text_completion_count": certificate.text_completion_count,
        "completion_count": certificate.completion_count,
        "term_coverage": (
            None
            if coverage is None
            else {
                "branch_term_count": len(coverage.branch_terms),
                "terminal_term_count": len(coverage.terminal_terms),
                "branch_completion_count": coverage.branch_completion_count,
                "terminal_completion_count": (
                    coverage.terminal_completion_count
                ),
                "completion_count": coverage.completion_count,
                "branch_terms": [
                    {
                        "projection_branch_digest": (
                            _branch_certificate_identity_envelope(
                                term.projection_branch_certificate
                            )["digest"]
                        ),
                        "cursor_weight": term.cursor_weight,
                        "projection_parent_weight": (
                            term.projection_parent_weight
                        ),
                        "count_parent_weight": term.count_parent_weight,
                        "successor_completion_count": (
                            term.successor_completion_count
                        ),
                        "weighted_completion_count": (
                            term.weighted_completion_count
                        ),
                    }
                    for term in coverage.branch_terms
                ],
                "terminal_terms": [
                    {
                        "projection_terminal_digest": (
                            _terminal_support_identity_envelope_from_certificate(
                                term.projection_terminal_certificate
                            )["digest"]
                        ),
                        "cursor_weight": term.cursor_weight,
                        "projection_parent_weight": (
                            term.projection_parent_weight
                        ),
                        "count_parent_weight": term.count_parent_weight,
                        "terminal_completion_count": (
                            term.terminal_completion_count
                        ),
                        "weighted_completion_count": (
                            term.weighted_completion_count
                        ),
                    }
                    for term in coverage.terminal_terms
                ],
            }
        ),
    }
    envelope["digest"] = _identity_digest(envelope)
    return envelope


def _choice_count_coverage_envelope(certificate):
    if certificate is None:
        return None
    envelope = {
        "projection_certificate_digest": (
            _frontier_projection_certificate_identity_envelope(
                certificate.projection_certificate
            )["digest"]
        ),
        "support_count_term_coverage_digest": (
            _support_count_coverage_envelope(
                certificate.support_count_term_coverage_certificate
            )["digest"]
        ),
        "completion_count_term_coverage_digest": (
            _completion_term_coverage_digest(
                certificate.completion_count_term_coverage_certificate
            )
        ),
        "text_choice_terms": [
            {
                "text_projection_digest": (
                    _text_projection_certificate_identity_envelope(
                        term.text_projection_certificate
                    )["digest"]
                ),
                "support_count": term.support_count,
                "completion_count": term.completion_count,
                "completion_branch_count": len(
                    term.completion_coverage_terms
                ),
            }
            for term in certificate.text_choice_terms
        ],
        "terminal_choice_term": (
            None
            if certificate.terminal_choice_term is None
            else {
                "terminal_projection_digest": (
                    _terminal_projection_certificate_identity_envelope(
                        certificate
                        .terminal_choice_term
                        .terminal_projection_certificate
                    )["digest"]
                ),
                "support_count": (
                    certificate.terminal_choice_term.support_count
                ),
                "completion_count": (
                    certificate.terminal_choice_term.completion_count
                ),
            }
        ),
        "support_count": certificate.support_count,
        "completion_count": certificate.completion_count,
    }
    envelope["digest"] = _identity_digest(envelope)
    return envelope


def _completion_term_coverage_digest(certificate):
    if certificate is None:
        return None
    envelope = {
        "projection_certificate_digest": (
            _frontier_projection_certificate_identity_envelope(
                certificate.projection_certificate
            )["digest"]
        ),
        "count_certificate_digest": (
            _completion_count_certificate_envelope(
                certificate.count_certificate
            )["digest"]
        ),
        "branch_completion_count": certificate.branch_completion_count,
        "terminal_completion_count": certificate.terminal_completion_count,
        "completion_count": certificate.completion_count,
        "branch_term_count": len(certificate.branch_terms),
        "terminal_term_count": len(certificate.terminal_terms),
    }
    return _identity_digest(envelope)


def _support_count_certificate_envelope(certificate):
    if certificate is None:
        return None
    envelope = {
        "cursor": _cursor_envelope(certificate.cursor),
        "support_count": certificate.support_count,
        "state_support_count_certificate_digest": _full_term_digest(
            certificate.state_support_count_certificate
        ),
    }
    envelope["digest"] = _identity_digest(envelope)
    return envelope


def _completion_count_certificate_envelope(certificate):
    if certificate is None:
        return None
    envelope = {
        "cursor": _cursor_envelope(certificate.cursor),
        "completion_count": certificate.completion_count,
        "state_count_certificate_digests": [
            _identity_digest((state_key, weight, state_certificate))
            for state_key, weight, state_certificate in (
                certificate.state_count_certificates
            )
        ],
    }
    envelope["digest"] = _identity_digest(envelope)
    return envelope


def _branch_support_identity_envelope(support):
    certificate = support.checked_branch_certificate
    envelope = _branch_certificate_identity_envelope(certificate)
    if envelope["parent_weight"] != support.parent_weight:
        _prefix_envelope_violation("branch_support_parent_weight_mismatch")
    if envelope["branch_ordinal"] != support.branch_ordinal:
        _prefix_envelope_violation("branch_support_ordinal_mismatch")
    return envelope


def _branch_certificate_identity_envelope(certificate):
    envelope = {
        "source_state_digest": _identity_digest(certificate.source_state),
        "successor_state_digest": _identity_digest(certificate.successor_state),
        "emitted_text": certificate.emitted_text,
        "parent_weight": certificate.parent_weight,
        "branch_ordinal": certificate.branch_ordinal,
        "transition_kind": _term(certificate.transition_kind),
        "graph_action_surface_digest": _full_term_digest(
            certificate.graph_action_surface
        ),
        "policy_family_digest": _full_term_digest(certificate.policy_family),
        "events_digest": _full_term_digest(certificate.events),
        "successor_state_certificate_digest": _full_term_digest(
            certificate.successor_state_certificate
        ),
    }
    envelope["digest"] = _identity_digest(envelope)
    return envelope


def _terminal_support_identity_envelope(support):
    certificate = support.checked_terminal_certificate
    envelope = _terminal_support_identity_envelope_from_certificate(
        certificate
    )
    if envelope["parent_weight"] != support.parent_weight:
        _prefix_envelope_violation("terminal_support_parent_weight_mismatch")
    if envelope["terminal_ordinal"] != support.terminal_ordinal:
        _prefix_envelope_violation("terminal_support_ordinal_mismatch")
    if envelope["terminal_support_key_digest"] != _identity_digest(
        support.terminal_support_key,
        budget=WriterEnvelopeWorkBudget(),
        operation="prefix.terminal_support.key",
    ):
        _prefix_envelope_violation("terminal_support_key_mismatch")
    return envelope


def _terminal_support_identity_envelope_from_certificate(certificate):
    envelope = {
        "source_state_digest": _identity_digest(certificate.source_state),
        "finalized_state_digest": _identity_digest(certificate.finalized_state),
        "parent_weight": certificate.parent_weight,
        "terminal_ordinal": certificate.terminal_ordinal,
        "terminal_support_key_digest": _identity_digest(
            certificate.terminal_support_key,
            budget=WriterEnvelopeWorkBudget(),
            operation="prefix.terminal_support.key",
        ),
        "terminal_execution_capabilities_digest": _full_term_digest(
            certificate.terminal_execution_capabilities
        ),
        "terminal_residual_work_evidence_digest": _full_term_digest(
            certificate.terminal_residual_work_evidence
        ),
        "terminal_stereo_lifecycle_evidence_digest": _full_term_digest(
            certificate.terminal_stereo_lifecycle_evidence
        ),
        "graph_obligation_work_evidence_digest": _full_term_digest(
            certificate.graph_obligation_work_evidence
        ),
        "terminal_certificate_digests": [
            _terminal_certificate_identity_envelope(terminal)["digest"]
            for terminal in certificate.terminal_certificates
        ],
    }
    envelope["digest"] = _identity_digest(envelope)
    return envelope


def _terminal_certificate_identity_envelope(certificate):
    envelope = {
        "kind": _term(certificate.kind),
        "source_state_digest": _identity_digest(certificate.source_state),
        "finalized_state_digest": _identity_digest(certificate.finalized_state),
        "digest": _full_term_digest(certificate),
    }
    return envelope


def _public_frontier_envelope(product) -> dict[str, object]:
    terminal = product.choices.terminal
    return {
        "terminal": (
            None
            if terminal is None
            else {
                "support_count": terminal.support_count,
                "completion_count": terminal.completion_count,
                "multiplicity": terminal.multiplicity,
                "finalized_cursor": _cursor_envelope(
                    terminal.finalized_cursor
                ),
                "terminal_support_identities": [
                    _terminal_support_identity_envelope(support)
                    for support in product.terminal_supports
                ],
            }
        ),
        "choices": [
            {
                "emitted_text": choice.emitted_text,
                "immediate_multiplicity": choice.immediate_multiplicity,
                "successor_cursor": _cursor_envelope(choice.successor),
                "support_count": choice.support_count,
                "completion_count": choice.completion_count,
            }
            for choice in product.choices.choices
        ],
    }


def _validate_envelope_shape(envelope: object) -> None:
    if not isinstance(envelope, Mapping):
        _prefix_envelope_violation("envelope_not_mapping")
    if frozenset(envelope) != _TOP_LEVEL_FIELDS:
        _prefix_envelope_violation("top_level_fields_mismatch")
    if envelope["schema_name"] != SCHEMA_NAME:
        _prefix_envelope_violation("unknown_schema_name")
    if envelope["schema_version"] != SCHEMA_VERSION:
        _prefix_envelope_violation("unknown_schema_version")
    if envelope["read_kind"] not in _READ_KINDS:
        _prefix_envelope_violation("unknown_read_kind")
    if not isinstance(envelope["include_counts"], bool):
        _prefix_envelope_violation("include_counts_not_bool")
    if not isinstance(envelope["emitted_texts"], list):
        _prefix_envelope_violation("emitted_texts_not_list")
    if envelope["read_kind"] == "readable":
        _validate_readable_shape(envelope)
    elif envelope["read_kind"] in (
        "replay_blocked",
        "invalid_emitted_text",
    ):
        _validate_failed_replay_shape(envelope)
    else:
        _validate_final_frontier_blocked_shape(envelope)


def _validate_readable_shape(envelope) -> None:
    if envelope["replay_envelope"]["outcome_kind"] != "advanced":
        _prefix_envelope_violation("readable_replay_kind_mismatch")
    if envelope["final_snapshot"] is None:
        _prefix_envelope_violation("readable_missing_final_snapshot")
    if envelope["final_frontier_product_kind"] != "legal":
        _prefix_envelope_violation("readable_product_kind_mismatch")
    product = envelope["final_frontier_product"]
    if not isinstance(product, Mapping) or product.get("kind") != "legal":
        _prefix_envelope_violation("readable_product_identity_mismatch")
    if envelope["prefix_read_certificate"] is None:
        _prefix_envelope_violation("readable_missing_prefix_certificate")
    if envelope["public_frontier"] is None:
        _prefix_envelope_violation("readable_missing_public_frontier")
    if envelope["failure"] is not None:
        _prefix_envelope_violation("readable_failure_mismatch")
    if envelope["include_counts"]:
        if envelope["support_count"] is None:
            _prefix_envelope_violation("readable_missing_support_count")
        if envelope["completion_count"] is None:
            _prefix_envelope_violation("readable_missing_completion_count")
    elif (
        envelope["support_count"] is not None
        or envelope["completion_count"] is not None
    ):
        _prefix_envelope_violation("readable_count_without_include_counts")


def _validate_failed_replay_shape(envelope) -> None:
    expected = (
        "blocked"
        if envelope["read_kind"] == "replay_blocked"
        else "invalid_emitted_text"
    )
    if envelope["replay_envelope"]["outcome_kind"] != expected:
        _prefix_envelope_violation("failed_replay_kind_mismatch")
    if envelope["final_snapshot"] is not None:
        _prefix_envelope_violation("failed_replay_final_snapshot_mismatch")
    if envelope["final_frontier_product_kind"] is not None:
        _prefix_envelope_violation("failed_replay_product_kind_mismatch")
    if envelope["final_frontier_product"] is not None:
        _prefix_envelope_violation("failed_replay_product_identity_mismatch")
    if envelope["prefix_read_certificate"] is not None:
        _prefix_envelope_violation("failed_replay_prefix_certificate_mismatch")
    if envelope["public_frontier"] is not None:
        _prefix_envelope_violation("failed_replay_public_frontier_mismatch")
    if envelope["support_count"] is not None or envelope["completion_count"] is not None:
        _prefix_envelope_violation("failed_replay_count_mismatch")
    failure = envelope["failure"]
    if not isinstance(failure, Mapping) or failure.get("kind") != envelope["read_kind"]:
        _prefix_envelope_violation("failed_replay_failure_mismatch")


def _validate_final_frontier_blocked_shape(envelope) -> None:
    if envelope["replay_envelope"]["outcome_kind"] != "advanced":
        _prefix_envelope_violation("final_blocked_replay_kind_mismatch")
    if envelope["final_snapshot"] is None:
        _prefix_envelope_violation("final_blocked_missing_final_snapshot")
    if envelope["final_frontier_product_kind"] != "blocked":
        _prefix_envelope_violation("final_blocked_product_kind_mismatch")
    product = envelope["final_frontier_product"]
    if not isinstance(product, Mapping) or product.get("kind") != "blocked":
        _prefix_envelope_violation("final_blocked_product_identity_mismatch")
    if envelope["prefix_read_certificate"] is not None:
        _prefix_envelope_violation("final_blocked_prefix_certificate_mismatch")
    if envelope["public_frontier"] is not None:
        _prefix_envelope_violation("final_blocked_public_frontier_mismatch")
    if envelope["support_count"] is not None or envelope["completion_count"] is not None:
        _prefix_envelope_violation("final_blocked_count_mismatch")
    failure = envelope["failure"]
    if not isinstance(failure, Mapping) or failure.get("kind") != "final_frontier_blocked":
        _prefix_envelope_violation("final_blocked_failure_mismatch")


def _assert_prepared_identity_matches(prepared, envelope) -> None:
    runtime_options = _runtime_options_from_terms(
        envelope["source_snapshot"]["runtime_options"]
    )
    actual = _identity_envelope(_prepared_identity(prepared, runtime_options))
    if envelope["prepared_identity"] != actual:
        _prefix_envelope_violation("prepared_identity_mismatch")
    if (
        envelope["source_snapshot"]["prepared_identity_digest"]
        != actual["digest"]
    ):
        _prefix_envelope_violation("source_snapshot_prepared_identity_mismatch")


def _prefix_envelope_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer snapshot prefix read envelope violation: {kind}",
    )


__all__ = (
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "WriterSnapshotPrefixReadEnvelopeVerification",
    "verify_writer_snapshot_prefix_read_envelope",
    "writer_snapshot_prefix_read_envelope_for_emitted_texts",
)
