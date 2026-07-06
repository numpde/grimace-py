"""Durable envelopes for checked writer frontier count certificates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .prepared_runtime import SouthStarPreparedMol
from .writer_envelope_terms import _cursor_envelope
from .writer_envelope_terms import _digest
from .writer_envelope_terms import _identity_envelope
from .writer_envelope_terms import _runtime_options_from_terms
from .writer_envelope_terms import _snapshot_identity_envelope
from .writer_envelope_terms import _term
from .writer_frontier import _checked_writer_frontier_product
from .writer_snapshot import _capture_writer_frontier_snapshot_unchecked
from .writer_snapshot import _prepared_identity
from .writer_snapshot_envelope import _source_snapshot_from_envelope
from .writer_snapshot_prefix_envelope import (
    _branch_certificate_identity_envelope,
)
from .writer_snapshot_prefix_envelope import (
    _terminal_projection_certificate_identity_envelope,
)
from .writer_snapshot_prefix_envelope import (
    _terminal_support_identity_envelope_from_certificate,
)
from .writer_snapshot_prefix_envelope import (
    _text_projection_certificate_identity_envelope,
)
from .writer_snapshot_prefix_envelope import (
    _writer_frontier_product_identity_envelope,
)
from .writer_snapshot_prefix_envelope import (
    verify_writer_snapshot_prefix_read_envelope,
)


SCHEMA_NAME = "writer_frontier_count"
SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset((
    "schema_name",
    "schema_version",
    "prepared_identity",
    "source_kind",
    "source_snapshot",
    "prefix_read_envelope",
    "frontier_snapshot",
    "frontier_product",
    "support_count",
    "completion_count",
    "support_count_certificate",
    "completion_count_certificate",
    "choice_count_certificates",
    "terminal_choice_count_certificate",
    "coverage",
))
_SOURCE_KINDS = frozenset(("snapshot", "prefix_read"))


@dataclass(frozen=True, slots=True)
class WriterFrontierCountEnvelopeVerification:
    accepted: bool
    source_kind: str
    support_count: int | None
    completion_count: int | None
    frontier_snapshot: object | None
    reason: str | None = None


def writer_frontier_count_envelope_for_snapshot(
    *,
    prepared: SouthStarPreparedMol,
    snapshot,
) -> dict[str, object]:
    product = _counted_frontier_product(prepared=prepared, snapshot=snapshot)
    envelope = _envelope_from_product(
        prepared=prepared,
        source_kind="snapshot",
        source_snapshot=snapshot,
        prefix_read_envelope=None,
        frontier_snapshot=snapshot,
        product=product,
    )
    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope)
    return envelope


def writer_frontier_count_envelope_for_prefix_read(
    *,
    prepared: SouthStarPreparedMol,
    prefix_read_envelope: Mapping[str, object],
) -> dict[str, object]:
    verification = verify_writer_snapshot_prefix_read_envelope(
        prepared=prepared,
        envelope=prefix_read_envelope,
    )
    if not verification.accepted:
        _count_envelope_violation("prefix_read_envelope_rejected")
    if verification.read_kind != "readable":
        _count_envelope_violation("prefix_read_envelope_not_readable")
    if verification.final_snapshot is None:
        _count_envelope_violation("prefix_read_envelope_lacks_final_snapshot")

    product = _counted_frontier_product(
        prepared=prepared,
        snapshot=verification.final_snapshot,
    )
    envelope = _envelope_from_product(
        prepared=prepared,
        source_kind="prefix_read",
        source_snapshot=None,
        prefix_read_envelope=prefix_read_envelope,
        frontier_snapshot=verification.final_snapshot,
        product=product,
    )
    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope)
    return envelope


def verify_writer_frontier_count_envelope(
    *,
    prepared: SouthStarPreparedMol,
    envelope: object,
) -> WriterFrontierCountEnvelopeVerification:
    try:
        _validate_envelope_shape(envelope)
        assert isinstance(envelope, Mapping)
        _assert_prepared_identity_matches(prepared, envelope)
        source_kind = str(envelope["source_kind"])
        if source_kind == "snapshot":
            frontier_snapshot = _source_snapshot_from_envelope(
                prepared=prepared,
                envelope=envelope,
            )
            expected = writer_frontier_count_envelope_for_snapshot(
                prepared=prepared,
                snapshot=frontier_snapshot,
            )
        elif source_kind == "prefix_read":
            prefix_envelope = envelope["prefix_read_envelope"]
            verification = verify_writer_snapshot_prefix_read_envelope(
                prepared=prepared,
                envelope=prefix_envelope,
            )
            if not verification.accepted:
                _count_envelope_violation("prefix_read_envelope_rejected")
            if verification.read_kind != "readable":
                _count_envelope_violation("prefix_read_envelope_not_readable")
            frontier_snapshot = verification.final_snapshot
            if frontier_snapshot is None:
                _count_envelope_violation(
                    "prefix_read_envelope_lacks_final_snapshot"
                )
            expected = writer_frontier_count_envelope_for_prefix_read(
                prepared=prepared,
                prefix_read_envelope=prefix_envelope,
            )
        else:
            _count_envelope_violation("unknown_source_kind")

        if expected != envelope:
            return WriterFrontierCountEnvelopeVerification(
                accepted=False,
                source_kind=source_kind,
                support_count=None,
                completion_count=None,
                frontier_snapshot=frontier_snapshot,
                reason="envelope_terms_mismatch",
            )
        return WriterFrontierCountEnvelopeVerification(
            accepted=True,
            source_kind=source_kind,
            support_count=envelope["support_count"],
            completion_count=envelope["completion_count"],
            frontier_snapshot=frontier_snapshot,
        )
    except SouthStarError as exc:
        return WriterFrontierCountEnvelopeVerification(
            accepted=False,
            source_kind=(
                envelope.get("source_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            support_count=None,
            completion_count=None,
            frontier_snapshot=None,
            reason=exc.args[-1] if exc.args else "verification_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterFrontierCountEnvelopeVerification(
            accepted=False,
            source_kind=(
                envelope.get("source_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            support_count=None,
            completion_count=None,
            frontier_snapshot=None,
            reason=f"malformed_envelope:{type(exc).__name__}",
        )


def _counted_frontier_product(*, prepared, snapshot):
    return _checked_writer_frontier_product(
        prepared,
        snapshot.cursor,
        include_counts=True,
        include_frontier_certificate=True,
        include_count_certificate=True,
    )


def _envelope_from_product(
    *,
    prepared,
    source_kind: str,
    source_snapshot,
    prefix_read_envelope,
    frontier_snapshot,
    product,
) -> dict[str, object]:
    if product.blocked:
        _count_envelope_violation("count_envelope_requires_legal_frontier")
    checked = product.checked_frontier_certificate
    if checked is None:
        _count_envelope_violation("missing_checked_frontier_certificate")
    support_count = product.support_count_certificate.support_count
    completion_count = product.count_certificate.completion_count
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "prepared_identity": _identity_envelope(frontier_snapshot.prepared_identity),
        "source_kind": source_kind,
        "source_snapshot": (
            None
            if source_snapshot is None
            else _snapshot_identity_envelope(source_snapshot)
        ),
        "prefix_read_envelope": prefix_read_envelope,
        "frontier_snapshot": _snapshot_identity_envelope(frontier_snapshot),
        "frontier_product": _writer_frontier_product_identity_envelope(product),
        "support_count": support_count,
        "completion_count": completion_count,
        "support_count_certificate": _text_support_count_certificate_envelope(
            product.support_count_certificate
        ),
        "completion_count_certificate": (
            _cursor_completion_count_certificate_envelope(product.count_certificate)
        ),
        "choice_count_certificates": [
            _text_choice_count_certificate_envelope(certificate)
            for certificate in product.text_choice_count_certificates
        ],
        "terminal_choice_count_certificate": (
            _terminal_choice_count_certificate_envelope(
                product.terminal_choice_count_certificate
            )
        ),
        "coverage": _coverage_envelope(product),
    }


def _text_support_count_certificate_envelope(certificate):
    if certificate is None:
        return None
    envelope = {
        "kind": "writer_text_support_count",
        "source_snapshot": _snapshot_or_cursor_envelope(
            certificate.source_snapshot
        ),
        "cursor": _cursor_envelope(certificate.cursor),
        "state_support_count_certificate": (
            _text_state_support_count_certificate_envelope(
                certificate.state_support_count_certificate
            )
        ),
        "support_count": certificate.support_count,
    }
    envelope["digest"] = _digest(_term(envelope))
    return envelope


def _text_state_support_count_certificate_envelope(certificate):
    if certificate is None:
        return None
    envelope = {
        "kind": "writer_text_state_support_count",
        "cursor": _cursor_envelope(certificate.cursor),
        "terminal_projection": (
            _terminal_projection_certificate_identity_envelope(
                certificate.terminal_projection_certificate
            )
        ),
        "terminal_count": certificate.terminal_count,
        "choice_terms": [
            _text_choice_support_count_term_envelope(term)
            for term in certificate.choice_terms
        ],
        "support_count": certificate.support_count,
    }
    envelope["digest"] = _digest(_term(envelope))
    return envelope


def _text_choice_support_count_term_envelope(certificate):
    envelope = {
        "kind": "writer_text_choice_support_count_term",
        "text_projection": _text_projection_certificate_identity_envelope(
            certificate.text_projection_certificate
        ),
        "successor_support_count_certificate": (
            _text_state_support_count_certificate_envelope(
                certificate.successor_support_count_certificate
            )
        ),
        "support_count": certificate.support_count,
    }
    envelope["digest"] = _digest(_term(envelope))
    return envelope


def _cursor_completion_count_certificate_envelope(certificate):
    if certificate is None:
        return None
    envelope = {
        "kind": "writer_cursor_completion_count",
        "cursor": _cursor_envelope(certificate.cursor),
        "state_count_certificates": [
            {
                "state_key_digest": _digest(_term(state_key)),
                "cursor_weight": weight,
                "state_count_certificate": (
                    _state_completion_count_certificate_envelope(
                        state_certificate
                    )
                ),
            }
            for state_key, weight, state_certificate in (
                certificate.state_count_certificates
            )
        ],
        "completion_count": certificate.completion_count,
    }
    envelope["digest"] = _digest(_term(envelope))
    return envelope


def _state_completion_count_certificate_envelope(certificate):
    if certificate is None:
        return None
    envelope = {
        "kind": "writer_state_completion_count",
        "state_key_digest": _digest(_term(certificate.state_key)),
        "terminal_projection": (
            _terminal_projection_certificate_identity_envelope(
                certificate.terminal_projection_certificate
            )
        ),
        "terminal_count": certificate.terminal_count,
        "branch_terms": [
            _branch_completion_term_certificate_envelope(term)
            for term in certificate.branch_terms
        ],
        "completion_count": certificate.completion_count,
    }
    envelope["digest"] = _digest(_term(envelope))
    return envelope


def _branch_completion_term_certificate_envelope(certificate):
    envelope = {
        "kind": "writer_branch_completion_term",
        "branch_certificate": _branch_certificate_identity_envelope(
            certificate.branch_certificate
        ),
        "successor_count_certificate": (
            _cursor_completion_count_certificate_envelope(
                certificate.successor_count_certificate
            )
        ),
        "successor_count": certificate.successor_count,
    }
    envelope["digest"] = _digest(_term(envelope))
    return envelope


def _text_choice_count_certificate_envelope(certificate):
    envelope = {
        "kind": "writer_text_choice_count",
        "text_projection": _text_projection_certificate_identity_envelope(
            certificate.text_projection_certificate
        ),
        "support_count_certificate": (
            _text_state_support_count_certificate_envelope(
                certificate.support_count_certificate
            )
        ),
        "completion_count_certificate": (
            _cursor_completion_count_certificate_envelope(
                certificate.completion_count_certificate
            )
        ),
        "emitted_text": certificate.emitted_text,
        "support_count": certificate.support_count,
        "completion_count": certificate.completion_count,
    }
    envelope["digest"] = _digest(_term(envelope))
    return envelope


def _terminal_choice_count_certificate_envelope(certificate):
    if certificate is None:
        return None
    envelope = {
        "kind": "writer_terminal_choice_count",
        "terminal_projection": _terminal_projection_certificate_identity_envelope(
            certificate.terminal_projection_certificate
        ),
        "support_count": certificate.support_count,
        "completion_count": certificate.completion_count,
    }
    envelope["digest"] = _digest(_term(envelope))
    return envelope


def _coverage_envelope(product):
    checked = product.checked_frontier_certificate
    support_coverage = checked.support_count_term_coverage_certificate
    completion_aggregate = checked.frontier_completion_count_certificate
    completion_coverage = completion_aggregate.term_coverage_certificate
    choice_coverage = checked.choice_count_coverage_certificate
    envelope = {
        "frontier_projection_digest": (
            product
            .checked_frontier_certificate
            .choice_count_coverage_certificate
            and _writer_frontier_product_identity_envelope(product)[
                "frontier_projection_certificate"
            ]["digest"]
        ),
        "terminal_covered": product.terminal_projection_certificate is not None,
        "text_choices_covered": [
            {
                "emitted_text": term.text_projection_certificate.emitted_text,
                "projection_digest": (
                    _text_projection_certificate_identity_envelope(
                        term.text_projection_certificate
                    )["digest"]
                ),
                "support_count": term.support_count,
                "completion_count": term.completion_count,
                "successor_support_count_digest": (
                    _text_state_support_count_certificate_envelope(
                        term
                        .support_coverage_term
                        .successor_support_count_certificate
                    )["digest"]
                ),
                "completion_branch_digests": [
                    _branch_certificate_identity_envelope(
                        completion_term.projection_branch_certificate
                    )["digest"]
                    for completion_term in term.completion_coverage_terms
                ],
            }
            for term in choice_coverage.text_choice_terms
        ],
        "terminal_choice_coverage": (
            None
            if choice_coverage.terminal_choice_term is None
            else {
                "terminal_projection_digest": (
                    _terminal_projection_certificate_identity_envelope(
                        choice_coverage
                        .terminal_choice_term
                        .terminal_projection_certificate
                    )["digest"]
                ),
                "terminal_support_identities": [
                    _terminal_support_identity_envelope_from_certificate(
                        certificate
                    )
                    for certificate in (
                        choice_coverage
                        .terminal_choice_term
                        .terminal_projection_certificate
                        .terminal_certificates
                    )
                ],
                "support_count": (
                    choice_coverage.terminal_choice_term.support_count
                ),
                "completion_count": (
                    choice_coverage.terminal_choice_term.completion_count
                ),
            }
        ),
        "branch_terms_covered": [
            {
                "branch_support_identity": _branch_certificate_identity_envelope(
                    term.projection_branch_certificate
                ),
                "successor_count_certificate_digest": (
                    _cursor_completion_count_certificate_envelope(
                        term
                        .count_branch_term_certificate
                        .successor_count_certificate
                    )["digest"]
                ),
                "successor_completion_count": term.successor_completion_count,
                "weighted_completion_count": term.weighted_completion_count,
            }
            for term in completion_coverage.branch_terms
        ],
        "support_text_term_count": len(support_coverage.text_terms),
        "support_terminal_covered": support_coverage.terminal_term is not None,
        "completion_branch_term_count": len(completion_coverage.branch_terms),
        "completion_terminal_term_count": len(
            completion_coverage.terminal_terms
        ),
        "support_count_total": support_coverage.support_count,
        "completion_count_total": completion_coverage.completion_count,
    }
    envelope["digest"] = _digest(_term(envelope))
    return envelope


def _snapshot_or_cursor_envelope(value):
    if hasattr(value, "decoder_boundary"):
        return _snapshot_identity_envelope(value)
    if hasattr(value, "weighted_states"):
        return _cursor_envelope(value)
    return {"digest": _digest(_term(value)), "terms": _term(value)}


def _validate_envelope_shape(envelope: object) -> None:
    if not isinstance(envelope, Mapping):
        _count_envelope_violation("envelope_not_mapping")
    if frozenset(envelope) != _TOP_LEVEL_FIELDS:
        _count_envelope_violation("top_level_fields_mismatch")
    if envelope["schema_name"] != SCHEMA_NAME:
        _count_envelope_violation("unknown_schema_name")
    if envelope["schema_version"] != SCHEMA_VERSION:
        _count_envelope_violation("unknown_schema_version")
    if envelope["source_kind"] not in _SOURCE_KINDS:
        _count_envelope_violation("unknown_source_kind")
    if envelope["source_kind"] == "snapshot":
        if envelope["source_snapshot"] is None:
            _count_envelope_violation("snapshot_source_missing_snapshot")
        if envelope["prefix_read_envelope"] is not None:
            _count_envelope_violation("snapshot_source_has_prefix_envelope")
    else:
        if envelope["source_snapshot"] is not None:
            _count_envelope_violation("prefix_source_has_source_snapshot")
        if envelope["prefix_read_envelope"] is None:
            _count_envelope_violation("prefix_source_missing_prefix_envelope")


def _assert_prepared_identity_matches(prepared, envelope) -> None:
    snapshot_terms = (
        envelope["source_snapshot"]
        if envelope["source_kind"] == "snapshot"
        else envelope["frontier_snapshot"]
    )
    runtime_options = _runtime_options_from_terms(
        snapshot_terms["runtime_options"]
    )
    actual = _identity_envelope(_prepared_identity(prepared, runtime_options))
    if envelope["prepared_identity"] != actual:
        _count_envelope_violation("prepared_identity_mismatch")
    if snapshot_terms["prepared_identity_digest"] != actual["digest"]:
        _count_envelope_violation("snapshot_prepared_identity_mismatch")


def _count_envelope_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer frontier count envelope violation: {kind}",
    )


__all__ = (
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "WriterFrontierCountEnvelopeVerification",
    "verify_writer_frontier_count_envelope",
    "writer_frontier_count_envelope_for_prefix_read",
    "writer_frontier_count_envelope_for_snapshot",
)
