"""Durable envelopes for checked writer snapshot prefix reads."""

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
) -> dict[str, object]:
    replay_envelope = writer_snapshot_replay_envelope_for_emitted_texts(
        prepared=prepared,
        snapshot=snapshot,
        emitted_texts=emitted_texts,
    )
    replay_verification = verify_writer_snapshot_replay_envelope(
        prepared=prepared,
        envelope=replay_envelope,
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
) -> WriterSnapshotPrefixReadEnvelopeVerification:
    try:
        _validate_envelope_shape(envelope)
        assert isinstance(envelope, Mapping)
        _assert_prepared_identity_matches(prepared, envelope)
        replay = verify_writer_snapshot_replay_envelope(
            prepared=prepared,
            envelope=envelope["replay_envelope"],
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
        prefix_read_certificate=None,
        public_frontier=None,
        support_count=None,
        completion_count=None,
        failure={
            "kind": "final_frontier_blocked",
            "blocked_frontier_certificate": {
                "cursor": _cursor_envelope(blocked.cursor),
                "blocked": blocked.blocked,
                "diagnostic_certificate_digest": _digest(_term(diagnostic)),
                "digest": _digest(_term(blocked)),
            },
            "diagnostic_certificate_digest": _digest(_term(diagnostic)),
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
        prefix_read_certificate=_prefix_read_certificate_envelope(
            certificate
        ),
        public_frontier=_public_frontier_envelope(
            prefix.frontier_product.choices
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
        "replay_certificate_digest": _certificate_digest(
            certificate.replay_certificate
        ),
        "final_snapshot": _snapshot_identity_envelope(
            certificate.final_snapshot
        ),
        "final_frontier_projection_certificate_digest": (
            _certificate_digest(
                certificate.final_frontier_projection_certificate
            )
        ),
        "checked_frontier_certificate_digest": _certificate_digest(
            certificate.checked_frontier_certificate
        ),
        "support_count_certificate_digest": _certificate_digest(
            certificate.support_count_certificate
        ),
        "completion_count_certificate_digest": _certificate_digest(
            certificate.completion_count_certificate
        ),
        "support_count": certificate.support_count,
        "completion_count": certificate.completion_count,
    }
    envelope["digest"] = _digest(_term(envelope))
    return envelope


def _certificate_digest(certificate) -> str | None:
    if certificate is None:
        return None
    terms = {
        "__class__": (
            f"{certificate.__class__.__module__}."
            f"{certificate.__class__.__name__}"
        ),
    }
    for name in (
        "cursor",
        "source_cursor",
        "successor_cursor",
        "finalized_cursor",
        "source_snapshot",
        "final_snapshot",
        "emitted_text",
        "emitted_texts",
        "support_count",
        "completion_count",
        "multiplicity",
        "blocked",
    ):
        if hasattr(certificate, name):
            terms[name] = _compact_certificate_term(getattr(certificate, name))
    return _digest(_term(terms))


def _compact_certificate_term(value):
    if hasattr(value, "cursor") and hasattr(value, "decoder_boundary"):
        return _snapshot_identity_envelope(value)
    if hasattr(value, "weighted_states"):
        return _cursor_envelope(value)
    if isinstance(value, tuple):
        return [_compact_certificate_term(item) for item in value]
    if value is None or isinstance(value, (str, bool, int)):
        return value
    return _term(value)


def _public_frontier_envelope(choices) -> dict[str, object]:
    terminal = choices.terminal
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
            for choice in choices.choices
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
