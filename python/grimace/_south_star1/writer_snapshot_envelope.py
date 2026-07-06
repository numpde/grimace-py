"""Durable envelopes for single writer snapshot advances."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import fields
from dataclasses import is_dataclass
from enum import Enum
import hashlib
import json

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .policy import SerializationLanguageMode
from .prepared_runtime import SouthStarPreparedMol
from .prepared_runtime import SouthStarRuntimeOptions
from .writer_frontier import initial_writer_frontier_cursor
from .writer_frontier import _snapshot_advance_writer_frontier_product
from .writer_snapshot import WriterDecoderBoundary
from .writer_snapshot import _capture_writer_frontier_snapshot_unchecked
from .writer_snapshot import _prepared_identity
from .writer_snapshot import _writer_snapshot_advance_outcome_by_emitted_text


SCHEMA_NAME = "writer_snapshot_advance"
SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset((
    "schema_name",
    "schema_version",
    "prepared_identity",
    "source_snapshot",
    "emitted_text",
    "outcome_kind",
    "frontier_product_kind",
    "advance_certificate",
))
_OUTCOME_KINDS = frozenset(("advanced", "invalid_emitted_text", "blocked"))
_PRODUCT_KINDS = frozenset(("legal", "blocked"))


@dataclass(frozen=True, slots=True)
class WriterSnapshotAdvanceEnvelopeVerification:
    accepted: bool
    outcome_kind: str
    source_snapshot: object | None
    advanced_snapshot: object | None
    reason: str | None = None


def writer_snapshot_advance_envelope_for_emitted_text(
    *,
    prepared: SouthStarPreparedMol,
    snapshot,
    emitted_text: str,
) -> dict[str, object]:
    outcome = _writer_snapshot_advance_outcome_by_emitted_text(
        snapshot,
        prepared=prepared,
        emitted_text=emitted_text,
    )
    return _envelope_from_outcome(
        prepared=prepared,
        outcome=outcome,
    )


def verify_writer_snapshot_advance_envelope(
    *,
    prepared: SouthStarPreparedMol,
    envelope: object,
) -> WriterSnapshotAdvanceEnvelopeVerification:
    try:
        _validate_envelope_shape(envelope)
        assert isinstance(envelope, Mapping)
        outcome_kind = envelope["outcome_kind"]
        source_snapshot = _source_snapshot_from_envelope(
            prepared=prepared,
            envelope=envelope,
        )
        expected = writer_snapshot_advance_envelope_for_emitted_text(
            prepared=prepared,
            snapshot=source_snapshot,
            emitted_text=envelope["emitted_text"],
        )
        if expected != envelope:
            return WriterSnapshotAdvanceEnvelopeVerification(
                accepted=False,
                outcome_kind=str(outcome_kind),
                source_snapshot=source_snapshot,
                advanced_snapshot=None,
                reason="envelope_terms_mismatch",
            )

        advanced_snapshot = None
        if outcome_kind == "advanced":
            outcome = _writer_snapshot_advance_outcome_by_emitted_text(
                source_snapshot,
                prepared=prepared,
                emitted_text=envelope["emitted_text"],
            )
            advanced_snapshot = outcome.advanced_snapshot
        return WriterSnapshotAdvanceEnvelopeVerification(
            accepted=True,
            outcome_kind=str(outcome_kind),
            source_snapshot=source_snapshot,
            advanced_snapshot=advanced_snapshot,
        )
    except SouthStarError as exc:
        return WriterSnapshotAdvanceEnvelopeVerification(
            accepted=False,
            outcome_kind=(
                envelope.get("outcome_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            source_snapshot=None,
            advanced_snapshot=None,
            reason=exc.args[-1] if exc.args else "verification_error",
        )
    except (KeyError, TypeError, ValueError) as exc:
        return WriterSnapshotAdvanceEnvelopeVerification(
            accepted=False,
            outcome_kind=(
                envelope.get("outcome_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            source_snapshot=None,
            advanced_snapshot=None,
            reason=f"malformed_envelope:{type(exc).__name__}",
        )


def _envelope_from_outcome(*, prepared, outcome) -> dict[str, object]:
    source_snapshot = outcome.source_snapshot
    product_kind = "blocked" if outcome.frontier_product.blocked else "legal"
    envelope = {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "prepared_identity": _identity_envelope(source_snapshot.prepared_identity),
        "source_snapshot": _snapshot_identity_envelope(source_snapshot),
        "emitted_text": outcome.emitted_text,
        "outcome_kind": outcome.kind.value,
        "frontier_product_kind": product_kind,
        "advance_certificate": _advance_certificate_envelope(outcome),
    }
    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope)
    return envelope


def _advance_certificate_envelope(outcome) -> dict[str, object]:
    if outcome.kind.value == "advanced":
        step = outcome.step_certificate
        return {
            "kind": "advanced",
            "frontier_projection": _frontier_projection_envelope(
                outcome.frontier_projection_certificate
            ),
            "selected_text_projection": _text_projection_envelope(
                outcome.text_projection_certificate
            ),
            "step_certificate": {
                "source_snapshot": _snapshot_identity_envelope(
                    step.source_snapshot
                ),
                "source_cursor": _cursor_envelope(step.source_cursor),
                "successor_cursor": _cursor_envelope(step.successor_cursor),
                "advanced_snapshot": _snapshot_identity_envelope(
                    step.advanced_snapshot
                ),
                "decoder_boundary_before": _term(
                    step.decoder_boundary_before
                ),
                "decoder_boundary_after": _term(step.decoder_boundary_after),
                "frontier_projection_digest": _digest(
                    _term(step.frontier_projection_certificate)
                ),
                "text_projection_digest": _digest(
                    _term(step.text_projection_certificate)
                ),
                "branch_certificate_digests": [
                    _digest(_term(certificate))
                    for certificate in step.branch_certificates
                ],
            },
            "advanced_snapshot": _snapshot_identity_envelope(
                outcome.advanced_snapshot
            ),
        }
    if outcome.kind.value == "invalid_emitted_text":
        projection = outcome.invalid_text_frontier_projection_certificate
        return {
            "kind": "invalid_emitted_text",
            "frontier_projection": _frontier_projection_envelope(projection),
            "invalid_text_certificate": {
                "source_snapshot": _snapshot_identity_envelope(
                    outcome.invalid_text_certificate.source_snapshot
                ),
                "emitted_text": outcome.invalid_text_certificate.emitted_text,
                "frontier_projection_digest": _digest(_term(projection)),
            },
            "projected_emitted_texts": [
                item.emitted_text
                for item in projection.text_choice_projection_certificates
            ],
            "projected_text_projection_digests": [
                _digest(_term(item))
                for item in projection.text_choice_projection_certificates
            ],
        }
    if outcome.kind.value == "blocked":
        blocked = outcome.blocked_frontier_certificate
        diagnostic = blocked.diagnostic_certificate
        return {
            "kind": "blocked",
            "blocked_frontier_certificate": {
                "cursor": _cursor_envelope(blocked.cursor),
                "blocked": blocked.blocked,
                "diagnostic_certificate_digest": _digest(_term(diagnostic)),
            },
            "blocked_advance_certificate": {
                "source_snapshot": _snapshot_identity_envelope(
                    outcome.blocked_advance_certificate.source_snapshot
                ),
                "emitted_text": (
                    outcome.blocked_advance_certificate.emitted_text
                ),
                "blocked_frontier_certificate_digest": _digest(
                    _term(blocked)
                ),
            },
            "diagnostic_certificate": _diagnostic_envelope(diagnostic),
        }
    _envelope_violation("unknown_outcome_kind")


def _frontier_projection_envelope(projection) -> dict[str, object]:
    return {
        "cursor": _cursor_envelope(projection.cursor),
        "text_projection_digests": [
            _digest(_term(item))
            for item in projection.text_choice_projection_certificates
        ],
        "text_projection_keys": [
            _text_projection_key(item)
            for item in projection.text_choice_projection_certificates
        ],
        "terminal_projection_digest": (
            None
            if projection.terminal_projection_certificate is None
            else _digest(_term(projection.terminal_projection_certificate))
        ),
        "digest": _digest(_term(projection)),
    }


def _text_projection_envelope(projection) -> dict[str, object]:
    return {
        "source_cursor": _cursor_envelope(projection.source_cursor),
        "emitted_text": projection.emitted_text,
        "successor_cursor": _cursor_envelope(projection.successor_cursor),
        "immediate_multiplicity": projection.immediate_multiplicity,
        "projection_key": _text_projection_key(projection),
        "branch_certificate_digests": [
            _digest(_term(item)) for item in projection.branch_certificates
        ],
        "digest": _digest(_term(projection)),
    }


def _diagnostic_envelope(diagnostic) -> dict[str, object]:
    return {
        "cursor": _cursor_envelope(diagnostic.cursor),
        "unsupported_execution_capabilities": [
            *sorted(
                _term(item.capability)
                for item in (
                    diagnostic
                    .unsupported_execution_capability_certificates
                )
            )
        ],
        "unsupported_terminal_execution_capabilities": [
            *sorted(
                _term(item.capability)
                for item in (
                    diagnostic
                    .unsupported_terminal_execution_capability_certificates
                )
            )
        ],
        "residual_work_envelope_violations": [
            _term(item.violation)
            for item in diagnostic.work_envelope_violation_certificates
            if item.category == "residual_work"
        ],
        "terminal_residual_work_envelope_violations": [
            _term(item.violation)
            for item in diagnostic.work_envelope_violation_certificates
            if item.category == "terminal_residual_work"
        ],
        "finite_relation_work_envelope_violations": [
            _term(item.violation)
            for item in diagnostic.work_envelope_violation_certificates
            if item.category == "finite_relation_work"
        ],
        "graph_obligation_work_envelope_violations": [
            _term(item.violation)
            for item in diagnostic.work_envelope_violation_certificates
            if item.category == "graph_obligation"
        ],
        "digest": _digest(_term(diagnostic)),
    }


def _source_snapshot_from_envelope(*, prepared, envelope) -> object:
    _assert_prepared_identity_matches(prepared, envelope)
    snapshot_terms = envelope["source_snapshot"]
    runtime_options = _runtime_options_from_terms(
        snapshot_terms["runtime_options"]
    )
    cursor_digest = snapshot_terms["cursor"]["digest"]
    expected_boundary = snapshot_terms["decoder_boundary"]
    for cursor in _reachable_cursors(prepared, runtime_options):
        snapshot = _capture_writer_frontier_snapshot_unchecked(
            prepared=prepared,
            runtime_options=runtime_options,
            cursor=cursor,
            decoder_boundary=WriterDecoderBoundary(
                consumed_token_count=(
                    expected_boundary["consumed_token_count"]
                ),
            ),
        )
        if _cursor_envelope(snapshot.cursor)["digest"] == cursor_digest:
            if _snapshot_identity_envelope(snapshot) != snapshot_terms:
                _envelope_violation("source_snapshot_identity_mismatch")
            return snapshot
    _envelope_violation("source_snapshot_cursor_not_reachable")


def _reachable_cursors(prepared, runtime_options):
    pending = [initial_writer_frontier_cursor(prepared, runtime_options)]
    seen = set()
    while pending and len(seen) < 5000:
        cursor = pending.pop(0)
        if cursor in seen:
            continue
        seen.add(cursor)
        yield cursor
        product = _snapshot_advance_writer_frontier_product(
            prepared,
            cursor,
        )
        if product.blocked:
            continue
        for projection in (
            product.projection_certificate.text_choice_projection_certificates
        ):
            pending.append(projection.successor_cursor)


def _runtime_options_from_terms(terms: Mapping[str, object]):
    return SouthStarRuntimeOptions(
        rooted_at_atom=terms["rooted_at_atom"],
        canonical=terms["canonical"],
        do_random=terms["do_random"],
        serialization_language=SerializationLanguageMode(
            terms["serialization_language"]
        ),
    )


def _assert_prepared_identity_matches(prepared, envelope) -> None:
    runtime_options = _runtime_options_from_terms(
        envelope["source_snapshot"]["runtime_options"]
    )
    identity = envelope["prepared_identity"]
    actual = _identity_envelope(_prepared_identity(prepared, runtime_options))
    if identity != actual:
        _envelope_violation("prepared_identity_mismatch")
    if (
        envelope["source_snapshot"]["prepared_identity_digest"]
        != actual["digest"]
    ):
        _envelope_violation("source_snapshot_prepared_identity_mismatch")


def _identity_envelope(identity) -> dict[str, object]:
    terms = _term(identity)
    return {
        "terms": terms,
        "digest": _digest(terms),
    }


def _snapshot_identity_envelope(snapshot) -> dict[str, object]:
    prepared_terms = _term(snapshot.prepared_identity)
    return {
        "serialization_language": snapshot.serialization_language.value,
        "runtime_options": _runtime_options_terms(snapshot.runtime_options),
        "prepared_identity_terms": prepared_terms,
        "prepared_identity_digest": _digest(prepared_terms),
        "cursor": _cursor_envelope(snapshot.cursor),
        "decoder_boundary": _decoder_boundary_terms(snapshot.decoder_boundary),
        "frame_stack_cursors": [
            _cursor_envelope(frame.cursor) for frame in snapshot.frame_stack
        ],
        "digest": _digest(_term(snapshot)),
    }


def _runtime_options_terms(options) -> dict[str, object]:
    return {
        "rooted_at_atom": options.rooted_at_atom,
        "canonical": options.canonical,
        "do_random": options.do_random,
        "serialization_language": options.serialization_language.value,
    }


def _decoder_boundary_terms(boundary) -> dict[str, object]:
    return {
        "consumed_token_count": boundary.consumed_token_count,
    }


def _cursor_envelope(cursor) -> dict[str, object]:
    terms = _term(cursor)
    return {
        "terms": terms,
        "digest": _digest(terms),
    }


def _text_projection_key(projection) -> dict[str, object]:
    return {
        "source_cursor_digest": _digest(_term(projection.source_cursor)),
        "emitted_text": projection.emitted_text,
        "successor_cursor_digest": _digest(_term(projection.successor_cursor)),
        "immediate_multiplicity": projection.immediate_multiplicity,
    }


def _validate_envelope_shape(envelope: object) -> None:
    if not isinstance(envelope, Mapping):
        _envelope_violation("envelope_not_mapping")
    keys = frozenset(envelope)
    if keys != _TOP_LEVEL_FIELDS:
        _envelope_violation("top_level_fields_mismatch")
    if envelope["schema_name"] != SCHEMA_NAME:
        _envelope_violation("unknown_schema_name")
    if envelope["schema_version"] != SCHEMA_VERSION:
        _envelope_violation("unknown_schema_version")
    if envelope["outcome_kind"] not in _OUTCOME_KINDS:
        _envelope_violation("unknown_outcome_kind")
    if envelope["frontier_product_kind"] not in _PRODUCT_KINDS:
        _envelope_violation("unknown_frontier_product_kind")
    certificate = envelope["advance_certificate"]
    if not isinstance(certificate, Mapping):
        _envelope_violation("advance_certificate_not_mapping")
    if certificate.get("kind") != envelope["outcome_kind"]:
        _envelope_violation("advance_certificate_kind_mismatch")
    if envelope["outcome_kind"] == "blocked":
        if envelope["frontier_product_kind"] != "blocked":
            _envelope_violation("blocked_product_kind_mismatch")
    elif envelope["frontier_product_kind"] != "legal":
        _envelope_violation("legal_product_kind_mismatch")


def _term(value):
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, int):
        return int(value)
    if isinstance(value, Enum):
        return {
            "__enum__": f"{value.__class__.__module__}.{value.__class__.__name__}",
            "value": value.value,
        }
    if isinstance(value, (tuple, list)):
        return [_term(item) for item in value]
    if isinstance(value, (frozenset, set)):
        return [
            *sorted(
                (_term(item) for item in value),
                key=_canonical_json,
            )
        ]
    if isinstance(value, Mapping):
        return [
            [str(key), _term(item)]
            for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
        ]
    if is_dataclass(value):
        return {
            "__dataclass__": (
                f"{value.__class__.__module__}.{value.__class__.__name__}"
            ),
            "fields": [
                [field.name, _term(getattr(value, field.name))]
                for field in fields(value)
            ],
        }
    _envelope_violation(f"unsupported_term_type:{type(value).__name__}")


def _digest(term) -> str:
    return hashlib.sha256(_canonical_json(term).encode("utf-8")).hexdigest()


def _canonical_json(term) -> str:
    return json.dumps(term, sort_keys=True, separators=(",", ":"))


def _envelope_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer snapshot advance envelope violation: {kind}",
    )


__all__ = (
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "WriterSnapshotAdvanceEnvelopeVerification",
    "verify_writer_snapshot_advance_envelope",
    "writer_snapshot_advance_envelope_for_emitted_text",
)
