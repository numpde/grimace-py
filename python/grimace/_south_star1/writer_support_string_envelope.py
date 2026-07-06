"""Durable envelopes for individual certified writer support strings."""

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
from .writer_frontier import _snapshot_advance_writer_frontier_product
from .writer_frontier_count_envelope import (
    verify_writer_frontier_count_envelope,
)
from .writer_frontier_count_envelope import (
    writer_frontier_count_envelope_for_prefix_read,
)
from .writer_frontier_count_envelope import (
    writer_frontier_count_envelope_for_snapshot,
)
from .writer_snapshot import _iter_writer_snapshot_certified_support_strings
from .writer_snapshot import _prepared_identity
from .writer_snapshot import (
    _writer_snapshot_advance_sequence_outcome_by_emitted_texts,
)
from .writer_snapshot_envelope import _source_snapshot_from_envelope
from .writer_snapshot_prefix_envelope import (
    _branch_certificate_identity_envelope,
)
from .writer_snapshot_prefix_envelope import (
    _frontier_projection_certificate_identity_envelope,
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
from .writer_snapshot_replay_envelope import (
    verify_writer_snapshot_replay_envelope,
)
from .writer_snapshot_replay_envelope import (
    writer_snapshot_replay_envelope_for_emitted_texts,
)
from .writer_support_certificates import writer_support_string_certificate


SCHEMA_NAME = "writer_support_string"
SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset((
    "schema_name",
    "schema_version",
    "prepared_identity",
    "source_kind",
    "source_snapshot",
    "prefix_read_envelope",
    "count_envelope",
    "string",
    "emitted_texts",
    "replay_envelope",
    "final_snapshot",
    "terminal_frontier_product",
    "terminal_frontier_projection",
    "terminal_projection",
    "support_string_certificate",
    "terminal_support_identities",
    "text_projection_chain",
))
_SOURCE_KINDS = frozenset(("snapshot", "prefix_read"))


@dataclass(frozen=True, slots=True)
class WriterSupportStringEnvelopeVerification:
    accepted: bool
    source_kind: str
    string: str | None
    source_snapshot: object | None
    final_snapshot: object | None
    reason: str | None = None


def writer_support_string_envelope_for_string(
    *,
    prepared: SouthStarPreparedMol,
    snapshot,
    string: str,
) -> dict[str, object]:
    certificate = _certified_support_string_for_string(
        prepared=prepared,
        snapshot=snapshot,
        string=string,
    )
    count_envelope = writer_frontier_count_envelope_for_snapshot(
        prepared=prepared,
        snapshot=snapshot,
    )
    replay_envelope = writer_snapshot_replay_envelope_for_emitted_texts(
        prepared=prepared,
        snapshot=snapshot,
        emitted_texts=certificate.emitted_texts,
    )
    terminal_product = _terminal_product_for_certificate(
        prepared=prepared,
        certificate=certificate,
    )
    envelope = _envelope_from_certificate_with_product(
        source_kind="snapshot",
        source_snapshot=snapshot,
        prefix_read_envelope=None,
        count_envelope=count_envelope,
        replay_envelope=replay_envelope,
        certificate=certificate,
        terminal_product=terminal_product,
    )
    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope)
    return envelope


def writer_support_string_envelope_for_prefix_read(
    *,
    prepared: SouthStarPreparedMol,
    prefix_read_envelope: Mapping[str, object],
    string: str,
) -> dict[str, object]:
    prefix = verify_writer_snapshot_prefix_read_envelope(
        prepared=prepared,
        envelope=prefix_read_envelope,
    )
    if not prefix.accepted:
        _support_string_envelope_violation("prefix_read_envelope_rejected")
    if prefix.read_kind != "readable":
        _support_string_envelope_violation("prefix_read_envelope_not_readable")
    if prefix.final_snapshot is None:
        _support_string_envelope_violation(
            "prefix_read_envelope_lacks_final_snapshot"
        )
    certificate = _certified_support_string_for_string(
        prepared=prepared,
        snapshot=prefix.final_snapshot,
        string=string,
    )
    count_envelope = writer_frontier_count_envelope_for_prefix_read(
        prepared=prepared,
        prefix_read_envelope=prefix_read_envelope,
    )
    replay_envelope = writer_snapshot_replay_envelope_for_emitted_texts(
        prepared=prepared,
        snapshot=prefix.final_snapshot,
        emitted_texts=certificate.emitted_texts,
    )
    terminal_product = _terminal_product_for_certificate(
        prepared=prepared,
        certificate=certificate,
    )
    envelope = _envelope_from_certificate_with_product(
        source_kind="prefix_read",
        source_snapshot=None,
        prefix_read_envelope=prefix_read_envelope,
        count_envelope=count_envelope,
        replay_envelope=replay_envelope,
        certificate=certificate,
        terminal_product=terminal_product,
    )
    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope)
    return envelope


def _writer_support_string_envelope_from_certificate(
    *,
    prepared: SouthStarPreparedMol,
    source_kind: str,
    source_snapshot,
    prefix_read_envelope,
    count_envelope,
    replay_envelope,
    certificate,
) -> dict[str, object]:
    terminal_product = _terminal_product_for_certificate(
        prepared=prepared,
        certificate=certificate,
    )
    envelope = _envelope_from_certificate_with_product(
        source_kind=source_kind,
        source_snapshot=source_snapshot,
        prefix_read_envelope=prefix_read_envelope,
        count_envelope=count_envelope,
        replay_envelope=replay_envelope,
        certificate=certificate,
        terminal_product=terminal_product,
    )
    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope)
    return envelope


def verify_writer_support_string_envelope(
    *,
    prepared: SouthStarPreparedMol,
    envelope: object,
) -> WriterSupportStringEnvelopeVerification:
    try:
        _validate_envelope_shape(envelope)
        assert isinstance(envelope, Mapping)
        _assert_prepared_identity_matches(prepared, envelope)
        source_kind = str(envelope["source_kind"])
        source_snapshot = _source_snapshot_for_envelope(
            prepared=prepared,
            envelope=envelope,
        )
        count = verify_writer_frontier_count_envelope(
            prepared=prepared,
            envelope=envelope["count_envelope"],
        )
        if not count.accepted:
            _support_string_envelope_violation("count_envelope_rejected")
        if count.frontier_snapshot != source_snapshot:
            _support_string_envelope_violation(
                "count_envelope_source_mismatch"
            )
        replay = verify_writer_snapshot_replay_envelope(
            prepared=prepared,
            envelope=envelope["replay_envelope"],
        )
        if not replay.accepted:
            _support_string_envelope_violation("replay_envelope_rejected")
        if replay.outcome_kind != "advanced":
            _support_string_envelope_violation("replay_envelope_not_advanced")
        if replay.source_snapshot != source_snapshot:
            _support_string_envelope_violation(
                "replay_envelope_source_mismatch"
            )
        if tuple(envelope["emitted_texts"]) != tuple(
            envelope["replay_envelope"]["emitted_texts"]
        ):
            _support_string_envelope_violation("replay_texts_mismatch")
        if envelope["string"] != "".join(envelope["emitted_texts"]):
            _support_string_envelope_violation("string_emitted_texts_mismatch")
        if replay.current_snapshot is None:
            _support_string_envelope_violation("replay_lacks_final_snapshot")
        expected = _expected_envelope_from_replay(
            source_kind=source_kind,
            source_snapshot=source_snapshot,
            prefix_read_envelope=envelope["prefix_read_envelope"],
            count_envelope=envelope["count_envelope"],
            replay_envelope=envelope["replay_envelope"],
            prepared=prepared,
            emitted_texts=tuple(envelope["emitted_texts"]),
        )
        if expected != envelope:
            return WriterSupportStringEnvelopeVerification(
                accepted=False,
                source_kind=source_kind,
                string=str(envelope["string"]),
                source_snapshot=source_snapshot,
                final_snapshot=replay.current_snapshot,
                reason="envelope_terms_mismatch",
            )
        return WriterSupportStringEnvelopeVerification(
            accepted=True,
            source_kind=source_kind,
            string=str(envelope["string"]),
            source_snapshot=source_snapshot,
            final_snapshot=replay.current_snapshot,
        )
    except SouthStarError as exc:
        return WriterSupportStringEnvelopeVerification(
            accepted=False,
            source_kind=(
                envelope.get("source_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            string=(
                envelope.get("string")
                if isinstance(envelope, Mapping)
                and isinstance(envelope.get("string"), str)
                else None
            ),
            source_snapshot=None,
            final_snapshot=None,
            reason=exc.args[-1] if exc.args else "verification_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterSupportStringEnvelopeVerification(
            accepted=False,
            source_kind=(
                envelope.get("source_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            string=(
                envelope.get("string")
                if isinstance(envelope, Mapping)
                and isinstance(envelope.get("string"), str)
                else None
            ),
            source_snapshot=None,
            final_snapshot=None,
            reason=f"malformed_envelope:{type(exc).__name__}",
        )


def _certified_support_string_for_string(*, prepared, snapshot, string: str):
    matches = tuple(
        item
        for item in _iter_writer_snapshot_certified_support_strings(
            snapshot,
            prepared=prepared,
        )
        if item.string == string
    )
    if len(matches) != 1:
        _support_string_envelope_violation("support_string_not_unique")
    return matches[0].certificate


def _expected_envelope_from_replay(
    *,
    source_kind: str,
    source_snapshot,
    prefix_read_envelope,
    count_envelope,
    replay_envelope,
    prepared,
    emitted_texts: tuple[str, ...],
) -> dict[str, object]:
    outcome = _writer_snapshot_advance_sequence_outcome_by_emitted_texts(
        source_snapshot,
        prepared=prepared,
        emitted_texts=emitted_texts,
    )
    if outcome.kind.value != "advanced" or outcome.replay_certificate is None:
        _support_string_envelope_violation("replay_outcome_not_advanced")
    product = _snapshot_advance_writer_frontier_product(
        prepared,
        outcome.current_snapshot.cursor,
    )
    if product.blocked:
        _support_string_envelope_violation("terminal_frontier_blocked")
    if product.terminal_projection_certificate is None:
        _support_string_envelope_violation("missing_terminal_projection")
    certificate = writer_support_string_certificate(
        source_snapshot=source_snapshot,
        string="".join(emitted_texts),
        emitted_texts=emitted_texts,
        replay_certificate=outcome.replay_certificate,
        terminal_frontier_projection_certificate=product.projection_certificate,
        terminal_projection_certificate=product.terminal_projection_certificate,
        text_projection_certificates=tuple(
            step.text_projection_certificate
            for step in outcome.advanced_step_outcomes
        ),
    )
    return _envelope_from_certificate_with_product(
        source_kind=source_kind,
        source_snapshot=(
            source_snapshot if source_kind == "snapshot" else None
        ),
        prefix_read_envelope=prefix_read_envelope,
        count_envelope=count_envelope,
        replay_envelope=replay_envelope,
        certificate=certificate,
        terminal_product=product,
    )


def _envelope_from_certificate_with_product(
    *,
    source_kind: str,
    source_snapshot,
    prefix_read_envelope,
    count_envelope,
    replay_envelope,
    certificate,
    terminal_product,
) -> dict[str, object]:
    text_chain = _text_projection_chain_envelope(certificate)
    terminal_projection = _terminal_projection_certificate_identity_envelope(
        certificate.terminal_projection_certificate
    )
    support_certificate = _support_string_certificate_envelope(
        certificate=certificate,
        text_projection_chain=text_chain,
        terminal_projection=terminal_projection,
    )
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "prepared_identity": _identity_envelope(
            certificate.replay_certificate.source_snapshot.prepared_identity
        ),
        "source_kind": source_kind,
        "source_snapshot": (
            None
            if source_snapshot is None
            else _snapshot_identity_envelope(source_snapshot)
        ),
        "prefix_read_envelope": prefix_read_envelope,
        "count_envelope": count_envelope,
        "string": certificate.string,
        "emitted_texts": list(certificate.emitted_texts),
        "replay_envelope": replay_envelope,
        "final_snapshot": _snapshot_identity_envelope(certificate.final_snapshot),
        "terminal_frontier_product": (
            _writer_frontier_product_identity_envelope(terminal_product)
        ),
        "terminal_frontier_projection": (
            _frontier_projection_certificate_identity_envelope(
                certificate.terminal_frontier_projection_certificate
            )
        ),
        "terminal_projection": terminal_projection,
        "support_string_certificate": support_certificate,
        "terminal_support_identities": [
            _terminal_support_identity_envelope_from_certificate(terminal)
            for terminal in (
                certificate
                .terminal_projection_certificate
                .terminal_certificates
            )
        ],
        "text_projection_chain": text_chain,
    }


def _terminal_product_for_certificate(*, prepared, certificate):
    product = _snapshot_advance_writer_frontier_product(
        prepared,
        certificate.final_snapshot.cursor,
    )
    if product.blocked:
        _support_string_envelope_violation("terminal_frontier_blocked")
    if product.terminal_projection_certificate is None:
        _support_string_envelope_violation("missing_terminal_projection")
    return product


def _support_string_certificate_envelope(
    *,
    certificate,
    text_projection_chain,
    terminal_projection,
) -> dict[str, object]:
    envelope = {
        "string": certificate.string,
        "emitted_texts": list(certificate.emitted_texts),
        "replay_certificate_digest": _digest(_term(certificate.replay_certificate)),
        "final_snapshot": _snapshot_identity_envelope(certificate.final_snapshot),
        "terminal_frontier_projection_digest": (
            _frontier_projection_certificate_identity_envelope(
                certificate.terminal_frontier_projection_certificate
            )["digest"]
        ),
        "terminal_projection_digest": terminal_projection["digest"],
        "terminal_certificate_digests": [
            _terminal_support_identity_envelope_from_certificate(
                terminal
            )["digest"]
            for terminal in certificate.terminal_certificates
        ],
        "text_projection_chain_digests": [
            step["text_projection"]["digest"] for step in text_projection_chain
        ],
    }
    envelope["digest"] = _digest(_term(envelope))
    return envelope


def _text_projection_chain_envelope(certificate):
    chain = []
    for index, step in enumerate(
        certificate.replay_certificate.step_certificates
    ):
        text_projection = _text_projection_certificate_identity_envelope(
            step.text_projection_certificate
        )
        frontier_projection = _frontier_projection_certificate_identity_envelope(
            step.frontier_projection_certificate
        )
        chain.append(
            {
                "step_index": index,
                "emitted_text": step.emitted_text,
                "source_cursor": _cursor_envelope(step.source_cursor),
                "successor_cursor": _cursor_envelope(step.successor_cursor),
                "text_projection": text_projection,
                "frontier_projection": frontier_projection,
                "step_certificate_digest": _digest(_term(step)),
                "branch_certificate_identities": [
                    _branch_certificate_identity_envelope(branch)
                    for branch in step.branch_certificates
                ],
            }
        )
    return chain


def _source_snapshot_for_envelope(*, prepared, envelope):
    if envelope["source_kind"] == "snapshot":
        return _source_snapshot_from_envelope(
            prepared=prepared,
            envelope=envelope,
        )
    prefix = verify_writer_snapshot_prefix_read_envelope(
        prepared=prepared,
        envelope=envelope["prefix_read_envelope"],
    )
    if not prefix.accepted:
        _support_string_envelope_violation("prefix_read_envelope_rejected")
    if prefix.read_kind != "readable":
        _support_string_envelope_violation("prefix_read_envelope_not_readable")
    if prefix.final_snapshot is None:
        _support_string_envelope_violation(
            "prefix_read_envelope_lacks_final_snapshot"
        )
    return prefix.final_snapshot


def _validate_envelope_shape(envelope: object) -> None:
    if not isinstance(envelope, Mapping):
        _support_string_envelope_violation("envelope_not_mapping")
    if frozenset(envelope) != _TOP_LEVEL_FIELDS:
        _support_string_envelope_violation("top_level_fields_mismatch")
    if envelope["schema_name"] != SCHEMA_NAME:
        _support_string_envelope_violation("unknown_schema_name")
    if envelope["schema_version"] != SCHEMA_VERSION:
        _support_string_envelope_violation("unknown_schema_version")
    if envelope["source_kind"] not in _SOURCE_KINDS:
        _support_string_envelope_violation("unknown_source_kind")
    if not isinstance(envelope["string"], str):
        _support_string_envelope_violation("string_not_text")
    if not isinstance(envelope["emitted_texts"], list):
        _support_string_envelope_violation("emitted_texts_not_list")
    if envelope["source_kind"] == "snapshot":
        if envelope["source_snapshot"] is None:
            _support_string_envelope_violation("snapshot_source_missing")
        if envelope["prefix_read_envelope"] is not None:
            _support_string_envelope_violation(
                "snapshot_source_has_prefix_read"
            )
    else:
        if envelope["source_snapshot"] is not None:
            _support_string_envelope_violation(
                "prefix_source_has_source_snapshot"
            )
        if envelope["prefix_read_envelope"] is None:
            _support_string_envelope_violation("prefix_source_missing_prefix")


def _assert_prepared_identity_matches(prepared, envelope) -> None:
    snapshot_terms = (
        envelope["source_snapshot"]
        if envelope["source_kind"] == "snapshot"
        else envelope["final_snapshot"]
    )
    runtime_options = _runtime_options_from_terms(
        snapshot_terms["runtime_options"]
    )
    actual = _identity_envelope(_prepared_identity(prepared, runtime_options))
    if envelope["prepared_identity"] != actual:
        _support_string_envelope_violation("prepared_identity_mismatch")
    if snapshot_terms["prepared_identity_digest"] != actual["digest"]:
        _support_string_envelope_violation(
            "snapshot_prepared_identity_mismatch"
        )


def _support_string_envelope_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer support string envelope violation: {kind}",
    )


__all__ = (
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "WriterSupportStringEnvelopeVerification",
    "verify_writer_support_string_envelope",
    "writer_support_string_envelope_for_prefix_read",
    "writer_support_string_envelope_for_string",
)
