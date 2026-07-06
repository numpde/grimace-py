"""Durable envelopes for complete writer support images."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .prepared_runtime import SouthStarPreparedMol
from .writer_envelope_terms import _digest
from .writer_envelope_terms import _identity_envelope
from .writer_envelope_terms import _runtime_options_from_terms
from .writer_envelope_terms import _snapshot_identity_envelope
from .writer_envelope_terms import _term
from .writer_frontier import _checked_writer_frontier_product
from .writer_frontier_count_envelope import (
    _cursor_completion_count_certificate_envelope,
)
from .writer_frontier_count_envelope import (
    _text_support_count_certificate_envelope,
)
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
from .writer_snapshot_envelope import _source_snapshot_from_envelope
from .writer_snapshot_prefix_envelope import (
    _checked_frontier_certificate_identity_envelope,
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
    writer_snapshot_replay_envelope_for_emitted_texts,
)
from .writer_support_certificates import writer_support_image_certificate
from .writer_support_string_envelope import (
    _writer_support_string_envelope_from_certificate,
)
from .writer_support_string_envelope import (
    verify_writer_support_string_envelope,
)


SCHEMA_NAME = "writer_support_image"
SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset((
    "schema_name",
    "schema_version",
    "prepared_identity",
    "source_kind",
    "source_snapshot",
    "prefix_read_envelope",
    "count_envelope",
    "support_strings",
    "support_string_envelopes",
    "distinct_count",
    "witness_count",
    "support_image_certificate",
    "enumeration_coverage",
    "frontier_product",
    "checked_frontier_certificate",
    "support_count_certificate",
    "witness_count_certificate",
))
_SOURCE_KINDS = frozenset(("snapshot", "prefix_read"))


@dataclass(frozen=True, slots=True)
class WriterSupportImageEnvelopeVerification:
    accepted: bool
    source_kind: str
    source_snapshot: object | None
    distinct_count: int | None
    witness_count: int | None
    reason: str | None = None


def writer_support_image_envelope_for_snapshot(
    *,
    prepared: SouthStarPreparedMol,
    snapshot,
) -> dict[str, object]:
    count_envelope = writer_frontier_count_envelope_for_snapshot(
        prepared=prepared,
        snapshot=snapshot,
    )
    product = _checked_product(prepared=prepared, snapshot=snapshot)
    image = _support_image_certificate_for_source(
        prepared=prepared,
        snapshot=snapshot,
        product=product,
    )
    envelope = _envelope_from_image(
        prepared=prepared,
        source_kind="snapshot",
        source_snapshot=snapshot,
        prefix_read_envelope=None,
        count_envelope=count_envelope,
        product=product,
        image=image,
    )
    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope)
    return envelope


def writer_support_image_envelope_for_prefix_read(
    *,
    prepared: SouthStarPreparedMol,
    prefix_read_envelope: Mapping[str, object],
) -> dict[str, object]:
    prefix = verify_writer_snapshot_prefix_read_envelope(
        prepared=prepared,
        envelope=prefix_read_envelope,
    )
    if not prefix.accepted:
        _image_envelope_violation("prefix_read_envelope_rejected")
    if prefix.read_kind != "readable":
        _image_envelope_violation("prefix_read_envelope_not_readable")
    if prefix.final_snapshot is None:
        _image_envelope_violation("prefix_read_envelope_lacks_final_snapshot")
    count_envelope = writer_frontier_count_envelope_for_prefix_read(
        prepared=prepared,
        prefix_read_envelope=prefix_read_envelope,
    )
    product = _checked_product(prepared=prepared, snapshot=prefix.final_snapshot)
    image = _support_image_certificate_for_source(
        prepared=prepared,
        snapshot=prefix.final_snapshot,
        product=product,
    )
    envelope = _envelope_from_image(
        prepared=prepared,
        source_kind="prefix_read",
        source_snapshot=None,
        prefix_read_envelope=prefix_read_envelope,
        count_envelope=count_envelope,
        product=product,
        image=image,
    )
    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope)
    return envelope


def verify_writer_support_image_envelope(
    *,
    prepared: SouthStarPreparedMol,
    envelope: object,
) -> WriterSupportImageEnvelopeVerification:
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
            _image_envelope_violation("count_envelope_rejected")
        if count.frontier_snapshot != source_snapshot:
            _image_envelope_violation("count_envelope_source_mismatch")
        strings = tuple(envelope["support_strings"])
        string_envelopes = tuple(envelope["support_string_envelopes"])
        if len(strings) != len(string_envelopes):
            _image_envelope_violation("support_string_count_mismatch")
        if len(set(strings)) != len(strings):
            _image_envelope_violation("duplicate_support_string")
        for expected_string, string_envelope in zip(strings, string_envelopes):
            verification = verify_writer_support_string_envelope(
                prepared=prepared,
                envelope=string_envelope,
            )
            if not verification.accepted:
                _image_envelope_violation("support_string_envelope_rejected")
            if verification.source_snapshot != source_snapshot:
                _image_envelope_violation("support_string_source_mismatch")
            if verification.string != expected_string:
                _image_envelope_violation("support_string_order_mismatch")
        if envelope["distinct_count"] != len(strings):
            _image_envelope_violation("distinct_count_mismatch")
        if envelope["distinct_count"] != envelope["count_envelope"]["support_count"]:
            _image_envelope_violation("support_count_mismatch")
        if envelope["witness_count"] != envelope["count_envelope"]["completion_count"]:
            _image_envelope_violation("witness_count_mismatch")

        product = _checked_product(prepared=prepared, snapshot=source_snapshot)
        expected = _envelope_from_verified_strings(
            prepared=prepared,
            source_kind=source_kind,
            source_snapshot=(
                source_snapshot if source_kind == "snapshot" else None
            ),
            prefix_read_envelope=envelope["prefix_read_envelope"],
            count_envelope=envelope["count_envelope"],
            product=product,
            support_string_envelopes=string_envelopes,
        )
        if expected != envelope:
            return WriterSupportImageEnvelopeVerification(
                accepted=False,
                source_kind=source_kind,
                source_snapshot=source_snapshot,
                distinct_count=None,
                witness_count=None,
                reason="envelope_terms_mismatch",
            )
        return WriterSupportImageEnvelopeVerification(
            accepted=True,
            source_kind=source_kind,
            source_snapshot=source_snapshot,
            distinct_count=envelope["distinct_count"],
            witness_count=envelope["witness_count"],
        )
    except SouthStarError as exc:
        return WriterSupportImageEnvelopeVerification(
            accepted=False,
            source_kind=(
                envelope.get("source_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            source_snapshot=None,
            distinct_count=None,
            witness_count=None,
            reason=exc.args[-1] if exc.args else "verification_error",
        )
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterSupportImageEnvelopeVerification(
            accepted=False,
            source_kind=(
                envelope.get("source_kind", "unknown")
                if isinstance(envelope, Mapping)
                else "unknown"
            ),
            source_snapshot=None,
            distinct_count=None,
            witness_count=None,
            reason=f"malformed_envelope:{type(exc).__name__}",
        )


def _support_image_certificate_for_source(*, prepared, snapshot, product):
    certified = tuple(
        _iter_writer_snapshot_certified_support_strings(
            snapshot,
            prepared=prepared,
        )
    )
    return writer_support_image_certificate(
        source_snapshot=snapshot,
        string_certificates=tuple(item.certificate for item in certified),
        witness_count=product.count_certificate.completion_count,
        witness_count_certificate=product.count_certificate,
        support_count_certificate=product.support_count_certificate,
        checked_frontier_certificate=product.checked_frontier_certificate,
    )


def _envelope_from_image(
    *,
    prepared,
    source_kind: str,
    source_snapshot,
    prefix_read_envelope,
    count_envelope,
    product,
    image,
) -> dict[str, object]:
    string_envelopes = _support_string_envelopes_from_image(
        prepared=prepared,
        source_kind=source_kind,
        source_snapshot=source_snapshot,
        prefix_read_envelope=prefix_read_envelope,
        count_envelope=count_envelope,
        image=image,
    )
    return _envelope_from_verified_strings(
        prepared=prepared,
        source_kind=source_kind,
        source_snapshot=source_snapshot,
        prefix_read_envelope=prefix_read_envelope,
        count_envelope=count_envelope,
        product=product,
        support_string_envelopes=tuple(string_envelopes),
    )


def _envelope_from_verified_strings(
    *,
    prepared,
    source_kind: str,
    source_snapshot,
    prefix_read_envelope,
    count_envelope,
    product,
    support_string_envelopes,
) -> dict[str, object]:
    del prepared
    strings = [item["string"] for item in support_string_envelopes]
    source_snapshot_identity = (
        _snapshot_identity_envelope(source_snapshot)
        if source_snapshot is not None
        else count_envelope["frontier_snapshot"]
    )
    coverage = _enumeration_coverage_envelope_from_product(
        product,
        string_envelopes=support_string_envelopes,
        source_snapshot_identity=source_snapshot_identity,
    )
    return {
        "schema_name": SCHEMA_NAME,
        "schema_version": SCHEMA_VERSION,
        "prepared_identity": count_envelope["prepared_identity"],
        "source_kind": source_kind,
        "source_snapshot": (
            None
            if source_snapshot is None
            else source_snapshot_identity
        ),
        "prefix_read_envelope": prefix_read_envelope,
        "count_envelope": count_envelope,
        "support_strings": strings,
        "support_string_envelopes": list(support_string_envelopes),
        "distinct_count": len(strings),
        "witness_count": product.count_certificate.completion_count,
        "support_image_certificate": _support_image_certificate_envelope(
            source_snapshot=source_snapshot_identity,
            strings=strings,
            support_string_envelopes=support_string_envelopes,
            product=product,
            coverage=coverage,
        ),
        "enumeration_coverage": coverage,
        "frontier_product": _writer_frontier_product_identity_envelope(product),
        "checked_frontier_certificate": (
            _checked_frontier_certificate_identity_envelope(
                product.checked_frontier_certificate
            )
        ),
        "support_count_certificate": _text_support_count_certificate_envelope(
            product.support_count_certificate
        ),
        "witness_count_certificate": (
            _cursor_completion_count_certificate_envelope(product.count_certificate)
        ),
    }


def _support_string_envelopes_from_image(
    *,
    prepared,
    source_kind: str,
    source_snapshot,
    prefix_read_envelope,
    count_envelope,
    image,
) -> list[dict[str, object]]:
    return [
        _writer_support_string_envelope_from_certificate(
            prepared=prepared,
            source_kind=source_kind,
            source_snapshot=source_snapshot,
            prefix_read_envelope=prefix_read_envelope,
            count_envelope=count_envelope,
            replay_envelope=writer_snapshot_replay_envelope_for_emitted_texts(
                prepared=prepared,
                snapshot=image.source_snapshot,
                emitted_texts=certificate.emitted_texts,
            ),
            certificate=certificate,
        )
        for certificate in image.string_certificates
    ]


def _support_image_certificate_envelope(
    *,
    source_snapshot,
    strings,
    support_string_envelopes,
    product,
    coverage,
):
    envelope = {
        "source_snapshot": source_snapshot,
        "strings": list(strings),
        "string_certificate_digests": [
            envelope["support_string_certificate"]["digest"]
            for envelope in support_string_envelopes
        ],
        "distinct_count": len(strings),
        "witness_count": product.count_certificate.completion_count,
        "support_count_certificate_digest": (
            _text_support_count_certificate_envelope(
                product.support_count_certificate
            )["digest"]
        ),
        "witness_count_certificate_digest": (
            _cursor_completion_count_certificate_envelope(
                product.count_certificate
            )["digest"]
        ),
        "checked_frontier_certificate_digest": (
            _checked_frontier_certificate_identity_envelope(
                product.checked_frontier_certificate
            )["digest"]
        ),
        "enumeration_coverage_digest": coverage["digest"],
    }
    envelope["digest"] = _digest(_term(envelope))
    return envelope


def _enumeration_coverage_envelope_from_product(
    product,
    *,
    string_envelopes,
    source_snapshot_identity,
):
    checked = product.checked_frontier_certificate
    coverage = checked.support_count_term_coverage_certificate
    envelope = {
        "source_snapshot": source_snapshot_identity,
        "checked_frontier_certificate": (
            _checked_frontier_certificate_identity_envelope(
                checked
            )
        ),
        "support_count_certificate": _text_support_count_certificate_envelope(
            product.support_count_certificate
        ),
        "support_count_term_coverage_digest": _digest(_term(coverage)),
        "text_buckets": [
            _text_bucket_envelope_from_term(term, string_envelopes)
            for term in coverage.text_terms
        ],
        "terminal_bucket": (
            None
            if coverage.terminal_term is None
            else _terminal_bucket_envelope_from_term(
                coverage.terminal_term,
                string_envelopes,
            )
        ),
        "distinct_count": len(string_envelopes),
        "support_count": coverage.support_count,
    }
    _validate_bucket_partition(envelope, len(string_envelopes))
    envelope["digest"] = _digest(_term(envelope))
    return envelope


def _text_bucket_envelope_from_term(term, string_envelopes):
    projection = term.text_projection_certificate
    projection_identity = _text_projection_certificate_identity_envelope(
        projection
    )
    projection_key = _text_projection_bucket_key(projection_identity)
    string_indices = [
        index
        for index, envelope in enumerate(string_envelopes)
        if envelope["emitted_texts"]
        and _text_projection_bucket_key(
            envelope["text_projection_chain"][0]["text_projection"]
        )
        == projection_key
    ]
    envelope = {
        "text_projection": projection_identity,
        "support_count_term_digest": _digest(_term(term)),
        "support_count": term.support_count,
        "string_indices": string_indices,
        "string_digests": [
            string_envelopes[index]["support_string_certificate"]["digest"]
            for index in string_indices
        ],
    }
    envelope["digest"] = _digest(_term(envelope))
    return envelope


def _text_projection_bucket_key(identity):
    return (
        identity["source_cursor"]["digest"],
        identity["emitted_text"],
        identity["successor_cursor"]["digest"],
        identity["immediate_multiplicity"],
        tuple(identity["branch_certificate_digests"]),
    )


def _terminal_bucket_envelope_from_term(term, string_envelopes):
    empty_indices = [
        index
        for index, envelope in enumerate(string_envelopes)
        if not envelope["emitted_texts"]
    ]
    if len(empty_indices) > 1:
        _image_envelope_violation("terminal_bucket_count_mismatch")
    string_index = empty_indices[0] if empty_indices else None
    string_digest = (
        None
        if string_index is None
        else string_envelopes[string_index]["support_string_certificate"][
            "digest"
        ]
    )
    terminal_projection = term.terminal_projection_certificate
    envelope = {
        "terminal_projection": (
            _terminal_projection_certificate_identity_envelope(
                terminal_projection
            )
        ),
        "terminal_support_term_digest": _digest(_term(term)),
        "terminal_support_identities": (
            []
            if terminal_projection is None
            else [
                _terminal_support_identity_envelope_from_certificate(
                    certificate
                )
                for certificate in terminal_projection.terminal_certificates
            ]
        ),
        "support_count": term.terminal_count,
        "string_index": string_index,
        "string_digest": string_digest,
    }
    envelope["digest"] = _digest(_term(envelope))
    return envelope


def _validate_bucket_partition(envelope, expected_count: int) -> None:
    indices = []
    for bucket in envelope["text_buckets"]:
        if bucket["support_count"] != len(bucket["string_indices"]):
            _image_envelope_violation("text_bucket_count_mismatch")
        indices.extend(bucket["string_indices"])
    terminal = envelope["terminal_bucket"]
    if terminal is not None and terminal["string_index"] is not None:
        if terminal["support_count"] != 1:
            _image_envelope_violation("terminal_bucket_count_mismatch")
        indices.append(terminal["string_index"])
    if sorted(indices) != list(range(expected_count)):
        _image_envelope_violation("bucket_partition_mismatch")


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
        _image_envelope_violation("prefix_read_envelope_rejected")
    if prefix.read_kind != "readable":
        _image_envelope_violation("prefix_read_envelope_not_readable")
    if prefix.final_snapshot is None:
        _image_envelope_violation("prefix_read_envelope_lacks_final_snapshot")
    return prefix.final_snapshot


def _checked_product(*, prepared, snapshot):
    return _checked_writer_frontier_product(
        prepared,
        snapshot.cursor,
        include_counts=True,
        include_frontier_certificate=True,
        include_count_certificate=True,
    )


def _validate_envelope_shape(envelope: object) -> None:
    if not isinstance(envelope, Mapping):
        _image_envelope_violation("envelope_not_mapping")
    if frozenset(envelope) != _TOP_LEVEL_FIELDS:
        _image_envelope_violation("top_level_fields_mismatch")
    if envelope["schema_name"] != SCHEMA_NAME:
        _image_envelope_violation("unknown_schema_name")
    if envelope["schema_version"] != SCHEMA_VERSION:
        _image_envelope_violation("unknown_schema_version")
    if envelope["source_kind"] not in _SOURCE_KINDS:
        _image_envelope_violation("unknown_source_kind")
    if envelope["source_kind"] == "snapshot":
        if envelope["source_snapshot"] is None:
            _image_envelope_violation("snapshot_source_missing")
        if envelope["prefix_read_envelope"] is not None:
            _image_envelope_violation("snapshot_source_has_prefix")
    else:
        if envelope["source_snapshot"] is not None:
            _image_envelope_violation("prefix_source_has_source_snapshot")
        if envelope["prefix_read_envelope"] is None:
            _image_envelope_violation("prefix_source_missing_prefix")


def _assert_prepared_identity_matches(prepared, envelope) -> None:
    snapshot_terms = (
        envelope["source_snapshot"]
        if envelope["source_kind"] == "snapshot"
        else envelope["count_envelope"]["frontier_snapshot"]
    )
    runtime_options = _runtime_options_from_terms(
        snapshot_terms["runtime_options"]
    )
    actual = _identity_envelope(_prepared_identity(prepared, runtime_options))
    if envelope["prepared_identity"] != actual:
        _image_envelope_violation("prepared_identity_mismatch")
    if snapshot_terms["prepared_identity_digest"] != actual["digest"]:
        _image_envelope_violation("snapshot_prepared_identity_mismatch")


def _image_envelope_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer support image envelope violation: {kind}",
    )


__all__ = (
    "SCHEMA_NAME",
    "SCHEMA_VERSION",
    "WriterSupportImageEnvelopeVerification",
    "verify_writer_support_image_envelope",
    "writer_support_image_envelope_for_prefix_read",
    "writer_support_image_envelope_for_snapshot",
)
