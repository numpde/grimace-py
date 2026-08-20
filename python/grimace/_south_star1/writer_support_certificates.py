"""Certificates for materialized writer support strings and images."""

from __future__ import annotations

from dataclasses import dataclass

from .errors import SouthStarError
from .errors import SouthStarErrorKind


@dataclass(frozen=True, slots=True)
class WriterSupportStringCertificate:
    string: str
    emitted_texts: tuple[str, ...]
    replay_certificate: object
    final_snapshot: object
    terminal_frontier_projection_certificate: object
    terminal_projection_certificate: object
    terminal_certificates: tuple[object, ...]
    text_projection_certificates: tuple[object, ...] = ()


@dataclass(frozen=True, slots=True)
class WriterFrontierSupportStringCertificate:
    source_cursor: object
    string: str
    emitted_texts: tuple[str, ...]
    text_projection_certificates: tuple[object, ...]
    terminal_projection_certificate: object
    terminal_certificates: tuple[object, ...]
    final_cursor: object


@dataclass(frozen=True, slots=True)
class WriterSupportImageTextBucketCoverage:
    text_projection_key: tuple[object, ...]
    support_count_term: object
    string_certificates: tuple[WriterSupportStringCertificate, ...]
    support_count: int


@dataclass(frozen=True, slots=True)
class WriterSupportImageTerminalBucketCoverage:
    terminal_projection_key: tuple[object, ...]
    terminal_support_term: object | None
    string_certificate: WriterSupportStringCertificate | None
    support_count: int


@dataclass(frozen=True, slots=True)
class WriterSupportImageEnumerationCoverageCertificate:
    source_snapshot: object
    checked_frontier_certificate: object
    support_count_certificate: object
    support_count_term_coverage_certificate: object
    string_certificates: tuple[WriterSupportStringCertificate, ...]
    text_buckets: tuple[WriterSupportImageTextBucketCoverage, ...]
    terminal_bucket: WriterSupportImageTerminalBucketCoverage | None
    distinct_count: int
    support_count: int


@dataclass(frozen=True, slots=True)
class WriterSupportImageCertificate:
    source_snapshot: object
    strings: tuple[str, ...]
    string_certificates: tuple[WriterSupportStringCertificate, ...]
    distinct_count: int
    witness_count: int
    support_count_certificate: object | None = None
    witness_count_certificate: object | None = None
    checked_frontier_certificate: object | None = None
    enumeration_coverage_certificate: object | None = None


def writer_support_string_certificate(
    *,
    source_snapshot,
    string: str,
    emitted_texts: tuple[str, ...],
    replay_certificate,
    terminal_frontier_projection_certificate,
    terminal_projection_certificate,
    text_projection_certificates: tuple[object, ...] = (),
) -> WriterSupportStringCertificate:
    if string != "".join(emitted_texts):
        _string_violation("string_emitted_texts_mismatch")
    if replay_certificate.source_snapshot != source_snapshot:
        _string_violation("replay_source_snapshot_mismatch")
    if replay_certificate.emitted_texts != emitted_texts:
        _string_violation("replay_emitted_texts_mismatch")
    if terminal_projection_certificate is None:
        _string_violation("missing_terminal_projection_certificate")
    if terminal_frontier_projection_certificate is None:
        _string_violation("missing_terminal_frontier_projection_certificate")
    if terminal_frontier_projection_certificate.cursor != (
        replay_certificate.final_snapshot.cursor
    ):
        _string_violation("terminal_frontier_projection_cursor_mismatch")
    if terminal_projection_certificate is not (
        terminal_frontier_projection_certificate.terminal_projection_certificate
    ):
        _string_violation("terminal_projection_not_in_frontier_projection")
    if terminal_projection_certificate.terminal is None:
        _string_violation("terminal_projection_lacks_terminal")
    if not terminal_projection_certificate.terminal_certificates:
        _string_violation("terminal_projection_lacks_certificates")
    if terminal_projection_certificate.terminal.finalized_cursor != (
        terminal_projection_certificate.finalized_cursor
    ):
        _string_violation("terminal_finalized_cursor_mismatch")
    if terminal_projection_certificate.source_cursor != (
        replay_certificate.final_snapshot.cursor
    ):
        _string_violation("terminal_projection_final_cursor_mismatch")
    if len(text_projection_certificates) != len(emitted_texts):
        _string_violation("support_string_projection_count_mismatch")
    if tuple(
        certificate.emitted_text
        for certificate in text_projection_certificates
    ) != emitted_texts:
        _string_violation("support_string_projection_text_mismatch")
    if replay_certificate.emitted_texts != emitted_texts:
        _string_violation("support_string_replay_text_mismatch")
    if tuple(
        id(step.text_projection_certificate)
        for step in replay_certificate.step_certificates
    ) != tuple(id(certificate) for certificate in text_projection_certificates):
        _string_violation(
            "support_string_replay_projection_chain_mismatch"
        )

    return WriterSupportStringCertificate(
        string=string,
        emitted_texts=emitted_texts,
        replay_certificate=replay_certificate,
        final_snapshot=replay_certificate.final_snapshot,
        terminal_frontier_projection_certificate=(
            terminal_frontier_projection_certificate
        ),
        terminal_projection_certificate=terminal_projection_certificate,
        terminal_certificates=terminal_projection_certificate.terminal_certificates,
        text_projection_certificates=text_projection_certificates,
    )


def validate_writer_support_string_certificate(certificate) -> None:
    if certificate.string != "".join(certificate.emitted_texts):
        _string_violation("string_emitted_texts_mismatch")
    replay_certificate = certificate.replay_certificate
    if replay_certificate.final_snapshot != certificate.final_snapshot:
        _string_violation("replay_final_snapshot_mismatch")
    if replay_certificate.emitted_texts != certificate.emitted_texts:
        _string_violation("replay_emitted_texts_mismatch")
    if tuple(
        step.text_projection_certificate
        for step in replay_certificate.step_certificates
    ) != tuple(certificate.text_projection_certificates):
        _string_violation(
            "support_string_replay_projection_chain_mismatch"
        )

    terminal_frontier = certificate.terminal_frontier_projection_certificate
    terminal_projection = certificate.terminal_projection_certificate
    if terminal_frontier.cursor != replay_certificate.final_snapshot.cursor:
        _string_violation("terminal_frontier_projection_cursor_mismatch")
    if terminal_projection is not (
        terminal_frontier.terminal_projection_certificate
    ):
        _string_violation("terminal_projection_not_in_frontier_projection")
    if terminal_projection.source_cursor != (
        replay_certificate.final_snapshot.cursor
    ):
        _string_violation("terminal_projection_final_cursor_mismatch")


def writer_frontier_support_string_certificate(
    *,
    source_cursor,
    string: str,
    emitted_texts: tuple[str, ...],
    text_projection_certificates: tuple[object, ...],
    terminal_projection_certificate,
) -> WriterFrontierSupportStringCertificate:
    if string != "".join(emitted_texts):
        _string_violation("frontier_string_text_mismatch")
    if len(emitted_texts) != len(text_projection_certificates):
        _string_violation("frontier_projection_count_mismatch")

    current = source_cursor
    for emitted_text, projection in zip(
        emitted_texts,
        text_projection_certificates,
    ):
        if projection.emitted_text != emitted_text:
            _string_violation("frontier_projection_text_mismatch")
        if projection.source_cursor != current:
            _string_violation("frontier_projection_source_cursor_mismatch")
        if projection.choice.successor != projection.successor_cursor:
            _string_violation("frontier_projection_successor_mismatch")
        current = projection.successor_cursor

    if terminal_projection_certificate is None:
        _string_violation("frontier_support_lacks_terminal_projection")
    if terminal_projection_certificate.terminal is None:
        _string_violation("frontier_terminal_projection_lacks_terminal")
    if not terminal_projection_certificate.terminal_certificates:
        _string_violation("frontier_terminal_projection_lacks_certificates")
    if terminal_projection_certificate.source_cursor != current:
        _string_violation(
            "frontier_terminal_projection_source_cursor_mismatch"
        )

    return WriterFrontierSupportStringCertificate(
        source_cursor=source_cursor,
        string=string,
        emitted_texts=emitted_texts,
        text_projection_certificates=text_projection_certificates,
        terminal_projection_certificate=terminal_projection_certificate,
        terminal_certificates=terminal_projection_certificate.terminal_certificates,
        final_cursor=current,
    )


def writer_support_image_enumeration_coverage_certificate(
    *,
    source_snapshot,
    checked_frontier_certificate,
    support_count_certificate,
    string_certificates: tuple[WriterSupportStringCertificate, ...],
) -> WriterSupportImageEnumerationCoverageCertificate:
    if checked_frontier_certificate is None:
        _image_violation("missing_checked_frontier_certificate")
    if checked_frontier_certificate.cursor != source_snapshot.cursor:
        _image_violation("checked_frontier_source_cursor_mismatch")
    if support_count_certificate is None:
        _image_violation("missing_support_count_certificate")
    if support_count_certificate is not (
        checked_frontier_certificate.support_count_certificate
    ):
        _image_violation("support_count_certificate_identity_mismatch")

    support_coverage = (
        checked_frontier_certificate
        .support_count_term_coverage_certificate
    )
    if support_coverage is None:
        _image_violation("missing_support_count_term_coverage_certificate")
    if support_coverage.support_count_certificate is not support_count_certificate:
        _image_violation("support_count_coverage_certificate_mismatch")

    text_bucket: dict[
        tuple[object, ...],
        list[WriterSupportStringCertificate],
    ] = {}
    terminal_bucket: list[WriterSupportStringCertificate] = []
    for certificate in string_certificates:
        validate_writer_support_string_certificate(certificate)
        if certificate.replay_certificate.source_snapshot != source_snapshot:
            _image_violation("string_certificate_source_snapshot_mismatch")
        if certificate.emitted_texts:
            first_step = certificate.replay_certificate.step_certificates[0]
            if first_step.source_snapshot != source_snapshot:
                _image_violation("string_first_step_source_snapshot_mismatch")
            if first_step.frontier_projection_certificate.cursor != (
                source_snapshot.cursor
            ):
                _image_violation("string_first_step_frontier_cursor_mismatch")
            key = _text_projection_support_key(
                first_step.text_projection_certificate
            )
            text_bucket.setdefault(key, []).append(certificate)
        else:
            terminal_bucket.append(certificate)

    coverage_terms_by_key = {
        _text_projection_support_key(term.text_projection_certificate): term
        for term in support_coverage.text_terms
    }
    if set(text_bucket) - set(coverage_terms_by_key):
        _image_violation("support_image_text_bucket_without_coverage")

    text_buckets = []
    for key, term in coverage_terms_by_key.items():
        certificates = tuple(text_bucket.get(key, ()))
        if len(certificates) != term.support_count:
            _image_violation("support_image_text_bucket_count_mismatch")
        text_buckets.append(
            WriterSupportImageTextBucketCoverage(
                text_projection_key=key,
                support_count_term=term,
                string_certificates=certificates,
                support_count=len(certificates),
            )
        )

    terminal_term = support_coverage.terminal_term
    if terminal_term is None:
        if terminal_bucket:
            _image_violation("support_image_terminal_bucket_without_coverage")
        terminal_coverage = None
        terminal_support_count = 0
    else:
        if len(terminal_bucket) != terminal_term.terminal_count:
            _image_violation("support_image_terminal_bucket_count_mismatch")
        if terminal_term.terminal_count not in (0, 1):
            _image_violation("support_image_terminal_count_invalid")
        terminal_coverage = WriterSupportImageTerminalBucketCoverage(
            terminal_projection_key=_terminal_projection_support_key(
                terminal_term.terminal_projection_certificate
            ),
            terminal_support_term=terminal_term,
            string_certificate=terminal_bucket[0] if terminal_bucket else None,
            support_count=len(terminal_bucket),
        )
        terminal_support_count = len(terminal_bucket)

    distinct_count = len({certificate.string for certificate in string_certificates})
    support_count = terminal_support_count + sum(
        bucket.support_count for bucket in text_buckets
    )
    if distinct_count != support_count:
        _image_violation("support_image_distinct_count_coverage_mismatch")
    if support_count != support_count_certificate.support_count:
        _image_violation("support_image_support_count_coverage_mismatch")

    return WriterSupportImageEnumerationCoverageCertificate(
        source_snapshot=source_snapshot,
        checked_frontier_certificate=checked_frontier_certificate,
        support_count_certificate=support_count_certificate,
        support_count_term_coverage_certificate=support_coverage,
        string_certificates=string_certificates,
        text_buckets=tuple(text_buckets),
        terminal_bucket=terminal_coverage,
        distinct_count=distinct_count,
        support_count=support_count,
    )


def writer_support_image_certificate(
    *,
    source_snapshot,
    string_certificates: tuple[WriterSupportStringCertificate, ...],
    witness_count: int,
    support_count_certificate: object | None = None,
    witness_count_certificate: object | None = None,
    checked_frontier_certificate: object | None = None,
    enumeration_coverage_certificate: object | None = None,
) -> WriterSupportImageCertificate:
    strings = tuple(certificate.string for certificate in string_certificates)
    if len(set(strings)) != len(strings):
        _image_violation("duplicate_support_string_certificate")
    for certificate in string_certificates:
        validate_writer_support_string_certificate(certificate)
        if certificate.replay_certificate.source_snapshot != source_snapshot:
            _image_violation("string_certificate_source_snapshot_mismatch")

    if support_count_certificate is not None:
        support_source = getattr(
            support_count_certificate,
            "source_snapshot",
            None,
        )
        support_source_cursor = getattr(
            support_source,
            "cursor",
            support_source,
        )
        if support_source_cursor != getattr(source_snapshot, "cursor", None):
            _image_violation("support_count_source_snapshot_mismatch")
        if support_count_certificate.support_count != len(strings):
            _image_violation("support_count_certificate_mismatch")

    if witness_count_certificate is not None:
        source_cursor = getattr(source_snapshot, "cursor", None)
        if (
            source_cursor is not None
            and witness_count_certificate.cursor != source_cursor
        ):
            _image_violation("witness_count_cursor_mismatch")
        if witness_count_certificate.completion_count != witness_count:
            _image_violation("witness_count_certificate_mismatch")

    if checked_frontier_certificate is not None:
        if support_count_certificate is None:
            _image_violation("missing_support_count_certificate")
        if enumeration_coverage_certificate is None:
            enumeration_coverage_certificate = (
                writer_support_image_enumeration_coverage_certificate(
                    source_snapshot=source_snapshot,
                    checked_frontier_certificate=checked_frontier_certificate,
                    support_count_certificate=support_count_certificate,
                    string_certificates=string_certificates,
                )
            )
        _validate_support_image_enumeration_coverage_certificate(
            coverage=enumeration_coverage_certificate,
            source_snapshot=source_snapshot,
            checked_frontier_certificate=checked_frontier_certificate,
            support_count_certificate=support_count_certificate,
            string_certificates=string_certificates,
        )
        if enumeration_coverage_certificate.support_count != len(strings):
            _image_violation("support_image_coverage_count_mismatch")
    elif enumeration_coverage_certificate is not None:
        _image_violation("support_image_coverage_without_checked_frontier")

    return WriterSupportImageCertificate(
        source_snapshot=source_snapshot,
        strings=strings,
        string_certificates=string_certificates,
        distinct_count=len(strings),
        witness_count=witness_count,
        support_count_certificate=support_count_certificate,
        witness_count_certificate=witness_count_certificate,
        checked_frontier_certificate=checked_frontier_certificate,
        enumeration_coverage_certificate=enumeration_coverage_certificate,
    )


def _validate_support_image_enumeration_coverage_certificate(
    *,
    coverage,
    source_snapshot,
    checked_frontier_certificate,
    support_count_certificate,
    string_certificates: tuple[WriterSupportStringCertificate, ...],
) -> None:
    if coverage.source_snapshot != source_snapshot:
        _image_violation("support_image_coverage_source_snapshot_mismatch")
    if coverage.checked_frontier_certificate is not checked_frontier_certificate:
        _image_violation("support_image_coverage_frontier_mismatch")
    if coverage.support_count_certificate is not support_count_certificate:
        _image_violation("support_image_coverage_support_count_mismatch")
    support_coverage = (
        checked_frontier_certificate.support_count_term_coverage_certificate
    )
    if coverage.support_count_term_coverage_certificate is not support_coverage:
        _image_violation("support_image_coverage_term_mismatch")
    if coverage.string_certificates != string_certificates:
        _image_violation("support_image_coverage_string_certificates_mismatch")
    if coverage.distinct_count != len(
        {certificate.string for certificate in string_certificates}
    ):
        _image_violation("support_image_coverage_distinct_count_mismatch")
    if coverage.support_count != support_count_certificate.support_count:
        _image_violation("support_image_coverage_support_count_mismatch")
    text_terms = {
        _text_projection_support_key(term.text_projection_certificate): term
        for term in support_coverage.text_terms
    }
    covered_string_ids = set()
    text_support_count = 0
    for bucket in coverage.text_buckets:
        term = text_terms.get(bucket.text_projection_key)
        if term is None or bucket.support_count_term is not term:
            _image_violation("support_image_coverage_text_bucket_mismatch")
        if bucket.support_count != len(bucket.string_certificates):
            _image_violation("support_image_coverage_text_bucket_count_mismatch")
        text_support_count += bucket.support_count
        covered_string_ids.update(id(item) for item in bucket.string_certificates)
    terminal_support_count = 0
    terminal_term = support_coverage.terminal_term
    terminal_bucket = coverage.terminal_bucket
    if terminal_term is None:
        if terminal_bucket is not None:
            _image_violation("support_image_coverage_terminal_bucket_mismatch")
    else:
        if terminal_bucket is None:
            _image_violation("support_image_coverage_terminal_bucket_mismatch")
        if terminal_bucket.terminal_support_term is not terminal_term:
            _image_violation("support_image_coverage_terminal_bucket_mismatch")
        terminal_support_count = terminal_bucket.support_count
        if terminal_bucket.string_certificate is not None:
            covered_string_ids.add(id(terminal_bucket.string_certificate))
    if text_support_count + terminal_support_count != coverage.support_count:
        _image_violation("support_image_coverage_count_mismatch")
    if covered_string_ids != {id(item) for item in string_certificates}:
        _image_violation("support_image_coverage_string_partition_mismatch")


def _text_projection_support_key(projection) -> tuple[object, ...]:
    return (
        projection.source_cursor,
        projection.emitted_text,
        projection.successor_cursor,
        projection.immediate_multiplicity,
    )


def _terminal_projection_support_key(projection) -> tuple[object, ...]:
    if projection is None:
        return ()
    return (
        projection.source_cursor,
        projection.finalized_cursor,
        projection.multiplicity,
        projection.support_count,
    )


def _string_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer support string certificate violation: {kind}",
    )


def _image_violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer support image certificate violation: {kind}",
    )


__all__ = (
    "WriterFrontierSupportStringCertificate",
    "WriterSupportImageEnumerationCoverageCertificate",
    "WriterSupportImageCertificate",
    "WriterSupportImageTerminalBucketCoverage",
    "WriterSupportImageTextBucketCoverage",
    "WriterSupportStringCertificate",
    "validate_writer_support_string_certificate",
    "writer_frontier_support_string_certificate",
    "writer_support_image_enumeration_coverage_certificate",
    "writer_support_image_certificate",
    "writer_support_string_certificate",
)
