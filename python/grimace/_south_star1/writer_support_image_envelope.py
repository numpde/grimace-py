"""Durable envelopes for complete writer support images."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .prepared_runtime import SouthStarPreparedMol
from .writer_envelope_terms import _identity_digest
from .writer_envelope_terms import _identity_envelope
from .writer_envelope_terms import _runtime_options_from_terms
from .writer_envelope_terms import _snapshot_identity_envelope
from .writer_envelope_work import WriterEnvelopeWorkBudget
from .writer_envelope_work import WriterEnvelopeWorkExceeded
from .writer_envelope_work import check_writer_envelope_work
from .writer_envelope_work import default_writer_envelope_work_budget
from .writer_envelope_work import writer_envelope_work_reason
from .writer_frontier import _checked_writer_frontier_product
from .writer_frontier_count_envelope import verify_writer_frontier_count_envelope
from .writer_frontier_count_envelope import writer_frontier_count_envelope_for_prefix_read
from .writer_frontier_count_envelope import writer_frontier_count_envelope_for_snapshot
from .writer_snapshot import _iter_writer_snapshot_certified_support_strings
from .writer_snapshot import _prepared_identity
from .writer_snapshot_envelope import _source_snapshot_from_envelope
from .writer_snapshot_prefix_envelope import _terminal_projection_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import _terminal_support_identity_envelope_from_certificate
from .writer_snapshot_prefix_envelope import _text_projection_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import verify_writer_snapshot_prefix_read_envelope
from .writer_snapshot_replay_envelope import writer_snapshot_replay_envelope_for_emitted_texts
from .writer_support_certificates import writer_support_image_certificate
from .writer_support_string_envelope import _writer_support_string_envelope_from_certificate
from .writer_support_string_envelope import verify_writer_support_string_envelope
SCHEMA_NAME = 'writer_support_image'
SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset(('schema_name', 'schema_version', 'prepared_identity', 'source_kind', 'source_snapshot', 'prefix_read_envelope', 'count_envelope', 'support_strings', 'support_string_envelopes', 'distinct_count', 'witness_count', 'support_image_certificate', 'enumeration_coverage', 'frontier_product', 'checked_frontier_certificate', 'support_count_certificate', 'witness_count_certificate'))
_SOURCE_KINDS = frozenset(('snapshot', 'prefix_read'))

@dataclass(frozen=True, slots=True)
class WriterSupportImageEnvelopeVerification:
    accepted: bool
    source_kind: str
    source_snapshot: object | None
    distinct_count: int | None
    witness_count: int | None
    reason: str | None = None

def writer_support_image_envelope_for_snapshot(*, prepared: SouthStarPreparedMol, snapshot, budget: WriterEnvelopeWorkBudget | None=None) -> dict[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    count_envelope = writer_frontier_count_envelope_for_snapshot(prepared=prepared, snapshot=snapshot, budget=budget)
    product = _checked_product(prepared=prepared, snapshot=snapshot)
    image = _support_image_certificate_for_source(prepared=prepared, snapshot=snapshot, product=product)
    envelope = _envelope_from_image(prepared=prepared, source_kind='snapshot', source_snapshot=snapshot, prefix_read_envelope=None, count_envelope=count_envelope, product=product, image=image, budget=budget)
    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope, budget=budget)
    return envelope

def writer_support_image_envelope_for_prefix_read(*, prepared: SouthStarPreparedMol, prefix_read_envelope: Mapping[str, object], budget: WriterEnvelopeWorkBudget | None=None) -> dict[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    prefix = verify_writer_snapshot_prefix_read_envelope(prepared=prepared, envelope=prefix_read_envelope, budget=budget)
    if not prefix.accepted:
        _image_envelope_violation('prefix_read_envelope_rejected')
    if prefix.read_kind != 'readable':
        _image_envelope_violation('prefix_read_envelope_not_readable')
    if prefix.final_snapshot is None:
        _image_envelope_violation('prefix_read_envelope_lacks_final_snapshot')
    count_envelope = writer_frontier_count_envelope_for_prefix_read(prepared=prepared, prefix_read_envelope=prefix_read_envelope, budget=budget)
    product = _checked_product(prepared=prepared, snapshot=prefix.final_snapshot)
    image = _support_image_certificate_for_source(prepared=prepared, snapshot=prefix.final_snapshot, product=product)
    envelope = _envelope_from_image(prepared=prepared, source_kind='prefix_read', source_snapshot=None, prefix_read_envelope=prefix_read_envelope, count_envelope=count_envelope, product=product, image=image, budget=budget)
    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope, budget=budget)
    return envelope

def verify_writer_support_image_envelope(*, prepared: SouthStarPreparedMol, envelope: object, budget: WriterEnvelopeWorkBudget | None=None) -> WriterSupportImageEnvelopeVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        _validate_envelope_shape(envelope)
        assert isinstance(envelope, Mapping)
        _check_support_image_work(envelope, budget=budget)
        _assert_prepared_identity_matches(prepared, envelope, budget=budget)
        source_kind = str(envelope['source_kind'])
        source_snapshot = _source_snapshot_for_envelope(prepared=prepared, envelope=envelope, budget=budget)
        count = verify_writer_frontier_count_envelope(prepared=prepared, envelope=envelope['count_envelope'], budget=budget)
        if not count.accepted:
            _image_envelope_violation('count_envelope_rejected')
        if count.frontier_snapshot != source_snapshot:
            _image_envelope_violation('count_envelope_source_mismatch')
        strings = tuple(envelope['support_strings'])
        string_envelopes = tuple(envelope['support_string_envelopes'])
        if len(strings) != len(string_envelopes):
            _image_envelope_violation('support_string_count_mismatch')
        if len(set(strings)) != len(strings):
            _image_envelope_violation('duplicate_support_string')
        for expected_string, string_envelope in zip(strings, string_envelopes):
            verification = verify_writer_support_string_envelope(prepared=prepared, envelope=string_envelope, budget=budget)
            if not verification.accepted:
                _image_envelope_violation('support_string_envelope_rejected')
            if verification.source_snapshot != source_snapshot:
                _image_envelope_violation('support_string_source_mismatch')
            if verification.string != expected_string:
                _image_envelope_violation('support_string_order_mismatch')
        if envelope['distinct_count'] != len(strings):
            _image_envelope_violation('distinct_count_mismatch')
        if envelope['distinct_count'] != envelope['count_envelope']['support_count']:
            _image_envelope_violation('support_count_mismatch')
        if envelope['witness_count'] != envelope['count_envelope']['completion_count']:
            _image_envelope_violation('witness_count_mismatch')
        product = _checked_product(prepared=prepared, snapshot=source_snapshot)
        expected = _envelope_from_verified_strings(prepared=prepared, source_kind=source_kind, source_snapshot=source_snapshot if source_kind == 'snapshot' else None, prefix_read_envelope=envelope['prefix_read_envelope'], count_envelope=envelope['count_envelope'], product=product, support_string_envelopes=string_envelopes, budget=budget)
        if expected != envelope:
            return WriterSupportImageEnvelopeVerification(accepted=False, source_kind=source_kind, source_snapshot=source_snapshot, distinct_count=None, witness_count=None, reason='envelope_terms_mismatch')
        return WriterSupportImageEnvelopeVerification(accepted=True, source_kind=source_kind, source_snapshot=source_snapshot, distinct_count=envelope['distinct_count'], witness_count=envelope['witness_count'])
    except WriterEnvelopeWorkExceeded as exc:
        return WriterSupportImageEnvelopeVerification(accepted=False, source_kind=envelope.get('source_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', source_snapshot=None, distinct_count=None, witness_count=None, reason=writer_envelope_work_reason(exc))
    except SouthStarError as exc:
        return WriterSupportImageEnvelopeVerification(accepted=False, source_kind=envelope.get('source_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', source_snapshot=None, distinct_count=None, witness_count=None, reason=exc.args[-1] if exc.args else 'verification_error')
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterSupportImageEnvelopeVerification(accepted=False, source_kind=envelope.get('source_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', source_snapshot=None, distinct_count=None, witness_count=None, reason=f'malformed_envelope:{type(exc).__name__}')

def _support_image_certificate_for_source(*, prepared, snapshot, product):
    certified = tuple(_iter_writer_snapshot_certified_support_strings(snapshot, prepared=prepared))
    return writer_support_image_certificate(source_snapshot=snapshot, string_certificates=tuple((item.certificate for item in certified)), witness_count=product.count_certificate.completion_count, witness_count_certificate=product.count_certificate, support_count_certificate=product.support_count_certificate, checked_frontier_certificate=product.checked_frontier_certificate)

def _envelope_from_image(*, prepared, source_kind: str, source_snapshot, prefix_read_envelope, count_envelope, product, image, budget) -> dict[str, object]:
    _check_materialized_image_work(image, budget=budget)
    string_envelopes = _support_string_envelopes_from_image(prepared=prepared, source_kind=source_kind, source_snapshot=source_snapshot, prefix_read_envelope=prefix_read_envelope, count_envelope=count_envelope, image=image, budget=budget)
    return _envelope_from_verified_strings(prepared=prepared, source_kind=source_kind, source_snapshot=source_snapshot, prefix_read_envelope=prefix_read_envelope, count_envelope=count_envelope, product=product, support_string_envelopes=tuple(string_envelopes), budget=budget)

def _envelope_from_verified_strings(*, prepared, source_kind: str, source_snapshot, prefix_read_envelope, count_envelope, product, support_string_envelopes, budget) -> dict[str, object]:
    del prepared
    _check_support_string_envelope_work(support_string_envelopes, budget=budget, operation='support_image_envelope')
    strings = [item['string'] for item in support_string_envelopes]
    source_snapshot_identity = _snapshot_identity_envelope(source_snapshot, budget=budget, operation='support_image.source_snapshot.digest') if source_snapshot is not None else count_envelope['frontier_snapshot']
    coverage = _enumeration_coverage_envelope_from_product(product, string_envelopes=support_string_envelopes, source_snapshot_identity=source_snapshot_identity, count_envelope=count_envelope, budget=budget)
    return {'schema_name': SCHEMA_NAME, 'schema_version': SCHEMA_VERSION, 'prepared_identity': count_envelope['prepared_identity'], 'source_kind': source_kind, 'source_snapshot': None if source_snapshot is None else source_snapshot_identity, 'prefix_read_envelope': prefix_read_envelope, 'count_envelope': count_envelope, 'support_strings': strings, 'support_string_envelopes': list(support_string_envelopes), 'distinct_count': len(strings), 'witness_count': product.count_certificate.completion_count, 'support_image_certificate': _support_image_certificate_envelope(source_snapshot=source_snapshot_identity, strings=strings, support_string_envelopes=support_string_envelopes, count_envelope=count_envelope, coverage=coverage, budget=budget), 'enumeration_coverage': coverage, 'frontier_product': count_envelope['frontier_product'], 'checked_frontier_certificate': count_envelope['frontier_product']['checked_frontier_certificate'], 'support_count_certificate': count_envelope['support_count_certificate'], 'witness_count_certificate': count_envelope['completion_count_certificate']}

def _support_string_envelopes_from_image(*, prepared, source_kind: str, source_snapshot, prefix_read_envelope, count_envelope, image, budget) -> list[dict[str, object]]:
    return [_writer_support_string_envelope_from_certificate(prepared=prepared, source_kind=source_kind, source_snapshot=source_snapshot, prefix_read_envelope=prefix_read_envelope, count_envelope=count_envelope, replay_envelope=writer_snapshot_replay_envelope_for_emitted_texts(prepared=prepared, snapshot=image.source_snapshot, emitted_texts=certificate.emitted_texts, budget=budget), certificate=certificate, budget=budget) for certificate in image.string_certificates]

def _support_image_certificate_envelope(*, source_snapshot, strings, support_string_envelopes, count_envelope, coverage, budget):
    envelope = {'source_snapshot': source_snapshot, 'strings': list(strings), 'string_certificate_digests': [envelope['support_string_certificate']['digest'] for envelope in support_string_envelopes], 'support_string_envelope_digests': [envelope['digest'] for envelope in support_string_envelopes], 'distinct_count': len(strings), 'witness_count': count_envelope['completion_count'], 'support_count_certificate_digest': count_envelope['support_count_certificate']['digest'], 'witness_count_certificate_digest': count_envelope['completion_count_certificate']['digest'], 'checked_frontier_certificate_digest': count_envelope['frontier_product']['checked_frontier_certificate']['digest'], 'count_dag_digest': count_envelope['count_dag']['digest'], 'frontier_product_digest': count_envelope['frontier_product']['digest'], 'enumeration_coverage_digest': coverage['digest']}
    envelope['digest'] = _identity_digest(_support_image_certificate_manifest(envelope), budget=budget, operation='support_image.certificate.digest')
    return envelope

def _enumeration_coverage_envelope_from_product(product, *, string_envelopes, source_snapshot_identity, count_envelope, budget):
    checked = product.checked_frontier_certificate
    coverage = checked.support_count_term_coverage_certificate
    envelope = {'source_snapshot': source_snapshot_identity, 'checked_frontier_certificate': count_envelope['frontier_product']['checked_frontier_certificate'], 'support_count_certificate': count_envelope['support_count_certificate'], 'support_count_term_coverage_digest': _identity_digest(coverage, budget=budget, operation='support_image.support_count_term_coverage.digest'), 'text_buckets': [_text_bucket_envelope_from_term(term, string_envelopes, budget=budget) for term in coverage.text_terms], 'terminal_bucket': None if coverage.terminal_term is None else _terminal_bucket_envelope_from_term(coverage.terminal_term, string_envelopes, budget=budget), 'distinct_count': len(string_envelopes), 'support_count': coverage.support_count}
    _validate_bucket_partition(envelope, len(string_envelopes))
    _check_coverage_work(envelope, budget=budget)
    envelope['digest'] = _identity_digest(_enumeration_coverage_manifest(envelope), budget=budget, operation='support_image.enumeration_coverage.digest')
    return envelope

def _text_bucket_envelope_from_term(term, string_envelopes, *, budget):
    projection = term.text_projection_certificate
    projection_identity = _text_projection_certificate_identity_envelope(projection, budget=budget)
    projection_key = _text_projection_bucket_key(projection_identity)
    string_indices = [index for index, envelope in enumerate(string_envelopes) if envelope['emitted_texts'] and _text_projection_bucket_key(envelope['text_projection_chain'][0]['text_projection']) == projection_key]
    envelope = {'text_projection': projection_identity, 'support_count_term_digest': _identity_digest(term, budget=budget, operation='support_image.text_bucket.support_count_term.digest'), 'support_count': term.support_count, 'string_indices': string_indices, 'string_digests': [string_envelopes[index]['support_string_certificate']['digest'] for index in string_indices], 'support_string_envelope_digests': [string_envelopes[index]['digest'] for index in string_indices]}
    envelope['digest'] = _identity_digest(_text_bucket_manifest(envelope), budget=budget, operation='support_image.text_bucket.digest')
    return envelope

def _text_projection_bucket_key(identity):
    return (identity['source_cursor']['digest'], identity['emitted_text'], identity['successor_cursor']['digest'], identity['immediate_multiplicity'], tuple(identity['branch_certificate_digests']))

def _terminal_bucket_envelope_from_term(term, string_envelopes, *, budget):
    empty_indices = [index for index, envelope in enumerate(string_envelopes) if not envelope['emitted_texts']]
    if len(empty_indices) > 1:
        _image_envelope_violation('terminal_bucket_count_mismatch')
    string_index = empty_indices[0] if empty_indices else None
    string_digest = None if string_index is None else string_envelopes[string_index]['support_string_certificate']['digest']
    terminal_projection = term.terminal_projection_certificate
    envelope = {'terminal_projection': _terminal_projection_certificate_identity_envelope(terminal_projection, budget=budget), 'terminal_support_term_digest': _identity_digest(term, budget=budget, operation='support_image.terminal_bucket.support_term.digest'), 'terminal_support_identities': [] if terminal_projection is None else [_terminal_support_identity_envelope_from_certificate(certificate, budget=budget) for certificate in terminal_projection.terminal_certificates], 'support_count': term.terminal_count, 'string_index': string_index, 'string_digest': string_digest, 'support_string_envelope_digest': None if string_index is None else string_envelopes[string_index]['digest']}
    envelope['digest'] = _identity_digest(_terminal_bucket_manifest(envelope), budget=budget, operation='support_image.terminal_bucket.digest')
    return envelope

def _support_image_certificate_manifest(envelope: Mapping[str, object]) -> dict[str, object]:
    return {'source_snapshot_digest': envelope['source_snapshot']['digest'], 'strings': envelope['strings'], 'string_certificate_digests': envelope['string_certificate_digests'], 'support_string_envelope_digests': envelope['support_string_envelope_digests'], 'distinct_count': envelope['distinct_count'], 'witness_count': envelope['witness_count'], 'support_count_certificate_digest': envelope['support_count_certificate_digest'], 'witness_count_certificate_digest': envelope['witness_count_certificate_digest'], 'checked_frontier_certificate_digest': envelope['checked_frontier_certificate_digest'], 'count_dag_digest': envelope['count_dag_digest'], 'frontier_product_digest': envelope['frontier_product_digest'], 'enumeration_coverage_digest': envelope['enumeration_coverage_digest']}

def _enumeration_coverage_manifest(envelope: Mapping[str, object]) -> dict[str, object]:
    return {'source_snapshot_digest': envelope['source_snapshot']['digest'], 'checked_frontier_certificate_digest': envelope['checked_frontier_certificate']['digest'], 'support_count_certificate_digest': envelope['support_count_certificate']['digest'], 'support_count_term_coverage_digest': envelope['support_count_term_coverage_digest'], 'text_bucket_digests': [bucket['digest'] for bucket in envelope['text_buckets']], 'terminal_bucket_digest': None if envelope['terminal_bucket'] is None else envelope['terminal_bucket']['digest'], 'distinct_count': envelope['distinct_count'], 'support_count': envelope['support_count']}

def _text_bucket_manifest(envelope: Mapping[str, object]) -> dict[str, object]:
    return {'text_projection_digest': envelope['text_projection']['digest'], 'support_count_term_digest': envelope['support_count_term_digest'], 'support_count': envelope['support_count'], 'string_indices': envelope['string_indices'], 'string_digests': envelope['string_digests'], 'support_string_envelope_digests': envelope['support_string_envelope_digests']}

def _terminal_bucket_manifest(envelope: Mapping[str, object]) -> dict[str, object]:
    return {'terminal_projection_digest': None if envelope['terminal_projection'] is None else envelope['terminal_projection']['digest'], 'terminal_support_term_digest': envelope['terminal_support_term_digest'], 'terminal_support_identity_digests': [identity['digest'] for identity in envelope['terminal_support_identities']], 'support_count': envelope['support_count'], 'string_index': envelope['string_index'], 'string_digest': envelope['string_digest'], 'support_string_envelope_digest': envelope['support_string_envelope_digest']}

def _validate_bucket_partition(envelope, expected_count: int) -> None:
    indices = []
    for bucket in envelope['text_buckets']:
        if bucket['support_count'] != len(bucket['string_indices']):
            _image_envelope_violation('text_bucket_count_mismatch')
        indices.extend(bucket['string_indices'])
    terminal = envelope['terminal_bucket']
    if terminal is not None and terminal['string_index'] is not None:
        if terminal['support_count'] != 1:
            _image_envelope_violation('terminal_bucket_count_mismatch')
        indices.append(terminal['string_index'])
    if sorted(indices) != list(range(expected_count)):
        _image_envelope_violation('bucket_partition_mismatch')

def _check_materialized_image_work(image, *, budget: WriterEnvelopeWorkBudget) -> None:
    check_writer_envelope_work(budget=budget, operation='support_image_envelope', metric='support_string_count', actual=len(image.string_certificates), limit=budget.max_support_strings)
    check_writer_envelope_work(budget=budget, operation='support_image_envelope', metric='total_emitted_text_bytes', actual=sum((len(text.encode('utf-8')) for certificate in image.string_certificates for text in certificate.emitted_texts)), limit=budget.max_total_emitted_text_bytes)

def _check_support_string_envelope_work(support_string_envelopes, *, budget: WriterEnvelopeWorkBudget, operation: str) -> None:
    check_writer_envelope_work(budget=budget, operation=operation, metric='support_string_envelope_count', actual=len(support_string_envelopes), limit=budget.max_support_string_envelopes)
    check_writer_envelope_work(budget=budget, operation=operation, metric='support_string_count', actual=len(support_string_envelopes), limit=budget.max_support_strings)
    check_writer_envelope_work(budget=budget, operation=operation, metric='total_emitted_text_bytes', actual=sum((len(text.encode('utf-8')) for envelope in support_string_envelopes for text in envelope['emitted_texts'])), limit=budget.max_total_emitted_text_bytes)

def _check_coverage_work(coverage, *, budget: WriterEnvelopeWorkBudget) -> None:
    bucket_count = len(coverage['text_buckets']) + (0 if coverage['terminal_bucket'] is None else 1)
    assignment_count = sum((len(bucket['string_indices']) for bucket in coverage['text_buckets']))
    terminal = coverage['terminal_bucket']
    if terminal is not None and terminal['string_index'] is not None:
        assignment_count += 1
    check_writer_envelope_work(budget=budget, operation='support_image_envelope', metric='coverage_bucket_count', actual=bucket_count, limit=budget.max_coverage_buckets)
    check_writer_envelope_work(budget=budget, operation='support_image_envelope', metric='bucket_assignment_count', actual=assignment_count, limit=budget.max_bucket_assignments)

def _source_snapshot_for_envelope(*, prepared, envelope, budget):
    if envelope['source_kind'] == 'snapshot':
        return _source_snapshot_from_envelope(prepared=prepared, envelope=envelope, budget=budget)
    prefix = verify_writer_snapshot_prefix_read_envelope(prepared=prepared, envelope=envelope['prefix_read_envelope'], budget=budget)
    if not prefix.accepted:
        _image_envelope_violation('prefix_read_envelope_rejected')
    if prefix.read_kind != 'readable':
        _image_envelope_violation('prefix_read_envelope_not_readable')
    if prefix.final_snapshot is None:
        _image_envelope_violation('prefix_read_envelope_lacks_final_snapshot')
    return prefix.final_snapshot

def _checked_product(*, prepared, snapshot):
    return _checked_writer_frontier_product(prepared, snapshot.cursor, include_counts=True, include_frontier_certificate=True, include_count_certificate=True)

def _validate_envelope_shape(envelope: object) -> None:
    if not isinstance(envelope, Mapping):
        _image_envelope_violation('envelope_not_mapping')
    if frozenset(envelope) != _TOP_LEVEL_FIELDS:
        _image_envelope_violation('top_level_fields_mismatch')
    if envelope['schema_name'] != SCHEMA_NAME:
        _image_envelope_violation('unknown_schema_name')
    if envelope['schema_version'] != SCHEMA_VERSION:
        _image_envelope_violation('unknown_schema_version')
    if envelope['source_kind'] not in _SOURCE_KINDS:
        _image_envelope_violation('unknown_source_kind')
    if envelope['source_kind'] == 'snapshot':
        if envelope['source_snapshot'] is None:
            _image_envelope_violation('snapshot_source_missing')
        if envelope['prefix_read_envelope'] is not None:
            _image_envelope_violation('snapshot_source_has_prefix')
    else:
        if envelope['source_snapshot'] is not None:
            _image_envelope_violation('prefix_source_has_source_snapshot')
        if envelope['prefix_read_envelope'] is None:
            _image_envelope_violation('prefix_source_missing_prefix')

def _check_support_image_work(envelope: Mapping[str, object], *, budget: WriterEnvelopeWorkBudget) -> None:
    _check_support_string_envelope_work(envelope['support_string_envelopes'], budget=budget, operation='support_image_verify')
    check_writer_envelope_work(budget=budget, operation='support_image_verify', metric='support_string_count', actual=len(envelope['support_strings']), limit=budget.max_support_strings)
    check_writer_envelope_work(budget=budget, operation='support_image_verify', metric='total_emitted_text_bytes', actual=sum((len(text.encode('utf-8')) for string_envelope in envelope['support_string_envelopes'] for text in string_envelope['emitted_texts'])), limit=budget.max_total_emitted_text_bytes)
    _check_coverage_work(envelope['enumeration_coverage'], budget=budget)

def _assert_prepared_identity_matches(prepared, envelope, *, budget) -> None:
    snapshot_terms = envelope['source_snapshot'] if envelope['source_kind'] == 'snapshot' else envelope['count_envelope']['frontier_snapshot']
    runtime_options = _runtime_options_from_terms(snapshot_terms['runtime_options'])
    actual = _identity_envelope(_prepared_identity(prepared, runtime_options), budget=budget, operation='envelope.identity')
    if envelope['prepared_identity'] != actual:
        _image_envelope_violation('prepared_identity_mismatch')
    if snapshot_terms['prepared_identity_digest'] != actual['digest']:
        _image_envelope_violation('snapshot_prepared_identity_mismatch')

def _image_envelope_violation(kind: str) -> None:
    raise SouthStarError(SouthStarErrorKind.INTERNAL_INVARIANT, f'writer support image envelope violation: {kind}')
__all__ = ('SCHEMA_NAME', 'SCHEMA_VERSION', 'WriterSupportImageEnvelopeVerification', 'verify_writer_support_image_envelope', 'writer_support_image_envelope_for_prefix_read', 'writer_support_image_envelope_for_snapshot')
