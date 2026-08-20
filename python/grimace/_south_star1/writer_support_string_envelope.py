"""Durable envelopes for individual certified writer support strings."""
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
from .writer_frontier import _snapshot_advance_writer_frontier_product
from .writer_frontier_count_envelope import verify_writer_frontier_count_envelope
from .writer_frontier_count_envelope import writer_frontier_count_envelope_for_prefix_read
from .writer_frontier_count_envelope import writer_frontier_count_envelope_for_snapshot
from .writer_snapshot import _iter_writer_snapshot_certified_support_strings
from .writer_snapshot import _prepared_identity
from .writer_snapshot import _writer_snapshot_advance_sequence_outcome_by_emitted_texts
from .writer_snapshot_envelope import _source_snapshot_from_envelope
from .writer_snapshot_prefix_envelope import _branch_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import _frontier_projection_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import _terminal_projection_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import _terminal_support_identity_envelope_from_certificate
from .writer_snapshot_prefix_envelope import _text_projection_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import _writer_frontier_product_identity_envelope
from .writer_snapshot_prefix_envelope import verify_writer_snapshot_prefix_read_envelope
from .writer_snapshot_replay_envelope import verify_writer_snapshot_replay_envelope
from .writer_snapshot_replay_envelope import writer_snapshot_replay_envelope_for_emitted_texts
from .writer_snapshot_replay_envelope import _frontier_projection_digest
from .writer_snapshot_replay_envelope import _step_certificate_digest
from .writer_support_certificates import writer_support_string_certificate
SCHEMA_NAME = 'writer_support_string'
SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset(('schema_name', 'schema_version', 'prepared_identity', 'source_kind', 'source_snapshot', 'prefix_read_envelope', 'count_envelope', 'string', 'emitted_texts', 'replay_envelope', 'final_snapshot', 'terminal_frontier_product', 'terminal_frontier_projection', 'terminal_projection', 'support_string_certificate', 'terminal_support_identities', 'text_projection_chain', 'digest'))
_SOURCE_KINDS = frozenset(('snapshot', 'prefix_read'))

@dataclass(frozen=True, slots=True)
class WriterSupportStringEnvelopeVerification:
    accepted: bool
    source_kind: str
    string: str | None
    source_snapshot: object | None
    final_snapshot: object | None
    reason: str | None = None

def writer_support_string_envelope_for_string(*, prepared: SouthStarPreparedMol, snapshot, string: str, budget: WriterEnvelopeWorkBudget | None=None) -> dict[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    certificate = _certified_support_string_for_string(prepared=prepared, snapshot=snapshot, string=string, budget=budget)
    count_envelope = writer_frontier_count_envelope_for_snapshot(prepared=prepared, snapshot=snapshot, budget=budget)
    replay_envelope = writer_snapshot_replay_envelope_for_emitted_texts(prepared=prepared, snapshot=snapshot, emitted_texts=certificate.emitted_texts, budget=budget)
    terminal_product = _terminal_product_for_certificate(prepared=prepared, certificate=certificate)
    envelope = _envelope_from_certificate_with_product(source_kind='snapshot', source_snapshot=snapshot, prefix_read_envelope=None, count_envelope=count_envelope, replay_envelope=replay_envelope, certificate=certificate, terminal_product=terminal_product, budget=budget)
    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope, budget=budget)
    return envelope

def writer_support_string_envelope_for_prefix_read(*, prepared: SouthStarPreparedMol, prefix_read_envelope: Mapping[str, object], string: str, budget: WriterEnvelopeWorkBudget | None=None) -> dict[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    prefix = verify_writer_snapshot_prefix_read_envelope(prepared=prepared, envelope=prefix_read_envelope, budget=budget)
    if not prefix.accepted:
        _support_string_envelope_violation('prefix_read_envelope_rejected')
    if prefix.read_kind != 'readable':
        _support_string_envelope_violation('prefix_read_envelope_not_readable')
    if prefix.final_snapshot is None:
        _support_string_envelope_violation('prefix_read_envelope_lacks_final_snapshot')
    certificate = _certified_support_string_for_string(prepared=prepared, snapshot=prefix.final_snapshot, string=string, budget=budget)
    count_envelope = writer_frontier_count_envelope_for_prefix_read(prepared=prepared, prefix_read_envelope=prefix_read_envelope, budget=budget)
    replay_envelope = writer_snapshot_replay_envelope_for_emitted_texts(prepared=prepared, snapshot=prefix.final_snapshot, emitted_texts=certificate.emitted_texts, budget=budget)
    terminal_product = _terminal_product_for_certificate(prepared=prepared, certificate=certificate)
    envelope = _envelope_from_certificate_with_product(source_kind='prefix_read', source_snapshot=None, prefix_read_envelope=prefix_read_envelope, count_envelope=count_envelope, replay_envelope=replay_envelope, certificate=certificate, terminal_product=terminal_product, budget=budget)
    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope, budget=budget)
    return envelope

def _writer_support_string_envelope_from_certificate(*, prepared: SouthStarPreparedMol, source_kind: str, source_snapshot, prefix_read_envelope, count_envelope, replay_envelope, certificate, budget: WriterEnvelopeWorkBudget | None=None) -> dict[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    terminal_product = _terminal_product_for_certificate(prepared=prepared, certificate=certificate)
    envelope = _envelope_from_certificate_with_product(source_kind=source_kind, source_snapshot=source_snapshot, prefix_read_envelope=prefix_read_envelope, count_envelope=count_envelope, replay_envelope=replay_envelope, certificate=certificate, terminal_product=terminal_product, budget=budget)
    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope, budget=budget)
    return envelope

def verify_writer_support_string_envelope(*, prepared: SouthStarPreparedMol, envelope: object, budget: WriterEnvelopeWorkBudget | None=None) -> WriterSupportStringEnvelopeVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        _validate_envelope_shape(envelope)
        assert isinstance(envelope, Mapping)
        _check_support_string_work(envelope, budget=budget)
        _assert_prepared_identity_matches(prepared, envelope, budget=budget)
        source_kind = str(envelope['source_kind'])
        source_snapshot = _source_snapshot_for_envelope(prepared=prepared, envelope=envelope, budget=budget)
        count = verify_writer_frontier_count_envelope(prepared=prepared, envelope=envelope['count_envelope'], budget=budget)
        if not count.accepted:
            _support_string_envelope_violation('count_envelope_rejected')
        if count.frontier_snapshot != source_snapshot:
            _support_string_envelope_violation('count_envelope_source_mismatch')
        replay = verify_writer_snapshot_replay_envelope(prepared=prepared, envelope=envelope['replay_envelope'], budget=budget)
        if not replay.accepted:
            _support_string_envelope_violation('replay_envelope_rejected')
        if replay.outcome_kind != 'advanced':
            _support_string_envelope_violation('replay_envelope_not_advanced')
        if replay.source_snapshot != source_snapshot:
            _support_string_envelope_violation('replay_envelope_source_mismatch')
        if tuple(envelope['emitted_texts']) != tuple(envelope['replay_envelope']['emitted_texts']):
            _support_string_envelope_violation('replay_texts_mismatch')
        if envelope['string'] != ''.join(envelope['emitted_texts']):
            _support_string_envelope_violation('string_emitted_texts_mismatch')
        if replay.current_snapshot is None:
            _support_string_envelope_violation('replay_lacks_final_snapshot')
        expected = _expected_envelope_from_replay(source_kind=source_kind, source_snapshot=source_snapshot, prefix_read_envelope=envelope['prefix_read_envelope'], count_envelope=envelope['count_envelope'], replay_envelope=envelope['replay_envelope'], prepared=prepared, emitted_texts=tuple(envelope['emitted_texts']), budget=budget)
        if expected != envelope:
            return WriterSupportStringEnvelopeVerification(accepted=False, source_kind=source_kind, string=str(envelope['string']), source_snapshot=source_snapshot, final_snapshot=replay.current_snapshot, reason='envelope_terms_mismatch')
        return WriterSupportStringEnvelopeVerification(accepted=True, source_kind=source_kind, string=str(envelope['string']), source_snapshot=source_snapshot, final_snapshot=replay.current_snapshot)
    except WriterEnvelopeWorkExceeded as exc:
        return WriterSupportStringEnvelopeVerification(accepted=False, source_kind=envelope.get('source_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', string=envelope.get('string') if isinstance(envelope, Mapping) and isinstance(envelope.get('string'), str) else None, source_snapshot=None, final_snapshot=None, reason=writer_envelope_work_reason(exc))
    except SouthStarError as exc:
        return WriterSupportStringEnvelopeVerification(accepted=False, source_kind=envelope.get('source_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', string=envelope.get('string') if isinstance(envelope, Mapping) and isinstance(envelope.get('string'), str) else None, source_snapshot=None, final_snapshot=None, reason=exc.args[-1] if exc.args else 'verification_error')
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterSupportStringEnvelopeVerification(accepted=False, source_kind=envelope.get('source_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', string=envelope.get('string') if isinstance(envelope, Mapping) and isinstance(envelope.get('string'), str) else None, source_snapshot=None, final_snapshot=None, reason=f'malformed_envelope:{type(exc).__name__}')

def _certified_support_string_for_string(*, prepared, snapshot, string: str, budget: WriterEnvelopeWorkBudget):
    matches = []
    for visited, item in enumerate(_iter_writer_snapshot_certified_support_strings(snapshot, prepared=prepared), start=1):
        check_writer_envelope_work(budget=budget, operation='support_string_search', metric='visited_support_strings', actual=visited, limit=budget.max_support_search_strings)
        if item.string == string:
            matches.append(item)
    if len(matches) != 1:
        _support_string_envelope_violation('support_string_not_unique')
    return matches[0].certificate

def _expected_envelope_from_replay(*, source_kind: str, source_snapshot, prefix_read_envelope, count_envelope, replay_envelope, prepared, emitted_texts: tuple[str, ...], budget: WriterEnvelopeWorkBudget) -> dict[str, object]:
    _check_emitted_text_work(emitted_texts, budget=budget, operation='support_string_envelope')
    outcome = _writer_snapshot_advance_sequence_outcome_by_emitted_texts(source_snapshot, prepared=prepared, emitted_texts=emitted_texts)
    if outcome.kind.value != 'advanced' or outcome.replay_certificate is None:
        _support_string_envelope_violation('replay_outcome_not_advanced')
    product = _snapshot_advance_writer_frontier_product(prepared, outcome.current_snapshot.cursor)
    if product.blocked:
        _support_string_envelope_violation('terminal_frontier_blocked')
    if product.terminal_projection_certificate is None:
        _support_string_envelope_violation('missing_terminal_projection')
    certificate = writer_support_string_certificate(source_snapshot=source_snapshot, string=''.join(emitted_texts), emitted_texts=emitted_texts, replay_certificate=outcome.replay_certificate, terminal_frontier_projection_certificate=product.projection_certificate, terminal_projection_certificate=product.terminal_projection_certificate, text_projection_certificates=tuple((step.text_projection_certificate for step in outcome.advanced_step_outcomes)))
    return _envelope_from_certificate_with_product(source_kind=source_kind, source_snapshot=source_snapshot if source_kind == 'snapshot' else None, prefix_read_envelope=prefix_read_envelope, count_envelope=count_envelope, replay_envelope=replay_envelope, certificate=certificate, terminal_product=product, budget=budget)

def _envelope_from_certificate_with_product(*, source_kind: str, source_snapshot, prefix_read_envelope, count_envelope, replay_envelope, certificate, terminal_product, budget: WriterEnvelopeWorkBudget) -> dict[str, object]:
    _check_certificate_work(certificate, budget=budget)
    text_chain = _text_projection_chain_envelope(certificate, budget=budget)
    terminal_projection = _terminal_projection_certificate_identity_envelope(certificate.terminal_projection_certificate, budget=budget)
    support_certificate = _support_string_certificate_envelope(certificate=certificate, text_projection_chain=text_chain, terminal_projection=terminal_projection, budget=budget)
    envelope = {'schema_name': SCHEMA_NAME, 'schema_version': SCHEMA_VERSION, 'prepared_identity': _identity_envelope(certificate.replay_certificate.source_snapshot.prepared_identity, budget=budget, operation='support_string.prepared_identity.digest'), 'source_kind': source_kind, 'source_snapshot': None if source_snapshot is None else _snapshot_identity_envelope(source_snapshot, budget=budget, operation='support_string.source_snapshot.digest'), 'prefix_read_envelope': prefix_read_envelope, 'count_envelope': count_envelope, 'string': certificate.string, 'emitted_texts': list(certificate.emitted_texts), 'replay_envelope': replay_envelope, 'final_snapshot': _snapshot_identity_envelope(certificate.final_snapshot, budget=budget, operation='support_string.final_snapshot.digest'), 'terminal_frontier_product': _writer_frontier_product_identity_envelope(terminal_product, budget=budget), 'terminal_frontier_projection': _frontier_projection_certificate_identity_envelope(certificate.terminal_frontier_projection_certificate, budget=budget), 'terminal_projection': terminal_projection, 'support_string_certificate': support_certificate, 'terminal_support_identities': [_terminal_support_identity_envelope_from_certificate(terminal, budget=budget) for terminal in certificate.terminal_projection_certificate.terminal_certificates], 'text_projection_chain': text_chain}
    envelope['digest'] = _identity_digest(_support_string_manifest(envelope), budget=budget, operation='support_string.manifest.digest')
    return envelope

def _terminal_product_for_certificate(*, prepared, certificate):
    product = _snapshot_advance_writer_frontier_product(prepared, certificate.final_snapshot.cursor)
    if product.blocked:
        _support_string_envelope_violation('terminal_frontier_blocked')
    if product.terminal_projection_certificate is None:
        _support_string_envelope_violation('missing_terminal_projection')
    return product

def _support_string_certificate_envelope(*, certificate, text_projection_chain, terminal_projection, budget) -> dict[str, object]:
    envelope = {'string': certificate.string, 'emitted_texts': list(certificate.emitted_texts), 'replay_certificate_digest': _support_string_replay_certificate_digest(certificate.replay_certificate, budget=budget), 'final_snapshot': _snapshot_identity_envelope(certificate.final_snapshot, budget=budget, operation='support_string.certificate.final_snapshot.digest'), 'terminal_frontier_projection_digest': _frontier_projection_certificate_identity_envelope(certificate.terminal_frontier_projection_certificate, budget=budget)['digest'], 'terminal_projection_digest': terminal_projection['digest'], 'terminal_certificate_digests': [_terminal_support_identity_envelope_from_certificate(terminal, budget=budget)['digest'] for terminal in certificate.terminal_certificates], 'text_projection_chain_digests': [step['text_projection']['digest'] for step in text_projection_chain]}
    envelope['digest'] = _identity_digest(envelope, budget=budget, operation='support_string.certificate.digest')
    return envelope

def _support_string_replay_certificate_digest(certificate, *, budget) -> str:
    manifest = {'source_snapshot_digest': _snapshot_identity_envelope(certificate.source_snapshot, budget=budget, operation='support_string.replay_certificate.source_snapshot.digest')['digest'], 'emitted_texts': list(certificate.emitted_texts), 'step_certificate_digests': [_step_certificate_digest(step, budget=budget, operation='support_string.replay_certificate.step.digest') for step in certificate.step_certificates], 'frontier_projection_digests': [_frontier_projection_digest(projection, budget=budget, operation='support_string.replay_certificate.frontier_projection.digest') for projection in certificate.frontier_projection_certificates], 'final_snapshot_digest': _snapshot_identity_envelope(certificate.final_snapshot, budget=budget, operation='support_string.replay_certificate.final_snapshot.digest')['digest']}
    return _identity_digest(manifest, budget=budget, operation='support_string.replay_certificate.digest')

def _text_projection_chain_envelope(certificate, *, budget):
    chain = []
    for index, step in enumerate(certificate.replay_certificate.step_certificates):
        text_projection = _text_projection_certificate_identity_envelope(step.text_projection_certificate, budget=budget)
        frontier_projection = _frontier_projection_certificate_identity_envelope(step.frontier_projection_certificate, budget=budget)
        chain.append({'step_index': index, 'emitted_text': step.emitted_text, 'source_cursor': _cursor_envelope(step.source_cursor, budget=budget, operation='support_string.text_chain.source_cursor.digest'), 'successor_cursor': _cursor_envelope(step.successor_cursor, budget=budget, operation='support_string.text_chain.successor_cursor.digest'), 'text_projection': text_projection, 'frontier_projection': frontier_projection, 'step_certificate_digest': _identity_digest(step, budget=budget, operation='support_string.step_certificate.digest'), 'branch_certificate_identities': [_branch_certificate_identity_envelope(branch, budget=budget) for branch in step.branch_certificates]})
    return chain

def _support_string_manifest(envelope: Mapping[str, object]) -> dict[str, object]:
    return {'schema_name': envelope['schema_name'], 'schema_version': envelope['schema_version'], 'source_kind': envelope['source_kind'], 'source_snapshot_digest': None if envelope['source_snapshot'] is None else envelope['source_snapshot']['digest'], 'prefix_read_kind': None if envelope['prefix_read_envelope'] is None else envelope['prefix_read_envelope']['read_kind'], 'count_dag_digest': envelope['count_envelope']['count_dag']['digest'], 'frontier_product_digest': envelope['count_envelope']['frontier_product']['digest'], 'string': envelope['string'], 'emitted_texts': envelope['emitted_texts'], 'replay_certificate_digest': envelope['support_string_certificate']['replay_certificate_digest'], 'final_snapshot_digest': envelope['final_snapshot']['digest'], 'terminal_frontier_product_digest': envelope['terminal_frontier_product']['digest'], 'terminal_frontier_projection_digest': envelope['terminal_frontier_projection']['digest'], 'terminal_projection_digest': envelope['terminal_projection']['digest'], 'support_string_certificate_digest': envelope['support_string_certificate']['digest'], 'terminal_support_identity_digests': [identity['digest'] for identity in envelope['terminal_support_identities']], 'text_projection_chain': [{'step_index': step['step_index'], 'emitted_text': step['emitted_text'], 'text_projection_digest': step['text_projection']['digest'], 'frontier_projection_digest': step['frontier_projection']['digest'], 'step_certificate_digest': step['step_certificate_digest'], 'branch_certificate_digests': [identity['digest'] for identity in step['branch_certificate_identities']]} for step in envelope['text_projection_chain']]}

def _source_snapshot_for_envelope(*, prepared, envelope, budget):
    if envelope['source_kind'] == 'snapshot':
        return _source_snapshot_from_envelope(prepared=prepared, envelope=envelope, budget=budget)
    prefix = verify_writer_snapshot_prefix_read_envelope(prepared=prepared, envelope=envelope['prefix_read_envelope'], budget=budget)
    if not prefix.accepted:
        _support_string_envelope_violation('prefix_read_envelope_rejected')
    if prefix.read_kind != 'readable':
        _support_string_envelope_violation('prefix_read_envelope_not_readable')
    if prefix.final_snapshot is None:
        _support_string_envelope_violation('prefix_read_envelope_lacks_final_snapshot')
    return prefix.final_snapshot

def _validate_envelope_shape(envelope: object) -> None:
    if not isinstance(envelope, Mapping):
        _support_string_envelope_violation('envelope_not_mapping')
    if frozenset(envelope) != _TOP_LEVEL_FIELDS:
        _support_string_envelope_violation('top_level_fields_mismatch')
    if envelope['schema_name'] != SCHEMA_NAME:
        _support_string_envelope_violation('unknown_schema_name')
    if envelope['schema_version'] != SCHEMA_VERSION:
        _support_string_envelope_violation('unknown_schema_version')
    if envelope['source_kind'] not in _SOURCE_KINDS:
        _support_string_envelope_violation('unknown_source_kind')
    if not isinstance(envelope['string'], str):
        _support_string_envelope_violation('string_not_text')
    if not isinstance(envelope['emitted_texts'], list):
        _support_string_envelope_violation('emitted_texts_not_list')
    if envelope['source_kind'] == 'snapshot':
        if envelope['source_snapshot'] is None:
            _support_string_envelope_violation('snapshot_source_missing')
        if envelope['prefix_read_envelope'] is not None:
            _support_string_envelope_violation('snapshot_source_has_prefix_read')
    else:
        if envelope['source_snapshot'] is not None:
            _support_string_envelope_violation('prefix_source_has_source_snapshot')
        if envelope['prefix_read_envelope'] is None:
            _support_string_envelope_violation('prefix_source_missing_prefix')

def _check_support_string_work(envelope: Mapping[str, object], *, budget: WriterEnvelopeWorkBudget) -> None:
    _check_emitted_text_work(tuple(envelope['emitted_texts']), budget=budget, operation='support_string_verify')
    check_writer_envelope_work(budget=budget, operation='support_string_verify', metric='text_projection_chain_length', actual=len(envelope['text_projection_chain']), limit=budget.max_text_projection_chain_length)
    check_writer_envelope_work(budget=budget, operation='support_string_verify', metric='terminal_support_identity_count', actual=len(envelope['terminal_support_identities']), limit=budget.max_terminal_support_identities)

def _check_certificate_work(certificate, *, budget: WriterEnvelopeWorkBudget) -> None:
    _check_emitted_text_work(tuple(certificate.emitted_texts), budget=budget, operation='support_string_envelope')
    check_writer_envelope_work(budget=budget, operation='support_string_envelope', metric='text_projection_chain_length', actual=len(certificate.replay_certificate.step_certificates), limit=budget.max_text_projection_chain_length)
    check_writer_envelope_work(budget=budget, operation='support_string_envelope', metric='terminal_support_identity_count', actual=len(certificate.terminal_projection_certificate.terminal_certificates), limit=budget.max_terminal_support_identities)

def _check_emitted_text_work(emitted_texts: tuple[str, ...], *, budget: WriterEnvelopeWorkBudget, operation: str) -> None:
    check_writer_envelope_work(budget=budget, operation=operation, metric='emitted_text_count', actual=len(emitted_texts), limit=budget.max_text_projection_chain_length)
    check_writer_envelope_work(budget=budget, operation=operation, metric='total_emitted_text_bytes', actual=sum((len(text.encode('utf-8')) for text in emitted_texts)), limit=budget.max_total_emitted_text_bytes)

def _assert_prepared_identity_matches(prepared, envelope, *, budget) -> None:
    snapshot_terms = envelope['source_snapshot'] if envelope['source_kind'] == 'snapshot' else envelope['final_snapshot']
    runtime_options = _runtime_options_from_terms(snapshot_terms['runtime_options'])
    actual = _identity_envelope(_prepared_identity(prepared, runtime_options), budget=budget, operation='envelope.identity')
    if envelope['prepared_identity'] != actual:
        _support_string_envelope_violation('prepared_identity_mismatch')
    if snapshot_terms['prepared_identity_digest'] != actual['digest']:
        _support_string_envelope_violation('snapshot_prepared_identity_mismatch')

def _support_string_envelope_violation(kind: str) -> None:
    raise SouthStarError(SouthStarErrorKind.INTERNAL_INVARIANT, f'writer support string envelope violation: {kind}')
__all__ = ('SCHEMA_NAME', 'SCHEMA_VERSION', 'WriterSupportStringEnvelopeVerification', 'verify_writer_support_string_envelope', 'writer_support_string_envelope_for_prefix_read', 'writer_support_string_envelope_for_string')
