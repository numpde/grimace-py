"""Durable envelopes for writer snapshot replay chains."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .prepared_runtime import SouthStarPreparedMol
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
from .writer_snapshot import _prepared_identity
from .writer_snapshot import _writer_snapshot_advance_sequence_outcome_by_emitted_texts
from .writer_snapshot_envelope import _source_snapshot_from_envelope
from .writer_snapshot_envelope import _verify_writer_snapshot_advance_envelope_from_known_source
from .writer_snapshot_envelope import writer_snapshot_advance_envelope_for_emitted_text
SCHEMA_NAME = 'writer_snapshot_replay'
SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset(('schema_name', 'schema_version', 'prepared_identity', 'source_snapshot', 'emitted_texts', 'outcome_kind', 'consumed_emitted_texts', 'remaining_emitted_texts', 'step_advance_envelopes', 'current_snapshot', 'replay_certificate', 'failed_advance_envelope'))
_OUTCOME_KINDS = frozenset(('advanced', 'invalid_emitted_text', 'blocked'))

@dataclass(frozen=True, slots=True)
class WriterSnapshotReplayEnvelopeVerification:
    accepted: bool
    outcome_kind: str
    source_snapshot: object | None
    current_snapshot: object | None
    failed_step_index: int | None = None
    reason: str | None = None

def writer_snapshot_replay_envelope_for_emitted_texts(*, prepared: SouthStarPreparedMol, snapshot, emitted_texts: tuple[str, ...], budget: WriterEnvelopeWorkBudget | None=None) -> dict[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    _check_emitted_text_work(emitted_texts, budget=budget, operation='snapshot_replay_envelope')
    outcome = _writer_snapshot_advance_sequence_outcome_by_emitted_texts(snapshot, prepared=prepared, emitted_texts=emitted_texts)
    advanced_step_envelopes = tuple((writer_snapshot_advance_envelope_for_emitted_text(prepared=prepared, snapshot=step.source_snapshot, emitted_text=step.emitted_text, budget=budget) for step in outcome.advanced_step_outcomes))
    failed_envelope = None
    failed = outcome.failed_outcome
    if failed is not None:
        failed_envelope = writer_snapshot_advance_envelope_for_emitted_text(prepared=prepared, snapshot=failed.source_snapshot, emitted_text=failed.emitted_text, budget=budget)
    envelope = {'schema_name': SCHEMA_NAME, 'schema_version': SCHEMA_VERSION, 'prepared_identity': _identity_envelope(snapshot.prepared_identity, budget=budget, operation='snapshot_replay.prepared_identity.digest'), 'source_snapshot': _snapshot_identity_envelope(snapshot, budget=budget, operation='snapshot_replay.source_snapshot.digest'), 'emitted_texts': list(emitted_texts), 'outcome_kind': outcome.kind.value, 'consumed_emitted_texts': list(outcome.consumed_emitted_texts), 'remaining_emitted_texts': list(outcome.remaining_emitted_texts), 'step_advance_envelopes': list(advanced_step_envelopes), 'current_snapshot': _snapshot_identity_envelope(outcome.current_snapshot, budget=budget, operation='snapshot_replay.current_snapshot.digest'), 'replay_certificate': None if outcome.replay_certificate is None else _replay_certificate_envelope(outcome.replay_certificate, budget=budget), 'failed_advance_envelope': failed_envelope}
    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope, budget=budget)
    return envelope

def verify_writer_snapshot_replay_envelope(*, prepared: SouthStarPreparedMol, envelope: object, budget: WriterEnvelopeWorkBudget | None=None) -> WriterSnapshotReplayEnvelopeVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        _validate_envelope_shape(envelope)
        assert isinstance(envelope, Mapping)
        _check_emitted_text_work(tuple(envelope['emitted_texts']), budget=budget, operation='snapshot_replay_verify')
        outcome_kind = str(envelope['outcome_kind'])
        _assert_prepared_identity_matches(prepared, envelope, budget=budget)
        source_snapshot = _source_snapshot_from_envelope(prepared=prepared, envelope=envelope, budget=budget)
        current_snapshot, failed_step_index = _verify_step_chain(prepared=prepared, source_snapshot=source_snapshot, envelope=envelope, budget=budget)
        expected = writer_snapshot_replay_envelope_for_emitted_texts(prepared=prepared, snapshot=source_snapshot, emitted_texts=tuple(envelope['emitted_texts']), budget=budget)
        if expected != envelope:
            return WriterSnapshotReplayEnvelopeVerification(accepted=False, outcome_kind=outcome_kind, source_snapshot=source_snapshot, current_snapshot=current_snapshot, failed_step_index=failed_step_index, reason='envelope_terms_mismatch')
        return WriterSnapshotReplayEnvelopeVerification(accepted=True, outcome_kind=outcome_kind, source_snapshot=source_snapshot, current_snapshot=current_snapshot, failed_step_index=failed_step_index)
    except WriterEnvelopeWorkExceeded as exc:
        return WriterSnapshotReplayEnvelopeVerification(accepted=False, outcome_kind=envelope.get('outcome_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', source_snapshot=None, current_snapshot=None, reason=writer_envelope_work_reason(exc))
    except SouthStarError as exc:
        return WriterSnapshotReplayEnvelopeVerification(accepted=False, outcome_kind=envelope.get('outcome_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', source_snapshot=None, current_snapshot=None, reason=exc.args[-1] if exc.args else 'verification_error')
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterSnapshotReplayEnvelopeVerification(accepted=False, outcome_kind=envelope.get('outcome_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', source_snapshot=None, current_snapshot=None, reason=f'malformed_envelope:{type(exc).__name__}')

def _verify_step_chain(*, prepared, source_snapshot, envelope, budget):
    emitted_texts = tuple(envelope['emitted_texts'])
    step_envelopes = tuple(envelope['step_advance_envelopes'])
    check_writer_envelope_work(budget=budget, operation='snapshot_replay_verify', metric='replay_step_count', actual=len(step_envelopes), limit=budget.max_replay_steps)
    current = source_snapshot
    for index, step_envelope in enumerate(step_envelopes):
        _validate_step_source(step_envelope, current, budget=budget)
        if step_envelope['emitted_text'] != emitted_texts[index]:
            _replay_envelope_violation('step_emitted_text_mismatch')
        verification = _verify_writer_snapshot_advance_envelope_from_known_source(prepared=prepared, source_snapshot=current, envelope=step_envelope, budget=budget)
        if not verification.accepted:
            _replay_envelope_violation('step_advance_envelope_rejected')
        if verification.outcome_kind != 'advanced':
            _replay_envelope_violation('advanced_step_outcome_mismatch')
        if verification.advanced_snapshot is None:
            _replay_envelope_violation('advanced_step_lacks_snapshot')
        current = verification.advanced_snapshot
    failed = envelope['failed_advance_envelope']
    outcome_kind = envelope['outcome_kind']
    if outcome_kind == 'advanced':
        if failed is not None:
            _replay_envelope_violation('advanced_replay_has_failed_step')
        if len(step_envelopes) != len(emitted_texts):
            _replay_envelope_violation('advanced_step_count_mismatch')
        if envelope['remaining_emitted_texts']:
            _replay_envelope_violation('advanced_remaining_texts_mismatch')
        if envelope['replay_certificate'] is None:
            _replay_envelope_violation('missing_replay_certificate')
        if _snapshot_identity_envelope(current, budget=budget, operation='envelope.identity') != envelope['current_snapshot']:
            _replay_envelope_violation('current_snapshot_mismatch')
        return (current, None)
    if failed is None:
        _replay_envelope_violation('missing_failed_advance_envelope')
    failed_index = len(step_envelopes)
    if failed_index >= len(emitted_texts):
        _replay_envelope_violation('failed_step_index_mismatch')
    _validate_step_source(failed, current, budget=budget)
    if failed['emitted_text'] != emitted_texts[failed_index]:
        _replay_envelope_violation('failed_step_emitted_text_mismatch')
    verification = _verify_writer_snapshot_advance_envelope_from_known_source(prepared=prepared, source_snapshot=current, envelope=failed, budget=budget)
    if not verification.accepted:
        _replay_envelope_violation('failed_advance_envelope_rejected')
    if verification.outcome_kind != outcome_kind:
        _replay_envelope_violation('failed_outcome_kind_mismatch')
    if envelope['replay_certificate'] is not None:
        _replay_envelope_violation('failed_replay_has_certificate')
    if _snapshot_identity_envelope(current, budget=budget, operation='envelope.identity') != envelope['current_snapshot']:
        _replay_envelope_violation('failed_current_snapshot_mismatch')
    return (current, failed_index)

def _validate_step_source(step_envelope, snapshot, *, budget) -> None:
    if step_envelope['source_snapshot'] != _snapshot_identity_envelope(snapshot, budget=budget, operation='envelope.identity'):
        _replay_envelope_violation('step_source_snapshot_mismatch')

def _check_emitted_text_work(emitted_texts: tuple[str, ...], *, budget: WriterEnvelopeWorkBudget, operation: str) -> None:
    check_writer_envelope_work(budget=budget, operation=operation, metric='replay_step_count', actual=len(emitted_texts), limit=budget.max_replay_steps)
    check_writer_envelope_work(budget=budget, operation=operation, metric='total_emitted_text_bytes', actual=sum((len(text.encode('utf-8')) for text in emitted_texts)), limit=budget.max_total_emitted_text_bytes)

def _replay_certificate_envelope(certificate, *, budget) -> dict[str, object]:
    envelope = {'source_snapshot': _snapshot_identity_envelope(certificate.source_snapshot, budget=budget, operation='snapshot_replay.certificate.source_snapshot.digest'), 'emitted_texts': list(certificate.emitted_texts), 'step_certificate_digests': [_step_certificate_digest(step, budget=budget, operation='snapshot_replay.step_certificate.digest') for step in certificate.step_certificates], 'frontier_projection_digests': [_frontier_projection_digest(projection, budget=budget, operation='snapshot_replay.frontier_projection.digest') for projection in certificate.frontier_projection_certificates], 'final_snapshot': _snapshot_identity_envelope(certificate.final_snapshot, budget=budget, operation='snapshot_replay.certificate.final_snapshot.digest')}
    envelope['digest'] = _identity_digest(_replay_certificate_manifest(envelope), budget=budget, operation='snapshot_replay.certificate.digest')
    return envelope

def _replay_certificate_manifest(envelope: Mapping[str, object]) -> dict[str, object]:
    return {'source_snapshot_digest': envelope['source_snapshot']['digest'], 'emitted_texts': envelope['emitted_texts'], 'step_certificate_digests': envelope['step_certificate_digests'], 'frontier_projection_digests': envelope['frontier_projection_digests'], 'final_snapshot_digest': envelope['final_snapshot']['digest']}

def _step_certificate_digest(step, *, budget, operation: str) -> str:
    manifest = {'source_snapshot_digest': _snapshot_identity_envelope(step.source_snapshot, budget=budget, operation=f'{operation}.source_snapshot')['digest'], 'emitted_text': step.emitted_text, 'frontier_projection_digest': _frontier_projection_digest(step.frontier_projection_certificate, budget=budget, operation=f'{operation}.frontier_projection'), 'text_projection_digest': _text_projection_digest(step.text_projection_certificate, budget=budget, operation=f'{operation}.text_projection'), 'source_cursor_digest': _identity_digest(step.source_cursor, budget=budget, operation=f'{operation}.source_cursor'), 'successor_cursor_digest': _identity_digest(step.successor_cursor, budget=budget, operation=f'{operation}.successor_cursor'), 'advanced_snapshot_digest': _snapshot_identity_envelope(step.advanced_snapshot, budget=budget, operation=f'{operation}.advanced_snapshot')['digest'], 'decoder_boundary_before': _term(step.decoder_boundary_before), 'decoder_boundary_after': _term(step.decoder_boundary_after), 'branch_count': len(step.branch_certificates)}
    return _identity_digest(manifest, budget=budget, operation=operation)

def _frontier_projection_digest(projection, *, budget, operation: str) -> str:
    manifest = {'cursor_digest': _identity_digest(projection.cursor, budget=budget, operation=f'{operation}.cursor'), 'text_projection_digests': [_text_projection_digest(item, budget=budget, operation=f'{operation}.text_projection') for item in projection.text_choice_projection_certificates], 'terminal_projection_present': projection.terminal_projection_certificate is not None, 'branch_count': len(projection.branch_certificates), 'terminal_count': len(projection.terminal_certificates)}
    return _identity_digest(manifest, budget=budget, operation=operation)

def _text_projection_digest(projection, *, budget, operation: str) -> str:
    manifest = {'source_cursor_digest': _identity_digest(projection.source_cursor, budget=budget, operation=f'{operation}.source_cursor'), 'emitted_text': projection.emitted_text, 'successor_cursor_digest': _identity_digest(projection.successor_cursor, budget=budget, operation=f'{operation}.successor_cursor'), 'immediate_multiplicity': projection.immediate_multiplicity, 'support_count': projection.support_count, 'completion_count': projection.completion_count, 'branch_count': len(projection.branch_certificates)}
    return _identity_digest(manifest, budget=budget, operation=operation)

def _validate_envelope_shape(envelope: object) -> None:
    if not isinstance(envelope, Mapping):
        _replay_envelope_violation('envelope_not_mapping')
    keys = frozenset(envelope)
    if keys != _TOP_LEVEL_FIELDS:
        _replay_envelope_violation('top_level_fields_mismatch')
    if envelope['schema_name'] != SCHEMA_NAME:
        _replay_envelope_violation('unknown_schema_name')
    if envelope['schema_version'] != SCHEMA_VERSION:
        _replay_envelope_violation('unknown_schema_version')
    if envelope['outcome_kind'] not in _OUTCOME_KINDS:
        _replay_envelope_violation('unknown_outcome_kind')
    for field in ('emitted_texts', 'consumed_emitted_texts', 'remaining_emitted_texts', 'step_advance_envelopes'):
        if not isinstance(envelope[field], list):
            _replay_envelope_violation(f'{field}_not_list')
    emitted = list(envelope['emitted_texts'])
    consumed = list(envelope['consumed_emitted_texts'])
    remaining = list(envelope['remaining_emitted_texts'])
    if consumed + remaining != emitted:
        _replay_envelope_violation('consumed_remaining_partition_mismatch')
    if envelope['outcome_kind'] == 'advanced':
        if remaining:
            _replay_envelope_violation('advanced_remaining_texts_mismatch')
        if envelope['failed_advance_envelope'] is not None:
            _replay_envelope_violation('advanced_failed_envelope_mismatch')
    else:
        if envelope['failed_advance_envelope'] is None:
            _replay_envelope_violation('missing_failed_advance_envelope')
        if not remaining:
            _replay_envelope_violation('failed_remaining_texts_mismatch')

def _assert_prepared_identity_matches(prepared, envelope, *, budget) -> None:
    runtime_options = _runtime_options_from_terms(envelope['source_snapshot']['runtime_options'])
    actual = _identity_envelope(_prepared_identity(prepared, runtime_options), budget=budget, operation='envelope.identity')
    if envelope['prepared_identity'] != actual:
        _replay_envelope_violation('prepared_identity_mismatch')
    if envelope['source_snapshot']['prepared_identity_digest'] != actual['digest']:
        _replay_envelope_violation('source_snapshot_prepared_identity_mismatch')

def _replay_envelope_violation(kind: str) -> None:
    raise SouthStarError(SouthStarErrorKind.INTERNAL_INVARIANT, f'writer snapshot replay envelope violation: {kind}')
__all__ = ('SCHEMA_NAME', 'SCHEMA_VERSION', 'WriterSnapshotReplayEnvelopeVerification', 'verify_writer_snapshot_replay_envelope', 'writer_snapshot_replay_envelope_for_emitted_texts')
