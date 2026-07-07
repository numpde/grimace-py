"""Durable envelopes for single writer snapshot advances."""
from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from .errors import SouthStarError
from .errors import SouthStarErrorKind
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
from .prepared_runtime import SouthStarPreparedMol
from .writer_frontier import initial_writer_frontier_cursor
from .writer_frontier import _snapshot_advance_writer_frontier_product
from .writer_snapshot import WriterDecoderBoundary
from .writer_snapshot import _capture_writer_frontier_snapshot_unchecked
from .writer_snapshot import _prepared_identity
from .writer_snapshot import _writer_snapshot_advance_outcome_by_emitted_text
SCHEMA_NAME = 'writer_snapshot_advance'
SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset(('schema_name', 'schema_version', 'prepared_identity', 'source_snapshot', 'emitted_text', 'outcome_kind', 'frontier_product_kind', 'advance_certificate'))
_OUTCOME_KINDS = frozenset(('advanced', 'invalid_emitted_text', 'blocked'))
_PRODUCT_KINDS = frozenset(('legal', 'blocked'))

@dataclass(frozen=True, slots=True)
class WriterSnapshotAdvanceEnvelopeVerification:
    accepted: bool
    outcome_kind: str
    source_snapshot: object | None
    advanced_snapshot: object | None
    reason: str | None = None

def writer_snapshot_advance_envelope_for_emitted_text(*, prepared: SouthStarPreparedMol, snapshot, emitted_text: str, budget: WriterEnvelopeWorkBudget | None=None) -> dict[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    outcome = _writer_snapshot_advance_outcome_by_emitted_text(snapshot, prepared=prepared, emitted_text=emitted_text)
    return _envelope_from_outcome(prepared=prepared, outcome=outcome, budget=budget)

def verify_writer_snapshot_advance_envelope(*, prepared: SouthStarPreparedMol, envelope: object, budget: WriterEnvelopeWorkBudget | None=None) -> WriterSnapshotAdvanceEnvelopeVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        _validate_envelope_shape(envelope)
        assert isinstance(envelope, Mapping)
        outcome_kind = envelope['outcome_kind']
        source_snapshot = _source_snapshot_from_envelope(prepared=prepared, envelope=envelope, budget=budget)
        expected = writer_snapshot_advance_envelope_for_emitted_text(prepared=prepared, snapshot=source_snapshot, emitted_text=envelope['emitted_text'], budget=budget)
        if expected != envelope:
            return WriterSnapshotAdvanceEnvelopeVerification(accepted=False, outcome_kind=str(outcome_kind), source_snapshot=source_snapshot, advanced_snapshot=None, reason='envelope_terms_mismatch')
        advanced_snapshot = None
        if outcome_kind == 'advanced':
            outcome = _writer_snapshot_advance_outcome_by_emitted_text(source_snapshot, prepared=prepared, emitted_text=envelope['emitted_text'])
            advanced_snapshot = outcome.advanced_snapshot
        return WriterSnapshotAdvanceEnvelopeVerification(accepted=True, outcome_kind=str(outcome_kind), source_snapshot=source_snapshot, advanced_snapshot=advanced_snapshot)
    except WriterEnvelopeWorkExceeded as exc:
        return WriterSnapshotAdvanceEnvelopeVerification(accepted=False, outcome_kind=envelope.get('outcome_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', source_snapshot=None, advanced_snapshot=None, reason=writer_envelope_work_reason(exc))
    except SouthStarError as exc:
        return WriterSnapshotAdvanceEnvelopeVerification(accepted=False, outcome_kind=envelope.get('outcome_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', source_snapshot=None, advanced_snapshot=None, reason=exc.args[-1] if exc.args else 'verification_error')
    except (KeyError, TypeError, ValueError) as exc:
        return WriterSnapshotAdvanceEnvelopeVerification(accepted=False, outcome_kind=envelope.get('outcome_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', source_snapshot=None, advanced_snapshot=None, reason=f'malformed_envelope:{type(exc).__name__}')

def _verify_writer_snapshot_advance_envelope_from_known_source(*, prepared: SouthStarPreparedMol, source_snapshot, envelope: object, budget: WriterEnvelopeWorkBudget | None=None) -> WriterSnapshotAdvanceEnvelopeVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        _validate_envelope_shape(envelope)
        assert isinstance(envelope, Mapping)
        outcome_kind = envelope['outcome_kind']
        if envelope['source_snapshot'] != _snapshot_identity_envelope(source_snapshot, budget=budget, operation='envelope.identity'):
            _envelope_violation('known_source_snapshot_mismatch')
        _assert_prepared_identity_matches(prepared, envelope, budget=budget)
        expected = writer_snapshot_advance_envelope_for_emitted_text(prepared=prepared, snapshot=source_snapshot, emitted_text=envelope['emitted_text'], budget=budget)
        if expected != envelope:
            return WriterSnapshotAdvanceEnvelopeVerification(accepted=False, outcome_kind=str(outcome_kind), source_snapshot=source_snapshot, advanced_snapshot=None, reason='envelope_terms_mismatch')
        advanced_snapshot = None
        if outcome_kind == 'advanced':
            outcome = _writer_snapshot_advance_outcome_by_emitted_text(source_snapshot, prepared=prepared, emitted_text=envelope['emitted_text'])
            advanced_snapshot = outcome.advanced_snapshot
        return WriterSnapshotAdvanceEnvelopeVerification(accepted=True, outcome_kind=str(outcome_kind), source_snapshot=source_snapshot, advanced_snapshot=advanced_snapshot)
    except WriterEnvelopeWorkExceeded as exc:
        return WriterSnapshotAdvanceEnvelopeVerification(accepted=False, outcome_kind=envelope.get('outcome_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', source_snapshot=None, advanced_snapshot=None, reason=writer_envelope_work_reason(exc))
    except SouthStarError as exc:
        return WriterSnapshotAdvanceEnvelopeVerification(accepted=False, outcome_kind=envelope.get('outcome_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', source_snapshot=None, advanced_snapshot=None, reason=exc.args[-1] if exc.args else 'verification_error')
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterSnapshotAdvanceEnvelopeVerification(accepted=False, outcome_kind=envelope.get('outcome_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', source_snapshot=None, advanced_snapshot=None, reason=f'malformed_envelope:{type(exc).__name__}')

def _envelope_from_outcome(*, prepared, outcome, budget) -> dict[str, object]:
    source_snapshot = outcome.source_snapshot
    product_kind = 'blocked' if outcome.frontier_product.blocked else 'legal'
    envelope = {'schema_name': SCHEMA_NAME, 'schema_version': SCHEMA_VERSION, 'prepared_identity': _identity_envelope(source_snapshot.prepared_identity, budget=budget, operation='snapshot_advance.prepared_identity'), 'source_snapshot': _snapshot_identity_envelope(source_snapshot, budget=budget, operation='snapshot_advance.source_snapshot'), 'emitted_text': outcome.emitted_text, 'outcome_kind': outcome.kind.value, 'frontier_product_kind': product_kind, 'advance_certificate': _advance_certificate_envelope(outcome, budget=budget)}
    _validate_envelope_shape(envelope)
    _assert_prepared_identity_matches(prepared, envelope, budget=budget)
    return envelope

def _advance_certificate_envelope(outcome, *, budget) -> dict[str, object]:
    if outcome.kind.value == 'advanced':
        step = outcome.step_certificate
        return {'kind': 'advanced', 'frontier_projection': _frontier_projection_envelope(outcome.frontier_projection_certificate, budget=budget), 'selected_text_projection': _text_projection_envelope(outcome.text_projection_certificate, budget=budget), 'step_certificate': {'source_snapshot': _snapshot_identity_envelope(step.source_snapshot, budget=budget, operation='snapshot_advance.step.source_snapshot'), 'source_cursor': _cursor_envelope(step.source_cursor, budget=budget, operation='snapshot_advance.step.source_cursor'), 'successor_cursor': _cursor_envelope(step.successor_cursor, budget=budget, operation='snapshot_advance.step.successor_cursor'), 'advanced_snapshot': _snapshot_identity_envelope(step.advanced_snapshot, budget=budget, operation='snapshot_advance.step.advanced_snapshot'), 'decoder_boundary_before': _term(step.decoder_boundary_before), 'decoder_boundary_after': _term(step.decoder_boundary_after), 'frontier_projection_digest': _identity_digest(step.frontier_projection_certificate, budget=budget, operation='snapshot_advance.step.frontier_projection'), 'text_projection_digest': _identity_digest(step.text_projection_certificate, budget=budget, operation='snapshot_advance.step.text_projection'), 'branch_certificate_digests': [_identity_digest(certificate, budget=budget, operation='snapshot_advance.step.branch_certificate') for certificate in step.branch_certificates]}, 'advanced_snapshot': _snapshot_identity_envelope(outcome.advanced_snapshot, budget=budget, operation='snapshot_advance.advanced_snapshot')}
    if outcome.kind.value == 'invalid_emitted_text':
        projection = outcome.invalid_text_frontier_projection_certificate
        return {'kind': 'invalid_emitted_text', 'frontier_projection': _frontier_projection_envelope(projection, budget=budget), 'invalid_text_certificate': {'source_snapshot': _snapshot_identity_envelope(outcome.invalid_text_certificate.source_snapshot, budget=budget, operation='snapshot_advance.invalid.source_snapshot'), 'emitted_text': outcome.invalid_text_certificate.emitted_text, 'frontier_projection_digest': _identity_digest(projection, budget=budget, operation='snapshot_advance.invalid.frontier_projection')}, 'projected_emitted_texts': [item.emitted_text for item in projection.text_choice_projection_certificates], 'projected_text_projection_digests': [_identity_digest(item, budget=budget, operation='snapshot_advance.invalid.text_projection') for item in projection.text_choice_projection_certificates]}
    if outcome.kind.value == 'blocked':
        blocked = outcome.blocked_frontier_certificate
        diagnostic = blocked.diagnostic_certificate
        return {'kind': 'blocked', 'blocked_frontier_certificate': {'cursor': _cursor_envelope(blocked.cursor, budget=budget, operation='snapshot_advance.blocked.cursor'), 'blocked': blocked.blocked, 'diagnostic_certificate_digest': _identity_digest(diagnostic, budget=budget, operation='snapshot_advance.blocked.diagnostic')}, 'blocked_advance_certificate': {'source_snapshot': _snapshot_identity_envelope(outcome.blocked_advance_certificate.source_snapshot, budget=budget, operation='snapshot_advance.blocked.source_snapshot'), 'emitted_text': outcome.blocked_advance_certificate.emitted_text, 'blocked_frontier_certificate_digest': _identity_digest(blocked, budget=budget, operation='snapshot_advance.blocked.frontier_certificate')}, 'diagnostic_certificate': _diagnostic_envelope(diagnostic, budget=budget)}
    _envelope_violation('unknown_outcome_kind')

def _frontier_projection_envelope(projection, *, budget) -> dict[str, object]:
    return {'cursor': _cursor_envelope(projection.cursor, budget=budget, operation='snapshot_advance.frontier_projection.cursor'), 'text_projection_digests': [_identity_digest(item, budget=budget, operation='snapshot_advance.frontier_projection.text_projection') for item in projection.text_choice_projection_certificates], 'text_projection_keys': [_text_projection_key(item, budget=budget) for item in projection.text_choice_projection_certificates], 'terminal_projection_digest': None if projection.terminal_projection_certificate is None else _identity_digest(projection.terminal_projection_certificate, budget=budget, operation='snapshot_advance.frontier_projection.terminal_projection'), 'digest': _identity_digest(projection, budget=budget, operation='snapshot_advance.frontier_projection')}

def _text_projection_envelope(projection, *, budget) -> dict[str, object]:
    return {'source_cursor': _cursor_envelope(projection.source_cursor, budget=budget, operation='snapshot_advance.text_projection.source_cursor'), 'emitted_text': projection.emitted_text, 'successor_cursor': _cursor_envelope(projection.successor_cursor, budget=budget, operation='snapshot_advance.text_projection.successor_cursor'), 'immediate_multiplicity': projection.immediate_multiplicity, 'projection_key': _text_projection_key(projection, budget=budget), 'branch_certificate_digests': [_identity_digest(item, budget=budget, operation='snapshot_advance.text_projection.branch_certificate') for item in projection.branch_certificates], 'digest': _identity_digest(projection, budget=budget, operation='snapshot_advance.text_projection')}

def _diagnostic_envelope(diagnostic, *, budget) -> dict[str, object]:
    return {'cursor': _cursor_envelope(diagnostic.cursor, budget=budget, operation='snapshot_advance.diagnostic.cursor'), 'unsupported_execution_capabilities': [*sorted((_term(item.capability) for item in diagnostic.unsupported_execution_capability_certificates))], 'unsupported_terminal_execution_capabilities': [*sorted((_term(item.capability) for item in diagnostic.unsupported_terminal_execution_capability_certificates))], 'residual_work_envelope_violations': [_term(item.violation) for item in diagnostic.work_envelope_violation_certificates if item.category == 'residual_work'], 'terminal_residual_work_envelope_violations': [_term(item.violation) for item in diagnostic.work_envelope_violation_certificates if item.category == 'terminal_residual_work'], 'finite_relation_work_envelope_violations': [_term(item.violation) for item in diagnostic.work_envelope_violation_certificates if item.category == 'finite_relation_work'], 'graph_obligation_work_envelope_violations': [_term(item.violation) for item in diagnostic.work_envelope_violation_certificates if item.category == 'graph_obligation'], 'digest': _identity_digest(diagnostic, budget=budget, operation='snapshot_advance.diagnostic')}

def _source_snapshot_from_envelope(*, prepared, envelope, budget: WriterEnvelopeWorkBudget | None=None) -> object:
    budget = default_writer_envelope_work_budget(budget)
    _assert_prepared_identity_matches(prepared, envelope, budget=budget)
    snapshot_terms = envelope['source_snapshot']
    runtime_options = _runtime_options_from_terms(snapshot_terms['runtime_options'])
    cursor_digest = snapshot_terms['cursor']['digest']
    expected_boundary = snapshot_terms['decoder_boundary']
    expected_depth = expected_boundary['consumed_token_count']
    positions = 0
    for cursor, depth in _reachable_snapshot_positions(prepared, runtime_options, budget=budget):
        positions += 1
        if depth != expected_depth:
            continue
        snapshot = _capture_writer_frontier_snapshot_unchecked(prepared=prepared, runtime_options=runtime_options, cursor=cursor, decoder_boundary=WriterDecoderBoundary(consumed_token_count=expected_depth))
        if _cursor_envelope(snapshot.cursor, budget=budget, operation='envelope.identity')['digest'] == cursor_digest:
            if _snapshot_identity_envelope(snapshot, budget=budget, operation='envelope.identity') != snapshot_terms:
                _envelope_violation('source_snapshot_identity_mismatch')
            return snapshot
    _envelope_violation('source_snapshot_position_not_reachable')

def _reachable_snapshot_positions(prepared, runtime_options, *, budget):
    pending = [(initial_writer_frontier_cursor(prepared, runtime_options), 0)]
    seen = set()
    while pending:
        cursor, depth = pending.pop(0)
        cursor_digest = _cursor_envelope(cursor, budget=budget, operation='envelope.identity')['digest']
        key = (cursor_digest, depth)
        if key in seen:
            continue
        seen.add(key)
        check_writer_envelope_work(budget=budget, operation='source_snapshot_lookup', metric='source_lookup_positions', actual=len(seen), limit=budget.max_source_lookup_positions)
        yield (cursor, depth)
        product = _snapshot_advance_writer_frontier_product(prepared, cursor)
        if product.blocked:
            continue
        for projection in product.projection_certificate.text_choice_projection_certificates:
            pending.append((projection.successor_cursor, depth + 1))

def _assert_prepared_identity_matches(prepared, envelope, *, budget) -> None:
    runtime_options = _runtime_options_from_terms(envelope['source_snapshot']['runtime_options'])
    identity = envelope['prepared_identity']
    actual = _identity_envelope(_prepared_identity(prepared, runtime_options), budget=budget, operation='envelope.identity')
    if identity != actual:
        _envelope_violation('prepared_identity_mismatch')
    if envelope['source_snapshot']['prepared_identity_digest'] != actual['digest']:
        _envelope_violation('source_snapshot_prepared_identity_mismatch')

def _text_projection_key(projection, *, budget) -> dict[str, object]:
    return {'source_cursor_digest': _identity_digest(projection.source_cursor, budget=budget, operation='envelope.identity'), 'emitted_text': projection.emitted_text, 'successor_cursor_digest': _identity_digest(projection.successor_cursor, budget=budget, operation='envelope.identity'), 'immediate_multiplicity': projection.immediate_multiplicity}

def _validate_envelope_shape(envelope: object) -> None:
    if not isinstance(envelope, Mapping):
        _envelope_violation('envelope_not_mapping')
    keys = frozenset(envelope)
    if keys != _TOP_LEVEL_FIELDS:
        _envelope_violation('top_level_fields_mismatch')
    if envelope['schema_name'] != SCHEMA_NAME:
        _envelope_violation('unknown_schema_name')
    if envelope['schema_version'] != SCHEMA_VERSION:
        _envelope_violation('unknown_schema_version')
    if envelope['outcome_kind'] not in _OUTCOME_KINDS:
        _envelope_violation('unknown_outcome_kind')
    if envelope['frontier_product_kind'] not in _PRODUCT_KINDS:
        _envelope_violation('unknown_frontier_product_kind')
    certificate = envelope['advance_certificate']
    if not isinstance(certificate, Mapping):
        _envelope_violation('advance_certificate_not_mapping')
    if certificate.get('kind') != envelope['outcome_kind']:
        _envelope_violation('advance_certificate_kind_mismatch')
    if envelope['outcome_kind'] == 'blocked':
        if envelope['frontier_product_kind'] != 'blocked':
            _envelope_violation('blocked_product_kind_mismatch')
    elif envelope['frontier_product_kind'] != 'legal':
        _envelope_violation('legal_product_kind_mismatch')

def _envelope_violation(kind: str) -> None:
    raise SouthStarError(SouthStarErrorKind.INTERNAL_INVARIANT, f'writer snapshot advance envelope violation: {kind}')
__all__ = ('SCHEMA_NAME', 'SCHEMA_VERSION', 'WriterSnapshotAdvanceEnvelopeVerification', 'verify_writer_snapshot_advance_envelope', 'writer_snapshot_advance_envelope_for_emitted_text')
