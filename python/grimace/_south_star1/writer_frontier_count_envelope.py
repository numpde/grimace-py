"""Durable envelopes for checked writer frontier count certificates."""
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
from .writer_envelope_work import default_writer_envelope_work_budget
from .writer_envelope_work import writer_envelope_work_reason
from .writer_count_dag_envelope import count_dag_node_by_id
from .writer_count_dag_envelope import validate_writer_count_certificate_dag_envelope
from .writer_count_dag_envelope import writer_count_certificate_dag_envelope_for_product
from .writer_frontier import _checked_writer_frontier_product
from .writer_snapshot import _capture_writer_frontier_snapshot_unchecked
from .writer_snapshot import _prepared_identity
from .writer_snapshot_envelope import _source_snapshot_from_envelope
from .writer_snapshot_prefix_envelope import _branch_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import _terminal_projection_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import _terminal_support_identity_envelope_from_certificate
from .writer_snapshot_prefix_envelope import _text_projection_certificate_identity_envelope
from .writer_snapshot_prefix_envelope import _writer_frontier_product_identity_envelope
from .writer_snapshot_prefix_envelope import verify_writer_snapshot_prefix_read_envelope
SCHEMA_NAME = 'writer_frontier_count'
SCHEMA_VERSION = 1
_TOP_LEVEL_FIELDS = frozenset(('schema_name', 'schema_version', 'prepared_identity', 'source_kind', 'source_snapshot', 'prefix_read_envelope', 'frontier_snapshot', 'frontier_product', 'support_count', 'completion_count', 'count_dag', 'support_count_certificate', 'completion_count_certificate', 'choice_count_certificates', 'terminal_choice_count_certificate', 'coverage'))
_SOURCE_KINDS = frozenset(('snapshot', 'prefix_read'))

@dataclass(frozen=True, slots=True)
class WriterFrontierCountEnvelopeVerification:
    accepted: bool
    source_kind: str
    support_count: int | None
    completion_count: int | None
    frontier_snapshot: object | None
    reason: str | None = None

def writer_frontier_count_envelope_for_snapshot(*, prepared: SouthStarPreparedMol, snapshot, budget: WriterEnvelopeWorkBudget | None=None, count_dag_diagnostics=None) -> dict[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    product = _counted_frontier_product(prepared=prepared, snapshot=snapshot)
    envelope = _envelope_from_product(prepared=prepared, source_kind='snapshot', source_snapshot=snapshot, prefix_read_envelope=None, frontier_snapshot=snapshot, product=product, budget=budget, count_dag_diagnostics=count_dag_diagnostics)
    _validate_envelope_shape(envelope, budget=budget)
    _assert_prepared_identity_matches(prepared, envelope, budget=budget)
    return envelope

def writer_frontier_count_envelope_for_prefix_read(*, prepared: SouthStarPreparedMol, prefix_read_envelope: Mapping[str, object], budget: WriterEnvelopeWorkBudget | None=None, count_dag_diagnostics=None) -> dict[str, object]:
    budget = default_writer_envelope_work_budget(budget)
    verification = verify_writer_snapshot_prefix_read_envelope(prepared=prepared, envelope=prefix_read_envelope, budget=budget)
    if not verification.accepted:
        _count_envelope_violation('prefix_read_envelope_rejected')
    if verification.read_kind != 'readable':
        _count_envelope_violation('prefix_read_envelope_not_readable')
    if verification.final_snapshot is None:
        _count_envelope_violation('prefix_read_envelope_lacks_final_snapshot')
    product = _counted_frontier_product(prepared=prepared, snapshot=verification.final_snapshot)
    envelope = _envelope_from_product(prepared=prepared, source_kind='prefix_read', source_snapshot=None, prefix_read_envelope=prefix_read_envelope, frontier_snapshot=verification.final_snapshot, product=product, budget=budget, count_dag_diagnostics=count_dag_diagnostics)
    _validate_envelope_shape(envelope, budget=budget)
    _assert_prepared_identity_matches(prepared, envelope, budget=budget)
    return envelope

def verify_writer_frontier_count_envelope(*, prepared: SouthStarPreparedMol, envelope: object, budget: WriterEnvelopeWorkBudget | None=None) -> WriterFrontierCountEnvelopeVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        _validate_envelope_shape(envelope, budget=budget)
        assert isinstance(envelope, Mapping)
        _assert_prepared_identity_matches(prepared, envelope, budget=budget)
        source_kind = str(envelope['source_kind'])
        if source_kind == 'snapshot':
            frontier_snapshot = _source_snapshot_from_envelope(prepared=prepared, envelope=envelope, budget=budget)
        elif source_kind == 'prefix_read':
            prefix_envelope = envelope['prefix_read_envelope']
            verification = verify_writer_snapshot_prefix_read_envelope(prepared=prepared, envelope=prefix_envelope, budget=budget)
            if not verification.accepted:
                _count_envelope_violation('prefix_read_envelope_rejected')
            if verification.read_kind != 'readable':
                _count_envelope_violation('prefix_read_envelope_not_readable')
            frontier_snapshot = verification.final_snapshot
            if frontier_snapshot is None:
                _count_envelope_violation('prefix_read_envelope_lacks_final_snapshot')
        else:
            _count_envelope_violation('unknown_source_kind')
        product = _counted_frontier_product(prepared=prepared, snapshot=frontier_snapshot)
        _verify_writer_frontier_count_envelope_against_product(
            prepared=prepared,
            frontier_snapshot=frontier_snapshot,
            product=product,
            envelope=envelope,
            budget=budget,
        )
        return WriterFrontierCountEnvelopeVerification(accepted=True, source_kind=source_kind, support_count=envelope['support_count'], completion_count=envelope['completion_count'], frontier_snapshot=frontier_snapshot)
    except WriterEnvelopeWorkExceeded as exc:
        return WriterFrontierCountEnvelopeVerification(accepted=False, source_kind=envelope.get('source_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', support_count=None, completion_count=None, frontier_snapshot=None, reason=writer_envelope_work_reason(exc))
    except SouthStarError as exc:
        return WriterFrontierCountEnvelopeVerification(accepted=False, source_kind=envelope.get('source_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', support_count=None, completion_count=None, frontier_snapshot=None, reason=exc.args[-1] if exc.args else 'verification_error')
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterFrontierCountEnvelopeVerification(accepted=False, source_kind=envelope.get('source_kind', 'unknown') if isinstance(envelope, Mapping) else 'unknown', support_count=None, completion_count=None, frontier_snapshot=None, reason=f'malformed_envelope:{type(exc).__name__}')

def _verify_writer_frontier_count_envelope_against_product(*, prepared, frontier_snapshot, product, envelope, budget):
    """Verify count terms against a fresh product without rebuilding the DAG."""
    _validate_envelope_shape(envelope, budget=budget)
    assert isinstance(envelope, Mapping)
    _assert_prepared_identity_matches(prepared, envelope, budget=budget)
    nodes = count_dag_node_by_id(envelope['count_dag'])
    expected = {
        'schema_name': SCHEMA_NAME,
        'schema_version': SCHEMA_VERSION,
        'prepared_identity': _identity_envelope(frontier_snapshot.prepared_identity, budget=budget, operation='envelope.identity'),
        'source_kind': envelope['source_kind'],
        'source_snapshot': envelope['source_snapshot'],
        'prefix_read_envelope': envelope['prefix_read_envelope'],
        'frontier_snapshot': envelope['frontier_snapshot'],
        'frontier_product': _frontier_product_count_identity_envelope(product, count_dag=envelope['count_dag'], budget=budget),
        'support_count': product.support_count_certificate.support_count,
        'completion_count': product.count_certificate.completion_count,
        'count_dag': envelope['count_dag'],
        'support_count_certificate': nodes[envelope['count_dag']['roots']['support_count_root']],
        'completion_count_certificate': nodes[envelope['count_dag']['roots']['completion_count_root']],
        'choice_count_certificates': [nodes[node_id] for node_id in envelope['count_dag']['roots']['choice_count_roots']],
        'terminal_choice_count_certificate': None if envelope['count_dag']['roots']['terminal_choice_count_root'] is None else nodes[envelope['count_dag']['roots']['terminal_choice_count_root']],
        'coverage': _coverage_envelope(product, count_dag=envelope['count_dag'], budget=budget),
    }
    if expected != envelope:
        _count_envelope_violation('envelope_terms_mismatch')

def _counted_frontier_product(*, prepared, snapshot):
    return _checked_writer_frontier_product(prepared, snapshot.cursor, include_counts=True, include_frontier_certificate=True, include_count_certificate=True)

def _envelope_from_product(*, prepared, source_kind: str, source_snapshot, prefix_read_envelope, frontier_snapshot, product, budget, count_dag_diagnostics=None) -> dict[str, object]:
    if product.blocked:
        _count_envelope_violation('count_envelope_requires_legal_frontier')
    checked = product.checked_frontier_certificate
    if checked is None:
        _count_envelope_violation('missing_checked_frontier_certificate')
    support_count = product.support_count_certificate.support_count
    completion_count = product.count_certificate.completion_count
    count_dag = writer_count_certificate_dag_envelope_for_product(product, budget=budget, diagnostics=count_dag_diagnostics)
    nodes = count_dag_node_by_id(count_dag)
    return {'schema_name': SCHEMA_NAME, 'schema_version': SCHEMA_VERSION, 'prepared_identity': _identity_envelope(frontier_snapshot.prepared_identity, budget=budget, operation='envelope.identity'), 'source_kind': source_kind, 'source_snapshot': None if source_snapshot is None else _snapshot_identity_envelope(source_snapshot, budget=budget, operation='envelope.identity'), 'prefix_read_envelope': prefix_read_envelope, 'frontier_snapshot': _snapshot_identity_envelope(frontier_snapshot, budget=budget, operation='envelope.identity'), 'frontier_product': _frontier_product_count_identity_envelope(product, count_dag=count_dag, budget=budget), 'support_count': support_count, 'completion_count': completion_count, 'count_dag': count_dag, 'support_count_certificate': nodes[count_dag['roots']['support_count_root']], 'completion_count_certificate': nodes[count_dag['roots']['completion_count_root']], 'choice_count_certificates': [nodes[node_id] for node_id in count_dag['roots']['choice_count_roots']], 'terminal_choice_count_certificate': None if count_dag['roots']['terminal_choice_count_root'] is None else nodes[count_dag['roots']['terminal_choice_count_root']], 'coverage': _coverage_envelope(product, count_dag=count_dag, budget=budget)}

def _frontier_product_count_identity_envelope(product, *, count_dag, budget):
    if product.blocked:
        _count_envelope_violation('count_product_identity_requires_legal_frontier')
    envelope = {'kind': 'legal', 'cursor': _cursor_envelope(product.cursor, budget=budget, operation='envelope.identity'), 'frontier_projection_certificate': product.projection_certificate and {'cursor': _cursor_envelope(product.projection_certificate.cursor, budget=budget, operation='envelope.identity'), 'text_projection_digests': [_text_projection_certificate_identity_envelope(projection, budget=budget)['digest'] for projection in product.projection_certificate.text_choice_projection_certificates], 'terminal_projection_digest': None if product.terminal_projection_certificate is None else _terminal_projection_certificate_identity_envelope(product.terminal_projection_certificate, budget=budget)['digest'], 'digest': _identity_digest(product.projection_certificate, budget=budget, operation='envelope.identity')}, 'text_projection_certificates': [_text_projection_certificate_identity_envelope(projection, budget=budget) for projection in product.text_choice_projection_certificates], 'terminal_projection_certificate': _terminal_projection_certificate_identity_envelope(product.terminal_projection_certificate, budget=budget), 'checked_frontier_certificate': {'cursor': _cursor_envelope(product.checked_frontier_certificate.cursor, budget=budget, operation='envelope.identity'), 'projection_certificate_digest': _identity_digest(product.checked_frontier_certificate.projection_certificate, budget=budget, operation='frontier_count.checked_frontier.projection_certificate'), 'support_count_certificate_node_id': count_dag['roots']['support_count_root'], 'completion_count_certificate_node_id': count_dag['roots']['completion_count_root'], 'support_count': product.support_count_certificate.support_count, 'completion_count': product.count_certificate.completion_count}, 'support_count_certificate_node_id': count_dag['roots']['support_count_root'], 'completion_count_certificate_node_id': count_dag['roots']['completion_count_root'], 'choice_count_certificate_node_ids': count_dag['roots']['choice_count_roots'], 'terminal_choice_count_certificate_node_id': count_dag['roots']['terminal_choice_count_root'], 'count_dag_digest': count_dag['digest'], 'branch_support_identities': [_branch_certificate_identity_envelope(certificate, budget=budget) for certificate in product.projection_certificate.branch_certificates], 'terminal_support_identities': [_terminal_support_identity_envelope_from_certificate(certificate, budget=budget) for certificate in product.projection_certificate.terminal_certificates]}
    envelope['checked_frontier_certificate']['digest'] = _identity_digest(envelope['checked_frontier_certificate'], budget=budget, operation='frontier_count.checked_frontier.digest')
    envelope['digest'] = _identity_digest(envelope, budget=budget, operation='envelope.identity')
    return envelope

def _coverage_envelope(product, *, count_dag, budget):
    checked = product.checked_frontier_certificate
    support_coverage = checked.support_count_term_coverage_certificate
    completion_aggregate = checked.frontier_completion_count_certificate
    completion_coverage = completion_aggregate.term_coverage_certificate
    choice_coverage = checked.choice_count_coverage_certificate
    nodes = count_dag_node_by_id(count_dag)
    choice_node_by_projection = {node['text_projection']['digest']: node for node in nodes.values() if node['kind'] == 'writer_text_choice_count'}
    branch_term_node_by_key = {(node['branch_certificate']['digest'], node['successor_count']): node for node in nodes.values() if node['kind'] == 'writer_branch_completion_term'}

    def node_digest(node_id):
        return None if node_id is None else nodes[node_id]['digest']
    envelope = {'frontier_projection_digest': product.checked_frontier_certificate.choice_count_coverage_certificate and _frontier_product_count_identity_envelope(product, count_dag=count_dag, budget=budget)['frontier_projection_certificate']['digest'], 'terminal_covered': product.terminal_projection_certificate is not None, 'text_choices_covered': [_text_choice_coverage_envelope(term, choice_node_by_projection, nodes, budget=budget) for term in choice_coverage.text_choice_terms], 'terminal_choice_coverage': None if choice_coverage.terminal_choice_term is None else {'terminal_choice_count_node_id': count_dag['roots']['terminal_choice_count_root'], 'terminal_choice_count_node_digest': node_digest(count_dag['roots']['terminal_choice_count_root']), 'terminal_projection_digest': _terminal_projection_certificate_identity_envelope(choice_coverage.terminal_choice_term.terminal_projection_certificate, budget=budget)['digest'], 'terminal_support_identities': [_terminal_support_identity_envelope_from_certificate(certificate, budget=budget) for certificate in choice_coverage.terminal_choice_term.terminal_projection_certificate.terminal_certificates], 'support_count': choice_coverage.terminal_choice_term.support_count, 'completion_count': choice_coverage.terminal_choice_term.completion_count}, 'branch_terms_covered': [_branch_term_coverage_envelope(term, branch_term_node_by_key, nodes, budget=budget) for term in completion_coverage.branch_terms], 'support_text_term_count': len(support_coverage.text_terms), 'support_terminal_covered': support_coverage.terminal_term is not None, 'completion_branch_term_count': len(completion_coverage.branch_terms), 'completion_terminal_term_count': len(completion_coverage.terminal_terms), 'support_count_total': support_coverage.support_count, 'completion_count_total': completion_coverage.completion_count}
    envelope['digest'] = _identity_digest(envelope, budget=budget, operation='envelope.identity')
    return envelope

def _text_choice_coverage_envelope(term, choice_node_by_projection, nodes, *, budget):
    projection = _text_projection_certificate_identity_envelope(term.text_projection_certificate, budget=budget)
    choice_node = choice_node_by_projection[projection['digest']]
    support_node_id = choice_node['support_count_node_id']
    completion_node_id = choice_node['completion_count_node_id']
    return {'emitted_text': term.text_projection_certificate.emitted_text, 'projection_digest': projection['digest'], 'support_count': term.support_count, 'completion_count': term.completion_count, 'successor_support_count_digest': nodes[support_node_id]['digest'], 'successor_support_count_node_id': support_node_id, 'completion_count_node_id': completion_node_id, 'completion_count_node_digest': nodes[completion_node_id]['digest'], 'completion_branch_digests': [_branch_certificate_identity_envelope(completion_term.projection_branch_certificate, budget=budget)['digest'] for completion_term in term.completion_coverage_terms]}

def _branch_term_coverage_envelope(term, branch_term_node_by_key, nodes, *, budget):
    branch = _branch_certificate_identity_envelope(term.projection_branch_certificate, budget=budget)
    count_branch = _branch_certificate_identity_envelope(term.count_branch_term_certificate.branch_certificate, budget=budget)
    node = branch_term_node_by_key.get((count_branch['digest'], term.successor_completion_count))
    if node is None:
        _count_envelope_violation('branch_completion_node_match_mismatch')
    successor_node_id = node['successor_count_node_id']
    return {'branch_support_identity': branch, 'count_branch_support_identity': count_branch, 'successor_count_certificate_digest': nodes[successor_node_id]['digest'], 'successor_count_certificate_node_id': successor_node_id, 'successor_completion_count': term.successor_completion_count, 'weighted_completion_count': term.weighted_completion_count}

def _validate_envelope_shape(envelope: object, *, budget: WriterEnvelopeWorkBudget | None=None) -> None:
    if not isinstance(envelope, Mapping):
        _count_envelope_violation('envelope_not_mapping')
    if frozenset(envelope) != _TOP_LEVEL_FIELDS:
        _count_envelope_violation('top_level_fields_mismatch')
    if envelope['schema_name'] != SCHEMA_NAME:
        _count_envelope_violation('unknown_schema_name')
    if envelope['schema_version'] != SCHEMA_VERSION:
        _count_envelope_violation('unknown_schema_version')
    if envelope['source_kind'] not in _SOURCE_KINDS:
        _count_envelope_violation('unknown_source_kind')
    if envelope['source_kind'] == 'snapshot':
        if envelope['source_snapshot'] is None:
            _count_envelope_violation('snapshot_source_missing_snapshot')
        if envelope['prefix_read_envelope'] is not None:
            _count_envelope_violation('snapshot_source_has_prefix_envelope')
    else:
        if envelope['source_snapshot'] is not None:
            _count_envelope_violation('prefix_source_has_source_snapshot')
        if envelope['prefix_read_envelope'] is None:
            _count_envelope_violation('prefix_source_missing_prefix_envelope')
    try:
        validate_writer_count_certificate_dag_envelope(envelope['count_dag'], budget=budget)
    except ValueError as exc:
        _count_envelope_violation(str(exc))

def _assert_prepared_identity_matches(prepared, envelope, *, budget) -> None:
    snapshot_terms = envelope['source_snapshot'] if envelope['source_kind'] == 'snapshot' else envelope['frontier_snapshot']
    runtime_options = _runtime_options_from_terms(snapshot_terms['runtime_options'])
    actual = _identity_envelope(_prepared_identity(prepared, runtime_options), budget=budget, operation='envelope.identity')
    if envelope['prepared_identity'] != actual:
        _count_envelope_violation('prepared_identity_mismatch')
    if snapshot_terms['prepared_identity_digest'] != actual['digest']:
        _count_envelope_violation('snapshot_prepared_identity_mismatch')

def _count_envelope_violation(kind: str) -> None:
    raise SouthStarError(SouthStarErrorKind.INTERNAL_INVARIANT, f'writer frontier count envelope violation: {kind}')
__all__ = ('SCHEMA_NAME', 'SCHEMA_VERSION', 'WriterFrontierCountEnvelopeVerification', 'verify_writer_frontier_count_envelope', 'writer_frontier_count_envelope_for_prefix_read', 'writer_frontier_count_envelope_for_snapshot')
