"""Typed proof terms for exact writer residual transitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import SiteStatus
from .policy import TetraToken
from .residual_constraints import ResidualFactorKey
from .residual_constraints import ResidualPropagationKind
from .residual_constraints import ResidualStore
from .residual_constraints import ResidualStoreValueSnapshot
from .residual_constraints import TetraLocalParity
from .residual_constraints import TetraTokenParityFactorValueSnapshot
from .residual_constraints import VarId
from .residual_constraints import tetra_parity_var
from .residual_constraints import tetra_token_var
from .writer_capabilities import _WriterExecutionCapabilityKind
from .writer_events import WriterAtomEmitted
from .writer_events import WriterLocalOrderClosed
from .writer_execution_evidence import WriterResidualPropagationWorkEvidence
from .writer_execution_evidence import writer_residual_propagation_work_evidence
from .writer_stereo_branch_certificates import WriterStereoBranchCertificate
from .writer_stereo_branch_certificates import WriterStereoBranchCertificateKind

if TYPE_CHECKING:
    from .prepared_runtime import SouthStarPreparedMol
    from .stereo_templates import TetraTemplate


class WriterResidualTransitionKind(Enum):
    TETRA_TOKEN_RESTRICTION = "tetra_token_restriction"
    TETRA_LOCAL_ORDER_RESTRICTION = "tetra_local_order_restriction"


@dataclass(frozen=True, slots=True)
class WriterResidualTransitionTerm:
    kind: WriterResidualTransitionKind
    operation: str
    capability: _WriterExecutionCapabilityKind
    event: object
    restrictions: tuple[tuple[VarId, object], ...]
    discharged_factor_keys: tuple[ResidualFactorKey, ...]
    source_residual_snapshot: ResidualStoreValueSnapshot
    successor_residual_snapshot: ResidualStoreValueSnapshot
    execution_capabilities: frozenset[object]
    work_evidence: WriterResidualPropagationWorkEvidence


@dataclass(frozen=True, slots=True)
class _TetraTransitionSpec:
    transition_kind: WriterResidualTransitionKind
    certificate_kind: WriterStereoBranchCertificateKind
    capability: _WriterExecutionCapabilityKind
    operation: str
    event_type: type


_TETRA_TOKEN_SPEC = _TetraTransitionSpec(
    transition_kind=WriterResidualTransitionKind.TETRA_TOKEN_RESTRICTION,
    certificate_kind=WriterStereoBranchCertificateKind.TETRA_TOKEN_RESTRICTED,
    capability=_WriterExecutionCapabilityKind.TETRA_TOKEN_RESTRICTION,
    operation="tetrahedral atom-token restriction",
    event_type=WriterAtomEmitted,
)
_TETRA_LOCAL_ORDER_SPEC = _TetraTransitionSpec(
    transition_kind=WriterResidualTransitionKind.TETRA_LOCAL_ORDER_RESTRICTION,
    certificate_kind=(
        WriterStereoBranchCertificateKind.TETRA_LOCAL_ORDER_RESTRICTED
    ),
    capability=_WriterExecutionCapabilityKind.TETRA_LOCAL_ORDER_RESTRICTION,
    operation="tetrahedral local-order factor closure",
    event_type=WriterLocalOrderClosed,
)


def writer_residual_transition_term(
    *,
    prepared: SouthStarPreparedMol,
    certificate: WriterStereoBranchCertificate,
) -> WriterResidualTransitionTerm:
    if certificate.kind is _TETRA_TOKEN_SPEC.certificate_kind:
        return _tetra_token_transition_term(
            prepared=prepared,
            certificate=certificate,
        )
    if certificate.kind is _TETRA_LOCAL_ORDER_SPEC.certificate_kind:
        return _tetra_local_order_transition_term(
            prepared=prepared,
            certificate=certificate,
        )
    _violation(f"unsupported_certificate_kind:{certificate.kind.value}")


def verify_writer_residual_transition_term(
    term: WriterResidualTransitionTerm,
) -> None:
    spec = _spec_for_transition_kind(term.kind)
    if term.operation != spec.operation:
        _violation("operation_mismatch")
    if term.capability is not spec.capability:
        _violation("capability_mismatch")
    if not isinstance(term.event, spec.event_type):
        _violation("event_kind_mismatch")
    if spec.capability not in term.execution_capabilities:
        _violation("execution_capability_missing")
    if (
        _WriterExecutionCapabilityKind.RESIDUAL_PROPAGATION
        not in term.execution_capabilities
    ):
        _violation("residual_propagation_capability_missing")
    if not term.restrictions:
        _violation("restriction_missing")
    if len({var for var, _value in term.restrictions}) != len(term.restrictions):
        _violation("restriction_variable_duplicate")
    if term.source_residual_snapshot == term.successor_residual_snapshot:
        _violation("residual_snapshot_unchanged")
    if term.work_evidence.operation != term.operation:
        _violation("work_operation_mismatch")
    if (
        term.work_evidence.result_kind
        is not ResidualPropagationKind.CERTIFIED_CONSISTENT
    ):
        _violation("work_result_not_certified")

    store = ResidualStore.from_value_snapshot(term.source_residual_snapshot)
    result = store.restrict_many_and_propagate(term.restrictions)
    if result.kind is not ResidualPropagationKind.CERTIFIED_CONSISTENT:
        _violation("replayed_restriction_not_certified")
    replayed_work = writer_residual_propagation_work_evidence(
        operation=term.operation,
        result=result,
    )
    if replayed_work != term.work_evidence:
        _violation("replayed_work_evidence_mismatch")
    try:
        store.discharge_satisfied_factors(term.discharged_factor_keys)
    except ValueError:
        _violation("replayed_factor_discharge_mismatch")
    if store.value_snapshot() != term.successor_residual_snapshot:
        _violation("replayed_successor_snapshot_mismatch")


def _tetra_token_transition_term(
    *,
    prepared: SouthStarPreparedMol,
    certificate: WriterStereoBranchCertificate,
) -> WriterResidualTransitionTerm:
    lifecycle, work = _certificate_lifecycle_and_work(
        certificate=certificate,
        spec=_TETRA_TOKEN_SPEC,
    )
    event = certificate.event
    assert isinstance(event, WriterAtomEmitted)
    if event.tetra_token not in (TetraToken.AT, TetraToken.ATAT):
        _violation("tetra_token_missing")
    template = _specified_tetra_template(prepared, event.atom)
    factor_key = _validate_tetra_source_factor(
        lifecycle.source_residual_snapshot,
        template,
    )
    token_var = tetra_token_var(template.site)
    parity_var = tetra_parity_var(template.site)
    _require_exact_work_component(
        work=work,
        variables=(token_var, parity_var),
        factor_keys=(factor_key,),
    )
    _validate_atom_occurrence_delta(lifecycle=lifecycle, event=event)
    term = WriterResidualTransitionTerm(
        kind=_TETRA_TOKEN_SPEC.transition_kind,
        operation=_TETRA_TOKEN_SPEC.operation,
        capability=_TETRA_TOKEN_SPEC.capability,
        event=event,
        restrictions=((token_var, event.tetra_token),),
        discharged_factor_keys=(),
        source_residual_snapshot=lifecycle.source_residual_snapshot,
        successor_residual_snapshot=lifecycle.successor_residual_snapshot,
        execution_capabilities=frozenset(lifecycle.capabilities),
        work_evidence=work,
    )
    verify_writer_residual_transition_term(term)
    return term


def _tetra_local_order_transition_term(
    *,
    prepared: SouthStarPreparedMol,
    certificate: WriterStereoBranchCertificate,
) -> WriterResidualTransitionTerm:
    lifecycle, work = _certificate_lifecycle_and_work(
        certificate=certificate,
        spec=_TETRA_LOCAL_ORDER_SPEC,
    )
    event = certificate.event
    assert isinstance(event, WriterLocalOrderClosed)
    template = _specified_tetra_template(prepared, event.atom)
    factor_key = _validate_tetra_source_factor(
        lifecycle.source_residual_snapshot,
        template,
    )
    token_var = tetra_token_var(template.site)
    parity_var = tetra_parity_var(template.site)
    _require_exact_work_component(
        work=work,
        variables=(token_var, parity_var),
        factor_keys=(factor_key,),
    )
    order = _validate_local_order_delta(lifecycle=lifecycle, atom=event.atom)
    parity = _tetra_local_parity(template.reference_order, order)
    term = WriterResidualTransitionTerm(
        kind=_TETRA_LOCAL_ORDER_SPEC.transition_kind,
        operation=_TETRA_LOCAL_ORDER_SPEC.operation,
        capability=_TETRA_LOCAL_ORDER_SPEC.capability,
        event=event,
        restrictions=((parity_var, parity),),
        discharged_factor_keys=(factor_key,),
        source_residual_snapshot=lifecycle.source_residual_snapshot,
        successor_residual_snapshot=lifecycle.successor_residual_snapshot,
        execution_capabilities=frozenset(lifecycle.capabilities),
        work_evidence=work,
    )
    verify_writer_residual_transition_term(term)
    return term


def _certificate_lifecycle_and_work(
    *,
    certificate: WriterStereoBranchCertificate,
    spec: _TetraTransitionSpec,
):
    if certificate.kind is not spec.certificate_kind:
        _violation("certificate_kind_mismatch")
    if certificate.capability is not spec.capability:
        _violation("certificate_capability_mismatch")
    if not isinstance(certificate.event, spec.event_type):
        _violation("certificate_event_kind_mismatch")
    lifecycle = certificate.lifecycle_evidence
    if lifecycle.event != certificate.event:
        _violation("certificate_lifecycle_event_mismatch")
    if spec.capability not in lifecycle.capabilities:
        _violation("lifecycle_capability_missing")
    if tuple(certificate.residual_work_evidence) != tuple(
        lifecycle.residual_work_evidence
    ):
        _violation("certificate_lifecycle_work_mismatch")
    matches = tuple(
        evidence
        for evidence in certificate.residual_work_evidence
        if evidence.operation == spec.operation
    )
    if len(matches) != 1:
        _violation("certificate_work_evidence_not_unique")
    work = matches[0]
    if not isinstance(work, WriterResidualPropagationWorkEvidence):
        _violation("certificate_work_evidence_type_mismatch")
    if not isinstance(lifecycle.source_residual_snapshot, ResidualStoreValueSnapshot):
        _violation("source_residual_snapshot_type_mismatch")
    if not isinstance(
        lifecycle.successor_residual_snapshot,
        ResidualStoreValueSnapshot,
    ):
        _violation("successor_residual_snapshot_type_mismatch")
    return lifecycle, work


def _specified_tetra_template(
    prepared: SouthStarPreparedMol,
    atom,
) -> TetraTemplate:
    matches = tuple(
        template
        for template in prepared.tetra_templates
        if template.center == atom and template.status is SiteStatus.SPECIFIED
    )
    if len(matches) != 1:
        _violation("specified_tetra_template_not_unique")
    return matches[0]


def _validate_tetra_source_factor(
    snapshot: ResidualStoreValueSnapshot,
    template: TetraTemplate,
) -> ResidualFactorKey:
    factor_key = ResidualFactorKey("tetra_site", (int(template.site),))
    matches = tuple(
        factor
        for factor in snapshot.factors
        if getattr(factor, "key", None) == factor_key
    )
    if len(matches) != 1:
        _violation("tetra_source_factor_not_unique")
    factor = matches[0]
    if not isinstance(factor, TetraTokenParityFactorValueSnapshot):
        _violation("tetra_source_factor_type_mismatch")
    if factor.scope != (
        tetra_token_var(template.site),
        tetra_parity_var(template.site),
    ):
        _violation("tetra_source_factor_scope_mismatch")
    if factor.status is not template.status or factor.target is not template.target:
        _violation("tetra_source_factor_semantics_mismatch")
    return factor_key


def _require_exact_work_component(
    *,
    work: WriterResidualPropagationWorkEvidence,
    variables: tuple[VarId, ...],
    factor_keys: tuple[ResidualFactorKey, ...],
) -> None:
    if frozenset(work.component_variables) != frozenset(variables):
        _violation("residual_work_component_variables_mismatch")
    if frozenset(work.component_factor_keys) != frozenset(factor_keys):
        _violation("residual_work_component_factors_mismatch")


def _validate_atom_occurrence_delta(*, lifecycle, event: WriterAtomEmitted) -> None:
    source = tuple(lifecycle.source_atom_occurrences)
    successor = tuple(lifecycle.successor_atom_occurrences)
    if len(successor) != len(source) + 1 or successor[:-1] != source:
        _violation("tetra_atom_occurrence_delta_mismatch")
    record = successor[-1]
    if record.atom != event.atom or record.token is not event.tetra_token:
        _violation("tetra_atom_occurrence_record_mismatch")


def _validate_local_order_delta(*, lifecycle, atom) -> tuple[object, ...]:
    source = {record.atom: record for record in lifecycle.source_local_orders}
    successor = {record.atom: record for record in lifecycle.successor_local_orders}
    if len(source) != len(lifecycle.source_local_orders):
        _violation("source_local_order_duplicate")
    if len(successor) != len(lifecycle.successor_local_orders):
        _violation("successor_local_order_duplicate")
    if set(successor) - set(source) - {atom}:
        _violation("local_order_unrelated_record_created")
    for other_atom in set(source) | set(successor):
        if other_atom == atom:
            continue
        if source.get(other_atom) != successor.get(other_atom):
            _violation("local_order_unrelated_record_changed")
    source_record = source.get(atom)
    if source_record is not None and source_record.closed:
        _violation("local_order_source_already_closed")
    successor_record = successor.get(atom)
    if successor_record is None or not successor_record.closed:
        _violation("local_order_successor_not_closed")
    return tuple(successor_record.order)


def _tetra_local_parity(
    reference_order: tuple[object, ...],
    local_order: tuple[object, ...],
) -> TetraLocalParity:
    if len(reference_order) != len(local_order):
        _violation("tetra_local_order_length_mismatch")
    if set(reference_order) != set(local_order):
        _violation("tetra_local_order_domain_mismatch")
    positions = {item: index for index, item in enumerate(reference_order)}
    indices = tuple(positions[item] for item in local_order)
    inversions = sum(
        1
        for index, left in enumerate(indices)
        for right in indices[index + 1 :]
        if left > right
    )
    return TetraLocalParity.EVEN if inversions % 2 == 0 else TetraLocalParity.ODD


def _spec_for_transition_kind(
    kind: WriterResidualTransitionKind,
) -> _TetraTransitionSpec:
    if kind is WriterResidualTransitionKind.TETRA_TOKEN_RESTRICTION:
        return _TETRA_TOKEN_SPEC
    if kind is WriterResidualTransitionKind.TETRA_LOCAL_ORDER_RESTRICTION:
        return _TETRA_LOCAL_ORDER_SPEC
    _violation(f"unsupported_transition_kind:{kind.value}")


def _violation(kind: str) -> None:
    raise SouthStarError(
        SouthStarErrorKind.INTERNAL_INVARIANT,
        f"writer residual transition term violation: {kind}",
    )


__all__ = (
    "WriterResidualTransitionKind",
    "WriterResidualTransitionTerm",
    "verify_writer_residual_transition_term",
    "writer_residual_transition_term",
)
