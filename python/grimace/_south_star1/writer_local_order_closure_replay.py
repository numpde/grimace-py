"""Producer-free local-order closure replay shared by DOT and EOS proofs."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from typing import Literal

from .facts import LigandKind
from .facts import SiteStatus
from .ids import AtomId
from .residual_constraints import ResidualFactorKey
from .residual_constraints import ResidualStore
from .residual_constraints import TetraLocalParity
from .residual_constraints import tetra_parity_var
from .writer_capabilities import _WriterExecutionCapabilityKind
from .writer_events import WriterLocalOrderClosed
from .writer_execution_evidence import WriterResidualPropagationWorkEvidence
from .writer_residual_transition_terms import (
    TetraLocalOrderFactorClosureTransitionTerm,
)
from .writer_residual_transition_terms import WriterResidualTransitionKind
from .writer_stereo import WriterLocalOrderRecord
from .writer_stereo import WriterStereoLifecycleEvidence
from .writer_stereo import WriterStereoLifecycleOutcomeKind
from .writer_state import WriterStereoState


@dataclass(frozen=True, slots=True)
class WriterLocalOrderClosureReplay:
    kind: Literal["already_closed_noop", "record_only", "tetra_residual"]
    successor_stereo_state: WriterStereoState
    lifecycle: WriterStereoLifecycleEvidence | None
    residual_work: tuple[WriterResidualPropagationWorkEvidence, ...]
    capabilities: frozenset[_WriterExecutionCapabilityKind]
    semantically_replayed_operations: tuple[str, ...]


class WriterLocalOrderClosureReplayError(ValueError):
    """The supplied states or transition do not satisfy local-order closure."""


def replay_writer_local_order_closure_for_facts(
    *,
    facts,
    source_state,
    successor_state,
    atom: AtomId,
    transition_term: TetraLocalOrderFactorClosureTransitionTerm | None,
) -> WriterLocalOrderClosureReplay:
    if atom != source_state.active.atom:
        _fail("local_order_atom_mismatch")
    source_stereo = source_state.stereo_state
    successor_stereo = successor_state.stereo_state
    _validated_snapshot(source_stereo.residual_snapshot)
    _validated_snapshot(successor_stereo.residual_snapshot)

    source_record = _record_for_atom(source_stereo.local_orders, atom)
    if source_record is not None and source_record.closed:
        if transition_term is not None or successor_stereo != source_stereo:
            _fail("local_order_already_closed_mismatch")
        return WriterLocalOrderClosureReplay(
            kind="already_closed_noop",
            successor_stereo_state=source_stereo,
            lifecycle=None,
            residual_work=(),
            capabilities=frozenset(),
            semantically_replayed_operations=(),
        )

    expected_record = WriterLocalOrderRecord(
        atom=atom,
        order=_closed_order(facts=facts, atom=atom, source_record=source_record),
        closed=True,
    )
    expected_orders = _replace_record(source_stereo.local_orders, expected_record)
    specified_sites = tuple(
        site
        for site in facts.stereo.tetrahedral
        if site.center == atom and site.status is SiteStatus.SPECIFIED
    )
    if len(specified_sites) > 1:
        _fail("local_order_tetra_site_mismatch")

    if not specified_sites:
        if transition_term is not None:
            _fail("local_order_unexpected_transition")
        expected_stereo = replace(source_stereo, local_orders=expected_orders)
        if successor_stereo != expected_stereo:
            _fail("local_order_state_mismatch")
        lifecycle = WriterStereoLifecycleEvidence(
            event=WriterLocalOrderClosed(atom),
            source_residual_snapshot=source_stereo.residual_snapshot,
            successor_residual_snapshot=successor_stereo.residual_snapshot,
            source_atom_occurrences=source_stereo.atom_occurrences,
            successor_atom_occurrences=successor_stereo.atom_occurrences,
            source_bond_occurrences=source_stereo.bond_occurrences,
            successor_bond_occurrences=successor_stereo.bond_occurrences,
            source_local_orders=source_stereo.local_orders,
            successor_local_orders=successor_stereo.local_orders,
            capabilities=frozenset(),
            residual_work_evidence=(),
            outcome_kind=WriterStereoLifecycleOutcomeKind.EVENT_RECORDED,
        )
        return WriterLocalOrderClosureReplay(
            kind="record_only",
            successor_stereo_state=expected_stereo,
            lifecycle=lifecycle,
            residual_work=(),
            capabilities=frozenset(),
            semantically_replayed_operations=(),
        )

    if transition_term is None:
        _fail("local_order_transition_missing")
    site = specified_sites[0]
    term = transition_term
    if (
        term.kind is not WriterResidualTransitionKind.TETRA_LOCAL_ORDER_FACTOR_CLOSURE
        or term.source_snapshot != source_stereo.residual_snapshot
        or int(term.atom) != int(atom)
    ):
        _fail("local_order_transition_state_anchor_mismatch")
    if int(term.site) != int(site.id):
        _fail("local_order_tetra_site_mismatch")
    if tuple(term.reference_order) != tuple(site.reference_order):
        _fail("local_order_tetra_reference_order_mismatch")
    if tuple(term.local_order) != expected_record.order:
        _fail("local_order_tetra_local_order_mismatch")
    parity = _permutation_parity(
        reference_order=tuple(site.reference_order),
        local_order=term.local_order,
    )
    factor_key = ResidualFactorKey("tetra_site", (int(site.id),))
    if (
        term.target_parity is not parity
        or term.constraint_var != tetra_parity_var(site.id)
        or term.constraint_value is not parity
        or term.discharged_factor_keys != (factor_key,)
        or term.projected_variables != (term.constraint_var,)
    ):
        _fail("local_order_tetra_restriction_mismatch")
    store = ResidualStore.from_value_snapshot(source_stereo.residual_snapshot)
    result = store.restrict_many_and_propagate(((term.constraint_var, parity),))
    if (
        result != term.propagation_result
        or result.stats.component_variables != term.affected_variables
        or result.stats.component_factor_keys != term.affected_factor_keys
    ):
        _fail("local_order_tetra_propagation_mismatch")
    try:
        store.discharge_satisfied_factors(term.discharged_factor_keys)
    except ValueError as exc:
        raise WriterLocalOrderClosureReplayError(
            "local_order_tetra_discharge_mismatch"
        ) from exc
    successor_snapshot = store.value_snapshot()
    if successor_snapshot != term.successor_snapshot:
        _fail("local_order_tetra_successor_residual_mismatch")
    expected_stereo = replace(
        source_stereo,
        residual_snapshot=successor_snapshot,
        local_orders=expected_orders,
    )
    if successor_stereo != expected_stereo:
        _fail("local_order_state_mismatch")
    capabilities = frozenset((
        _WriterExecutionCapabilityKind.RESIDUAL_FACTOR_DISCHARGE,
        _WriterExecutionCapabilityKind.RESIDUAL_PROPAGATION,
        _WriterExecutionCapabilityKind.TETRA_LOCAL_ORDER_RESTRICTION,
    ))
    operation = "tetrahedral local-order factor closure"
    residual_work = (WriterResidualPropagationWorkEvidence(
        operation=operation,
        result_kind=result.kind,
        component_variables=result.stats.component_variables,
        component_factor_keys=result.stats.component_factor_keys,
        checked_candidate_rows=result.stats.checked_candidate_rows,
        largest_factor_scope=result.stats.largest_factor_scope,
        largest_candidate_row_count=result.stats.largest_candidate_row_count,
        transition_term=term,
    ),)
    lifecycle = WriterStereoLifecycleEvidence(
        event=WriterLocalOrderClosed(atom),
        source_residual_snapshot=source_stereo.residual_snapshot,
        successor_residual_snapshot=successor_snapshot,
        source_atom_occurrences=source_stereo.atom_occurrences,
        successor_atom_occurrences=successor_stereo.atom_occurrences,
        source_bond_occurrences=source_stereo.bond_occurrences,
        successor_bond_occurrences=successor_stereo.bond_occurrences,
        source_local_orders=source_stereo.local_orders,
        successor_local_orders=successor_stereo.local_orders,
        capabilities=capabilities,
        residual_work_evidence=residual_work,
        outcome_kind=WriterStereoLifecycleOutcomeKind.RECORD_AND_RESTRICT,
    )
    return WriterLocalOrderClosureReplay(
        kind="tetra_residual",
        successor_stereo_state=expected_stereo,
        lifecycle=lifecycle,
        residual_work=residual_work,
        capabilities=capabilities,
        semantically_replayed_operations=(operation,),
    )


def _validated_snapshot(snapshot) -> None:
    try:
        ResidualStore.from_value_snapshot(snapshot)
    except ValueError as exc:
        raise WriterLocalOrderClosureReplayError(
            "local_order_residual_snapshot_malformed"
        ) from exc


def _record_for_atom(records, atom):
    matches = tuple(record for record in records if record.atom == atom)
    if len(matches) > 1:
        _fail("local_order_record_duplicate")
    return None if not matches else matches[0]


def _closed_order(*, facts, atom, source_record):
    order = () if source_record is None else source_record.order
    implicit_h = tuple(
        occurrence.id
        for occurrence in facts.ligand_occurrences
        if occurrence.kind is LigandKind.IMPLICIT_H and occurrence.atom == atom
    )
    return order + tuple(item for item in implicit_h if item not in order)


def _replace_record(records, replacement):
    out = tuple(
        replacement if record.atom == replacement.atom else record
        for record in records
    )
    if not any(record.atom == replacement.atom for record in records):
        out += (replacement,)
    return tuple(sorted(out, key=lambda item: int(item.atom)))


def _permutation_parity(*, reference_order, local_order):
    if (
        len(reference_order) != len(local_order)
        or set(reference_order) != set(local_order)
    ):
        _fail("local_order_reference_order_mismatch")
    position = {item: index for index, item in enumerate(reference_order)}
    indices = tuple(position[item] for item in local_order)
    inversions = sum(
        1
        for left in range(len(indices))
        for right in range(left + 1, len(indices))
        if indices[left] > indices[right]
    )
    return TetraLocalParity.ODD if inversions % 2 else TetraLocalParity.EVEN


def _fail(reason):
    raise WriterLocalOrderClosureReplayError(reason)


__all__ = (
    "WriterLocalOrderClosureReplay",
    "WriterLocalOrderClosureReplayError",
    "replay_writer_local_order_closure_for_facts",
)
