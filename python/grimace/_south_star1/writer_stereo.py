"""Writer-owned residual stereo state advancement."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import LigandKind
from .facts import SiteStatus
from .writer_capabilities import _WriterExecutionCapabilityKind
from .writer_execution_evidence import WriterResidualPropagationWorkEvidence
from .writer_execution_evidence import writer_residual_propagation_work_evidence
from .ids import AtomId
from .ids import BondId
from .ids import OccurrenceId
from .ids import SiteId
from .policy import DirectionMark
from .policy import RingLabel
from .policy import TetraToken
from .residual_constraints import DirectionalBondEmissionFactor
from .residual_constraints import DirectionalBondEmissionFactorValueSnapshot
from .residual_constraints import DirectionalNormalizedSign
from .residual_constraints import DirectionalResidualFactorValueSnapshot
from .residual_constraints import DirectionalSiteCarrierModel
from .residual_constraints import DirectionalSiteFactor
from .residual_constraints import DirectionalSiteFactorValueSnapshot
from .residual_constraints import ResidualFactorKey
from .residual_constraints import ResidualPropagationKind
from .residual_constraints import ResidualPropagationResult
from .residual_constraints import ResidualStore
from .residual_constraints import ResidualStoreValueSnapshot
from .residual_constraints import TetraLocalParity
from .residual_constraints import TetraResidualFactorValueSnapshot
from .residual_constraints import TetraTokenParityFactor
from .residual_constraints import TetraTokenParityFactorValueSnapshot
from .residual_constraints import VarId
from .residual_constraints import add_factors_and_propagate
from .residual_constraints import directional_site_carrier_var
from .residual_constraints import normalized_sign_from_mark
from .residual_constraints import tetra_parity_var
from .residual_constraints import tetra_token_var
from .stereo_templates import DirectionalTemplate
from .stereo_templates import TetraTemplate
from .writer_events import WriterAtomEmitted
from .writer_events import WriterBondEmitted
from .writer_events import WriterEvent
from .writer_events import WriterLocalOrderClosed
from .writer_events import WriterRingEndpointEmitted
from .writer_events import WriterRingEndpointPaired

if TYPE_CHECKING:
    from .prepared_runtime import SouthStarPreparedMol
    from .writer_state import WriterRingStateKey
    from .writer_state import WriterStereoState
    from .writer_state import WriterStereoStateKey


EMPTY_RESIDUAL_SNAPSHOT = ResidualStore().value_snapshot()
_MAX_TETRA_RING_ENDPOINT_OCCURRENCES = 1
_MAX_DIRECTIONAL_RING_CARRIER_SITES = 2


@dataclass(frozen=True, slots=True)
class WriterAtomOccurrenceRecord:
    atom: AtomId
    token: TetraToken


@dataclass(frozen=True, slots=True)
class WriterBondOccurrenceRecord:
    bond: BondId
    parent: AtomId
    child: AtomId
    mark: DirectionMark


@dataclass(frozen=True, slots=True)
class WriterLocalOrderRecord:
    atom: AtomId
    order: tuple[OccurrenceId, ...]
    closed: bool = False


@dataclass(frozen=True, slots=True)
class WriterAtomTextChoice:
    text: str
    tetra_token: TetraToken
    atom: AtomId
    site: SiteId | None


@dataclass(frozen=True, slots=True)
class WriterBondTextChoice:
    text: str
    direction_mark: DirectionMark
    bond: BondId
    carrier_sites: tuple[SiteId, ...]


@dataclass(frozen=True, slots=True)
class WriterStereoPolicyBlocker:
    kind: str
    site: SiteId | None = None
    operation: str = ""


@dataclass(frozen=True, slots=True)
class _WriterStereoMutation:
    state: "WriterStereoState | None"
    capabilities: frozenset[_WriterExecutionCapabilityKind] = frozenset()
    residual_work_evidence: tuple[
        WriterResidualPropagationWorkEvidence,
        ...
    ] = ()
    stereo_policy_blockers: tuple[WriterStereoPolicyBlocker, ...] = ()


@dataclass(frozen=True, slots=True)
class _WriterStereoAdvanceOutcome:
    state: "WriterStereoState | None"
    execution_capabilities: frozenset[
        _WriterExecutionCapabilityKind
    ] = frozenset()
    residual_work_evidence: tuple[
        WriterResidualPropagationWorkEvidence,
        ...
    ] = ()
    stereo_policy_blockers: tuple[WriterStereoPolicyBlocker, ...] = ()


def empty_writer_stereo_state() -> "WriterStereoState":
    from .writer_state import WriterStereoState

    return WriterStereoState(
        residual_snapshot=EMPTY_RESIDUAL_SNAPSHOT,
        atom_occurrences=(),
        bond_occurrences=(),
        local_orders=(),
    )


def _writer_stereo_relation_definitions(
    prepared: SouthStarPreparedMol,
) -> tuple[tuple[tuple[VarId, tuple[object, ...]], ...], tuple[object, ...]]:
    domains: list[tuple[VarId, tuple[object, ...]]] = []
    factors: list[object] = []
    seen_vars: set[VarId] = set()

    def add_var(var: VarId, domain: tuple[object, ...]) -> None:
        if var in seen_vars:
            return
        seen_vars.add(var)
        domains.append((var, domain))

    for template in prepared.tetra_templates:
        token = tetra_token_var(template.site)
        parity = tetra_parity_var(template.site)
        add_var(token, _tetra_domain(template))
        add_var(parity, (TetraLocalParity.EVEN, TetraLocalParity.ODD))
        factors.append(
            TetraTokenParityFactor(
                key=_tetra_factor_key(template.site),
                scope=(token, parity),
                status=template.status,
                target=template.target,
            )
        )

    bond_models: dict[BondId, list[tuple[VarId, DirectionalSiteCarrierModel]]] = {}
    for template in prepared.directional_templates:
        site_models = _directional_site_carrier_models(prepared, template)
        scope = tuple(var for var, _ in site_models)
        for var in scope:
            add_var(var, _directional_normalized_domain())
        factors.append(
            DirectionalSiteFactor(
                key=_directional_site_factor_key(template.site),
                scope=scope,
                sides=tuple((var, model.side) for var, model in site_models),
                status=template.status,
                target=template.target,
            )
        )
        for var, model in site_models:
            bond_models.setdefault(model.bond, []).append((var, model))

    for bond, entries in bond_models.items():
        factors.append(
            DirectionalBondEmissionFactor(
                key=_directional_bond_factor_key(bond),
                scope=tuple(var for var, _ in entries),
                models=tuple(model for _, model in entries),
                allowed_marks=_allowed_direction_marks(prepared, bond),
            )
        )

    return tuple(domains), tuple(factors)


def initial_writer_stereo_state(prepared: SouthStarPreparedMol) -> "WriterStereoState":
    from .writer_state import WriterStereoState

    store = ResidualStore()
    domains, factors = _writer_stereo_relation_definitions(prepared)
    for var, domain in domains:
        store.add_var(var, domain)

    result = add_factors_and_propagate(store, tuple(factors))
    if not _writer_residual_mutation_is_legal(
        result,
        operation="initial stereo relation construction",
    ):
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "initial stereo relation is contradictory",
        )

    return WriterStereoState(
        residual_snapshot=store.value_snapshot(),
        atom_occurrences=(),
        bond_occurrences=(),
        local_orders=(),
    )


def reconstruct_writer_stereo_residual_snapshot(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoStateKey",
    *,
    ring_state: "WriterRingStateKey | None" = None,
) -> ResidualStoreValueSnapshot:
    store = ResidualStore()
    domains, factors = _writer_stereo_relation_definitions(prepared)
    for var, domain in domains:
        store.add_var(var, domain)

    result = add_factors_and_propagate(store, tuple(factors))
    _require_certified_reconstruction(result, "initial relation")

    restrictions: list[tuple[VarId, object]] = []
    for record in stereo_state.atom_occurrences:
        restriction = _tetra_token_restriction(
            prepared,
            atom=record.atom,
            token=record.token,
        )
        if restriction is not None:
            restrictions.append(restriction)

    for record in stereo_state.bond_occurrences:
        restrictions.extend(
            _directional_bond_restrictions(
                prepared,
                bond=record.bond,
                parent=record.parent,
                child=record.child,
                mark=record.mark,
            )
        )

    for record in stereo_state.local_orders:
        if not record.closed:
            continue
        restriction = _tetra_parity_restriction(
            prepared,
            atom=record.atom,
            order=record.order,
        )
        if restriction is not None:
            restrictions.append(restriction)

    result = store.restrict_many_and_propagate(tuple(restrictions))
    _require_certified_reconstruction(result, "recorded stereo history")

    if ring_state is not None:
        _replay_directional_ring_state_for_reconstruction(
            prepared,
            store,
            stereo_state,
            ring_state,
        )

    emitted_directional_bonds = tuple(
        record.bond
        for record in stereo_state.bond_occurrences
        if _directional_models_for_bond(prepared, record.bond)
    )
    for bond in emitted_directional_bonds:
        store.discharge_satisfied_factors((_directional_bond_factor_key(bond),))

    emitted_bond_set = frozenset(record.bond for record in stereo_state.bond_occurrences)
    for template in sorted(prepared.directional_templates, key=lambda item: int(item.site)):
        if _directional_template_substituent_bonds(prepared, template).issubset(
            emitted_bond_set,
        ):
            store.discharge_satisfied_factors(
                (_directional_site_factor_key(template.site),)
            )

    for record in stereo_state.local_orders:
        if not record.closed:
            continue
        template = _tetra_template_by_center(prepared).get(record.atom)
        if template is not None:
            store.discharge_satisfied_factors((_tetra_factor_key(template.site),))

    return store.value_snapshot()


def _replay_directional_ring_state_for_reconstruction(
    prepared: SouthStarPreparedMol,
    store: ResidualStore,
    stereo_state: "WriterStereoStateKey",
    ring_state: "WriterRingStateKey",
) -> None:
    for endpoint in ring_state.open_endpoints:
        if not _bounded_directional_ring_models(prepared, endpoint.bond):
            continue
        if any(record.bond == endpoint.bond for record in stereo_state.bond_occurrences):
            raise ValueError("open directional ring endpoint is already recorded")

        event = WriterRingEndpointEmitted(
            bond=endpoint.bond,
            endpoint_atom=endpoint.first_atom,
            partner_atom=endpoint.second_atom,
            label=endpoint.label,
            endpoint_text=endpoint.label.text,
            bond_text=endpoint.first_endpoint_bond_text,
            direction_mark=endpoint.first_endpoint_direction_mark,
        )
        restriction = _directional_ring_endpoint_projection(prepared, event)
        if restriction is None:
            raise ValueError("directional ring endpoint has no compatible partner")
        result = store.intersect_domains_and_propagate(restriction)
        _require_certified_reconstruction(
            result,
            "open directional ring endpoint",
        )

    for closure in ring_state.closed_closures:
        if not _bounded_directional_ring_models(prepared, closure.bond):
            continue
        expected = _directional_ring_pair_bond_occurrence(prepared, closure)
        if expected is None:
            raise ValueError("directional ring closure pair is incompatible")
        actual = tuple(
            record
            for record in stereo_state.bond_occurrences
            if record.bond == closure.bond
        )
        if actual != (expected,):
            raise ValueError("directional ring closure bond record mismatch")


def reconstruct_writer_local_order_records(
    prepared: SouthStarPreparedMol,
    *,
    atom_occurrences: tuple[WriterAtomOccurrenceRecord, ...],
    parent_by_child: Mapping[AtomId, AtomId],
    closed_atoms: frozenset[AtomId],
    ring_incidences_by_atom: (
        Mapping[AtomId, tuple[tuple[BondId, AtomId], ...]] | None
    ) = None,
) -> tuple[WriterLocalOrderRecord, ...]:
    records: tuple[WriterLocalOrderRecord, ...] = ()
    ring_incidences_by_atom = ring_incidences_by_atom or {}

    for occurrence in atom_occurrences:
        parent = parent_by_child.get(occurrence.atom)

        if parent is not None:
            records = _record_parent_occurrence(
                prepared,
                records,
                atom=occurrence.atom,
                parent=parent,
            )
            records = _record_child_occurrence(
                prepared,
                records,
                parent=parent,
                child=occurrence.atom,
            )

        for occurrence_id in _resolved_tetra_ring_endpoint_occurrences(
            prepared,
            endpoint_atom=occurrence.atom,
            incidences=ring_incidences_by_atom.get(occurrence.atom, ()),
        ):
            records = _append_local_order(records, occurrence.atom, occurrence_id)

    for atom in sorted(closed_atoms, key=int):
        record = _local_order_record(records, atom)
        records = _replace_local_order(
            records,
            _close_local_order(
                prepared,
                record,
                atom=atom,
            ),
        )

    return records


def advance_writer_stereo_state(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    events: tuple[WriterEvent, ...],
) -> "WriterStereoState | None":
    return advance_writer_stereo_state_with_evidence(
        prepared,
        stereo_state,
        events,
    ).state


def advance_writer_stereo_state_with_evidence(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    events: tuple[WriterEvent, ...],
) -> _WriterStereoAdvanceOutcome:
    state = stereo_state
    capabilities: set[_WriterExecutionCapabilityKind] = set()
    work_evidence: list[WriterResidualPropagationWorkEvidence] = []
    stereo_policy_blockers: list[WriterStereoPolicyBlocker] = []
    for event in events:
        if isinstance(event, WriterAtomEmitted):
            mutation = _on_atom_emitted(prepared, state, event)
        elif isinstance(event, WriterBondEmitted):
            mutation = _on_bond_emitted(prepared, state, event)
        elif isinstance(event, WriterLocalOrderClosed):
            mutation = _on_local_order_closed(prepared, state, event.atom)
        elif isinstance(event, WriterRingEndpointEmitted):
            mutation = _on_ring_endpoint_emitted(prepared, state, event)
        elif isinstance(event, WriterRingEndpointPaired):
            mutation = _on_ring_endpoint_paired(prepared, state, event)
        else:
            continue
        if mutation.state is None:
            return _WriterStereoAdvanceOutcome(
                state=None,
                stereo_policy_blockers=mutation.stereo_policy_blockers,
            )
        state = mutation.state
        capabilities.update(mutation.capabilities)
        work_evidence.extend(mutation.residual_work_evidence)
        stereo_policy_blockers.extend(mutation.stereo_policy_blockers)
    return _WriterStereoAdvanceOutcome(
        state=state,
        execution_capabilities=frozenset(capabilities),
        residual_work_evidence=tuple(work_evidence),
        stereo_policy_blockers=tuple(stereo_policy_blockers),
    )


def terminal_writer_stereo_state(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    atom: AtomId,
) -> "WriterStereoState | None":
    return terminal_writer_stereo_state_with_evidence(
        prepared,
        stereo_state,
        atom,
    ).state


def terminal_writer_stereo_state_with_evidence(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    atom: AtomId,
) -> _WriterStereoAdvanceOutcome:
    outcome = advance_writer_stereo_state_with_evidence(
        prepared,
        stereo_state,
        (WriterLocalOrderClosed(atom=atom),),
    )
    if outcome.state is None:
        return _WriterStereoAdvanceOutcome(state=None)
    if outcome.state.residual_snapshot != EMPTY_RESIDUAL_SNAPSHOT:
        return _WriterStereoAdvanceOutcome(state=None)
    return outcome


def _writer_residual_mutation_is_legal(
    result: ResidualPropagationResult,
    *,
    operation: str,
) -> bool:
    if result.kind is ResidualPropagationKind.CERTIFIED_CONSISTENT:
        return True

    if result.kind is ResidualPropagationKind.CONTRADICTION:
        return False

    if result.kind is ResidualPropagationKind.LOCALLY_CONSISTENT_UNCERTIFIED:
        stats = result.stats
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "WRITER_SHAPED residual propagation exceeded the supported "
            f"complexity envelope during {operation}: "
            f"variables={len(stats.component_variables)}, "
            f"factors={len(stats.component_factor_keys)}, "
            f"largest_scope={stats.largest_factor_scope}, "
            f"largest_candidate_rows={stats.largest_candidate_row_count}",
        )

    raise AssertionError(f"unknown propagation result: {result.kind!r}")


def _require_certified_reconstruction(
    result: ResidualPropagationResult,
    operation: str,
) -> None:
    if result.kind is ResidualPropagationKind.CERTIFIED_CONSISTENT:
        return
    if result.kind is ResidualPropagationKind.CONTRADICTION:
        raise ValueError(f"writer stereo reconstruction contradicted: {operation}")
    if result.kind is ResidualPropagationKind.LOCALLY_CONSISTENT_UNCERTIFIED:
        stats = result.stats
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "WRITER_SHAPED residual propagation exceeded the supported "
            f"complexity envelope during {operation}: "
            f"variables={len(stats.component_variables)}, "
            f"factors={len(stats.component_factor_keys)}, "
            f"largest_scope={stats.largest_factor_scope}, "
            f"largest_candidate_rows={stats.largest_candidate_row_count}",
        )
    raise AssertionError(f"unknown propagation result: {result.kind!r}")


def _tetra_token_restriction(
    prepared: SouthStarPreparedMol,
    *,
    atom: AtomId,
    token: TetraToken,
) -> tuple[VarId, object] | None:
    template = _tetra_template_by_center(prepared).get(atom)
    if template is None:
        if token is not TetraToken.NONE:
            raise ValueError("non-tetra atom occurrence carries tetra token")
        return None
    return (tetra_token_var(template.site), token)


def _directional_bond_restrictions(
    prepared: SouthStarPreparedMol,
    *,
    bond: BondId,
    parent: AtomId,
    child: AtomId,
    mark: DirectionMark,
) -> tuple[tuple[VarId, object], ...]:
    models = _directional_models_for_bond(prepared, bond)
    if not models:
        if mark is not DirectionMark.ABSENT:
            raise ValueError("non-directional bond occurrence carries direction mark")
        return ()
    event = WriterBondEmitted(
        bond=bond,
        parent=parent,
        child=child,
        text="",
        direction_mark=mark,
    )
    orientation = _canonical_bond_orientation(prepared, event)
    return tuple(
        (
            directional_site_carrier_var(model.site, bond),
            normalized_sign_from_mark(
                mark=mark,
                canonical_orientation=orientation,
                model=model,
            ),
        )
        for model in models
    )


def _normalized_directional_value(
    prepared: SouthStarPreparedMol,
    *,
    bond: BondId,
    parent: AtomId,
    child: AtomId,
    mark: DirectionMark,
    model: DirectionalSiteCarrierModel,
) -> DirectionalNormalizedSign:
    event = WriterBondEmitted(
        bond=bond,
        parent=parent,
        child=child,
        text="",
        direction_mark=mark,
    )
    return normalized_sign_from_mark(
        mark=mark,
        canonical_orientation=_canonical_bond_orientation(prepared, event),
        model=model,
    )


def _directional_ring_pair_value_for_model(
    prepared: SouthStarPreparedMol,
    *,
    model: DirectionalSiteCarrierModel,
    bond: BondId,
    first_atom: AtomId,
    second_atom: AtomId,
    first_mark: DirectionMark,
    second_mark: DirectionMark,
) -> DirectionalNormalizedSign | None:
    values = []
    if first_mark is not DirectionMark.ABSENT:
        values.append(
            _normalized_directional_value(
                prepared,
                bond=bond,
                parent=first_atom,
                child=second_atom,
                mark=first_mark,
                model=model,
            )
        )
    if second_mark is not DirectionMark.ABSENT:
        values.append(
            _normalized_directional_value(
                prepared,
                bond=bond,
                parent=second_atom,
                child=first_atom,
                mark=second_mark,
                model=model,
            )
        )

    if not values:
        return DirectionalNormalizedSign.ABSENT
    if len(frozenset(values)) != 1:
        return None
    return values[0]


def _directional_ring_pair_restrictions(
    prepared: SouthStarPreparedMol,
    *,
    bond: BondId,
    first_atom: AtomId,
    second_atom: AtomId,
    first_mark: DirectionMark,
    second_mark: DirectionMark,
) -> tuple[tuple[VarId, DirectionalNormalizedSign], ...] | None:
    models = _bounded_directional_ring_models(prepared, bond)
    if not models:
        if (
            first_mark is not DirectionMark.ABSENT
            or second_mark is not DirectionMark.ABSENT
        ):
            return None
        return ()

    restrictions: list[tuple[VarId, DirectionalNormalizedSign]] = []
    for model in models:
        value = _directional_ring_pair_value_for_model(
            prepared,
            model=model,
            bond=bond,
            first_atom=first_atom,
            second_atom=second_atom,
            first_mark=first_mark,
            second_mark=second_mark,
        )
        if value is None:
            return None
        restrictions.append((
            directional_site_carrier_var(model.site, bond),
            value,
        ))
    return tuple(restrictions)


def _directional_ring_pair_value(
    prepared: SouthStarPreparedMol,
    *,
    bond: BondId,
    first_atom: AtomId,
    second_atom: AtomId,
    first_mark: DirectionMark,
    second_mark: DirectionMark,
) -> DirectionalNormalizedSign | None:
    restrictions = _directional_ring_pair_restrictions(
        prepared,
        bond=bond,
        first_atom=first_atom,
        second_atom=second_atom,
        first_mark=first_mark,
        second_mark=second_mark,
    )
    if restrictions is None:
        return None
    if not restrictions:
        return DirectionalNormalizedSign.ABSENT
    values = tuple(value for _var, value in restrictions)
    if len(frozenset(values)) != 1:
        return None
    return values[0]


def _tetra_parity_restriction(
    prepared: SouthStarPreparedMol,
    *,
    atom: AtomId,
    order: tuple[OccurrenceId, ...],
) -> tuple[VarId, object] | None:
    template = _tetra_template_by_center(prepared).get(atom)
    if template is None:
        return None
    return (
        tetra_parity_var(template.site),
        _tetra_local_parity(template, order),
    )


def writer_atom_text_choices(
    prepared: SouthStarPreparedMol,
    atom: AtomId,
) -> tuple[WriterAtomTextChoice, ...]:
    site = _tetra_template_by_center(prepared).get(atom)
    choices: list[WriterAtomTextChoice] = []
    for atom_choice in prepared.policy.atom_text_domain_unchecked(atom):
        for token, text in atom_choice.text_by_tetra:
            if site is None and token is not TetraToken.NONE:
                continue
            if site is not None and site.status is SiteStatus.UNSPECIFIED:
                if token is not TetraToken.NONE:
                    continue
            choices.append(
                WriterAtomTextChoice(
                    text=text,
                    tetra_token=token,
                    atom=atom,
                    site=None if site is None else site.site,
                )
            )
    if not choices:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            f"WRITER_SHAPED has no atom text for {atom!r}",
        )
    return tuple(choices)


def writer_bond_text_choices(
    prepared: SouthStarPreparedMol,
    bond: BondId,
) -> tuple[WriterBondTextChoice, ...]:
    try:
        choices = prepared.policy.bond_text_domain_unchecked(
            bond,
            slot_kind="tree",
        )
    except KeyError as exc:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            f"WRITER_SHAPED has no acyclic writer bond text for {bond!r}",
        ) from exc
    if not choices:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            f"WRITER_SHAPED has empty acyclic writer bond text domain for {bond!r}",
        )
    eligible_sites = _directional_sites_for_carrier_bond(prepared, bond)
    out: list[WriterBondTextChoice] = []
    for choice in choices:
        out.append(
            WriterBondTextChoice(
                text=choice.base_text,
                direction_mark=DirectionMark.ABSENT,
                bond=bond,
                carrier_sites=eligible_sites,
            )
        )
        if eligible_sites and choice.permits_direction:
            out.append(
                WriterBondTextChoice(
                    text="/",
                    direction_mark=DirectionMark.FWD,
                    bond=bond,
                    carrier_sites=eligible_sites,
                )
            )
            out.append(
                WriterBondTextChoice(
                    text="\\",
                    direction_mark=DirectionMark.REV,
                    bond=bond,
                    carrier_sites=eligible_sites,
                )
            )
    return tuple(out)


def writer_closure_endpoint_relation(
    prepared: SouthStarPreparedMol,
    *,
    bond: BondId,
    first_atom: AtomId,
    second_atom: AtomId,
):
    from .writer_graph_obligations import WriterClosureEndpointRelation
    from .writer_graph_obligations import writer_closure_endpoint_relation as base_relation

    models = _bounded_directional_ring_models(prepared, bond)
    if not models:
        return base_relation(prepared, bond)
    relation = base_relation(
        prepared,
        bond,
        include_direction_marks=True,
    )

    rows = []
    for first, seconds in relation.rows:
        compatible_seconds = tuple(
            second
            for second in seconds
            if _directional_ring_pair_restrictions(
                prepared,
                bond=bond,
                first_atom=first_atom,
                second_atom=second_atom,
                first_mark=first.direction_mark,
                second_mark=second.direction_mark,
            )
            is not None
        )
        rows.append((first, compatible_seconds))
    return WriterClosureEndpointRelation(rows=tuple(rows))


def writer_stereo_state_sort_tuple(state: "WriterStereoState") -> tuple[object, ...]:
    return (
        _residual_snapshot_sort_tuple(state.residual_snapshot),
        tuple(_atom_record_sort_tuple(record) for record in state.atom_occurrences),
        tuple(_bond_record_sort_tuple(record) for record in state.bond_occurrences),
        tuple(_local_order_sort_tuple(record) for record in state.local_orders),
    )


def validate_writer_stereo_supported_prepared(prepared: SouthStarPreparedMol) -> None:
    return None


def _on_atom_emitted(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    event: WriterAtomEmitted,
) -> _WriterStereoMutation:
    from .writer_state import WriterStereoState

    store = ResidualStore.from_value_snapshot(stereo_state.residual_snapshot)
    local_orders = _record_parent_occurrence(
        prepared,
        stereo_state.local_orders,
        atom=event.atom,
        parent=event.parent,
    )
    local_orders = _record_child_occurrence(
        prepared,
        local_orders,
        parent=event.parent,
        child=event.atom,
    )
    checkpoint = store.checkpoint()
    try:
        restriction = _tetra_token_restriction(
            prepared,
            atom=event.atom,
            token=event.tetra_token,
        )
    except ValueError:
        return _WriterStereoMutation(state=None)
    capabilities: set[_WriterExecutionCapabilityKind] = set()
    work_evidence: list[WriterResidualPropagationWorkEvidence] = []
    if restriction is not None:
        operation = "tetrahedral atom-token restriction"
        result = store.restrict_many_and_propagate(
            (restriction,)
        )
        evidence = writer_residual_propagation_work_evidence(
            operation=operation,
            result=result,
        )
        if not _writer_residual_mutation_is_legal(
            result,
            operation=operation,
        ):
            store.rollback(checkpoint)
            return _WriterStereoMutation(state=None)
        work_evidence.append(evidence)
        capabilities.update(
            {
                _WriterExecutionCapabilityKind.TETRA_TOKEN_RESTRICTION,
                _WriterExecutionCapabilityKind.RESIDUAL_PROPAGATION,
            }
        )
    return _WriterStereoMutation(
        state=WriterStereoState(
            residual_snapshot=store.value_snapshot(),
            atom_occurrences=stereo_state.atom_occurrences
            + (WriterAtomOccurrenceRecord(event.atom, event.tetra_token),),
            bond_occurrences=stereo_state.bond_occurrences,
            local_orders=local_orders,
        ),
        capabilities=frozenset(capabilities),
        residual_work_evidence=tuple(work_evidence),
    )


def _on_bond_emitted(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    event: WriterBondEmitted,
    *,
    operation: str = "directional carrier-mark restriction",
) -> _WriterStereoMutation:
    from .writer_state import WriterStereoState

    store = ResidualStore.from_value_snapshot(stereo_state.residual_snapshot)
    models = _directional_models_for_bond(prepared, event.bond)
    checkpoint = store.checkpoint()
    capabilities: set[_WriterExecutionCapabilityKind] = set()
    work_evidence: list[WriterResidualPropagationWorkEvidence] = []
    if models:
        blocker = _unsupported_directional_non_neighbor_ligand_blocker_for_bond(
            prepared,
            event.bond,
            operation=operation,
        )
        if blocker is not None:
            return _WriterStereoMutation(
                state=None,
                stereo_policy_blockers=(blocker,),
            )
        restrictions = _directional_bond_restrictions(
            prepared,
            bond=event.bond,
            parent=event.parent,
            child=event.child,
            mark=event.direction_mark,
        )
        result = store.restrict_many_and_propagate(restrictions)
        evidence = writer_residual_propagation_work_evidence(
            operation=operation,
            result=result,
        )
        if not _writer_residual_mutation_is_legal(
            result,
            operation=operation,
        ):
            store.rollback(checkpoint)
            return _WriterStereoMutation(state=None)
        work_evidence.append(evidence)
        capabilities.update(
            {
                _WriterExecutionCapabilityKind.DIRECTIONAL_CARRIER_RESTRICTION,
                _WriterExecutionCapabilityKind.DIRECTIONAL_SITE_COMPATIBILITY,
                _WriterExecutionCapabilityKind.RESIDUAL_PROPAGATION,
            }
        )
        if len(models) > 1:
            capabilities.add(
                _WriterExecutionCapabilityKind
                .SHARED_DIRECTIONAL_CARRIER_RESTRICTION
            )
    elif event.direction_mark is not DirectionMark.ABSENT:
        return _WriterStereoMutation(state=None)
    if models:
        try:
            store.discharge_satisfied_factors((_directional_bond_factor_key(event.bond),))
            capabilities.add(_WriterExecutionCapabilityKind.RESIDUAL_FACTOR_DISCHARGE)
            emitted_bonds = {
                record.bond
                for record in stereo_state.bond_occurrences
            } | {event.bond}
            for site in sorted({model.site for model in models}, key=int):
                template = _directional_template_by_site(prepared)[site]
                if _directional_template_substituent_bonds(
                    prepared,
                    template,
                ).issubset(emitted_bonds):
                    store.discharge_satisfied_factors(
                        (_directional_site_factor_key(site),)
                    )
                    capabilities.add(
                        _WriterExecutionCapabilityKind.RESIDUAL_FACTOR_DISCHARGE
                    )
        except ValueError:
            store.rollback(checkpoint)
            return _WriterStereoMutation(state=None)

    return _WriterStereoMutation(
        state=WriterStereoState(
            residual_snapshot=store.value_snapshot(),
            atom_occurrences=stereo_state.atom_occurrences,
            bond_occurrences=stereo_state.bond_occurrences
            + (
                WriterBondOccurrenceRecord(
                    event.bond,
                    event.parent,
                    event.child,
                    event.direction_mark,
                ),
            ),
            local_orders=stereo_state.local_orders,
        ),
        capabilities=frozenset(capabilities),
        residual_work_evidence=tuple(work_evidence),
    )


def _on_local_order_closed(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    atom: AtomId,
) -> _WriterStereoMutation:
    from .writer_state import WriterStereoState

    template = _tetra_template_by_center(prepared).get(atom)
    record = _local_order_record(stereo_state.local_orders, atom)
    if record is not None and record.closed:
        return _WriterStereoMutation(state=stereo_state)
    closed_order = _close_local_order(prepared, record, atom=atom)
    local_orders = _replace_local_order(stereo_state.local_orders, closed_order)
    store = ResidualStore.from_value_snapshot(stereo_state.residual_snapshot)
    if template is not None:
        checkpoint = store.checkpoint()
        capabilities: set[_WriterExecutionCapabilityKind] = set()
        restriction = _tetra_parity_restriction(
            prepared,
            atom=atom,
            order=closed_order.order,
        )
        assert restriction is not None
        operation = "tetrahedral local-order factor closure"
        result = store.restrict_many_and_propagate((restriction,))
        evidence = writer_residual_propagation_work_evidence(
            operation=operation,
            result=result,
        )
        if not _writer_residual_mutation_is_legal(
            result,
            operation=operation,
        ):
            store.rollback(checkpoint)
            return _WriterStereoMutation(state=None)
        work_evidence = (evidence,)
        capabilities.update(
            {
                _WriterExecutionCapabilityKind.TETRA_LOCAL_ORDER_RESTRICTION,
                _WriterExecutionCapabilityKind.RESIDUAL_PROPAGATION,
            }
        )
        try:
            store.discharge_satisfied_factors((_tetra_factor_key(template.site),))
            capabilities.add(_WriterExecutionCapabilityKind.RESIDUAL_FACTOR_DISCHARGE)
        except ValueError:
            store.rollback(checkpoint)
            return _WriterStereoMutation(state=None)
    else:
        capabilities = set()
        work_evidence = ()
    return _WriterStereoMutation(
        state=WriterStereoState(
            residual_snapshot=store.value_snapshot(),
            atom_occurrences=stereo_state.atom_occurrences,
            bond_occurrences=stereo_state.bond_occurrences,
            local_orders=local_orders,
        ),
        capabilities=frozenset(capabilities),
        residual_work_evidence=work_evidence,
    )


def _on_ring_endpoint_emitted(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    event: WriterRingEndpointEmitted,
) -> _WriterStereoMutation:
    if not _ring_event_text_ok(prepared, event):
        return _WriterStereoMutation(state=None)
    mutation = _project_directional_ring_endpoint(
        prepared,
        stereo_state,
        event,
    )
    if mutation.state is None:
        return mutation
    tetra_mutation = _record_tetra_ring_endpoint(
        prepared,
        mutation.state,
        event,
    )
    if tetra_mutation.state is None:
        return tetra_mutation
    return _WriterStereoMutation(
        state=tetra_mutation.state,
        capabilities=(
            mutation.capabilities | tetra_mutation.capabilities
        ),
        residual_work_evidence=(
            mutation.residual_work_evidence
            + tetra_mutation.residual_work_evidence
        ),
    )


def _on_ring_endpoint_paired(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    event: WriterRingEndpointPaired,
) -> _WriterStereoMutation:
    if not _ring_event_text_ok(prepared, event):
        return _WriterStereoMutation(state=None)
    mutation = _restrict_directional_ring_pair(
        prepared,
        stereo_state,
        event,
    )
    if mutation.state is None:
        return mutation
    tetra_mutation = _record_tetra_ring_endpoint(
        prepared,
        mutation.state,
        event,
    )
    if tetra_mutation.state is None:
        return tetra_mutation
    return _WriterStereoMutation(
        state=tetra_mutation.state,
        capabilities=(
            mutation.capabilities | tetra_mutation.capabilities
        ),
        residual_work_evidence=(
            mutation.residual_work_evidence
            + tetra_mutation.residual_work_evidence
        ),
    )


def _ring_event_text_ok(prepared: SouthStarPreparedMol, event) -> bool:
    from .writer_graph_obligations import WriterClosureEndpointChoice

    try:
        policy_label = RingLabel(event.label.value)
        expected_label = policy_label.text()
    except ValueError:
        return False
    if policy_label not in prepared.policy.ring_labels:
        return False
    if event.label.text != expected_label:
        return False
    if event.endpoint_text != event.label.text:
        return False
    try:
        if isinstance(event, WriterRingEndpointPaired):
            relation = writer_closure_endpoint_relation(
                prepared,
                bond=event.bond,
                first_atom=event.partner_atom,
                second_atom=event.endpoint_atom,
            )
            return relation.pair_ok(
                WriterClosureEndpointChoice(
                    event.first_endpoint_bond_text,
                    event.first_endpoint_direction_mark,
                ),
                WriterClosureEndpointChoice(
                    event.bond_text,
                    event.direction_mark,
                ),
            )
        relation = writer_closure_endpoint_relation(
            prepared,
            bond=event.bond,
            first_atom=event.endpoint_atom,
            second_atom=event.partner_atom,
        )
    except SouthStarError:
        return False
    return (
        WriterClosureEndpointChoice(event.bond_text, event.direction_mark)
        in relation.openable_first_choices
    )


def _project_directional_ring_endpoint(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    event: WriterRingEndpointEmitted,
) -> _WriterStereoMutation:
    from .writer_state import WriterStereoState

    models = _bounded_directional_ring_models(prepared, event.bond)
    if not models:
        if event.direction_mark is not DirectionMark.ABSENT:
            return _WriterStereoMutation(state=None)
        return _WriterStereoMutation(state=stereo_state)

    restriction = _directional_ring_endpoint_projection(prepared, event)
    if restriction is None:
        return _WriterStereoMutation(state=None)

    store = ResidualStore.from_value_snapshot(stereo_state.residual_snapshot)
    checkpoint = store.checkpoint()
    operation = "directional ring endpoint projection"
    result = store.intersect_domains_and_propagate(restriction)
    evidence = writer_residual_propagation_work_evidence(
        operation=operation,
        result=result,
    )
    if not _writer_residual_mutation_is_legal(
        result,
        operation=operation,
    ):
        store.rollback(checkpoint)
        return _WriterStereoMutation(state=None)

    capabilities = {
        _WriterExecutionCapabilityKind.DIRECTIONAL_RING_PAIR_COMPATIBILITY,
        _WriterExecutionCapabilityKind.RESIDUAL_PROPAGATION,
    }
    if len(models) > 1:
        capabilities.add(
            _WriterExecutionCapabilityKind
            .SHARED_DIRECTIONAL_CARRIER_RESTRICTION
        )

    return _WriterStereoMutation(
        state=WriterStereoState(
            residual_snapshot=store.value_snapshot(),
            atom_occurrences=stereo_state.atom_occurrences,
            bond_occurrences=stereo_state.bond_occurrences,
            local_orders=stereo_state.local_orders,
        ),
        capabilities=frozenset(capabilities),
        residual_work_evidence=(evidence,),
    )


def _directional_ring_endpoint_projection(
    prepared: SouthStarPreparedMol,
    event: WriterRingEndpointEmitted,
) -> tuple[tuple[VarId, tuple[object, ...]], ...] | None:
    from .writer_graph_obligations import WriterClosureEndpointChoice

    models = _bounded_directional_ring_models(prepared, event.bond)
    if not models:
        if event.direction_mark is not DirectionMark.ABSENT:
            return None
        return None

    relation = writer_closure_endpoint_relation(
        prepared,
        bond=event.bond,
        first_atom=event.endpoint_atom,
        second_atom=event.partner_atom,
    )
    first = WriterClosureEndpointChoice(event.bond_text, event.direction_mark)
    projected: dict[VarId, list[object]] = {
        directional_site_carrier_var(model.site, event.bond): []
        for model in models
    }
    for second in relation.compatible_seconds(first):
        restrictions = _directional_ring_pair_restrictions(
            prepared,
            bond=event.bond,
            first_atom=event.endpoint_atom,
            second_atom=event.partner_atom,
            first_mark=event.direction_mark,
            second_mark=second.direction_mark,
        )
        if restrictions is None:
            continue
        for var, value in restrictions:
            projected.setdefault(var, []).append(value)

    projection = tuple(
        (var, tuple(dict.fromkeys(values)))
        for var, values in sorted(
            projected.items(),
            key=lambda item: _var_sort_tuple(item[0]),
        )
    )
    if not projection or any(not values for _var, values in projection):
        return None
    return projection


def _restrict_directional_ring_pair(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    event: WriterRingEndpointPaired,
) -> _WriterStereoMutation:
    models = _bounded_directional_ring_models(prepared, event.bond)
    if not models:
        if event.direction_mark is not DirectionMark.ABSENT:
            return _WriterStereoMutation(state=None)
        return _WriterStereoMutation(state=stereo_state)

    restrictions = _directional_ring_pair_restrictions(
        prepared,
        bond=event.bond,
        first_atom=event.partner_atom,
        second_atom=event.endpoint_atom,
        first_mark=event.first_endpoint_direction_mark,
        second_mark=event.direction_mark,
    )
    if restrictions is None:
        return _WriterStereoMutation(state=None)

    record = _directional_ring_pair_event_bond_occurrence(prepared, event)
    if record is None:
        return _WriterStereoMutation(state=None)

    mutation = _on_bond_emitted(
        prepared,
        stereo_state,
        WriterBondEmitted(
            bond=event.bond,
            parent=record.parent,
            child=record.child,
            text=event.bond_text,
            direction_mark=record.mark,
        ),
        operation="directional ring pair restriction",
    )
    if mutation.state is None:
        return mutation
    capabilities = set(mutation.capabilities)
    if len(models) > 1:
        capabilities.add(
            _WriterExecutionCapabilityKind
            .SHARED_DIRECTIONAL_CARRIER_RESTRICTION
        )
    return _WriterStereoMutation(
        state=mutation.state,
        capabilities=frozenset(capabilities)
        | frozenset((
            _WriterExecutionCapabilityKind.DIRECTIONAL_RING_PAIR_COMPATIBILITY,
        )),
        residual_work_evidence=mutation.residual_work_evidence,
    )


def _directional_ring_pair_event_bond_occurrence(
    prepared: SouthStarPreparedMol,
    event: WriterRingEndpointPaired,
) -> WriterBondOccurrenceRecord | None:
    restrictions = _directional_ring_pair_restrictions(
        prepared,
        bond=event.bond,
        first_atom=event.partner_atom,
        second_atom=event.endpoint_atom,
        first_mark=event.first_endpoint_direction_mark,
        second_mark=event.direction_mark,
    )
    if restrictions is None:
        return None

    if event.first_endpoint_direction_mark is not DirectionMark.ABSENT:
        return WriterBondOccurrenceRecord(
            bond=event.bond,
            parent=event.partner_atom,
            child=event.endpoint_atom,
            mark=event.first_endpoint_direction_mark,
        )
    if event.direction_mark is not DirectionMark.ABSENT:
        return WriterBondOccurrenceRecord(
            bond=event.bond,
            parent=event.endpoint_atom,
            child=event.partner_atom,
            mark=event.direction_mark,
        )
    return WriterBondOccurrenceRecord(
        bond=event.bond,
        parent=event.partner_atom,
        child=event.endpoint_atom,
        mark=DirectionMark.ABSENT,
    )


def _directional_ring_pair_bond_occurrence(
    prepared: SouthStarPreparedMol,
    closure,
) -> WriterBondOccurrenceRecord | None:
    event = WriterRingEndpointPaired(
        bond=closure.bond,
        endpoint_atom=closure.second_atom,
        partner_atom=closure.first_atom,
        label=closure.label,
        endpoint_text=closure.label.text,
        bond_text=closure.second_endpoint_bond_text,
        direction_mark=closure.second_endpoint_direction_mark,
        first_endpoint_bond_text=closure.first_endpoint_bond_text,
        first_endpoint_direction_mark=closure.first_endpoint_direction_mark,
    )
    return _directional_ring_pair_event_bond_occurrence(prepared, event)


def _record_tetra_ring_endpoint(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    event: WriterRingEndpointEmitted | WriterRingEndpointPaired,
) -> _WriterStereoMutation:
    from .writer_state import WriterStereoState

    occurrence_id = _tetra_ring_endpoint_occurrence_id(
        prepared,
        endpoint_atom=event.endpoint_atom,
        partner_atom=event.partner_atom,
        bond=event.bond,
    )
    if occurrence_id is None:
        return _WriterStereoMutation(state=stereo_state)

    record = _local_order_record(stereo_state.local_orders, event.endpoint_atom)
    if record is not None and (record.closed or occurrence_id in record.order):
        return _WriterStereoMutation(state=None)
    if _recorded_tetra_ring_endpoint_occurrences(
        prepared,
        stereo_state,
        event.endpoint_atom,
    ):
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "multiple tetrahedral ring-endpoint incidences are unsupported",
        )

    return _WriterStereoMutation(
        state=WriterStereoState(
            residual_snapshot=stereo_state.residual_snapshot,
            atom_occurrences=stereo_state.atom_occurrences,
            bond_occurrences=stereo_state.bond_occurrences,
            local_orders=_append_local_order(
                stereo_state.local_orders,
                event.endpoint_atom,
                occurrence_id,
            ),
        ),
        capabilities=frozenset((
            _WriterExecutionCapabilityKind.TETRA_RING_ENDPOINT_ORDER_OCCURRENCE,
        )),
    )


def _record_parent_occurrence(
    prepared: SouthStarPreparedMol,
    records: tuple[WriterLocalOrderRecord, ...],
    *,
    atom: AtomId,
    parent: AtomId | None,
) -> tuple[WriterLocalOrderRecord, ...]:
    if parent is None:
        return records
    occurrence = _neighbor_occurrence_by_atom(prepared, atom).get(parent)
    if occurrence is None:
        return records
    return _append_local_order(records, atom, occurrence)


def _record_child_occurrence(
    prepared: SouthStarPreparedMol,
    records: tuple[WriterLocalOrderRecord, ...],
    *,
    parent: AtomId | None,
    child: AtomId,
) -> tuple[WriterLocalOrderRecord, ...]:
    if parent is None:
        return records
    occurrence = _neighbor_occurrence_by_atom(prepared, parent).get(child)
    if occurrence is None:
        return records
    return _append_local_order(records, parent, occurrence)


def _append_local_order(
    records: tuple[WriterLocalOrderRecord, ...],
    atom: AtomId,
    occurrence: OccurrenceId,
) -> tuple[WriterLocalOrderRecord, ...]:
    record = _local_order_record(records, atom)
    if record is None:
        return records + (WriterLocalOrderRecord(atom, (occurrence,), closed=False),)
    if record.closed or occurrence in record.order:
        return records
    return _replace_local_order(
        records,
        WriterLocalOrderRecord(
            atom=atom,
            order=record.order + (occurrence,),
            closed=False,
        ),
    )


def _close_local_order(
    prepared: SouthStarPreparedMol,
    record: WriterLocalOrderRecord | None,
    *,
    atom: AtomId,
) -> WriterLocalOrderRecord:
    order = () if record is None else record.order
    implicit_h = tuple(
        occurrence.id
        for occurrence in prepared.facts.ligand_occurrences
        if occurrence.kind is LigandKind.IMPLICIT_H and occurrence.atom == atom
    )
    return WriterLocalOrderRecord(
        atom=atom,
        order=order + tuple(item for item in implicit_h if item not in order),
        closed=True,
    )


def _replace_local_order(
    records: tuple[WriterLocalOrderRecord, ...],
    replacement: WriterLocalOrderRecord,
) -> tuple[WriterLocalOrderRecord, ...]:
    found = False
    out = []
    for record in records:
        if record.atom == replacement.atom:
            out.append(replacement)
            found = True
        else:
            out.append(record)
    if not found:
        out.append(replacement)
    return tuple(sorted(out, key=lambda item: int(item.atom)))


def _local_order_record(
    records: tuple[WriterLocalOrderRecord, ...],
    atom: AtomId,
) -> WriterLocalOrderRecord | None:
    for record in records:
        if record.atom == atom:
            return record
    return None


def _tetra_domain(template: TetraTemplate) -> tuple[TetraToken, ...]:
    if template.status is SiteStatus.UNSPECIFIED:
        return (TetraToken.NONE,)
    return (TetraToken.AT, TetraToken.ATAT)


def _direction_domain(
    prepared: SouthStarPreparedMol,
    sites: tuple[SiteId, ...],
) -> tuple[DirectionMark, ...]:
    template_by_site = _directional_template_by_site(prepared)
    if any(
        template_by_site[site].status is SiteStatus.SPECIFIED
        for site in sites
    ):
        return (DirectionMark.ABSENT, DirectionMark.FWD, DirectionMark.REV)
    return (DirectionMark.ABSENT,)


def _tetra_template_by_center(
    prepared: SouthStarPreparedMol,
) -> dict[AtomId, TetraTemplate]:
    return {template.center: template for template in prepared.tetra_templates}


def _tetra_ring_endpoint_occurrence_id(
    prepared: SouthStarPreparedMol,
    *,
    endpoint_atom: AtomId,
    partner_atom: AtomId,
    bond: BondId,
) -> OccurrenceId | None:
    template = _tetra_template_by_center(prepared).get(endpoint_atom)
    if template is None:
        return None

    occurrence_by_id = _occurrence_by_id(prepared)
    occurrence_id = _neighbor_ligands_by_bond(
        occurrence_by_id,
        template.ligand_occurrences,
    ).get(bond)
    if occurrence_id is None:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "tetrahedral ring endpoint is not a ligand occurrence",
        )

    occurrence = occurrence_by_id[occurrence_id]
    if occurrence.atom != partner_atom:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "tetrahedral ring endpoint has the wrong partner atom",
        )

    return occurrence_id


def _resolved_tetra_ring_endpoint_occurrences(
    prepared: SouthStarPreparedMol,
    *,
    endpoint_atom: AtomId,
    incidences: tuple[tuple[BondId, AtomId], ...],
) -> tuple[OccurrenceId, ...]:
    resolved = tuple(
        occurrence_id
        for bond, partner in incidences
        if (
            occurrence_id := _tetra_ring_endpoint_occurrence_id(
                prepared,
                endpoint_atom=endpoint_atom,
                partner_atom=partner,
                bond=bond,
            )
        )
        is not None
    )
    if len(frozenset(resolved)) > _MAX_TETRA_RING_ENDPOINT_OCCURRENCES:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "multiple tetrahedral ring-endpoint incidences are unsupported",
        )
    return resolved


def _recorded_tetra_ring_endpoint_occurrences(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    atom: AtomId,
) -> tuple[OccurrenceId, ...]:
    record = _local_order_record(stereo_state.local_orders, atom)
    if record is None:
        return ()

    occurrence_by_id = _occurrence_by_id(prepared)
    emitted_tree_bonds = frozenset(
        item.bond for item in stereo_state.bond_occurrences
    )
    return tuple(
        occurrence_id
        for occurrence_id in record.order
        if (
            (occurrence := occurrence_by_id[occurrence_id]).kind
            is LigandKind.NEIGHBOR_ATOM
            and occurrence.bond not in emitted_tree_bonds
        )
    )


def _directional_template_by_site(
    prepared: SouthStarPreparedMol,
) -> dict[SiteId, DirectionalTemplate]:
    return {template.site: template for template in prepared.directional_templates}


def _directional_sites_for_carrier_bond(
    prepared: SouthStarPreparedMol,
    bond: BondId,
) -> tuple[SiteId, ...]:
    sites = []
    for template in prepared.directional_templates:
        if bond in _directional_template_substituent_bonds(prepared, template):
            sites.append(template.site)
    return tuple(sites)


def _directional_template_substituent_bonds(
    prepared: SouthStarPreparedMol,
    template: DirectionalTemplate,
) -> frozenset[BondId]:
    occurrence_by_id = _occurrence_by_id(prepared)
    bonds: set[BondId] = set()
    for occurrence_id in template.left_ligands + template.right_ligands:
        occurrence = occurrence_by_id[occurrence_id]
        if occurrence.kind is LigandKind.NEIGHBOR_ATOM:
            if occurrence.bond is None:
                raise SouthStarError(
                    SouthStarErrorKind.UNSUPPORTED_STEREO,
                    "directional neighbor occurrence lacks a bond",
                )
            bonds.add(occurrence.bond)
    return frozenset(bonds)


def _unsupported_directional_non_neighbor_ligand_blocker_for_bond(
    prepared: SouthStarPreparedMol,
    bond: BondId,
    *,
    operation: str,
) -> WriterStereoPolicyBlocker | None:
    occurrence_by_id = _occurrence_by_id(prepared)
    for template in sorted(
        prepared.directional_templates,
        key=lambda item: int(item.site),
    ):
        ligand_ids = template.left_ligands + template.right_ligands
        if not any(
            occurrence_by_id[item].kind is not LigandKind.NEIGHBOR_ATOM
            for item in ligand_ids
        ):
            continue
        if bond not in _directional_template_substituent_bonds(
            prepared,
            template,
        ):
            continue
        return WriterStereoPolicyBlocker(
            kind="unsupported_directional_non_neighbor_ligand",
            site=template.site,
            operation=operation,
        )

    return None


def _tetra_factor_key(site: SiteId) -> ResidualFactorKey:
    return ResidualFactorKey("tetra_site", (int(site),))


def _directional_site_factor_key(site: SiteId) -> ResidualFactorKey:
    return ResidualFactorKey("directional_site", (int(site),))


def _directional_bond_factor_key(bond: BondId) -> ResidualFactorKey:
    return ResidualFactorKey("directional_bond_emission", (int(bond),))


def _directional_normalized_domain() -> tuple[DirectionalNormalizedSign, ...]:
    return (
        DirectionalNormalizedSign.ABSENT,
        DirectionalNormalizedSign.POSITIVE,
        DirectionalNormalizedSign.NEGATIVE,
    )


def _directional_site_carrier_models(
    prepared: SouthStarPreparedMol,
    template: DirectionalTemplate,
) -> tuple[tuple[VarId, DirectionalSiteCarrierModel], ...]:
    occurrence_by_id = _occurrence_by_id(prepared)
    left_reference, right_reference = _directional_reference_pair(template)
    left_by_bond = _neighbor_ligands_by_bond(occurrence_by_id, template.left_ligands)
    right_by_bond = _neighbor_ligands_by_bond(occurrence_by_id, template.right_ligands)
    entries: list[tuple[VarId, DirectionalSiteCarrierModel]] = []
    for bond, occurrence in left_by_bond.items():
        model = DirectionalSiteCarrierModel(
            site=template.site,
            bond=bond,
            side="left",
            endpoint_orientation_factor=_bond_endpoint_orientation_factor(
                prepared,
                bond,
                template.left_endpoint,
            ),
            ligand_factor=_ligand_factor(
                occurrence,
                reference=left_reference,
                side_ligands=template.left_ligands,
            ),
        )
        entries.append((directional_site_carrier_var(template.site, bond), model))
    for bond, occurrence in right_by_bond.items():
        model = DirectionalSiteCarrierModel(
            site=template.site,
            bond=bond,
            side="right",
            endpoint_orientation_factor=_bond_endpoint_orientation_factor(
                prepared,
                bond,
                template.right_endpoint,
            ),
            ligand_factor=_ligand_factor(
                occurrence,
                reference=right_reference,
                side_ligands=template.right_ligands,
            ),
        )
        entries.append((directional_site_carrier_var(template.site, bond), model))
    return tuple(sorted(entries, key=lambda item: _var_sort_tuple(item[0])))


def _directional_models_for_bond(
    prepared: SouthStarPreparedMol,
    bond: BondId,
) -> tuple[DirectionalSiteCarrierModel, ...]:
    models = []
    for template in prepared.directional_templates:
        for _, model in _directional_site_carrier_models(prepared, template):
            if model.bond == bond:
                models.append(model)
    return tuple(
        sorted(
            models,
            key=lambda model: (
                int(model.site),
                int(model.bond),
                model.side,
                model.endpoint_orientation_factor,
                model.ligand_factor,
            ),
        )
    )


def _bounded_directional_ring_models(
    prepared: SouthStarPreparedMol,
    bond: BondId,
) -> tuple[DirectionalSiteCarrierModel, ...]:
    models = _directional_models_for_bond(prepared, bond)
    if len(models) > _MAX_DIRECTIONAL_RING_CARRIER_SITES:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "more than two directional sites for one ring carrier are unsupported",
        )
    return models


def _allowed_direction_marks(
    prepared: SouthStarPreparedMol,
    bond: BondId,
) -> tuple[DirectionMark, ...]:
    try:
        choices = prepared.policy.bond_text_domain_unchecked(
            bond,
            slot_kind="tree",
        )
    except KeyError as exc:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            f"WRITER_SHAPED has no acyclic writer bond text for {bond!r}",
        ) from exc
    allowed = [DirectionMark.ABSENT]
    if any(choice.permits_direction for choice in choices):
        allowed.extend((DirectionMark.FWD, DirectionMark.REV))
    return tuple(allowed)


def _canonical_bond_orientation(
    prepared: SouthStarPreparedMol,
    event: WriterBondEmitted,
) -> Literal[-1, 1]:
    bond = prepared.graph_index.bond_by_id[event.bond]
    if event.parent == bond.a and event.child == bond.b:
        return 1
    if event.parent == bond.b and event.child == bond.a:
        return -1
    raise SouthStarError(
        SouthStarErrorKind.UNSUPPORTED_STEREO,
        "writer bond event is not oriented along its graph bond",
    )


def _bond_endpoint_orientation_factor(
    prepared: SouthStarPreparedMol,
    bond: BondId,
    endpoint: AtomId,
) -> Literal[-1, 1]:
    graph_bond = prepared.graph_index.bond_by_id[bond]
    if graph_bond.a == endpoint:
        return 1
    if graph_bond.b == endpoint:
        return -1
    raise SouthStarError(
        SouthStarErrorKind.UNSUPPORTED_STEREO,
        "directional carrier is not incident to its alkene endpoint",
    )


def _tetra_local_parity(
    template: TetraTemplate,
    local_order: tuple[OccurrenceId, ...],
) -> TetraLocalParity:
    return (
        TetraLocalParity.EVEN
        if _is_even_permutation(template.reference_order, local_order)
        else TetraLocalParity.ODD
    )


def _is_even_permutation(
    reference_order: tuple[OccurrenceId, ...],
    local_order: tuple[OccurrenceId, ...],
) -> bool:
    if set(reference_order) != set(local_order):
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "tetrahedral local order does not match the template reference order",
        )
    positions = {item: index for index, item in enumerate(reference_order)}
    indices = tuple(positions[item] for item in local_order)
    inversions = 0
    for index, left in enumerate(indices):
        for right in indices[index + 1:]:
            if left > right:
                inversions += 1
    return inversions % 2 == 0


def _ligand_factor(
    occurrence: OccurrenceId,
    *,
    reference: OccurrenceId,
    side_ligands: tuple[OccurrenceId, ...],
) -> Literal[-1, 1]:
    if occurrence == reference:
        return 1
    if occurrence not in side_ligands:
        raise ValueError("occurrence is not on directional side")
    return -1


def _directional_reference_pair(
    template: DirectionalTemplate,
) -> tuple[OccurrenceId, OccurrenceId]:
    if template.reference_pair is not None:
        return template.reference_pair
    if template.status is SiteStatus.SPECIFIED:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "specified directional site lacks a reference pair",
        )
    return (min(template.left_ligands, key=int), min(template.right_ligands, key=int))


def _neighbor_ligands_by_bond(
    occurrence_by_id,
    ligand_ids: tuple[OccurrenceId, ...],
) -> dict[BondId, OccurrenceId]:
    out = {}
    for ligand_id in ligand_ids:
        occurrence = occurrence_by_id[ligand_id]
        if occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
            continue
        if occurrence.bond is None:
            raise ValueError("neighbor occurrence lacks bond")
        out[occurrence.bond] = ligand_id
    return out


def _neighbor_occurrence_by_atom(
    prepared: SouthStarPreparedMol,
    atom: AtomId,
) -> dict[AtomId, OccurrenceId]:
    out: dict[AtomId, OccurrenceId] = {}
    for template in prepared.tetra_templates:
        if template.center != atom:
            continue
        for occurrence_id in template.ligand_occurrences:
            occurrence = _occurrence_by_id(prepared)[occurrence_id]
            if occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
                continue
            if occurrence.atom is not None:
                out[occurrence.atom] = occurrence.id
    return out


def _occurrence_by_id(prepared: SouthStarPreparedMol):
    return {occurrence.id: occurrence for occurrence in prepared.facts.ligand_occurrences}


def _residual_snapshot_sort_tuple(
    snapshot: ResidualStoreValueSnapshot,
) -> tuple[object, ...]:
    return (
        tuple(
            (
                _var_sort_tuple(var),
                tuple(_value_sort_tuple(value) for value in domain),
            )
            for var, domain in snapshot.domains
        ),
        tuple(
            (_var_sort_tuple(var), _value_sort_tuple(value))
            for var, value in snapshot.assignments
        ),
        tuple(_factor_snapshot_sort_tuple(factor) for factor in snapshot.factors),
    )


def _factor_snapshot_sort_tuple(factor: object) -> tuple[object, ...]:
    if isinstance(factor, TetraResidualFactorValueSnapshot):
        return (
            "tetra",
            _factor_key_sort_tuple(factor.key),
            tuple(_var_sort_tuple(var) for var in factor.scope),
            factor.status.value,
            factor.target.value,
            tuple(int(item) for item in factor.reference_order),
            tuple(int(item) for item in factor.local_order),
        )
    if isinstance(factor, DirectionalResidualFactorValueSnapshot):
        return (
            "directional",
            _factor_key_sort_tuple(factor.key),
            tuple(_var_sort_tuple(var) for var in factor.scope),
            factor.status.value,
            factor.target.value,
            tuple(
                (
                    _var_sort_tuple(var),
                    model.side,
                    model.orientation,
                    model.ligand_factor,
                )
                for var, model in factor.carrier_models
            ),
        )
    if isinstance(factor, TetraTokenParityFactorValueSnapshot):
        return (
            "tetra_token_parity",
            _factor_key_sort_tuple(factor.key),
            tuple(_var_sort_tuple(var) for var in factor.scope),
            factor.status.value,
            factor.target.value,
        )
    if isinstance(factor, DirectionalSiteFactorValueSnapshot):
        return (
            "directional_site",
            _factor_key_sort_tuple(factor.key),
            tuple(_var_sort_tuple(var) for var in factor.scope),
            tuple((_var_sort_tuple(var), side) for var, side in factor.sides),
            factor.status.value,
            factor.target.value,
        )
    if isinstance(factor, DirectionalBondEmissionFactorValueSnapshot):
        return (
            "directional_bond_emission",
            _factor_key_sort_tuple(factor.key),
            tuple(_var_sort_tuple(var) for var in factor.scope),
            tuple(
                (
                    int(model.site),
                    int(model.bond),
                    model.side,
                    model.endpoint_orientation_factor,
                    model.ligand_factor,
                )
                for model in factor.models
            ),
            tuple(mark.value for mark in factor.allowed_marks),
        )
    raise TypeError(f"unknown residual factor snapshot: {factor!r}")


def _factor_key_sort_tuple(key: ResidualFactorKey) -> tuple[object, ...]:
    return (key.kind, tuple(_value_sort_tuple(item) for item in key.key))


def _var_sort_tuple(var: VarId) -> tuple[object, ...]:
    return (var.kind, tuple(_value_sort_tuple(item) for item in var.key))


def _value_sort_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, (int, str)):
        return (type(value).__name__, value)
    if isinstance(
        value,
        (TetraToken, DirectionMark, TetraLocalParity, DirectionalNormalizedSign),
    ):
        return (value.__class__.__name__, value.value)
    if isinstance(value, ResidualFactorKey):
        return ("ResidualFactorKey", _factor_key_sort_tuple(value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_value_sort_tuple(item) for item in value))
    return (value.__class__.__name__, str(value))


def _atom_record_sort_tuple(record: WriterAtomOccurrenceRecord) -> tuple[object, ...]:
    return (
        int(record.atom),
        record.token.value,
    )


def _bond_record_sort_tuple(record: WriterBondOccurrenceRecord) -> tuple[object, ...]:
    return (
        int(record.bond),
        int(record.parent),
        int(record.child),
        record.mark.value,
    )


def _local_order_sort_tuple(record: WriterLocalOrderRecord) -> tuple[object, ...]:
    return (int(record.atom), tuple(int(item) for item in record.order), record.closed)


__all__ = (
    "EMPTY_RESIDUAL_SNAPSHOT",
    "WriterAtomOccurrenceRecord",
    "WriterAtomTextChoice",
    "WriterBondOccurrenceRecord",
    "WriterBondTextChoice",
    "WriterLocalOrderRecord",
    "WriterStereoPolicyBlocker",
    "_WriterStereoAdvanceOutcome",
    "advance_writer_stereo_state",
    "advance_writer_stereo_state_with_evidence",
    "empty_writer_stereo_state",
    "initial_writer_stereo_state",
    "_writer_stereo_relation_definitions",
    "reconstruct_writer_local_order_records",
    "reconstruct_writer_stereo_residual_snapshot",
    "terminal_writer_stereo_state",
    "terminal_writer_stereo_state_with_evidence",
    "validate_writer_stereo_supported_prepared",
    "writer_atom_text_choices",
    "writer_bond_text_choices",
    "writer_closure_endpoint_relation",
    "writer_stereo_state_sort_tuple",
)
