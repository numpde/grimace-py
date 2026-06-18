"""Writer-owned residual stereo state advancement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import LigandKind
from .facts import SiteStatus
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
    from .writer_state import WriterStereoState


EMPTY_RESIDUAL_SNAPSHOT = ResidualStore().value_snapshot()


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


def advance_writer_stereo_state(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    events: tuple[WriterEvent, ...],
) -> "WriterStereoState | None":
    state = stereo_state
    for event in events:
        if isinstance(event, WriterAtomEmitted):
            state = _on_atom_emitted(prepared, state, event)
        elif isinstance(event, WriterBondEmitted):
            state = _on_bond_emitted(prepared, state, event)
        elif isinstance(event, WriterLocalOrderClosed):
            state = _on_local_order_closed(prepared, state, event.atom)
        elif isinstance(event, WriterRingEndpointEmitted):
            state = _on_ring_endpoint_emitted(prepared, state, event)
        elif isinstance(event, WriterRingEndpointPaired):
            state = _on_ring_endpoint_paired(prepared, state, event)
        else:
            continue
        if state is None:
            return None
    return state


def terminal_writer_stereo_state(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    atom: AtomId,
) -> "WriterStereoState | None":
    state = advance_writer_stereo_state(
        prepared,
        stereo_state,
        (WriterLocalOrderClosed(atom=atom),),
    )
    if state is None:
        return None
    if state.residual_snapshot != EMPTY_RESIDUAL_SNAPSHOT:
        return None
    return state


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


def writer_stereo_state_sort_tuple(state: "WriterStereoState") -> tuple[object, ...]:
    return (
        _residual_snapshot_sort_tuple(state.residual_snapshot),
        tuple(_atom_record_sort_tuple(record) for record in state.atom_occurrences),
        tuple(_bond_record_sort_tuple(record) for record in state.bond_occurrences),
        tuple(_local_order_sort_tuple(record) for record in state.local_orders),
    )


def validate_writer_stereo_supported_prepared(prepared: SouthStarPreparedMol) -> None:
    occurrence_by_id = _occurrence_by_id(prepared)
    if any(
        occurrence_by_id[item].kind is not LigandKind.NEIGHBOR_ATOM
        for template in prepared.directional_templates
        for item in template.left_ligands + template.right_ligands
    ):
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "WRITER_SHAPED directional stereo currently requires neighbor ligands",
        )


def _on_atom_emitted(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    event: WriterAtomEmitted,
) -> "WriterStereoState | None":
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
    template = _tetra_template_by_center(prepared).get(event.atom)
    checkpoint = store.checkpoint()
    if template is not None:
        result = store.restrict_many_and_propagate(
            ((tetra_token_var(template.site), event.tetra_token),)
        )
        if not _writer_residual_mutation_is_legal(
            result,
            operation="tetrahedral atom-token restriction",
        ):
            store.rollback(checkpoint)
            return None
    elif event.tetra_token is not TetraToken.NONE:
        return None
    return WriterStereoState(
        residual_snapshot=store.value_snapshot(),
        atom_occurrences=stereo_state.atom_occurrences
        + (WriterAtomOccurrenceRecord(event.atom, event.tetra_token),),
        bond_occurrences=stereo_state.bond_occurrences,
        local_orders=local_orders,
    )


def _on_bond_emitted(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    event: WriterBondEmitted,
) -> "WriterStereoState | None":
    from .writer_state import WriterStereoState

    store = ResidualStore.from_value_snapshot(stereo_state.residual_snapshot)
    models = _directional_models_for_bond(prepared, event.bond)
    checkpoint = store.checkpoint()
    if models:
        restrictions = tuple(
            (
                directional_site_carrier_var(model.site, event.bond),
                normalized_sign_from_mark(
                    mark=event.direction_mark,
                    canonical_orientation=_canonical_bond_orientation(
                        prepared,
                        event,
                    ),
                    model=model,
                ),
            )
            for model in models
        )
        result = store.restrict_many_and_propagate(restrictions)
        if not _writer_residual_mutation_is_legal(
            result,
            operation="directional carrier-mark restriction",
        ):
            store.rollback(checkpoint)
            return None
    elif event.direction_mark is not DirectionMark.ABSENT:
        return None
    if models:
        try:
            store.discharge_satisfied_factors((_directional_bond_factor_key(event.bond),))
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
        except ValueError:
            store.rollback(checkpoint)
            return None

    return WriterStereoState(
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
    )


def _on_local_order_closed(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    atom: AtomId,
) -> "WriterStereoState | None":
    from .writer_state import WriterStereoState

    template = _tetra_template_by_center(prepared).get(atom)
    record = _local_order_record(stereo_state.local_orders, atom)
    if record is not None and record.closed:
        return stereo_state
    closed_order = _close_local_order(prepared, record, atom=atom)
    local_orders = _replace_local_order(stereo_state.local_orders, closed_order)
    store = ResidualStore.from_value_snapshot(stereo_state.residual_snapshot)
    if template is not None:
        checkpoint = store.checkpoint()
        result = store.restrict_many_and_propagate(
            (
                (
                    tetra_parity_var(template.site),
                    _tetra_local_parity(template, closed_order.order),
                ),
            )
        )
        if not _writer_residual_mutation_is_legal(
            result,
            operation="tetrahedral local-order factor closure",
        ):
            store.rollback(checkpoint)
            return None
        try:
            store.discharge_satisfied_factors((_tetra_factor_key(template.site),))
        except ValueError:
            store.rollback(checkpoint)
            return None
    return WriterStereoState(
        residual_snapshot=store.value_snapshot(),
        atom_occurrences=stereo_state.atom_occurrences,
        bond_occurrences=stereo_state.bond_occurrences,
        local_orders=local_orders,
    )


def _on_ring_endpoint_emitted(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    event: WriterRingEndpointEmitted,
) -> "WriterStereoState | None":
    _reject_supported_ring_pair_stereo(prepared, event.bond)
    if not _ring_event_text_ok(prepared, event):
        return None
    from .writer_state import WriterStereoState

    return WriterStereoState(
        residual_snapshot=stereo_state.residual_snapshot,
        atom_occurrences=stereo_state.atom_occurrences,
        bond_occurrences=stereo_state.bond_occurrences,
        local_orders=stereo_state.local_orders,
    )


def _on_ring_endpoint_paired(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    event: WriterRingEndpointPaired,
) -> "WriterStereoState | None":
    _reject_supported_ring_pair_stereo(prepared, event.bond)
    if not _ring_event_text_ok(prepared, event):
        return None
    from .writer_state import WriterStereoState

    return WriterStereoState(
        residual_snapshot=stereo_state.residual_snapshot,
        atom_occurrences=stereo_state.atom_occurrences,
        bond_occurrences=stereo_state.bond_occurrences,
        local_orders=stereo_state.local_orders,
    )


def _ring_event_text_ok(prepared: SouthStarPreparedMol, event) -> bool:
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
    return event.bond_text in _ring_endpoint_bond_texts(prepared, event.bond)


def _ring_endpoint_bond_texts(
    prepared: SouthStarPreparedMol,
    bond: BondId,
) -> frozenset[str]:
    try:
        choices = prepared.policy.bond_text_domain_unchecked(
            bond,
            slot_kind="ring_endpoint",
        )
    except KeyError:
        return frozenset()
    return frozenset(
        choice.base_text
        for choice in choices
        if choice.base_text not in {"/", "\\"}
    )


def _reject_supported_ring_pair_stereo(
    prepared: SouthStarPreparedMol,
    bond: BondId,
) -> None:
    if _directional_sites_for_carrier_bond(prepared, bond):
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_STEREO,
            "WRITER_SHAPED ring-pair directional stereo is not supported yet",
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
    "advance_writer_stereo_state",
    "empty_writer_stereo_state",
    "initial_writer_stereo_state",
    "_writer_stereo_relation_definitions",
    "terminal_writer_stereo_state",
    "validate_writer_stereo_supported_prepared",
    "writer_atom_text_choices",
    "writer_bond_text_choices",
    "writer_stereo_state_sort_tuple",
)
