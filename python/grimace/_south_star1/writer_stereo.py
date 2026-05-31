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
from .policy import TetraToken
from .residual_constraints import DirectionalCarrierResidual
from .residual_constraints import DirectionalResidualFactor
from .residual_constraints import DirectionalResidualFactorValueSnapshot
from .residual_constraints import ResidualStore
from .residual_constraints import ResidualStoreValueSnapshot
from .residual_constraints import TetraResidualFactor
from .residual_constraints import TetraResidualFactorValueSnapshot
from .residual_constraints import VarId
from .residual_constraints import add_factor_checked
from .residual_constraints import direction_var
from .residual_constraints import tetra_var
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
    var: VarId | None


@dataclass(frozen=True, slots=True)
class WriterBondOccurrenceRecord:
    bond: BondId
    parent: AtomId
    child: AtomId
    mark: DirectionMark
    var: VarId | None


@dataclass(frozen=True, slots=True)
class WriterLocalOrderRecord:
    atom: AtomId
    order: tuple[OccurrenceId, ...]
    closed: bool = False


@dataclass(frozen=True, slots=True)
class WriterDelayedStereoFactor:
    kind: Literal["tetra", "directional", "ring_pair"]
    site: SiteId
    scope: tuple[VarId, ...] = ()
    evidence: tuple[tuple[object, ...], ...] = ()
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
        delayed_factors=(),
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
    return _on_local_order_closed(prepared, stereo_state, atom)


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
        tuple(_delayed_factor_sort_tuple(factor) for factor in state.delayed_factors),
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
    var = None
    delayed = stereo_state.delayed_factors
    if template is not None:
        var = tetra_var(("writer", int(template.site)))
        if store.assignment(var) is None:
            store.add_var(var, _tetra_domain(template))
        if not store.assign(var, event.tetra_token):
            return None
        delayed = _mark_factor_pending(
            delayed,
            WriterDelayedStereoFactor(
                kind="tetra",
                site=template.site,
                scope=(var,),
                evidence=(("atom", int(event.atom)),),
                closed=False,
            ),
        )
    elif event.tetra_token is not TetraToken.NONE:
        return None
    return WriterStereoState(
        residual_snapshot=store.value_snapshot(),
        atom_occurrences=stereo_state.atom_occurrences
        + (WriterAtomOccurrenceRecord(event.atom, event.tetra_token, var),),
        bond_occurrences=stereo_state.bond_occurrences,
        local_orders=local_orders,
        delayed_factors=delayed,
    )


def _on_bond_emitted(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    event: WriterBondEmitted,
) -> "WriterStereoState | None":
    from .writer_state import WriterStereoState

    store = ResidualStore.from_value_snapshot(stereo_state.residual_snapshot)
    eligible = _directional_sites_for_carrier_bond(prepared, event.bond)
    var = None
    delayed = stereo_state.delayed_factors
    if eligible:
        var = direction_var(("writer", int(event.bond)))
        if store.assignment(var) is None:
            store.add_var(var, _direction_domain(prepared, eligible))
        if not store.assign(var, event.direction_mark):
            return None
        for site in eligible:
            delayed = _mark_factor_pending(
                delayed,
                _updated_directional_pending(
                    delayed,
                    site=site,
                    var=var,
                    bond=event.bond,
                ),
            )
    elif event.direction_mark is not DirectionMark.ABSENT:
        return None
    next_state = WriterStereoState(
        residual_snapshot=store.value_snapshot(),
        atom_occurrences=stereo_state.atom_occurrences,
        bond_occurrences=stereo_state.bond_occurrences
        + (
            WriterBondOccurrenceRecord(
                event.bond,
                event.parent,
                event.child,
                event.direction_mark,
                var,
            ),
        ),
        local_orders=stereo_state.local_orders,
        delayed_factors=delayed,
    )
    return _close_ready_directional_factors(prepared, next_state)


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
    delayed = stereo_state.delayed_factors
    if template is not None:
        var = tetra_var(("writer", int(template.site)))
        if store.assignment(var) is None:
            return None
        factor = TetraResidualFactor(
            scope=(var,),
            status=template.status,
            target=template.target,
            reference_order=template.reference_order,
            local_order=closed_order.order,
        )
        if not add_factor_checked(store, factor):
            return None
        if not factor.close():
            return None
        delayed = _mark_factor_closed(
            delayed,
            WriterDelayedStereoFactor(
                kind="tetra",
                site=template.site,
                scope=(var,),
                evidence=(("atom", int(atom)),),
                closed=True,
            ),
        )
    return WriterStereoState(
        residual_snapshot=store.value_snapshot(),
        atom_occurrences=stereo_state.atom_occurrences,
        bond_occurrences=stereo_state.bond_occurrences,
        local_orders=local_orders,
        delayed_factors=delayed,
    )


def _on_ring_endpoint_emitted(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    event: WriterRingEndpointEmitted,
) -> "WriterStereoState | None":
    _reject_supported_ring_pair_stereo(prepared, event.bond)
    from .writer_state import WriterStereoState

    return WriterStereoState(
        residual_snapshot=stereo_state.residual_snapshot,
        atom_occurrences=stereo_state.atom_occurrences,
        bond_occurrences=stereo_state.bond_occurrences,
        local_orders=stereo_state.local_orders,
        delayed_factors=_mark_factor_pending(
            stereo_state.delayed_factors,
            WriterDelayedStereoFactor(
                kind="ring_pair",
                site=SiteId(int(event.bond)),
                scope=(),
                evidence=(_ring_endpoint_evidence(event),),
                closed=False,
            ),
        ),
    )


def _on_ring_endpoint_paired(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
    event: WriterRingEndpointPaired,
) -> "WriterStereoState | None":
    _reject_supported_ring_pair_stereo(prepared, event.bond)
    from .writer_state import WriterStereoState

    pending = next(
        (
            factor
            for factor in stereo_state.delayed_factors
            if factor.kind == "ring_pair"
            and factor.site == SiteId(int(event.bond))
            and not factor.closed
        ),
        None,
    )
    if pending is None or len(pending.evidence) != 1:
        return None
    first_evidence = pending.evidence[0]
    if (
        len(first_evidence) != 9
        or first_evidence[0] != "ring_endpoint"
        or first_evidence[1] != int(event.bond)
        or first_evidence[2] != "open"
        or first_evidence[3] != int(event.partner_atom)
        or first_evidence[4] != int(event.endpoint_atom)
        or first_evidence[5] != event.label.value
        or first_evidence[6] != event.label.text
    ):
        return None
    return WriterStereoState(
        residual_snapshot=stereo_state.residual_snapshot,
        atom_occurrences=stereo_state.atom_occurrences,
        bond_occurrences=stereo_state.bond_occurrences,
        local_orders=stereo_state.local_orders,
        delayed_factors=_mark_factor_closed(
            stereo_state.delayed_factors,
            WriterDelayedStereoFactor(
                kind="ring_pair",
                site=SiteId(int(event.bond)),
                scope=(),
                evidence=(_ring_pair_evidence(first_evidence, event),),
                closed=True,
            ),
        ),
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


def _ring_endpoint_evidence(event: WriterRingEndpointEmitted) -> tuple[object, ...]:
    return (
        "ring_endpoint",
        int(event.bond),
        event.side,
        int(event.endpoint_atom),
        int(event.partner_atom),
        event.label.value,
        event.label.text,
        event.endpoint_text,
        event.bond_text,
    )


def _ring_pair_evidence(
    first_evidence: tuple[object, ...],
    event: WriterRingEndpointPaired,
) -> tuple[object, ...]:
    return (
        "ring_pair",
        int(event.bond),
        int(event.partner_atom),
        int(event.endpoint_atom),
        event.label.value,
        event.label.text,
        first_evidence[7],
        event.endpoint_text,
        first_evidence[8],
        event.bond_text,
    )


def _close_ready_directional_factors(
    prepared: SouthStarPreparedMol,
    stereo_state: "WriterStereoState",
) -> "WriterStereoState | None":
    from .writer_state import WriterStereoState

    store = ResidualStore.from_value_snapshot(stereo_state.residual_snapshot)
    bond_records = {record.bond: record for record in stereo_state.bond_occurrences}
    delayed = stereo_state.delayed_factors
    changed = False
    for template in prepared.directional_templates:
        if _factor_already_closed(delayed, "directional", template.site):
            continue
        carrier_bonds = _directional_template_substituent_bonds(prepared, template)
        if not carrier_bonds.issubset(bond_records):
            continue
        models = _directional_models(prepared, template, bond_records)
        factor = DirectionalResidualFactor(
            scope=tuple(sorted(models, key=_var_sort_tuple)),
            status=template.status,
            target=template.target,
            carrier_models=models,
        )
        if not add_factor_checked(store, factor):
            return None
        if not factor.close():
            return None
        delayed = _mark_factor_closed(
            delayed,
            WriterDelayedStereoFactor(
                kind="directional",
                site=template.site,
                scope=tuple(sorted(models, key=_var_sort_tuple)),
                evidence=tuple(
                    sorted(
                        ("bond", int(bond))
                        for bond in carrier_bonds
                    )
                ),
                closed=True,
            ),
        )
        changed = True
    if not changed:
        return stereo_state
    return WriterStereoState(
        residual_snapshot=store.value_snapshot(),
        atom_occurrences=stereo_state.atom_occurrences,
        bond_occurrences=stereo_state.bond_occurrences,
        local_orders=stereo_state.local_orders,
        delayed_factors=delayed,
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


def _directional_models(
    prepared: SouthStarPreparedMol,
    template: DirectionalTemplate,
    bond_records: dict[BondId, WriterBondOccurrenceRecord],
) -> dict[VarId, DirectionalCarrierResidual]:
    occurrence_by_id = _occurrence_by_id(prepared)
    left_reference, right_reference = _directional_reference_pair(template)
    left_by_bond = _neighbor_ligands_by_bond(occurrence_by_id, template.left_ligands)
    right_by_bond = _neighbor_ligands_by_bond(occurrence_by_id, template.right_ligands)
    models: dict[VarId, DirectionalCarrierResidual] = {}
    for bond, occurrence in left_by_bond.items():
        record = bond_records[bond]
        var = direction_var(("writer", int(bond)))
        models[var] = DirectionalCarrierResidual(
            var=var,
            side="left",
            orientation=_carrier_orientation(record, template.left_endpoint),
            ligand_factor=_ligand_factor(
                occurrence,
                reference=left_reference,
                side_ligands=template.left_ligands,
            ),
        )
    for bond, occurrence in right_by_bond.items():
        record = bond_records[bond]
        var = direction_var(("writer", int(bond)))
        models[var] = DirectionalCarrierResidual(
            var=var,
            side="right",
            orientation=_carrier_orientation(record, template.right_endpoint),
            ligand_factor=_ligand_factor(
                occurrence,
                reference=right_reference,
                side_ligands=template.right_ligands,
            ),
        )
    return models


def _carrier_orientation(
    record: WriterBondOccurrenceRecord,
    endpoint: AtomId,
) -> Literal[-1, 1]:
    if record.parent == endpoint:
        return 1
    if record.child == endpoint:
        return -1
    raise SouthStarError(
        SouthStarErrorKind.UNSUPPORTED_STEREO,
        "directional carrier is not incident to its alkene endpoint",
    )


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


def _mark_factor_closed(
    factors: tuple[WriterDelayedStereoFactor, ...],
    replacement: WriterDelayedStereoFactor,
) -> tuple[WriterDelayedStereoFactor, ...]:
    out = tuple(
        factor
        for factor in factors
        if not (factor.kind == replacement.kind and factor.site == replacement.site)
    )
    return tuple(sorted(out + (replacement,), key=_delayed_factor_sort_tuple))


def _mark_factor_pending(
    factors: tuple[WriterDelayedStereoFactor, ...],
    replacement: WriterDelayedStereoFactor,
) -> tuple[WriterDelayedStereoFactor, ...]:
    if _factor_already_closed(factors, replacement.kind, replacement.site):
        return factors
    out = tuple(
        factor
        for factor in factors
        if not (factor.kind == replacement.kind and factor.site == replacement.site)
    )
    return tuple(sorted(out + (replacement,), key=_delayed_factor_sort_tuple))


def _updated_directional_pending(
    factors: tuple[WriterDelayedStereoFactor, ...],
    *,
    site: SiteId,
    var: VarId,
    bond: BondId,
) -> WriterDelayedStereoFactor:
    existing = next(
        (
            factor
            for factor in factors
            if factor.kind == "directional" and factor.site == site
        ),
        None,
    )
    scope = (var,) if existing is None else existing.scope + (var,)
    evidence = (
        (("bond", int(bond)),)
        if existing is None
        else existing.evidence + (("bond", int(bond)),)
    )
    return WriterDelayedStereoFactor(
        kind="directional",
        site=site,
        scope=tuple(sorted(set(scope), key=_var_sort_tuple)),
        evidence=tuple(sorted(set(evidence))),
        closed=False,
    )


def _factor_already_closed(
    factors: tuple[WriterDelayedStereoFactor, ...],
    kind: Literal["tetra", "directional", "ring_pair"],
    site: SiteId,
) -> bool:
    return any(
        factor.kind == kind and factor.site == site and factor.closed
        for factor in factors
    )


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
            tuple(_var_sort_tuple(var) for var in factor.scope),
            factor.status.value,
            factor.target.value,
            tuple(int(item) for item in factor.reference_order),
            tuple(int(item) for item in factor.local_order),
            _value_sort_tuple(factor.assigned),
        )
    if isinstance(factor, DirectionalResidualFactorValueSnapshot):
        return (
            "directional",
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
            tuple(
                (_var_sort_tuple(var), _value_sort_tuple(value))
                for var, value in factor.marks
            ),
        )
    raise TypeError(f"unknown residual factor snapshot: {factor!r}")


def _var_sort_tuple(var: VarId) -> tuple[object, ...]:
    return (var.kind, tuple(_value_sort_tuple(item) for item in var.key))


def _value_sort_tuple(value: object) -> tuple[object, ...]:
    if isinstance(value, (int, str)):
        return (type(value).__name__, value)
    if isinstance(value, (TetraToken, DirectionMark)):
        return (value.__class__.__name__, value.value)
    if isinstance(value, tuple):
        return ("tuple", tuple(_value_sort_tuple(item) for item in value))
    return (value.__class__.__name__, str(value))


def _atom_record_sort_tuple(record: WriterAtomOccurrenceRecord) -> tuple[object, ...]:
    return (
        int(record.atom),
        record.token.value,
        None if record.var is None else _var_sort_tuple(record.var),
    )


def _bond_record_sort_tuple(record: WriterBondOccurrenceRecord) -> tuple[object, ...]:
    return (
        int(record.bond),
        int(record.parent),
        int(record.child),
        record.mark.value,
        None if record.var is None else _var_sort_tuple(record.var),
    )


def _local_order_sort_tuple(record: WriterLocalOrderRecord) -> tuple[object, ...]:
    return (int(record.atom), tuple(int(item) for item in record.order), record.closed)


def _delayed_factor_sort_tuple(factor: WriterDelayedStereoFactor) -> tuple[object, ...]:
    return (
        factor.kind,
        int(factor.site),
        tuple(_var_sort_tuple(var) for var in factor.scope),
        factor.evidence,
        factor.closed,
    )


__all__ = (
    "EMPTY_RESIDUAL_SNAPSHOT",
    "WriterAtomOccurrenceRecord",
    "WriterAtomTextChoice",
    "WriterBondOccurrenceRecord",
    "WriterBondTextChoice",
    "WriterDelayedStereoFactor",
    "WriterLocalOrderRecord",
    "advance_writer_stereo_state",
    "empty_writer_stereo_state",
    "terminal_writer_stereo_state",
    "validate_writer_stereo_supported_prepared",
    "writer_atom_text_choices",
    "writer_bond_text_choices",
    "writer_stereo_state_sort_tuple",
)
