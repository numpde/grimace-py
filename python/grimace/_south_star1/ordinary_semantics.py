"""RDKit-free parser semantics for the bounded ordinary SMILES dialect."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

from .facts import BondFacts
from .facts import BondOrder
from .facts import DirectionalSiteFacts
from .facts import DirectionalValue
from .facts import LigandKind
from .facts import LigandOccurrence
from .facts import MoleculeFacts
from .facts import SiteStatus
from .facts import TetraValue
from .facts import TetrahedralSiteFacts
from .ids import AtomId
from .ids import BondId
from .ids import CarrierSlotId
from .ids import OccurrenceId
from .ids import SiteId
from .policy import AtomTextChoice
from .policy import BondTextChoice
from .policy import DirectionMark
from .policy import TetraToken
from .semantics import INVALID
from .semantics import Invalid
from .semantics import ParserSemantics
from .skeleton import ChildEvent
from .skeleton import RingEvent
from .skeleton import TraversalSkeleton
from .slots import CarrierSlot
from .slots import SlotBundle


class OrdinarySmilesSemantics(ParserSemantics):
    """Concrete finite semantics for the current bounded ordinary dialect.

    This module defines parser-side relations for the South Star proof kernel;
    it is not an RDKit writer model.  RDKit-specific behavior belongs at an
    adapter/audit boundary that snapshots molecules into ``MoleculeFacts``.
    """

    def atom_decode_ok(
        self,
        facts: MoleculeFacts,
        atom: AtomId,
        atom_text: AtomTextChoice,
        tetra_token: TetraToken,
        incident_bond_texts: tuple[BondTextChoice, ...],
    ) -> bool:
        del facts, atom, incident_bond_texts
        return atom_text.permits(tetra_token)

    def bond_decode_ok(
        self,
        facts: MoleculeFacts,
        bond: BondId,
        bond_text: BondTextChoice,
        direction_mark: DirectionMark,
    ) -> bool:
        bond_facts = _bond_by_id(facts)[bond]
        if direction_mark is not DirectionMark.ABSENT:
            if not bond_text.permits_direction:
                return False
            if bond_facts.order is not BondOrder.SINGLE:
                return False
        return _bond_text_matches_order(bond_text, bond_facts)

    def ring_pair_decode_ok(
        self,
        facts: MoleculeFacts,
        bond: BondId,
        endpoint_1: BondTextChoice,
        mark_1: DirectionMark,
        endpoint_2: BondTextChoice,
        mark_2: DirectionMark,
    ) -> bool:
        bond_facts = _bond_by_id(facts)[bond]
        if bond_facts.order is BondOrder.DOUBLE:
            return (
                mark_1 is DirectionMark.ABSENT
                and mark_2 is DirectionMark.ABSENT
                and (endpoint_1.base_text, endpoint_2.base_text)
                in {("=", ""), ("", "=")}
            )
        if bond_facts.order is BondOrder.TRIPLE:
            return (
                mark_1 is DirectionMark.ABSENT
                and mark_2 is DirectionMark.ABSENT
                and (endpoint_1.base_text, endpoint_2.base_text)
                in {("#", ""), ("", "#")}
            )
        return self.bond_decode_ok(facts, bond, endpoint_1, mark_1) and (
            self.bond_decode_ok(facts, bond, endpoint_2, mark_2)
        )

    def local_tetra_order(
        self,
        facts: MoleculeFacts,
        skel: object,
        slots: object,
        site: SiteId,
    ) -> tuple[OccurrenceId, ...]:
        del slots
        skeleton = _require_skeleton(skel)
        site_facts = _tetra_site_by_id(facts)[site]
        occurrence_by_atom = _neighbor_occurrences_by_atom(facts, site_facts)
        implicit_h = tuple(
            occurrence.id
            for occurrence in _occurrences_by_id(facts).values()
            if occurrence.site == site and occurrence.kind is LigandKind.IMPLICIT_H
        )

        order: list[OccurrenceId] = []
        parent = skeleton.parent[site_facts.center]
        if parent is not None:
            occurrence = occurrence_by_atom.get(parent)
            if occurrence is not None:
                order.append(occurrence.id)

        for event in skeleton.events_at[site_facts.center]:
            if isinstance(event, ChildEvent):
                occurrence = occurrence_by_atom.get(event.child)
            elif isinstance(event, RingEvent):
                occurrence = occurrence_by_atom.get(event.other_atom)
            else:  # pragma: no cover - defensive for future event extension
                raise TypeError(event)
            if occurrence is not None:
                order.append(occurrence.id)

        order.extend(implicit_h)
        return tuple(order)

    def tetra_value(
        self,
        facts: MoleculeFacts,
        site: SiteId,
        local_order: tuple[OccurrenceId, ...],
        token: TetraToken,
    ) -> TetraValue | Invalid:
        site_facts = _tetra_site_by_id(facts)[site]
        if token is TetraToken.NONE:
            return TetraValue.NONE
        reference = site_facts.reference_order
        if set(local_order) != set(reference) or len(local_order) != len(reference):
            return INVALID

        is_even = _is_even_permutation(
            tuple(reference.index(occurrence) for occurrence in local_order)
        )
        if token is TetraToken.AT:
            return TetraValue.PLUS if is_even else TetraValue.MINUS
        if token is TetraToken.ATAT:
            return TetraValue.MINUS if is_even else TetraValue.PLUS
        return INVALID

    def directional_scope(
        self,
        facts: MoleculeFacts,
        skel: object,
        slots: object,
        site: SiteId,
    ) -> tuple[CarrierSlotId, ...]:
        del skel
        slot_bundle = _require_slots(slots)
        site_facts = _directional_site_by_id(facts)[site]
        substituent_bonds = _directional_substituent_bonds(facts, site_facts)
        return tuple(
            carrier.id
            for carrier in slot_bundle.carrier_slots
            if carrier.bond in substituent_bonds
            and carrier.bond != site_facts.center_bond
        )

    def directional_value(
        self,
        facts: MoleculeFacts,
        skel: object,
        slots: object,
        site: SiteId,
        marks: Mapping[CarrierSlotId, DirectionMark],
    ) -> DirectionalValue | Invalid:
        del skel
        slot_bundle = _require_slots(slots)
        site_facts = _directional_site_by_id(facts)[site]
        carrier_by_id = {carrier.id: carrier for carrier in slot_bundle.carrier_slots}
        model_by_carrier = _directional_carrier_models(
            facts=facts,
            site=site_facts,
            slots=slot_bundle,
        )
        left_values: list[int] = []
        right_values: list[int] = []

        for carrier_id, mark in marks.items():
            carrier = carrier_by_id[carrier_id]
            if mark is DirectionMark.ABSENT:
                continue
            if carrier.bond == site_facts.center_bond:
                return INVALID
            model = model_by_carrier.get(carrier_id)
            if model is None:
                return INVALID

            raw = _signed_direction(mark, carrier, endpoint=model.endpoint)
            normalized = raw * model.ligand_factor

            if model.side == "left":
                left_values.append(normalized)
            else:
                right_values.append(normalized)

        if not left_values and not right_values:
            return DirectionalValue.NONE
        if not left_values or not right_values:
            return DirectionalValue.NONE
        if len(set(left_values)) != 1:
            return INVALID
        if len(set(right_values)) != 1:
            return INVALID

        if left_values[0] == right_values[0]:
            return DirectionalValue.TOGETHER
        return DirectionalValue.OPPOSITE


@dataclass(frozen=True, slots=True)
class _DirectionalCarrierModel:
    side: Literal["left", "right"]
    endpoint: AtomId
    occurrence: OccurrenceId
    ligand_factor: int


def _bond_text_matches_order(choice: BondTextChoice, bond: BondFacts) -> bool:
    if bond.order is BondOrder.SINGLE:
        return choice.base_text in {"", "-"}
    if bond.order is BondOrder.DOUBLE:
        return choice.base_text == "="
    if bond.order is BondOrder.TRIPLE:
        return choice.base_text == "#"
    if bond.order is BondOrder.AROMATIC:
        return choice.base_text in {"", ":"}
    return False


def _signed_direction(
    mark: DirectionMark,
    carrier: CarrierSlot,
    *,
    endpoint: AtomId,
) -> int:
    if mark is DirectionMark.ABSENT:
        raise ValueError("absent direction mark has no sign")
    if carrier.written_from == endpoint:
        orientation = 1
    elif carrier.written_to == endpoint:
        orientation = -1
    else:
        raise ValueError(
            f"carrier {carrier.id!r} is not incident to endpoint {endpoint!r}"
        )
    return _mark_sign(mark) * orientation


def _mark_sign(mark: DirectionMark) -> int:
    if mark is DirectionMark.FWD:
        return 1
    if mark is DirectionMark.REV:
        return -1
    raise ValueError(f"direction mark has no sign: {mark!r}")


def _directional_substituent_bonds(
    facts: MoleculeFacts,
    site: DirectionalSiteFacts,
) -> frozenset[BondId]:
    return frozenset(
        _ligand_bonds(facts, site.left_ligands)
        | _ligand_bonds(facts, site.right_ligands)
    )


def _directional_carrier_models(
    *,
    facts: MoleculeFacts,
    site: DirectionalSiteFacts,
    slots: SlotBundle,
) -> dict[CarrierSlotId, _DirectionalCarrierModel]:
    occurrences = _occurrences_by_id(facts)
    left_reference, right_reference = _directional_reference_pair(site)
    left_by_bond = _neighbor_ligands_by_bond(
        occurrences,
        site.left_ligands,
        side="left",
    )
    right_by_bond = _neighbor_ligands_by_bond(
        occurrences,
        site.right_ligands,
        side="right",
    )
    out: dict[CarrierSlotId, _DirectionalCarrierModel] = {}

    for carrier in slots.carrier_slots:
        if carrier.bond == site.center_bond:
            continue
        if carrier.bond in left_by_bond:
            occurrence = left_by_bond[carrier.bond]
            out[carrier.id] = _DirectionalCarrierModel(
                side="left",
                endpoint=site.left_endpoint,
                occurrence=occurrence,
                ligand_factor=_ligand_factor(
                    occurrence,
                    reference=left_reference,
                    side_ligands=site.left_ligands,
                ),
            )
            continue
        if carrier.bond in right_by_bond:
            occurrence = right_by_bond[carrier.bond]
            out[carrier.id] = _DirectionalCarrierModel(
                side="right",
                endpoint=site.right_endpoint,
                occurrence=occurrence,
                ligand_factor=_ligand_factor(
                    occurrence,
                    reference=right_reference,
                    side_ligands=site.right_ligands,
                ),
            )

    return out


def _directional_reference_pair(
    site: DirectionalSiteFacts,
) -> tuple[OccurrenceId, OccurrenceId]:
    if site.reference_pair is not None:
        return site.reference_pair
    if site.status is SiteStatus.SPECIFIED:
        raise ValueError(
            f"specified directional site lacks reference pair: {site.id!r}"
        )

    # Potential-but-unspecified sites only need a stable NONE/non-NONE
    # distinction.  The arbitrary local reference fixes names for non-NONE
    # values without changing whether accidental stereo exists.
    return (
        min(site.left_ligands, key=int),
        min(site.right_ligands, key=int),
    )


def _neighbor_ligands_by_bond(
    occurrences: Mapping[OccurrenceId, LigandOccurrence],
    ligand_ids: tuple[OccurrenceId, ...],
    *,
    side: str,
) -> dict[BondId, OccurrenceId]:
    out: dict[BondId, OccurrenceId] = {}
    for ligand_id in ligand_ids:
        occurrence = occurrences[ligand_id]
        if occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
            continue
        if occurrence.bond is None:
            raise ValueError(f"neighbor occurrence lacks bond: {occurrence.id!r}")
        if occurrence.bond in out:
            raise ValueError(
                f"directional {side} side has multiple ligands on bond "
                f"{occurrence.bond!r}"
            )
        out[occurrence.bond] = ligand_id
    return out


def _ligand_factor(
    occurrence: OccurrenceId,
    *,
    reference: OccurrenceId,
    side_ligands: tuple[OccurrenceId, ...],
) -> int:
    if len(side_ligands) > 2:
        raise ValueError(
            "ordinary directional endpoint has more than two ligand occurrences"
        )
    if occurrence == reference:
        return 1
    if occurrence not in side_ligands:
        raise ValueError(f"occurrence {occurrence!r} is not on directional side")
    return -1


def _ligand_bonds(
    facts: MoleculeFacts,
    ligand_ids: tuple[OccurrenceId, ...],
) -> frozenset[BondId]:
    occurrences = _occurrences_by_id(facts)
    bonds: set[BondId] = set()
    for ligand_id in ligand_ids:
        occurrence = occurrences[ligand_id]
        if occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
            continue
        if occurrence.bond is None:
            raise ValueError(f"neighbor occurrence lacks bond: {occurrence.id!r}")
        bonds.add(occurrence.bond)
    return frozenset(bonds)


def _neighbor_occurrences_by_atom(
    facts: MoleculeFacts,
    site: TetrahedralSiteFacts,
) -> dict[AtomId, LigandOccurrence]:
    occurrences = _occurrences_by_id(facts)
    out: dict[AtomId, LigandOccurrence] = {}
    for occurrence_id in site.ligand_occurrences:
        occurrence = occurrences[occurrence_id]
        if occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
            continue
        if occurrence.atom is None:
            raise ValueError(f"neighbor occurrence lacks atom: {occurrence.id!r}")
        if occurrence.atom in out:
            raise ValueError(
                f"site {site.id!r} has multiple occurrences for atom "
                f"{occurrence.atom!r}"
            )
        out[occurrence.atom] = occurrence
    return out


def _tetra_site_by_id(facts: MoleculeFacts) -> dict[SiteId, TetrahedralSiteFacts]:
    return {site.id: site for site in facts.stereo.tetrahedral}


def _directional_site_by_id(
    facts: MoleculeFacts,
) -> dict[SiteId, DirectionalSiteFacts]:
    return {site.id: site for site in facts.stereo.directional}


def _occurrences_by_id(facts: MoleculeFacts) -> dict[OccurrenceId, LigandOccurrence]:
    return {occurrence.id: occurrence for occurrence in facts.ligand_occurrences}


def _bond_by_id(facts: MoleculeFacts) -> dict[BondId, BondFacts]:
    return {bond.id: bond for bond in facts.bonds}


def _require_skeleton(value: object) -> TraversalSkeleton:
    if not isinstance(value, TraversalSkeleton):
        raise TypeError(f"expected TraversalSkeleton, got {type(value).__name__}")
    return value


def _require_slots(value: object) -> SlotBundle:
    if not isinstance(value, SlotBundle):
        raise TypeError(f"expected SlotBundle, got {type(value).__name__}")
    return value


def _is_even_permutation(indices: tuple[int, ...]) -> bool:
    inversions = 0
    for left, value in enumerate(indices):
        for other in indices[left + 1 :]:
            if value > other:
                inversions += 1
    return inversions % 2 == 0


__all__ = ("OrdinarySmilesSemantics",)
