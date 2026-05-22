"""RDKit-free parser semantics for the bounded ordinary SMILES dialect."""

from __future__ import annotations

from collections.abc import Mapping

from .facts import BondFacts
from .facts import BondOrder
from .facts import DirectionalSiteFacts
from .facts import DirectionalValue
from .facts import LigandKind
from .facts import LigandOccurrence
from .facts import MoleculeFacts
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
        if (
            direction_mark is not DirectionMark.ABSENT
            and not bond_text.permits_direction
        ):
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
        left_bonds = _ligand_bonds(facts, site_facts.left_ligands)
        right_bonds = _ligand_bonds(facts, site_facts.right_ligands)
        left_signed: list[int] = []
        right_signed: list[int] = []

        for carrier_id, mark in marks.items():
            carrier = carrier_by_id[carrier_id]
            if mark is DirectionMark.ABSENT:
                continue
            if carrier.bond == site_facts.center_bond:
                return INVALID
            if carrier.bond in left_bonds:
                left_signed.append(
                    _signed_direction(mark, carrier, endpoint=site_facts.left_endpoint)
                )
                continue
            if carrier.bond in right_bonds:
                right_signed.append(
                    _signed_direction(mark, carrier, endpoint=site_facts.right_endpoint)
                )
                continue
            return INVALID

        if not left_signed and not right_signed:
            return DirectionalValue.NONE
        if not left_signed or not right_signed:
            return DirectionalValue.NONE

        pair_values = {
            DirectionalValue.TOGETHER
            if left == right
            else DirectionalValue.OPPOSITE
            for left in left_signed
            for right in right_signed
        }
        if len(pair_values) != 1:
            return INVALID

        return next(iter(pair_values))


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
