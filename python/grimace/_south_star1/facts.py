"""Immutable molecule-fact records for the private proof kernel."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from .ids import AtomId
from .ids import BondId
from .ids import ComponentId
from .ids import OccurrenceId
from .ids import SiteId


class BondOrder(Enum):
    SINGLE = "single"
    DOUBLE = "double"
    TRIPLE = "triple"
    AROMATIC = "aromatic"


class LigandKind(Enum):
    NEIGHBOR_ATOM = "neighbor_atom"
    IMPLICIT_H = "implicit_h"
    PSEUDO = "pseudo"


class SiteStatus(Enum):
    SPECIFIED = "specified"
    UNSPECIFIED = "unspecified"


class TetraValue(Enum):
    NONE = "none"
    PLUS = "plus"
    MINUS = "minus"


class DirectionalValue(Enum):
    NONE = "none"
    TOGETHER = "together"
    OPPOSITE = "opposite"


@dataclass(frozen=True, slots=True)
class AtomFacts:
    id: AtomId
    atomic_num: int
    symbol: str
    isotope: int | None
    formal_charge: int
    is_aromatic: bool
    explicit_h_count: int
    implicit_h_count: int
    no_implicit: bool


@dataclass(frozen=True, slots=True)
class BondFacts:
    id: BondId
    a: AtomId
    b: AtomId
    order: BondOrder
    is_aromatic: bool
    is_conjugated: bool


@dataclass(frozen=True, slots=True)
class ComponentFacts:
    id: ComponentId
    atoms: tuple[AtomId, ...]
    bonds: tuple[BondId, ...]


@dataclass(frozen=True, slots=True)
class LigandOccurrence:
    id: OccurrenceId
    site: SiteId
    kind: LigandKind
    atom: AtomId | None
    bond: BondId | None
    ordinal: int = 0


@dataclass(frozen=True, slots=True)
class TetrahedralSiteFacts:
    id: SiteId
    center: AtomId
    status: SiteStatus
    target: TetraValue
    ligand_occurrences: tuple[OccurrenceId, ...]
    reference_order: tuple[OccurrenceId, ...]


@dataclass(frozen=True, slots=True)
class DirectionalSiteFacts:
    id: SiteId
    center_bond: BondId
    left_endpoint: AtomId
    right_endpoint: AtomId
    status: SiteStatus
    target: DirectionalValue
    left_ligands: tuple[OccurrenceId, ...]
    right_ligands: tuple[OccurrenceId, ...]
    reference_pair: tuple[OccurrenceId, OccurrenceId] | None = None


@dataclass(frozen=True, slots=True)
class StereoFacts:
    tetrahedral: tuple[TetrahedralSiteFacts, ...] = ()
    directional: tuple[DirectionalSiteFacts, ...] = ()


@dataclass(frozen=True, slots=True)
class MoleculeFacts:
    atoms: tuple[AtomFacts, ...]
    bonds: tuple[BondFacts, ...]
    components: tuple[ComponentFacts, ...]
    stereo: StereoFacts = StereoFacts()
    ligand_occurrences: tuple[LigandOccurrence, ...] = ()

    def validate(self) -> None:
        """Validate the finite molecule snapshot.

        The proof kernel deliberately validates strongly at the fact boundary so
        downstream skeleton, slot, and constraint code can assume ids and
        partitions are internally consistent.
        """

        atom_ids = _unique_ids("atom", (atom.id for atom in self.atoms))
        bond_ids = _unique_ids("bond", (bond.id for bond in self.bonds))
        component_ids = _unique_ids(
            "component",
            (component.id for component in self.components),
        )
        _unique_ids(
            "ligand occurrence",
            (occurrence.id for occurrence in self.ligand_occurrences),
        )

        if not atom_ids:
            raise ValueError("molecule facts must contain at least one atom")
        if not component_ids:
            raise ValueError("molecule facts must contain at least one component")

        bond_by_id = {bond.id: bond for bond in self.bonds}
        occurrence_by_id = {
            occurrence.id: occurrence for occurrence in self.ligand_occurrences
        }

        _validate_bonds(self.bonds, atom_ids)
        _validate_components(self.components, atom_ids, bond_ids, bond_by_id)
        site_ids = _validate_site_ids(self.stereo)
        _validate_ligand_occurrences(
            self.ligand_occurrences,
            atom_ids,
            bond_ids,
            bond_by_id,
            site_ids,
        )
        _validate_tetrahedral_sites(self.stereo.tetrahedral, atom_ids, occurrence_by_id)
        _validate_directional_sites(
            self.stereo.directional,
            atom_ids,
            bond_by_id,
            occurrence_by_id,
        )
        _validate_occurrence_coverage(self.stereo, set(occurrence_by_id))


def _unique_ids(label: str, values: Iterable[object]) -> set[object]:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {label} id: {value!r}")
        seen.add(value)
    return seen


def _validate_bonds(bonds: tuple[BondFacts, ...], atom_ids: set[object]) -> None:
    pairs: set[tuple[AtomId, AtomId]] = set()
    for bond in bonds:
        if bond.a not in atom_ids:
            raise ValueError(f"bond {bond.id!r} has unknown atom endpoint {bond.a!r}")
        if bond.b not in atom_ids:
            raise ValueError(f"bond {bond.id!r} has unknown atom endpoint {bond.b!r}")
        if bond.a == bond.b:
            raise ValueError(f"bond {bond.id!r} is a self bond")
        pair = atom_pair_key(bond.a, bond.b)
        if pair in pairs:
            raise ValueError(f"multiple bonds share atom pair {pair!r}")
        pairs.add(pair)


def _validate_components(
    components: tuple[ComponentFacts, ...],
    atom_ids: set[object],
    bond_ids: set[object],
    bond_by_id: dict[BondId, BondFacts],
) -> None:
    component_atoms: set[AtomId] = set()
    component_bonds: set[BondId] = set()

    for component in components:
        if len(set(component.atoms)) != len(component.atoms):
            raise ValueError(f"component {component.id!r} repeats atoms")
        if len(set(component.bonds)) != len(component.bonds):
            raise ValueError(f"component {component.id!r} repeats bonds")
        unknown_atoms = set(component.atoms) - atom_ids
        if unknown_atoms:
            raise ValueError(
                f"component {component.id!r} has unknown atoms {unknown_atoms!r}"
            )
        unknown_bonds = set(component.bonds) - bond_ids
        if unknown_bonds:
            raise ValueError(
                f"component {component.id!r} has unknown bonds {unknown_bonds!r}"
            )
        overlap_atoms = component_atoms.intersection(component.atoms)
        if overlap_atoms:
            raise ValueError(f"atoms appear in multiple components: {overlap_atoms!r}")
        overlap_bonds = component_bonds.intersection(component.bonds)
        if overlap_bonds:
            raise ValueError(f"bonds appear in multiple components: {overlap_bonds!r}")

        component_atom_set = set(component.atoms)
        for bond_id in component.bonds:
            bond = bond_by_id[bond_id]
            if bond.a not in component_atom_set or bond.b not in component_atom_set:
                raise ValueError(
                    f"component {component.id!r} contains bond {bond_id!r} "
                    "without both endpoints"
                )

        component_atoms.update(component.atoms)
        component_bonds.update(component.bonds)

    if component_atoms != atom_ids:
        missing = atom_ids - component_atoms
        extra = component_atoms - atom_ids
        raise ValueError(
            f"component atom partition mismatch: missing={missing!r}, "
            f"extra={extra!r}"
        )
    if component_bonds != bond_ids:
        missing = bond_ids - component_bonds
        extra = component_bonds - bond_ids
        raise ValueError(
            f"component bond partition mismatch: missing={missing!r}, "
            f"extra={extra!r}"
        )


def _validate_site_ids(stereo: StereoFacts) -> set[object]:
    return _unique_ids(
        "stereo site",
        tuple(site.id for site in stereo.tetrahedral)
        + tuple(site.id for site in stereo.directional),
    )


def _validate_ligand_occurrences(
    occurrences: tuple[LigandOccurrence, ...],
    atom_ids: set[object],
    bond_ids: set[object],
    bond_by_id: dict[BondId, BondFacts],
    site_ids: set[object],
) -> None:
    for occurrence in occurrences:
        if occurrence.site not in site_ids:
            raise ValueError(
                f"ligand occurrence {occurrence.id!r} has unknown site "
                f"{occurrence.site!r}"
            )
        if occurrence.ordinal < 0:
            raise ValueError(
                f"ligand occurrence {occurrence.id!r} has negative ordinal"
            )
        if occurrence.kind is LigandKind.NEIGHBOR_ATOM:
            if occurrence.atom not in atom_ids:
                raise ValueError(
                    f"neighbor occurrence {occurrence.id!r} has unknown atom "
                    f"{occurrence.atom!r}"
                )
            if occurrence.bond not in bond_ids:
                raise ValueError(
                    f"neighbor occurrence {occurrence.id!r} has unknown bond "
                    f"{occurrence.bond!r}"
                )
            bond = bond_by_id[occurrence.bond]
            if occurrence.atom not in (bond.a, bond.b):
                raise ValueError(
                    f"neighbor occurrence {occurrence.id!r} atom is not incident "
                    "to its bond"
                )
        elif occurrence.kind is LigandKind.IMPLICIT_H:
            if occurrence.atom not in atom_ids:
                raise ValueError(
                    f"implicit-H occurrence {occurrence.id!r} has unknown atom "
                    f"{occurrence.atom!r}"
                )
            if occurrence.bond is not None:
                raise ValueError(
                    f"implicit-H occurrence {occurrence.id!r} must not carry a bond"
                )
        elif occurrence.kind is LigandKind.PSEUDO:
            if occurrence.atom is not None or occurrence.bond is not None:
                raise ValueError(
                    f"pseudo occurrence {occurrence.id!r} must not carry atom/bond ids"
                )
        else:  # pragma: no cover - defensive for future enum extension
            raise ValueError(f"unknown ligand kind: {occurrence.kind!r}")


def _validate_tetrahedral_sites(
    sites: tuple[TetrahedralSiteFacts, ...],
    atom_ids: set[object],
    occurrence_by_id: dict[OccurrenceId, LigandOccurrence],
) -> None:
    for site in sites:
        if site.center not in atom_ids:
            raise ValueError(f"tetrahedral site {site.id!r} has unknown center")
        _validate_tetra_target(site)
        if len(site.ligand_occurrences) != 4:
            raise ValueError(f"tetrahedral site {site.id!r} must have four ligands")
        if len(site.reference_order) != 4:
            raise ValueError(
                f"tetrahedral site {site.id!r} must have four reference ligands"
            )
        if set(site.reference_order) != set(site.ligand_occurrences):
            raise ValueError(
                f"tetrahedral site {site.id!r} reference order is inconsistent"
            )
        _validate_site_occurrences(site.id, site.ligand_occurrences, occurrence_by_id)


def _validate_directional_sites(
    sites: tuple[DirectionalSiteFacts, ...],
    atom_ids: set[object],
    bond_by_id: dict[BondId, BondFacts],
    occurrence_by_id: dict[OccurrenceId, LigandOccurrence],
) -> None:
    for site in sites:
        if site.left_endpoint not in atom_ids:
            raise ValueError(f"directional site {site.id!r} has unknown left endpoint")
        if site.right_endpoint not in atom_ids:
            raise ValueError(f"directional site {site.id!r} has unknown right endpoint")
        bond = bond_by_id.get(site.center_bond)
        if bond is None:
            raise ValueError(f"directional site {site.id!r} has unknown center bond")
        if bond.order is not BondOrder.DOUBLE:
            raise ValueError(
                f"directional site {site.id!r} center bond is not double"
            )
        if {bond.a, bond.b} != {site.left_endpoint, site.right_endpoint}:
            raise ValueError(
                f"directional site {site.id!r} endpoints do not match center bond"
            )
        _validate_directional_target(site)
        if not site.left_ligands or not site.right_ligands:
            raise ValueError(
                f"directional site {site.id!r} must have ligands on both sides"
            )
        ligand_ids = site.left_ligands + site.right_ligands
        _validate_site_occurrences(site.id, ligand_ids, occurrence_by_id)
        if site.reference_pair is not None:
            left, right = site.reference_pair
            if left not in site.left_ligands:
                raise ValueError(
                    f"directional site {site.id!r} reference left ligand is invalid"
                )
            if right not in site.right_ligands:
                raise ValueError(
                    f"directional site {site.id!r} reference right ligand is invalid"
                )


def _validate_tetra_target(site: TetrahedralSiteFacts) -> None:
    if site.status is SiteStatus.SPECIFIED and site.target is TetraValue.NONE:
        raise ValueError(f"specified tetrahedral site {site.id!r} has no target")
    if site.status is SiteStatus.UNSPECIFIED and site.target is not TetraValue.NONE:
        raise ValueError(f"unspecified tetrahedral site {site.id!r} has a target")


def _validate_directional_target(site: DirectionalSiteFacts) -> None:
    if site.status is SiteStatus.SPECIFIED and site.target is DirectionalValue.NONE:
        raise ValueError(f"specified directional site {site.id!r} has no target")
    if (
        site.status is SiteStatus.UNSPECIFIED
        and site.target is not DirectionalValue.NONE
    ):
        raise ValueError(f"unspecified directional site {site.id!r} has a target")


def _validate_occurrence_coverage(
    stereo: StereoFacts,
    occurrence_ids: set[object],
) -> None:
    used: set[OccurrenceId] = set()
    for site in stereo.tetrahedral:
        used.update(site.ligand_occurrences)
    for site in stereo.directional:
        used.update(site.left_ligands)
        used.update(site.right_ligands)

    if used != occurrence_ids:
        missing = occurrence_ids - used
        unknown = used - occurrence_ids
        raise ValueError(
            "ligand occurrence coverage mismatch: "
            f"unused={missing!r}, unknown={unknown!r}"
        )


def _validate_site_occurrences(
    site_id: SiteId,
    occurrence_ids: tuple[OccurrenceId, ...],
    occurrence_by_id: dict[OccurrenceId, LigandOccurrence],
) -> None:
    if len(set(occurrence_ids)) != len(occurrence_ids):
        raise ValueError(f"site {site_id!r} repeats ligand occurrences")
    for occurrence_id in occurrence_ids:
        occurrence = occurrence_by_id.get(occurrence_id)
        if occurrence is None:
            raise ValueError(
                f"site {site_id!r} references unknown occurrence {occurrence_id!r}"
            )
        if occurrence.site != site_id:
            raise ValueError(
                f"occurrence {occurrence_id!r} belongs to site {occurrence.site!r}, "
                f"not {site_id!r}"
            )


def atom_pair_key(a: AtomId, b: AtomId) -> tuple[AtomId, AtomId]:
    return (a, b) if int(a) < int(b) else (b, a)


__all__ = (
    "AtomFacts",
    "atom_pair_key",
    "BondFacts",
    "BondOrder",
    "ComponentFacts",
    "DirectionalSiteFacts",
    "DirectionalValue",
    "LigandKind",
    "LigandOccurrence",
    "MoleculeFacts",
    "SiteStatus",
    "StereoFacts",
    "TetraValue",
    "TetrahedralSiteFacts",
)
