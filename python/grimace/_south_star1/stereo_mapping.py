"""Shared stereo compatibility under atom/bond mappings."""

from __future__ import annotations

from collections.abc import Iterable
from collections.abc import Mapping
from dataclasses import dataclass

from .facts import DirectionalSiteFacts
from .facts import DirectionalValue
from .facts import LigandOccurrence
from .facts import MoleculeFacts
from .facts import SiteStatus
from .facts import TetraValue
from .facts import TetrahedralSiteFacts
from .ids import AtomId
from .ids import BondId
from .ids import OccurrenceId
from .ids import SiteId


def map_ligand_occurrence(
    occurrence: LigandOccurrence,
    *,
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
    target_occurrences: Iterable[LigandOccurrence],
) -> LigandOccurrence | None:
    """Map a ligand occurrence by semantic identity, ignoring construction ids."""

    key = occurrence_key(occurrence, atom_map, bond_map)
    for target in target_occurrences:
        if identity_occurrence_key(target) == key:
            return target
    return None


def tetra_site_compatible_under_mapping(
    left: TetrahedralSiteFacts,
    right: TetrahedralSiteFacts,
    *,
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
    left_occurrences: Mapping[OccurrenceId, LigandOccurrence],
    right_occurrences: Mapping[OccurrenceId, LigandOccurrence],
) -> bool:
    """Return whether one mapped tetrahedral site preserves stereo semantics."""

    if tetrahedral_structure_key(
        left,
        left_occurrences,
        atom_map,
        bond_map,
    ) != identity_tetrahedral_structure_key(right, right_occurrences):
        return False

    mapped_left_order = tuple(
        occurrence_key(left_occurrences[occurrence], atom_map, bond_map)
        for occurrence in left.reference_order
    )
    return tetra_targets_compatible(
        left,
        mapped_left_order,
        right,
        right_occurrences,
    )


def directional_site_compatible_under_mapping(
    left: DirectionalSiteFacts,
    right: DirectionalSiteFacts,
    *,
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
    left_occurrences: Mapping[OccurrenceId, LigandOccurrence],
    right_occurrences: Mapping[OccurrenceId, LigandOccurrence],
) -> bool:
    """Return whether one mapped directional site preserves stereo semantics."""

    mapped = mapped_directional_site(left, left_occurrences, atom_map, bond_map)
    identity = identity_mapped_directional_site(right, right_occurrences)
    return directional_orientation_compatible(mapped, identity) or (
        directional_orientation_compatible(mapped, swap_directional_site_sides(identity))
    )


def specified_stereo_compatible_under_mapping(
    left: MoleculeFacts,
    right: MoleculeFacts,
    *,
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
    ignore_site_ids: frozenset[SiteId] = frozenset(),
) -> bool:
    """Return whether specified stereo labels are compatible under a mapping."""

    return stereo_compatible_under_mapping(
        left,
        right,
        atom_map=atom_map,
        bond_map=bond_map,
        compare_potential_sites=False,
        ignore_site_ids=ignore_site_ids,
    )


def stereo_compatible_under_mapping(
    left: MoleculeFacts,
    right: MoleculeFacts,
    *,
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
    compare_potential_sites: bool,
    ignore_site_ids: frozenset[SiteId] = frozenset(),
) -> bool:
    return tetrahedral_sites_compatible_under_mapping(
        left,
        right,
        atom_map=atom_map,
        bond_map=bond_map,
        compare_potential_sites=compare_potential_sites,
        ignore_site_ids=ignore_site_ids,
    ) and directional_sites_compatible_under_mapping(
        left,
        right,
        atom_map=atom_map,
        bond_map=bond_map,
        compare_potential_sites=compare_potential_sites,
        ignore_site_ids=ignore_site_ids,
    )


def tetrahedral_sites_compatible_under_mapping(
    left: MoleculeFacts,
    right: MoleculeFacts,
    *,
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
    compare_potential_sites: bool,
    ignore_site_ids: frozenset[SiteId] = frozenset(),
) -> bool:
    left_sites = _filtered_tetrahedral_sites(left, compare_potential_sites)
    right_sites = _filtered_tetrahedral_sites(right, compare_potential_sites)
    if ignore_site_ids:
        left_sites = tuple(site for site in left_sites if site.id not in ignore_site_ids)
        right_sites = tuple(site for site in right_sites if site.id not in ignore_site_ids)

    right_sites_list = list(right_sites)
    used: set[int] = set()
    left_occurrences = occurrences_by_id(left)
    right_occurrences = occurrences_by_id(right)

    for left_site in left_sites:
        found = False
        for index, right_site in enumerate(right_sites_list):
            if index in used:
                continue
            if not tetra_site_compatible_under_mapping(
                left_site,
                right_site,
                atom_map=atom_map,
                bond_map=bond_map,
                left_occurrences=left_occurrences,
                right_occurrences=right_occurrences,
            ):
                continue
            used.add(index)
            found = True
            break
        if not found:
            return False
    return len(used) == len(right_sites_list)


def directional_sites_compatible_under_mapping(
    left: MoleculeFacts,
    right: MoleculeFacts,
    *,
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
    compare_potential_sites: bool,
    ignore_site_ids: frozenset[SiteId] = frozenset(),
) -> bool:
    left_sites = _filtered_directional_sites(left, compare_potential_sites)
    right_sites = _filtered_directional_sites(right, compare_potential_sites)
    if ignore_site_ids:
        left_sites = tuple(site for site in left_sites if site.id not in ignore_site_ids)
        right_sites = tuple(site for site in right_sites if site.id not in ignore_site_ids)

    right_sites_list = list(right_sites)
    used: set[int] = set()
    left_occurrences = occurrences_by_id(left)
    right_occurrences = occurrences_by_id(right)

    for left_site in left_sites:
        found = False
        for index, right_site in enumerate(right_sites_list):
            if index in used:
                continue
            if not directional_site_compatible_under_mapping(
                left_site,
                right_site,
                atom_map=atom_map,
                bond_map=bond_map,
                left_occurrences=left_occurrences,
                right_occurrences=right_occurrences,
            ):
                continue
            used.add(index)
            found = True
            break
        if not found:
            return False
    return len(used) == len(right_sites_list)


def tetrahedral_structure_key(
    site: TetrahedralSiteFacts,
    occurrences: Mapping[OccurrenceId, LigandOccurrence],
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
) -> tuple[object, ...]:
    return (
        int(atom_map[site.center]),
        site.status.value,
        tuple(
            sorted(
                occurrence_key(occurrences[occurrence], atom_map, bond_map)
                for occurrence in site.ligand_occurrences
            )
        ),
    )


def identity_tetrahedral_structure_key(
    site: TetrahedralSiteFacts,
    occurrences: Mapping[OccurrenceId, LigandOccurrence],
) -> tuple[object, ...]:
    return (
        int(site.center),
        site.status.value,
        tuple(
            sorted(
                identity_occurrence_key(occurrences[occurrence])
                for occurrence in site.ligand_occurrences
            )
        ),
    )


def tetra_targets_compatible(
    left: TetrahedralSiteFacts,
    mapped_left_order: tuple[tuple[object, ...], ...],
    right: TetrahedralSiteFacts,
    right_occurrences: Mapping[OccurrenceId, LigandOccurrence],
) -> bool:
    if left.status != right.status:
        return False
    if left.status is SiteStatus.UNSPECIFIED:
        return right.target is TetraValue.NONE

    right_order = tuple(
        identity_occurrence_key(right_occurrences[occurrence])
        for occurrence in right.reference_order
    )
    if set(mapped_left_order) != set(right_order):
        return False
    parity = permutation_parity(
        tuple(mapped_left_order.index(occurrence) for occurrence in right_order)
    )
    expected = flip_tetra_value(left.target) if parity == -1 else left.target
    return expected == right.target


@dataclass(frozen=True, slots=True)
class MappedDirectionalSite:
    center_bond: int
    left_endpoint: int
    right_endpoint: int
    status: SiteStatus
    target: DirectionalValue
    left_ligands: tuple[tuple[object, ...], ...]
    right_ligands: tuple[tuple[object, ...], ...]
    reference_pair: tuple[tuple[object, ...], tuple[object, ...]] | None


def mapped_directional_site(
    site: DirectionalSiteFacts,
    occurrences: Mapping[OccurrenceId, LigandOccurrence],
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
) -> MappedDirectionalSite:
    reference_pair = (
        None
        if site.reference_pair is None
        else tuple(
            occurrence_key(occurrences[occurrence], atom_map, bond_map)
            for occurrence in site.reference_pair
        )
    )
    return MappedDirectionalSite(
        center_bond=int(bond_map[site.center_bond]),
        left_endpoint=int(atom_map[site.left_endpoint]),
        right_endpoint=int(atom_map[site.right_endpoint]),
        status=site.status,
        target=site.target,
        left_ligands=tuple(
            occurrence_key(occurrences[occurrence], atom_map, bond_map)
            for occurrence in site.left_ligands
        ),
        right_ligands=tuple(
            occurrence_key(occurrences[occurrence], atom_map, bond_map)
            for occurrence in site.right_ligands
        ),
        reference_pair=reference_pair,
    )


def identity_mapped_directional_site(
    site: DirectionalSiteFacts,
    occurrences: Mapping[OccurrenceId, LigandOccurrence],
) -> MappedDirectionalSite:
    reference_pair = (
        None
        if site.reference_pair is None
        else tuple(
            identity_occurrence_key(occurrences[occurrence])
            for occurrence in site.reference_pair
        )
    )
    return MappedDirectionalSite(
        center_bond=int(site.center_bond),
        left_endpoint=int(site.left_endpoint),
        right_endpoint=int(site.right_endpoint),
        status=site.status,
        target=site.target,
        left_ligands=tuple(
            identity_occurrence_key(occurrences[occurrence])
            for occurrence in site.left_ligands
        ),
        right_ligands=tuple(
            identity_occurrence_key(occurrences[occurrence])
            for occurrence in site.right_ligands
        ),
        reference_pair=reference_pair,
    )


def directional_orientation_compatible(
    left: MappedDirectionalSite,
    right: MappedDirectionalSite,
) -> bool:
    if (
        left.center_bond != right.center_bond
        or left.left_endpoint != right.left_endpoint
        or left.right_endpoint != right.right_endpoint
        or left.status is not right.status
        or set(left.left_ligands) != set(right.left_ligands)
        or set(left.right_ligands) != set(right.right_ligands)
    ):
        return False

    if left.status is SiteStatus.UNSPECIFIED:
        return right.target is DirectionalValue.NONE
    if left.reference_pair is None or right.reference_pair is None:
        return left.reference_pair is None and right.reference_pair is None and (
            left.target is right.target
        )

    expected = left.target
    left_changed = left.reference_pair[0] != right.reference_pair[0]
    right_changed = left.reference_pair[1] != right.reference_pair[1]
    if left_changed != right_changed:
        expected = flip_directional_value(expected)
    return expected is right.target


def swap_directional_site_sides(
    site: MappedDirectionalSite,
) -> MappedDirectionalSite:
    return MappedDirectionalSite(
        center_bond=site.center_bond,
        left_endpoint=site.right_endpoint,
        right_endpoint=site.left_endpoint,
        status=site.status,
        target=site.target,
        left_ligands=site.right_ligands,
        right_ligands=site.left_ligands,
        reference_pair=None
        if site.reference_pair is None
        else (site.reference_pair[1], site.reference_pair[0]),
    )


def occurrence_key(
    occurrence: LigandOccurrence,
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
) -> tuple[object, ...]:
    return (
        occurrence.kind.value,
        None if occurrence.atom is None else int(atom_map[occurrence.atom]),
        None if occurrence.bond is None else int(bond_map[occurrence.bond]),
    )


def identity_occurrence_key(occurrence: LigandOccurrence) -> tuple[object, ...]:
    return (
        occurrence.kind.value,
        None if occurrence.atom is None else int(occurrence.atom),
        None if occurrence.bond is None else int(occurrence.bond),
    )


def occurrences_by_id(facts: MoleculeFacts) -> dict[OccurrenceId, LigandOccurrence]:
    return {occurrence.id: occurrence for occurrence in facts.ligand_occurrences}


def permutation_parity(indices: tuple[int, ...]) -> int:
    inversions = 0
    for left, value in enumerate(indices):
        for other in indices[left + 1 :]:
            if value > other:
                inversions += 1
    return 1 if inversions % 2 == 0 else -1


def flip_tetra_value(value: TetraValue) -> TetraValue:
    if value is TetraValue.PLUS:
        return TetraValue.MINUS
    if value is TetraValue.MINUS:
        return TetraValue.PLUS
    return value


def flip_directional_value(value: DirectionalValue) -> DirectionalValue:
    if value is DirectionalValue.TOGETHER:
        return DirectionalValue.OPPOSITE
    if value is DirectionalValue.OPPOSITE:
        return DirectionalValue.TOGETHER
    return value


def _filtered_tetrahedral_sites(
    facts: MoleculeFacts,
    compare_potential_sites: bool,
) -> tuple[TetrahedralSiteFacts, ...]:
    if compare_potential_sites:
        return facts.stereo.tetrahedral
    return tuple(
        site
        for site in facts.stereo.tetrahedral
        if site.status is SiteStatus.SPECIFIED
    )


def _filtered_directional_sites(
    facts: MoleculeFacts,
    compare_potential_sites: bool,
) -> tuple[DirectionalSiteFacts, ...]:
    if compare_potential_sites:
        return facts.stereo.directional
    return tuple(
        site
        for site in facts.stereo.directional
        if site.status is SiteStatus.SPECIFIED
    )


__all__ = (
    "MappedDirectionalSite",
    "directional_site_compatible_under_mapping",
    "directional_sites_compatible_under_mapping",
    "map_ligand_occurrence",
    "specified_stereo_compatible_under_mapping",
    "stereo_compatible_under_mapping",
    "tetra_site_compatible_under_mapping",
    "tetrahedral_sites_compatible_under_mapping",
)
