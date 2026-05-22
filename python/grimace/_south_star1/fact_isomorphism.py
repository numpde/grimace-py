"""RDKit-free isomorphism checks for South Star molecule facts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping

from .facts import AtomFacts
from .facts import BondFacts
from .facts import DirectionalSiteFacts
from .facts import LigandOccurrence
from .facts import MoleculeFacts
from .facts import TetrahedralSiteFacts
from .ids import AtomId
from .ids import BondId
from .ids import OccurrenceId


def facts_are_isomorphic(
    left: MoleculeFacts,
    right: MoleculeFacts,
    *,
    compare_stereo: bool = True,
) -> bool:
    """Return whether two fact snapshots are equal up to atom/bond/site ids.

    This is a small-audit helper, not a graph-isomorphism engine.  It exists so
    parsed SMILES snapshots can be compared semantically even when the parser
    assigned different atom and bond ids.  The check preserves atom attributes,
    bond attributes, component partitions, and, when requested, stereo site
    structure/status/targets.
    """

    if len(left.atoms) != len(right.atoms):
        return False
    if len(left.bonds) != len(right.bonds):
        return False
    if len(left.components) != len(right.components):
        return False
    if Counter(map(_atom_signature, left.atoms)) != Counter(
        map(_atom_signature, right.atoms)
    ):
        return False
    if Counter(map(_bond_signature, left.bonds)) != Counter(
        map(_bond_signature, right.bonds)
    ):
        return False

    for atom_map in _atom_isomorphisms(left, right):
        bond_map = _bond_map_for_atom_map(left, right, atom_map)
        if bond_map is None:
            continue
        if not _components_match(left, right, atom_map, bond_map):
            continue
        if compare_stereo and not _stereo_matches(left, right, atom_map, bond_map):
            continue
        return True
    return False


def _atom_isomorphisms(
    left: MoleculeFacts,
    right: MoleculeFacts,
) -> tuple[dict[AtomId, AtomId], ...]:
    right_by_signature: dict[tuple[object, ...], list[AtomId]] = {}
    for atom in right.atoms:
        right_by_signature.setdefault(_atom_signature(atom), []).append(atom.id)

    left_order = tuple(
        sorted(
            left.atoms,
            key=lambda atom: (
                len(right_by_signature[_atom_signature(atom)]),
                -_atom_degree(left, atom.id),
                int(atom.id),
            ),
        )
    )
    left_adjacency = _bond_by_atom_pair(left)
    right_adjacency = _bond_by_atom_pair(right)
    left_bonds = {bond.id: bond for bond in left.bonds}
    right_bonds = {bond.id: bond for bond in right.bonds}
    mappings: list[dict[AtomId, AtomId]] = []

    def search(index: int, atom_map: dict[AtomId, AtomId], used: set[AtomId]) -> None:
        if index == len(left_order):
            mappings.append(dict(atom_map))
            return

        left_atom = left_order[index]
        for right_atom in right_by_signature[_atom_signature(left_atom)]:
            if right_atom in used:
                continue
            if not _mapped_incident_bonds_match(
                left_atom.id,
                right_atom,
                atom_map,
                left_adjacency,
                right_adjacency,
                left_bonds,
                right_bonds,
            ):
                continue
            atom_map[left_atom.id] = right_atom
            used.add(right_atom)
            search(index + 1, atom_map, used)
            used.remove(right_atom)
            del atom_map[left_atom.id]

    search(0, {}, set())
    return tuple(mappings)


def _mapped_incident_bonds_match(
    left_atom: AtomId,
    right_atom: AtomId,
    atom_map: Mapping[AtomId, AtomId],
    left_adjacency: Mapping[frozenset[AtomId], BondId],
    right_adjacency: Mapping[frozenset[AtomId], BondId],
    left_bonds: Mapping[BondId, BondFacts],
    right_bonds: Mapping[BondId, BondFacts],
) -> bool:
    for other_left, other_right in atom_map.items():
        left_bond_id = left_adjacency.get(frozenset((left_atom, other_left)))
        right_bond_id = right_adjacency.get(frozenset((right_atom, other_right)))
        if left_bond_id is None and right_bond_id is None:
            continue
        if left_bond_id is None or right_bond_id is None:
            return False
        if _bond_signature(left_bonds[left_bond_id]) != _bond_signature(
            right_bonds[right_bond_id]
        ):
            return False
    return True


def _bond_map_for_atom_map(
    left: MoleculeFacts,
    right: MoleculeFacts,
    atom_map: Mapping[AtomId, AtomId],
) -> dict[BondId, BondId] | None:
    right_by_pair = _bond_by_atom_pair(right)
    right_bonds = {bond.id: bond for bond in right.bonds}
    bond_map: dict[BondId, BondId] = {}
    used: set[BondId] = set()

    for left_bond in left.bonds:
        right_id = right_by_pair.get(
            frozenset((atom_map[left_bond.a], atom_map[left_bond.b]))
        )
        if right_id is None or right_id in used:
            return None
        if _bond_signature(left_bond) != _bond_signature(right_bonds[right_id]):
            return None
        bond_map[left_bond.id] = right_id
        used.add(right_id)

    return bond_map


def _components_match(
    left: MoleculeFacts,
    right: MoleculeFacts,
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
) -> bool:
    left_components = Counter(
        (
            frozenset(atom_map[atom] for atom in component.atoms),
            frozenset(bond_map[bond] for bond in component.bonds),
        )
        for component in left.components
    )
    right_components = Counter(
        (
            frozenset(component.atoms),
            frozenset(component.bonds),
        )
        for component in right.components
    )
    return left_components == right_components


def _stereo_matches(
    left: MoleculeFacts,
    right: MoleculeFacts,
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
) -> bool:
    return _tetrahedral_site_keys(left, atom_map, bond_map) == _tetrahedral_site_keys(
        right,
        _identity_atom_map(right),
        _identity_bond_map(right),
    ) and _directional_site_keys(left, atom_map, bond_map) == _directional_site_keys(
        right,
        _identity_atom_map(right),
        _identity_bond_map(right),
    )


def _tetrahedral_site_keys(
    facts: MoleculeFacts,
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
) -> Counter[tuple[object, ...]]:
    occurrences = {occurrence.id: occurrence for occurrence in facts.ligand_occurrences}
    return Counter(
        (
            "tetrahedral",
            int(atom_map[site.center]),
            site.status.value,
            site.target.value,
            tuple(
                sorted(
                    _occurrence_key(occurrences[occurrence], atom_map, bond_map)
                    for occurrence in site.ligand_occurrences
                )
            ),
            tuple(
                _occurrence_key(occurrences[occurrence], atom_map, bond_map)
                for occurrence in site.reference_order
            ),
        )
        for site in facts.stereo.tetrahedral
    )


def _directional_site_keys(
    facts: MoleculeFacts,
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
) -> Counter[tuple[object, ...]]:
    occurrences = {occurrence.id: occurrence for occurrence in facts.ligand_occurrences}
    return Counter(
        _directional_site_key(site, occurrences, atom_map, bond_map)
        for site in facts.stereo.directional
    )


def _directional_site_key(
    site: DirectionalSiteFacts,
    occurrences: Mapping[OccurrenceId, LigandOccurrence],
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
) -> tuple[object, ...]:
    left_ligands = tuple(
        sorted(
            _occurrence_key(occurrences[occurrence], atom_map, bond_map)
            for occurrence in site.left_ligands
        )
    )
    right_ligands = tuple(
        sorted(
            _occurrence_key(occurrences[occurrence], atom_map, bond_map)
            for occurrence in site.right_ligands
        )
    )
    reference_pair = (
        None
        if site.reference_pair is None
        else tuple(
            _occurrence_key(occurrences[occurrence], atom_map, bond_map)
            for occurrence in site.reference_pair
        )
    )
    forward = (
        bond_map[site.center_bond],
        int(atom_map[site.left_endpoint]),
        int(atom_map[site.right_endpoint]),
        site.status.value,
        site.target.value,
        left_ligands,
        right_ligands,
        reference_pair,
    )
    reverse = (
        bond_map[site.center_bond],
        int(atom_map[site.right_endpoint]),
        int(atom_map[site.left_endpoint]),
        site.status.value,
        site.target.value,
        right_ligands,
        left_ligands,
        None if reference_pair is None else (reference_pair[1], reference_pair[0]),
    )
    return ("directional",) + min(forward, reverse)


def _occurrence_key(
    occurrence: LigandOccurrence,
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
) -> tuple[object, ...]:
    return (
        occurrence.kind.value,
        None if occurrence.atom is None else int(atom_map[occurrence.atom]),
        None if occurrence.bond is None else int(bond_map[occurrence.bond]),
        occurrence.ordinal,
    )


def _atom_signature(atom: AtomFacts) -> tuple[object, ...]:
    return (
        atom.atomic_num,
        atom.symbol,
        atom.isotope,
        atom.formal_charge,
        atom.is_aromatic,
        atom.explicit_h_count,
        atom.implicit_h_count,
        atom.no_implicit,
    )


def _bond_signature(bond: BondFacts) -> tuple[object, ...]:
    return (
        bond.order,
        bond.is_aromatic,
        bond.is_conjugated,
    )


def _bond_by_atom_pair(facts: MoleculeFacts) -> dict[frozenset[AtomId], BondId]:
    return {frozenset((bond.a, bond.b)): bond.id for bond in facts.bonds}


def _atom_degree(facts: MoleculeFacts, atom_id: AtomId) -> int:
    return sum(atom_id in (bond.a, bond.b) for bond in facts.bonds)


def _identity_atom_map(facts: MoleculeFacts) -> dict[AtomId, AtomId]:
    return {atom.id: atom.id for atom in facts.atoms}


def _identity_bond_map(facts: MoleculeFacts) -> dict[BondId, BondId]:
    return {bond.id: bond.id for bond in facts.bonds}


__all__ = ("facts_are_isomorphic",)
