"""RDKit-free isomorphism checks for South Star molecule facts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass

from .facts import AtomFacts
from .facts import BondFacts
from .facts import MoleculeFacts
from .facts import SiteStatus
from .ids import AtomId
from .ids import BondId
from .stereo_mapping import directional_sites_compatible_under_mapping
from .stereo_mapping import tetrahedral_sites_compatible_under_mapping


@dataclass(frozen=True, slots=True)
class FactIsomorphism:
    """One graph isomorphism between two ``MoleculeFacts`` snapshots."""

    atom_map: dict[AtomId, AtomId]
    bond_map: dict[BondId, BondId]


@dataclass(frozen=True, slots=True)
class FactIsomorphismResult:
    """Structured result for fact isomorphism checks."""

    isomorphic: bool
    isomorphism: FactIsomorphism | None = None
    reason: str | None = None

    def __bool__(self) -> bool:
        return self.isomorphic


def facts_are_isomorphic(
    left: MoleculeFacts,
    right: MoleculeFacts,
    *,
    compare_stereo: bool = True,
    compare_potential_sites: bool = True,
) -> FactIsomorphismResult:
    """Return whether two fact snapshots are equal up to generated ids.

    This is a small-audit helper, not a graph-isomorphism engine.  It exists so
    parsed SMILES snapshots can be compared semantically even when the parser
    assigned different atom and bond ids.  The check preserves atom attributes,
    bond attributes, component partitions, and, when requested, stereo site
    structure/status/targets modulo reference-order parity.
    """

    left.validate()
    right.validate()

    cheap_failure = _cheap_invariant_failure(
        left,
        right,
        compare_stereo=compare_stereo,
        compare_potential_sites=compare_potential_sites,
    )
    if cheap_failure is not None:
        return _failure(cheap_failure)

    stereo_failure: str | None = None
    for atom_map in _atom_isomorphisms(left, right):
        bond_map = _bond_map_for_atom_map(left, right, atom_map)
        if bond_map is None:
            continue
        if not _components_match(left, right, atom_map, bond_map):
            continue
        if compare_stereo:
            matched, reason = _stereo_matches(
                left,
                right,
                atom_map,
                bond_map,
                compare_potential_sites=compare_potential_sites,
            )
            if not matched:
                stereo_failure = reason
                continue
        return FactIsomorphismResult(
            isomorphic=True,
            isomorphism=FactIsomorphism(
                atom_map=dict(atom_map),
                bond_map=dict(bond_map),
            ),
        )

    if stereo_failure is not None:
        return _failure(stereo_failure)
    return _failure("no atom/bond/component isomorphism found")


def _cheap_invariant_failure(
    left: MoleculeFacts,
    right: MoleculeFacts,
    *,
    compare_stereo: bool,
    compare_potential_sites: bool,
) -> str | None:
    if len(left.atoms) != len(right.atoms):
        return "atom count mismatch"
    if len(left.bonds) != len(right.bonds):
        return "bond count mismatch"
    if len(left.components) != len(right.components):
        return "component count mismatch"
    if Counter(map(_atom_signature, left.atoms)) != Counter(
        map(_atom_signature, right.atoms)
    ):
        return "atom color multiset mismatch"
    if Counter(map(_bond_signature, left.bonds)) != Counter(
        map(_bond_signature, right.bonds)
    ):
        return "bond color multiset mismatch"
    if Counter(_component_size(component) for component in left.components) != Counter(
        _component_size(component) for component in right.components
    ):
        return "component size multiset mismatch"

    if not compare_stereo:
        return None

    left_tetra = _site_status_counts(
        site.status for site in left.stereo.tetrahedral
    )
    right_tetra = _site_status_counts(
        site.status for site in right.stereo.tetrahedral
    )
    left_directional = _site_status_counts(
        site.status for site in left.stereo.directional
    )
    right_directional = _site_status_counts(
        site.status for site in right.stereo.directional
    )

    if not compare_potential_sites:
        left_tetra.pop(SiteStatus.UNSPECIFIED, None)
        right_tetra.pop(SiteStatus.UNSPECIFIED, None)
        left_directional.pop(SiteStatus.UNSPECIFIED, None)
        right_directional.pop(SiteStatus.UNSPECIFIED, None)

    if left_tetra != right_tetra:
        return "tetrahedral site status count mismatch"
    if left_directional != right_directional:
        return "directional site status count mismatch"
    return None


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
    *,
    compare_potential_sites: bool,
) -> tuple[bool, str | None]:
    if not tetrahedral_sites_compatible_under_mapping(
        left,
        right,
        atom_map=atom_map,
        bond_map=bond_map,
        compare_potential_sites=compare_potential_sites,
    ):
        return (False, "tetrahedral stereo mismatch")
    if not directional_sites_compatible_under_mapping(
        left,
        right,
        atom_map=atom_map,
        bond_map=bond_map,
        compare_potential_sites=compare_potential_sites,
    ):
        return (False, "directional stereo mismatch")
    return (True, None)


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


def _component_size(component) -> tuple[int, int]:
    return (len(component.atoms), len(component.bonds))


def _site_status_counts(statuses) -> Counter[SiteStatus]:
    return Counter(statuses)


def _bond_by_atom_pair(facts: MoleculeFacts) -> dict[frozenset[AtomId], BondId]:
    return {frozenset((bond.a, bond.b)): bond.id for bond in facts.bonds}


def _atom_degree(facts: MoleculeFacts, atom_id: AtomId) -> int:
    return sum(atom_id in (bond.a, bond.b) for bond in facts.bonds)


def _failure(reason: str) -> FactIsomorphismResult:
    return FactIsomorphismResult(
        isomorphic=False,
        reason=reason,
    )


__all__ = (
    "FactIsomorphism",
    "FactIsomorphismResult",
    "facts_are_isomorphic",
)
