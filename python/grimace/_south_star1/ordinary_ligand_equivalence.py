"""Exact RDKit-free ligand equivalence for ordinary stereo-site eligibility."""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Mapping
from dataclasses import dataclass
from dataclasses import field
from typing import Literal

from .facts import AtomFacts
from .facts import BondFacts
from .facts import LigandKind
from .facts import LigandOccurrence
from .facts import MoleculeFacts
from .ids import AtomId
from .ids import BondId
from .ids import SiteId
from .stereo_mapping import specified_stereo_compatible_under_mapping


@dataclass(frozen=True, slots=True)
class AutomorphismAnchor:
    """Atoms and bonds that an ordinary ligand-equivalence test must fix."""

    fixed_atoms: frozenset[AtomId] = frozenset()
    fixed_bonds: frozenset[BondId] = frozenset()


@dataclass(slots=True)
class LigandEquivalenceCache:
    """Memo table for exact anchored ligand-equivalence queries."""

    by_key: dict[tuple[object, ...], bool] = field(default_factory=dict)


@dataclass(slots=True)
class LigandEquivalenceStats:
    """Optional search counters for exact ligand-equivalence diagnostics."""

    cache_hits: int = 0
    cache_misses: int = 0
    searches_started: int = 0
    atom_maps_considered: int = 0
    complete_automorphisms_considered: int = 0


def ligand_occurrences_equivalent(
    facts: MoleculeFacts,
    *,
    anchor: AutomorphismAnchor,
    left: LigandOccurrence,
    right: LigandOccurrence,
    stereo_mode: Literal["graph", "stereochemical"] = "graph",
    ignore_site_ids: frozenset[SiteId] = frozenset(),
    cache: LigandEquivalenceCache | None = None,
    stats: LigandEquivalenceStats | None = None,
) -> bool:
    """Return whether an anchored graph automorphism maps ``left`` to ``right``.

    This is the exact semantic relation for opt-in ordinary stereo-site
    expansion.  Color refinement is used only as pruning; equivalence itself is
    existence of a label-preserving automorphism of ``facts`` that fixes the
    anchor and maps one ligand occurrence to the other.
    """

    facts.validate()
    _validate_stereo_mode(stereo_mode)
    key = _cache_key(
        facts,
        anchor,
        left,
        right,
        stereo_mode=stereo_mode,
        ignore_site_ids=ignore_site_ids,
    )
    if cache is not None and key in cache.by_key:
        if stats is not None:
            stats.cache_hits += 1
        return cache.by_key[key]
    if cache is not None and stats is not None:
        stats.cache_misses += 1

    if left.kind is not right.kind:
        return _cache_result(cache, key, False)
    if left.kind is LigandKind.PSEUDO:
        return _cache_result(cache, key, False)
    if left.kind is LigandKind.IMPLICIT_H:
        if left.atom is None or right.atom is None:
            return _cache_result(cache, key, False)
    if left.kind is LigandKind.NEIGHBOR_ATOM:
        if (
            left.atom is None
            or right.atom is None
            or left.bond is None
            or right.bond is None
        ):
            return _cache_result(cache, key, False)

    _validate_anchor(facts, anchor)
    if stats is not None:
        stats.searches_started += 1
    for atom_map in _iter_anchored_atom_automorphisms(facts, anchor, stats):
        bond_map = _bond_map_for_atom_map(facts, atom_map)
        if bond_map is None:
            continue
        if any(bond_map[bond] != bond for bond in anchor.fixed_bonds):
            continue
        if _occurrence_maps_to(left, right, atom_map=atom_map, bond_map=bond_map):
            if stereo_mode == "stereochemical" and not (
                specified_stereo_compatible_under_mapping(
                    facts,
                    facts,
                    atom_map=atom_map,
                    bond_map=bond_map,
                    ignore_site_ids=ignore_site_ids,
                )
            ):
                continue
            return _cache_result(cache, key, True)
    return _cache_result(cache, key, False)


def _cache_result(
    cache: LigandEquivalenceCache | None,
    key: tuple[object, ...],
    value: bool,
) -> bool:
    if cache is not None:
        cache.by_key[key] = value
    return value


def _cache_key(
    facts: MoleculeFacts,
    anchor: AutomorphismAnchor,
    left: LigandOccurrence,
    right: LigandOccurrence,
    *,
    stereo_mode: str,
    ignore_site_ids: frozenset[SiteId],
) -> tuple[object, ...]:
    return (
        id(facts),
        tuple(sorted(anchor.fixed_atoms, key=int)),
        tuple(sorted(anchor.fixed_bonds, key=int)),
        _occurrence_key(left),
        _occurrence_key(right),
        stereo_mode,
        tuple(sorted(ignore_site_ids, key=int)),
    )


def _occurrence_key(occurrence: LigandOccurrence) -> tuple[object, ...]:
    return (
        occurrence.kind,
        occurrence.atom,
        occurrence.bond,
        occurrence.ordinal,
    )


def _validate_anchor(facts: MoleculeFacts, anchor: AutomorphismAnchor) -> None:
    atom_ids = {atom.id for atom in facts.atoms}
    bond_ids = {bond.id for bond in facts.bonds}
    unknown_atoms = anchor.fixed_atoms - atom_ids
    unknown_bonds = anchor.fixed_bonds - bond_ids
    if unknown_atoms:
        raise ValueError(f"automorphism anchor has unknown atoms: {unknown_atoms!r}")
    if unknown_bonds:
        raise ValueError(f"automorphism anchor has unknown bonds: {unknown_bonds!r}")


def _validate_stereo_mode(stereo_mode: str) -> None:
    if stereo_mode not in {"graph", "stereochemical"}:
        raise ValueError(f"unsupported ligand equivalence stereo mode: {stereo_mode!r}")


def _iter_anchored_atom_automorphisms(
    facts: MoleculeFacts,
    anchor: AutomorphismAnchor,
    stats: LigandEquivalenceStats | None = None,
) -> Iterator[dict[AtomId, AtomId]]:
    candidates_by_atom: dict[AtomId, tuple[AtomId, ...]] = {}
    atoms_by_signature: dict[tuple[object, ...], list[AtomId]] = {}
    for atom in facts.atoms:
        atoms_by_signature.setdefault(_atom_signature(atom), []).append(atom.id)

    for atom in facts.atoms:
        if atom.id in anchor.fixed_atoms:
            candidates_by_atom[atom.id] = (atom.id,)
            continue
        candidates_by_atom[atom.id] = tuple(
            sorted(
                (
                    candidate
                    for candidate in atoms_by_signature[_atom_signature(atom)]
                    if candidate not in anchor.fixed_atoms
                ),
                key=int,
            )
        )

    adjacency = _bond_by_atom_pair(facts)
    bonds = {bond.id: bond for bond in facts.bonds}
    order = tuple(
        sorted(
            facts.atoms,
            key=lambda atom: (
                len(candidates_by_atom[atom.id]),
                -_atom_degree(facts, atom.id),
                int(atom.id),
            ),
        )
    )
    def search(
        index: int,
        atom_map: dict[AtomId, AtomId],
        used: set[AtomId],
    ) -> Iterator[dict[AtomId, AtomId]]:
        if index == len(order):
            if stats is not None:
                stats.complete_automorphisms_considered += 1
            yield dict(atom_map)
            return

        atom = order[index]
        for candidate in candidates_by_atom[atom.id]:
            if candidate in used:
                continue
            if stats is not None:
                stats.atom_maps_considered += 1
            if not _mapped_bonds_match(
                atom.id,
                candidate,
                atom_map,
                adjacency,
                bonds,
            ):
                continue
            atom_map[atom.id] = candidate
            used.add(candidate)
            yield from search(index + 1, atom_map, used)
            used.remove(candidate)
            del atom_map[atom.id]

    yield from search(0, {}, set())


def _mapped_bonds_match(
    left: AtomId,
    right: AtomId,
    atom_map: Mapping[AtomId, AtomId],
    adjacency: Mapping[frozenset[AtomId], BondId],
    bonds: Mapping[BondId, BondFacts],
) -> bool:
    for mapped_left, mapped_right in atom_map.items():
        left_bond = adjacency.get(frozenset((left, mapped_left)))
        right_bond = adjacency.get(frozenset((right, mapped_right)))
        if left_bond is None and right_bond is None:
            continue
        if left_bond is None or right_bond is None:
            return False
        if _bond_signature(bonds[left_bond]) != _bond_signature(bonds[right_bond]):
            return False
    return True


def _bond_map_for_atom_map(
    facts: MoleculeFacts,
    atom_map: Mapping[AtomId, AtomId],
) -> dict[BondId, BondId] | None:
    by_pair = _bond_by_atom_pair(facts)
    bonds = {bond.id: bond for bond in facts.bonds}
    out: dict[BondId, BondId] = {}
    used: set[BondId] = set()
    for bond in facts.bonds:
        mapped = by_pair.get(frozenset((atom_map[bond.a], atom_map[bond.b])))
        if mapped is None or mapped in used:
            return None
        if _bond_signature(bond) != _bond_signature(bonds[mapped]):
            return None
        out[bond.id] = mapped
        used.add(mapped)
    return out


def _occurrence_maps_to(
    left: LigandOccurrence,
    right: LigandOccurrence,
    *,
    atom_map: Mapping[AtomId, AtomId],
    bond_map: Mapping[BondId, BondId],
) -> bool:
    if left.kind is LigandKind.IMPLICIT_H:
        return left.atom is not None and atom_map[left.atom] == right.atom
    if left.kind is LigandKind.NEIGHBOR_ATOM:
        return (
            left.atom is not None
            and right.atom is not None
            and left.bond is not None
            and right.bond is not None
            and atom_map[left.atom] == right.atom
            and bond_map[left.bond] == right.bond
        )
    return False


def _atom_degree(facts: MoleculeFacts, atom: AtomId) -> int:
    return sum(atom in {bond.a, bond.b} for bond in facts.bonds)


def _bond_by_atom_pair(facts: MoleculeFacts) -> dict[frozenset[AtomId], BondId]:
    return {frozenset((bond.a, bond.b)): bond.id for bond in facts.bonds}


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


__all__ = (
    "AutomorphismAnchor",
    "LigandEquivalenceCache",
    "LigandEquivalenceStats",
    "ligand_occurrences_equivalent",
)
