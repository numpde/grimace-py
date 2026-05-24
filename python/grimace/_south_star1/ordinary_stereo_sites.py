"""RDKit-free ordinary stereo-site construction for South Star facts."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from itertools import combinations
from typing import Literal

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import AtomFacts
from .facts import BondFacts
from .facts import BondOrder
from .facts import DirectionalSiteFacts
from .facts import DirectionalValue
from .facts import LigandKind
from .facts import LigandOccurrence
from .facts import MoleculeFacts
from .facts import SiteStatus
from .facts import StereoFacts
from .facts import TetraValue
from .facts import TetrahedralSiteFacts
from .ids import AtomId
from .ids import BondId
from .ids import OccurrenceId
from .ids import SiteId
from .ordinary_ligand_equivalence import AutomorphismAnchor
from .ordinary_ligand_equivalence import LigandEquivalenceCache
from .ordinary_ligand_equivalence import ligand_occurrences_equivalent


@dataclass(frozen=True, slots=True)
class OrdinaryStereoSiteOptions:
    """Options for ordinary RDKit-free potential stereo-site construction."""

    ligand_equivalence: Literal[
        "immediate_color",
        "exact_graph_automorphism",
    ] = "immediate_color"


def ordinary_tetrahedral_candidates(
    facts: MoleculeFacts,
    options: OrdinaryStereoSiteOptions = OrdinaryStereoSiteOptions(),
    ligand_equivalence_cache: LigandEquivalenceCache | None = None,
) -> tuple[tuple[TetrahedralSiteFacts, ...], tuple[LigandOccurrence, ...]]:
    """Return ordinary potential tetrahedral sites implied by graph facts."""

    builder = _SiteBuilder(facts, options, ligand_equivalence_cache)
    builder.add_tetrahedral_candidates(skip_centers=frozenset())
    return tuple(builder.tetrahedral), tuple(builder.occurrences)


def ordinary_directional_candidates(
    facts: MoleculeFacts,
    options: OrdinaryStereoSiteOptions = OrdinaryStereoSiteOptions(),
    ligand_equivalence_cache: LigandEquivalenceCache | None = None,
) -> tuple[tuple[DirectionalSiteFacts, ...], tuple[LigandOccurrence, ...]]:
    """Return ordinary potential directional sites implied by graph facts."""

    builder = _SiteBuilder(facts, options, ligand_equivalence_cache)
    builder.add_directional_candidates(skip_center_bonds=frozenset())
    return tuple(builder.directional), tuple(builder.occurrences)


def add_ordinary_potential_sites(
    facts: MoleculeFacts,
    *,
    preserve_specified: bool = True,
    options: OrdinaryStereoSiteOptions = OrdinaryStereoSiteOptions(),
    ligand_equivalence_cache: LigandEquivalenceCache | None = None,
) -> MoleculeFacts:
    """Add ordinary potential stereo sites without using RDKit perception."""

    facts.validate()
    _validate_options(options)
    if preserve_specified:
        skip_tetra_centers = frozenset(site.center for site in facts.stereo.tetrahedral)
        skip_directional_bonds = frozenset(
            site.center_bond for site in facts.stereo.directional
        )
    else:
        skip_tetra_centers = frozenset()
        skip_directional_bonds = frozenset()

    builder = _SiteBuilder(facts, options, ligand_equivalence_cache)
    builder.add_tetrahedral_candidates(skip_centers=skip_tetra_centers)
    builder.add_directional_candidates(skip_center_bonds=skip_directional_bonds)

    out = replace(
        facts,
        stereo=StereoFacts(
            tetrahedral=facts.stereo.tetrahedral + tuple(builder.tetrahedral),
            directional=facts.stereo.directional + tuple(builder.directional),
        ),
        ligand_occurrences=facts.ligand_occurrences + tuple(builder.occurrences),
    )
    out.validate()
    return out


class _SiteBuilder:
    def __init__(
        self,
        facts: MoleculeFacts,
        options: OrdinaryStereoSiteOptions,
        ligand_equivalence_cache: LigandEquivalenceCache | None,
    ) -> None:
        facts.validate()
        _validate_options(options)
        self.facts = facts
        self.options = options
        if (
            ligand_equivalence_cache is None
            and options.ligand_equivalence == "exact_graph_automorphism"
        ):
            ligand_equivalence_cache = LigandEquivalenceCache()
        self.ligand_equivalence_cache = ligand_equivalence_cache
        self.bonds_by_atom = _bonds_by_atom(facts)
        self.tetrahedral: list[TetrahedralSiteFacts] = []
        self.directional: list[DirectionalSiteFacts] = []
        self.occurrences: list[LigandOccurrence] = []
        self.next_site = _next_site_id(facts)
        self.next_occurrence = _next_occurrence_id(facts)

    def add_tetrahedral_candidates(
        self,
        *,
        skip_centers: frozenset[AtomId],
    ) -> None:
        for atom in self.facts.atoms:
            if atom.id in skip_centers:
                continue
            if not _is_supported_tetrahedral_center(atom):
                continue

            occurrences = self._atom_ligands(atom)
            if not _eligible_tetrahedral_ligands(
                self.facts,
                center=atom.id,
                occurrences=occurrences,
                options=self.options,
                cache=self.ligand_equivalence_cache,
            ):
                continue

            site_id = self._new_site_id()
            occurrence_ids = tuple(
                self._new_ligand_occurrence(
                    replace(occurrence, site=site_id, ordinal=index)
                )
                for index, occurrence in enumerate(occurrences)
            )
            self.tetrahedral.append(
                TetrahedralSiteFacts(
                    id=site_id,
                    center=atom.id,
                    status=SiteStatus.UNSPECIFIED,
                    target=TetraValue.NONE,
                    ligand_occurrences=occurrence_ids,
                    reference_order=occurrence_ids,
                )
            )

    def add_directional_candidates(
        self,
        *,
        skip_center_bonds: frozenset[BondId],
    ) -> None:
        for bond in self.facts.bonds:
            if bond.id in skip_center_bonds:
                continue
            if not _is_supported_directional_center_bond(bond):
                continue

            left_ligands = self._endpoint_ligands(
                endpoint=bond.a,
                center_bond=bond.id,
            )
            right_ligands = self._endpoint_ligands(
                endpoint=bond.b,
                center_bond=bond.id,
            )
            if not left_ligands or not right_ligands:
                continue
            if len(left_ligands) > 2 or len(right_ligands) > 2:
                continue
            if not _eligible_directional_ligands(
                self.facts,
                endpoint=bond.a,
                center_bond=bond.id,
                occurrences=left_ligands,
                options=self.options,
                cache=self.ligand_equivalence_cache,
            ):
                continue
            if not _eligible_directional_ligands(
                self.facts,
                endpoint=bond.b,
                center_bond=bond.id,
                occurrences=right_ligands,
                options=self.options,
                cache=self.ligand_equivalence_cache,
            ):
                continue

            site_id = self._new_site_id()
            left_ids = tuple(
                self._new_ligand_occurrence(
                    replace(occurrence, site=site_id, ordinal=index)
                )
                for index, occurrence in enumerate(left_ligands)
            )
            right_ids = tuple(
                self._new_ligand_occurrence(
                    replace(occurrence, site=site_id, ordinal=index)
                )
                for index, occurrence in enumerate(right_ligands)
            )
            self.directional.append(
                DirectionalSiteFacts(
                    id=site_id,
                    center_bond=bond.id,
                    left_endpoint=bond.a,
                    right_endpoint=bond.b,
                    status=SiteStatus.UNSPECIFIED,
                    target=DirectionalValue.NONE,
                    left_ligands=left_ids,
                    right_ligands=right_ids,
                    reference_pair=None,
                )
            )

    def _atom_ligands(self, atom: AtomFacts) -> tuple[LigandOccurrence, ...]:
        occurrences: list[LigandOccurrence] = []
        for bond in sorted(
            self.bonds_by_atom.get(atom.id, ()),
            key=lambda candidate: (
                int(_other_atom(candidate, atom.id)),
                int(candidate.id),
            ),
        ):
            occurrences.append(
                LigandOccurrence(
                    id=OccurrenceId(-1),
                    site=SiteId(-1),
                    kind=LigandKind.NEIGHBOR_ATOM,
                    atom=_other_atom(bond, atom.id),
                    bond=bond.id,
                )
            )
        for _ in range(atom.implicit_h_count):
            occurrences.append(
                LigandOccurrence(
                    id=OccurrenceId(-1),
                    site=SiteId(-1),
                    kind=LigandKind.IMPLICIT_H,
                    atom=atom.id,
                    bond=None,
                )
            )
        return _sort_ligand_occurrences(
            self.facts,
            center=atom.id,
            occurrences=tuple(occurrences),
        )

    def _endpoint_ligands(
        self,
        *,
        endpoint: AtomId,
        center_bond: BondId,
    ) -> tuple[LigandOccurrence, ...]:
        occurrences: list[LigandOccurrence] = []
        for bond in sorted(
            (
                bond
                for bond in self.bonds_by_atom.get(endpoint, ())
                if bond.id != center_bond
            ),
            key=lambda candidate: (
                int(_other_atom(candidate, endpoint)),
                int(candidate.id),
            ),
        ):
            occurrences.append(
                LigandOccurrence(
                    id=OccurrenceId(-1),
                    site=SiteId(-1),
                    kind=LigandKind.NEIGHBOR_ATOM,
                    atom=_other_atom(bond, endpoint),
                    bond=bond.id,
                )
            )

        atom = _atom_by_id(self.facts)[endpoint]
        for _ in range(atom.implicit_h_count):
            occurrences.append(
                LigandOccurrence(
                    id=OccurrenceId(-1),
                    site=SiteId(-1),
                    kind=LigandKind.IMPLICIT_H,
                    atom=endpoint,
                    bond=None,
                )
            )
        return _sort_ligand_occurrences(
            self.facts,
            center=endpoint,
            occurrences=tuple(occurrences),
        )

    def _new_site_id(self) -> SiteId:
        site_id = SiteId(self.next_site)
        self.next_site += 1
        return site_id

    def _new_ligand_occurrence(self, occurrence: LigandOccurrence) -> OccurrenceId:
        occurrence_id = OccurrenceId(self.next_occurrence)
        self.next_occurrence += 1
        self.occurrences.append(replace(occurrence, id=occurrence_id))
        return occurrence_id


def _is_supported_tetrahedral_center(atom: AtomFacts) -> bool:
    return (
        _is_plain_neutral_atom(atom)
        and atom.atomic_num == 6
        and not atom.is_aromatic
    )


def _is_supported_directional_center_bond(bond: BondFacts) -> bool:
    return bond.order is BondOrder.DOUBLE and not bond.is_aromatic


def _eligible_tetrahedral_ligands(
    facts: MoleculeFacts,
    *,
    center: AtomId,
    occurrences: tuple[LigandOccurrence, ...],
    options: OrdinaryStereoSiteOptions,
    cache: LigandEquivalenceCache | None,
) -> bool:
    if len(occurrences) != 4:
        return False
    if _implicit_h_count(occurrences) > 1:
        return False
    return _ligands_are_distinguishable(
        facts,
        anchor=AutomorphismAnchor(fixed_atoms=frozenset({center})),
        center=center,
        occurrences=occurrences,
        options=options,
        cache=cache,
    )


def _eligible_directional_ligands(
    facts: MoleculeFacts,
    *,
    endpoint: AtomId,
    center_bond: BondId,
    occurrences: tuple[LigandOccurrence, ...],
    options: OrdinaryStereoSiteOptions,
    cache: LigandEquivalenceCache | None,
) -> bool:
    if len(occurrences) not in {1, 2}:
        return False
    if not any(
        occurrence.kind is LigandKind.NEIGHBOR_ATOM
        for occurrence in occurrences
    ):
        return False
    if _implicit_h_count(occurrences) > 1:
        return False
    bond = _bond_by_id(facts)[center_bond]
    return _ligands_are_distinguishable(
        facts,
        anchor=AutomorphismAnchor(
            fixed_atoms=frozenset({bond.a, bond.b}),
            fixed_bonds=frozenset({center_bond}),
        ),
        center=endpoint,
        occurrences=occurrences,
        options=options,
        cache=cache,
    )


def _implicit_h_count(occurrences: tuple[LigandOccurrence, ...]) -> int:
    return sum(
        occurrence.kind is LigandKind.IMPLICIT_H
        for occurrence in occurrences
    )


def _ligands_are_distinguishable(
    facts: MoleculeFacts,
    *,
    anchor: AutomorphismAnchor,
    center: AtomId,
    occurrences: tuple[LigandOccurrence, ...],
    options: OrdinaryStereoSiteOptions,
    cache: LigandEquivalenceCache | None,
) -> bool:
    if options.ligand_equivalence == "immediate_color":
        return _ligand_colors_are_unique(facts, center=center, occurrences=occurrences)
    if options.ligand_equivalence != "exact_graph_automorphism":
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "unsupported ordinary stereo-site ligand equivalence mode: "
            f"{options.ligand_equivalence!r}",
        )

    return all(
        not ligand_occurrences_equivalent(
            facts,
            anchor=anchor,
            left=left,
            right=right,
            cache=cache,
        )
        for left, right in combinations(occurrences, 2)
    )


def _ligand_colors_are_unique(
    facts: MoleculeFacts,
    *,
    center: AtomId,
    occurrences: tuple[LigandOccurrence, ...],
) -> bool:
    colors = tuple(
        _ligand_color(facts, center=center, occurrence=occurrence)
        for occurrence in occurrences
    )
    return len(set(colors)) == len(colors)


def _sort_ligand_occurrences(
    facts: MoleculeFacts,
    *,
    center: AtomId,
    occurrences: tuple[LigandOccurrence, ...],
) -> tuple[LigandOccurrence, ...]:
    return tuple(
        sorted(
            occurrences,
            key=lambda occurrence: _ligand_color(
                facts,
                center=center,
                occurrence=occurrence,
            ),
        )
    )


def _ligand_color(
    facts: MoleculeFacts,
    *,
    center: AtomId,
    occurrence: LigandOccurrence,
) -> tuple[object, ...]:
    if occurrence.kind is LigandKind.IMPLICIT_H:
        return ("implicit_h",)
    if occurrence.kind is LigandKind.NEIGHBOR_ATOM:
        if occurrence.atom is None or occurrence.bond is None:
            raise ValueError(f"neighbor occurrence is incomplete: {occurrence!r}")
        atom = _atom_by_id(facts)[occurrence.atom]
        bond = _bond_by_id(facts)[occurrence.bond]
        if center not in {bond.a, bond.b}:
            raise ValueError(
                f"neighbor occurrence bond {bond.id!r} is not incident to "
                f"center {center!r}"
            )
        return (
            "neighbor",
            atom.atomic_num,
            atom.symbol,
            atom.isotope,
            atom.formal_charge,
            atom.is_aromatic,
            atom.explicit_h_count,
            atom.implicit_h_count,
            atom.no_implicit,
            bond.order.value,
            bond.is_aromatic,
            bond.is_conjugated,
        )
    raise NotImplementedError(
        f"ordinary stereo-site ligand color does not support {occurrence.kind!r}"
    )


def _is_plain_neutral_atom(atom: AtomFacts) -> bool:
    return (
        atom.isotope is None
        and atom.formal_charge == 0
        and atom.explicit_h_count == 0
        and not atom.no_implicit
    )


def _bonds_by_atom(facts: MoleculeFacts) -> dict[AtomId, tuple[BondFacts, ...]]:
    out: dict[AtomId, list[BondFacts]] = {atom.id: [] for atom in facts.atoms}
    for bond in facts.bonds:
        out[bond.a].append(bond)
        out[bond.b].append(bond)
    return {atom: tuple(bonds) for atom, bonds in out.items()}


def _other_atom(bond: BondFacts, atom: AtomId) -> AtomId:
    if bond.a == atom:
        return bond.b
    if bond.b == atom:
        return bond.a
    raise ValueError(f"bond {bond.id!r} is not incident to atom {atom!r}")


def _atom_by_id(facts: MoleculeFacts) -> dict[AtomId, AtomFacts]:
    return {atom.id: atom for atom in facts.atoms}


def _bond_by_id(facts: MoleculeFacts) -> dict[BondId, BondFacts]:
    return {bond.id: bond for bond in facts.bonds}


def _next_site_id(facts: MoleculeFacts) -> int:
    ids = [int(site.id) for site in facts.stereo.tetrahedral]
    ids.extend(int(site.id) for site in facts.stereo.directional)
    return max(ids, default=-1) + 1


def _next_occurrence_id(facts: MoleculeFacts) -> int:
    return (
        max(
            (int(occurrence.id) for occurrence in facts.ligand_occurrences),
            default=-1,
        )
        + 1
    )


def _validate_options(options: OrdinaryStereoSiteOptions) -> None:
    if options.ligand_equivalence not in {
        "immediate_color",
        "exact_graph_automorphism",
    }:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "unsupported ordinary stereo-site ligand equivalence mode: "
            f"{options.ligand_equivalence!r}",
        )


__all__ = (
    "OrdinaryStereoSiteOptions",
    "add_ordinary_potential_sites",
    "ordinary_directional_candidates",
    "ordinary_tetrahedral_candidates",
)
