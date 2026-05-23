"""RDKit ingestion boundary for the private proof kernel.

This is the only South Star 1 module intended to snapshot RDKit ``Mol`` objects
into immutable molecule facts. It must remain a one-way adapter and must not be
called by core enumeration for candidate validation.
"""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from .facts import AtomFacts
from .facts import BondFacts
from .facts import BondOrder
from .facts import ComponentFacts
from .facts import DirectionalSiteFacts
from .facts import DirectionalValue
from .facts import LigandKind
from .facts import MoleculeFacts
from .facts import SiteStatus
from .facts import StereoFacts
from .facts import TetraValue
from .facts import TetrahedralSiteFacts
from .ids import AtomId
from .ids import BondId
from .ids import ComponentId
from .ids import OccurrenceId
from .ordinary_stereo_sites import add_ordinary_potential_sites


@dataclass(frozen=True, slots=True)
class RdkitOrdinaryExtractionOptions:
    include_potential_sites: bool = True
    extract_specified_tetrahedral: bool = True
    extract_specified_directional: bool = True
    normalize_non_graph_hydrogens: bool = True
    reject_unsupported_stereo: bool = True


def molecule_facts_from_rdkit(mol: Chem.Mol) -> MoleculeFacts:
    """Snapshot a supported non-stereo RDKit molecule into South Star facts."""

    _reject_rdkit_stereo(mol)
    atoms = tuple(
        _atom_facts(
            atom,
            normalize_non_graph_hydrogens=True,
        )
        for atom in mol.GetAtoms()
    )
    bonds = tuple(_bond_facts(bond) for bond in mol.GetBonds())
    facts = MoleculeFacts(
        atoms=atoms,
        bonds=bonds,
        components=_component_facts(mol),
    )
    facts.validate()
    return facts


def ordinary_molecule_facts_from_rdkit(
    mol: Chem.Mol,
    options: RdkitOrdinaryExtractionOptions = RdkitOrdinaryExtractionOptions(),
) -> MoleculeFacts:
    """Snapshot RDKit molecules into the ordinary South Star fact convention."""

    _validate_stereo_extraction_scope(mol, options)
    facts = _base_molecule_facts_from_rdkit(
        mol,
        normalize_non_graph_hydrogens=options.normalize_non_graph_hydrogens,
    )
    if options.include_potential_sites:
        facts = add_ordinary_potential_sites(facts)
    if options.extract_specified_tetrahedral:
        facts = _overlay_rdkit_tetrahedral_stereo(mol, facts)
    if options.extract_specified_directional:
        facts = _overlay_rdkit_directional_stereo(mol, facts)
    facts.validate()
    return facts


def _base_molecule_facts_from_rdkit(
    mol: Chem.Mol,
    *,
    normalize_non_graph_hydrogens: bool,
) -> MoleculeFacts:
    atoms = tuple(
        _atom_facts(
            atom,
            normalize_non_graph_hydrogens=normalize_non_graph_hydrogens,
        )
        for atom in mol.GetAtoms()
    )
    bonds = tuple(_bond_facts(bond) for bond in mol.GetBonds())
    facts = MoleculeFacts(
        atoms=atoms,
        bonds=bonds,
        components=_component_facts(mol),
    )
    facts.validate()
    return facts


def _validate_stereo_extraction_scope(
    mol: Chem.Mol,
    options: RdkitOrdinaryExtractionOptions,
) -> None:
    has_atom_stereo = _has_rdkit_atom_stereo(mol)
    has_bond_stereo = _has_rdkit_bond_stereo(mol)

    if has_atom_stereo and not options.extract_specified_tetrahedral:
        if options.reject_unsupported_stereo:
            _reject_rdkit_stereo(mol)
    if has_bond_stereo and not options.extract_specified_directional:
        if options.reject_unsupported_stereo:
            _reject_rdkit_stereo(mol)
    if options.reject_unsupported_stereo and (has_atom_stereo or has_bond_stereo):
        if has_bond_stereo and not options.extract_specified_directional:
            _reject_rdkit_stereo(mol)


def _has_rdkit_atom_stereo(mol: Chem.Mol) -> bool:
    return any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in mol.GetAtoms()
    )


def _has_rdkit_bond_stereo(mol: Chem.Mol) -> bool:
    return any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        for bond in mol.GetBonds()
    )


def _overlay_rdkit_tetrahedral_stereo(
    mol: Chem.Mol,
    facts: MoleculeFacts,
) -> MoleculeFacts:
    sites_by_center = {
        site.center: site
        for site in facts.stereo.tetrahedral
    }
    replacement_by_id: dict[object, TetrahedralSiteFacts] = {}

    for atom in mol.GetAtoms():
        tag = atom.GetChiralTag()
        if tag == Chem.ChiralType.CHI_UNSPECIFIED:
            continue
        if tag not in {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        }:
            raise NotImplementedError(f"unsupported RDKit atom stereo: {tag!r}")

        center = AtomId(atom.GetIdx())
        site = sites_by_center.get(center)
        if site is None:
            raise NotImplementedError(
                f"RDKit tetrahedral atom has no ordinary potential site: {center!r}"
            )

        replacement_by_id[site.id] = TetrahedralSiteFacts(
            id=site.id,
            center=site.center,
            status=SiteStatus.SPECIFIED,
            target=_tetra_target_from_rdkit_tag(tag),
            ligand_occurrences=site.ligand_occurrences,
            reference_order=_rdkit_tetra_reference_order(atom, facts, site),
        )

    if not replacement_by_id:
        return facts

    out = _replace_tetrahedral_sites(facts, replacement_by_id)
    out.validate()
    return out


def _tetra_target_from_rdkit_tag(tag) -> TetraValue:
    if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        return TetraValue.PLUS
    if tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        return TetraValue.MINUS
    raise NotImplementedError(f"unsupported RDKit tetrahedral tag: {tag!r}")


def _rdkit_tetra_reference_order(
    atom: Chem.Atom,
    facts: MoleculeFacts,
    site: TetrahedralSiteFacts,
) -> tuple[OccurrenceId, ...]:
    occurrence_by_neighbor = _neighbor_tetra_occurrences_by_atom(facts, site)
    implicit_h = tuple(
        occurrence.id
        for occurrence in facts.ligand_occurrences
        if occurrence.site == site.id and occurrence.kind is LigandKind.IMPLICIT_H
    )
    reference_order = tuple(
        occurrence_by_neighbor[AtomId(neighbor.GetIdx())]
        for neighbor in atom.GetNeighbors()
        if AtomId(neighbor.GetIdx()) in occurrence_by_neighbor
    ) + implicit_h
    if set(reference_order) != set(site.ligand_occurrences):
        raise ValueError(
            f"RDKit tetrahedral reference order does not cover site {site.id!r}"
        )
    return reference_order


def _neighbor_tetra_occurrences_by_atom(
    facts: MoleculeFacts,
    site: TetrahedralSiteFacts,
) -> dict[AtomId, OccurrenceId]:
    occurrence_by_id = {
        occurrence.id: occurrence
        for occurrence in facts.ligand_occurrences
    }
    out: dict[AtomId, OccurrenceId] = {}
    for occurrence_id in site.ligand_occurrences:
        occurrence = occurrence_by_id[occurrence_id]
        if occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
            continue
        if occurrence.atom is None:
            raise ValueError(f"neighbor occurrence lacks atom: {occurrence.id!r}")
        out[occurrence.atom] = occurrence.id
    return out


def _replace_tetrahedral_sites(
    facts: MoleculeFacts,
    replacement_by_id: dict[object, TetrahedralSiteFacts],
) -> MoleculeFacts:
    return MoleculeFacts(
        atoms=facts.atoms,
        bonds=facts.bonds,
        components=facts.components,
        stereo=StereoFacts(
            tetrahedral=tuple(
                replacement_by_id.get(site.id, site)
                for site in facts.stereo.tetrahedral
            ),
            directional=facts.stereo.directional,
        ),
        ligand_occurrences=facts.ligand_occurrences,
    )


def _overlay_rdkit_directional_stereo(
    mol: Chem.Mol,
    facts: MoleculeFacts,
) -> MoleculeFacts:
    sites_by_center_bond = {
        site.center_bond: site
        for site in facts.stereo.directional
    }
    replacement_by_id: dict[object, DirectionalSiteFacts] = {}

    for bond in mol.GetBonds():
        stereo = bond.GetStereo()
        if stereo == Chem.BondStereo.STEREONONE:
            continue
        if stereo not in {Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ}:
            raise NotImplementedError(f"unsupported RDKit bond stereo: {stereo!r}")

        center_bond = BondId(bond.GetIdx())
        site = sites_by_center_bond.get(center_bond)
        if site is None:
            raise NotImplementedError(
                f"RDKit stereo bond has no ordinary potential site: {center_bond!r}"
            )
        stereo_atoms = tuple(AtomId(atom_idx) for atom_idx in bond.GetStereoAtoms())
        if len(stereo_atoms) != 2:
            raise NotImplementedError(
                f"RDKit stereo bond {center_bond!r} lacks two stereo atoms"
            )
        reference_pair = _directional_reference_pair_from_stereo_atoms(
            facts,
            site,
            stereo_atoms,
        )
        replacement_by_id[site.id] = DirectionalSiteFacts(
            id=site.id,
            center_bond=site.center_bond,
            left_endpoint=site.left_endpoint,
            right_endpoint=site.right_endpoint,
            status=SiteStatus.SPECIFIED,
            target=_directional_target_from_rdkit_stereo(stereo),
            left_ligands=site.left_ligands,
            right_ligands=site.right_ligands,
            reference_pair=reference_pair,
        )

    if not replacement_by_id:
        return facts

    out = _replace_directional_sites(facts, replacement_by_id)
    out.validate()
    return out


def _directional_reference_pair_from_stereo_atoms(
    facts: MoleculeFacts,
    site: DirectionalSiteFacts,
    stereo_atoms: tuple[AtomId, AtomId],
) -> tuple[OccurrenceId, OccurrenceId]:
    left_by_atom = _neighbor_directional_occurrences_by_atom(
        facts,
        site.left_ligands,
    )
    right_by_atom = _neighbor_directional_occurrences_by_atom(
        facts,
        site.right_ligands,
    )
    first, second = stereo_atoms
    if first in left_by_atom and second in right_by_atom:
        return (left_by_atom[first], right_by_atom[second])
    if second in left_by_atom and first in right_by_atom:
        return (left_by_atom[second], right_by_atom[first])
    raise NotImplementedError(
        f"RDKit stereo atoms do not match ordinary directional site {site.id!r}"
    )


def _neighbor_directional_occurrences_by_atom(
    facts: MoleculeFacts,
    occurrence_ids: tuple[OccurrenceId, ...],
) -> dict[AtomId, OccurrenceId]:
    occurrence_by_id = {
        occurrence.id: occurrence
        for occurrence in facts.ligand_occurrences
    }
    out: dict[AtomId, OccurrenceId] = {}
    for occurrence_id in occurrence_ids:
        occurrence = occurrence_by_id[occurrence_id]
        if occurrence.kind is not LigandKind.NEIGHBOR_ATOM:
            continue
        if occurrence.atom is None:
            raise ValueError(f"neighbor occurrence lacks atom: {occurrence.id!r}")
        out[occurrence.atom] = occurrence.id
    return out


def _directional_target_from_rdkit_stereo(stereo) -> DirectionalValue:
    if stereo == Chem.BondStereo.STEREOZ:
        return DirectionalValue.TOGETHER
    if stereo == Chem.BondStereo.STEREOE:
        return DirectionalValue.OPPOSITE
    raise NotImplementedError(f"unsupported RDKit bond stereo: {stereo!r}")


def _replace_directional_sites(
    facts: MoleculeFacts,
    replacement_by_id: dict[object, DirectionalSiteFacts],
) -> MoleculeFacts:
    return MoleculeFacts(
        atoms=facts.atoms,
        bonds=facts.bonds,
        components=facts.components,
        stereo=StereoFacts(
            tetrahedral=facts.stereo.tetrahedral,
            directional=tuple(
                replacement_by_id.get(site.id, site)
                for site in facts.stereo.directional
            ),
        ),
        ligand_occurrences=facts.ligand_occurrences,
    )


def _reject_rdkit_stereo(mol: Chem.Mol) -> None:
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            raise NotImplementedError("South Star 1 RDKit adapter rejects atom stereo")
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            raise NotImplementedError("South Star 1 RDKit adapter rejects bond stereo")


def _atom_facts(
    atom: Chem.Atom,
    *,
    normalize_non_graph_hydrogens: bool,
) -> AtomFacts:
    explicit_h_count = atom.GetNumExplicitHs()
    implicit_h_count = atom.GetNumImplicitHs()
    no_implicit = atom.GetNoImplicit()
    if normalize_non_graph_hydrogens:
        implicit_h_count += explicit_h_count
        explicit_h_count = 0
        no_implicit = False

    return AtomFacts(
        id=AtomId(atom.GetIdx()),
        atomic_num=atom.GetAtomicNum(),
        symbol=atom.GetSymbol(),
        isotope=atom.GetIsotope() or None,
        formal_charge=atom.GetFormalCharge(),
        is_aromatic=atom.GetIsAromatic(),
        explicit_h_count=explicit_h_count,
        implicit_h_count=implicit_h_count,
        no_implicit=no_implicit,
    )


def _bond_facts(bond: Chem.Bond) -> BondFacts:
    return BondFacts(
        id=BondId(bond.GetIdx()),
        a=AtomId(bond.GetBeginAtomIdx()),
        b=AtomId(bond.GetEndAtomIdx()),
        order=_bond_order(bond),
        is_aromatic=bond.GetIsAromatic(),
        is_conjugated=bond.GetIsConjugated(),
    )


def _bond_order(bond: Chem.Bond) -> BondOrder:
    bond_type = bond.GetBondType()
    if bond_type == Chem.BondType.SINGLE:
        return BondOrder.SINGLE
    if bond_type == Chem.BondType.DOUBLE:
        return BondOrder.DOUBLE
    if bond_type == Chem.BondType.TRIPLE:
        return BondOrder.TRIPLE
    if bond_type == Chem.BondType.AROMATIC:
        return BondOrder.AROMATIC
    raise NotImplementedError(f"unsupported RDKit bond type: {bond_type!r}")


def _component_facts(mol: Chem.Mol) -> tuple[ComponentFacts, ...]:
    atom_count = mol.GetNumAtoms()
    bond_by_atom: dict[int, list[int]] = {idx: [] for idx in range(atom_count)}
    for bond in mol.GetBonds():
        bond_by_atom[bond.GetBeginAtomIdx()].append(bond.GetIdx())
        bond_by_atom[bond.GetEndAtomIdx()].append(bond.GetIdx())

    components: list[ComponentFacts] = []
    seen_atoms: set[int] = set()
    for start in range(atom_count):
        if start in seen_atoms:
            continue
        atom_ids = _reachable_atom_indices(mol, start, bond_by_atom, seen_atoms)
        atom_set = set(atom_ids)
        bond_ids = tuple(
            BondId(bond.GetIdx())
            for bond in mol.GetBonds()
            if bond.GetBeginAtomIdx() in atom_set and bond.GetEndAtomIdx() in atom_set
        )
        components.append(
            ComponentFacts(
                id=ComponentId(len(components)),
                atoms=tuple(AtomId(idx) for idx in atom_ids),
                bonds=bond_ids,
            )
        )
    return tuple(components)


def _reachable_atom_indices(
    mol: Chem.Mol,
    start: int,
    bond_by_atom: dict[int, list[int]],
    seen_atoms: set[int],
) -> tuple[int, ...]:
    component: list[int] = []
    stack = [start]
    while stack:
        atom_idx = stack.pop()
        if atom_idx in seen_atoms:
            continue
        seen_atoms.add(atom_idx)
        component.append(atom_idx)
        for bond_idx in reversed(bond_by_atom[atom_idx]):
            bond = mol.GetBondWithIdx(bond_idx)
            neighbor = (
                bond.GetEndAtomIdx()
                if bond.GetBeginAtomIdx() == atom_idx
                else bond.GetBeginAtomIdx()
            )
            if neighbor not in seen_atoms:
                stack.append(neighbor)
    return tuple(sorted(component))


__all__ = (
    "RdkitOrdinaryExtractionOptions",
    "molecule_facts_from_rdkit",
    "ordinary_molecule_facts_from_rdkit",
)
