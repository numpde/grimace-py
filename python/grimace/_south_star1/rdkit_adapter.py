"""RDKit ingestion boundary for the private proof kernel.

This is the only South Star 1 module intended to snapshot RDKit ``Mol`` objects
into immutable molecule facts. It must remain a one-way adapter and must not be
called by core enumeration for candidate validation.
"""

from __future__ import annotations

from rdkit import Chem

from .facts import AtomFacts
from .facts import BondFacts
from .facts import BondOrder
from .facts import ComponentFacts
from .facts import MoleculeFacts
from .ids import AtomId
from .ids import BondId
from .ids import ComponentId


def molecule_facts_from_rdkit(mol: Chem.Mol) -> MoleculeFacts:
    """Snapshot a supported non-stereo RDKit molecule into South Star facts."""

    _reject_rdkit_stereo(mol)
    atoms = tuple(_atom_facts(atom) for atom in mol.GetAtoms())
    bonds = tuple(_bond_facts(bond) for bond in mol.GetBonds())
    facts = MoleculeFacts(
        atoms=atoms,
        bonds=bonds,
        components=_component_facts(mol),
    )
    facts.validate()
    return facts


def _reject_rdkit_stereo(mol: Chem.Mol) -> None:
    for atom in mol.GetAtoms():
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            raise NotImplementedError("South Star 1 RDKit adapter rejects atom stereo")
    for bond in mol.GetBonds():
        if bond.GetStereo() != Chem.BondStereo.STEREONONE:
            raise NotImplementedError("South Star 1 RDKit adapter rejects bond stereo")


def _atom_facts(atom: Chem.Atom) -> AtomFacts:
    return AtomFacts(
        id=AtomId(atom.GetIdx()),
        atomic_num=atom.GetAtomicNum(),
        symbol=atom.GetSymbol(),
        isotope=atom.GetIsotope() or None,
        formal_charge=atom.GetFormalCharge(),
        is_aromatic=atom.GetIsAromatic(),
        explicit_h_count=atom.GetNumExplicitHs(),
        implicit_h_count=atom.GetNumImplicitHs(),
        no_implicit=atom.GetNoImplicit(),
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


__all__ = ("molecule_facts_from_rdkit",)
