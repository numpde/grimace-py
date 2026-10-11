"""Prepare the first bounded set of real RDKit molecules for South Star 2."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from importlib import import_module
from typing import TypeAlias


class VirtualHydrogen(Enum):
    """A tetrahedral ligand that is not an atom or bond in the prepared graph."""

    INSTANCE = "virtual_hydrogen"


TetrahedralLigandReference: TypeAlias = int | VirtualHydrogen


@dataclass(frozen=True, slots=True)
class TetrahedralAtom:
    """Native bond identities followed by non-graph ligands and parity spellings."""

    reference_order: tuple[TetrahedralLigandReference, ...]
    text_by_parity: tuple[str, str]

    def __post_init__(self) -> None:
        if type(self.reference_order) is not tuple or type(self.text_by_parity) is not tuple:
            raise TypeError("tetrahedral atom facts must use immutable tuples")
        if any(
            type(value) is not int and type(value) is not VirtualHydrogen
            for value in self.reference_order
        ):
            raise TypeError("tetrahedral reference order must contain bond IDs or VirtualHydrogen")
        if any(type(value) is not str for value in self.text_by_parity):
            raise TypeError("tetrahedral parity spellings must be strings")


@dataclass(frozen=True, slots=True)
class PreparedMolecule:
    """Immutable atom facts and indexed bonds for the Rust writer."""

    atoms: tuple[str | TetrahedralAtom, ...]
    bonds: tuple[tuple[int, int, str], ...]

    def __post_init__(self) -> None:
        if any(type(value) is not tuple for value in (self.atoms, self.bonds)):
            raise TypeError("prepared molecule facts must use immutable tuples")
        if any(
            type(value) is not str and type(value) is not TetrahedralAtom
            for value in self.atoms
        ):
            raise TypeError("prepared atoms must be fixed spellings or TetrahedralAtom facts")
        if any(type(value) is str and value not in ("C", "O") for value in self.atoms):
            raise ValueError("prepared atom spellings must be fixed aliphatic C or O tokens")
        for bond in self.bonds:
            if (
                type(bond) is not tuple
                or len(bond) != 3
                or type(bond[0]) is not int
                or type(bond[1]) is not int
                or type(bond[2]) is not str
                or not 0 <= bond[0] < len(self.atoms)
                or not 0 <= bond[1] < len(self.atoms)
                or bond[0] == bond[1]
            ):
                raise ValueError("prepared bonds must contain two valid atom indexes and a token")
            if bond[2] != "":
                raise ValueError("prepared bonds must use the elided single-bond token")

    def build_writer(self) -> object:
        """Construct the private Rust writer when its extension is available."""

        try:
            core = import_module("grimace._core")
        except ModuleNotFoundError as exc:
            if exc.name != "grimace._core":
                raise
            raise ImportError(
                "Cannot build the South Star 2 writer because the compiled "
                "grimace._core extension is unavailable."
            ) from exc

        atom_facts = []
        for atom in self.atoms:
            if type(atom) is str:
                atom_facts.append(atom)
            else:
                reference_order = tuple(
                    "virtual_hydrogen"
                    if ligand is VirtualHydrogen.INSTANCE
                    else ligand
                    for ligand in atom.reference_order
                )
                atom_facts.append(("tetrahedral", reference_order, atom.text_by_parity))
        return core._SouthStar2PreparedWriter(
            atom_facts, list(self.bonds), "rdkit_random_noncanonical"
        )


def prepare_molecule(mol: object) -> PreparedMolecule:
    """Prepare neutral, non-stereo aliphatic C/O graphs with single bonds.

    In this slice, the neutral C/O single-bond graph determines implicit
    hydrogens by standard valence. This checks RDKit's count against that rule
    and hands the graph and atom facts to Rust. Atom spelling does
    not come from an RDKit SMILES writer.
    """

    from rdkit import Chem

    if not isinstance(mol, Chem.Mol):
        raise TypeError("mol must be an RDKit Mol")
    if mol.NeedsUpdatePropertyCache():
        raise ValueError("South Star 2 preparation requires a current RDKit property cache")

    bonds: list[tuple[int, int, str]] = []
    for index, bond in enumerate(mol.GetBonds()):
        bond_type = bond.GetBondType()
        is_aromatic = bond.GetIsAromatic()
        if bond_type != Chem.BondType.SINGLE or is_aromatic:
            raise ValueError(
                f"Cannot prepare South Star 2 molecule: bond {index} has "
                f"{bond_type = !r}, {is_aromatic = }; expected a nonaromatic SINGLE bond."
            )
        if (
            bond.GetStereo() != Chem.BondStereo.STEREONONE
            or bond.GetBondDir() != Chem.BondDir.NONE
        ):
            raise ValueError(
                f"Cannot prepare South Star 2 molecule: bond {index} has "
                f"{bond.GetStereo() = !r}, {bond.GetBondDir() = !r}; "
                "configured bond stereo is not represented."
            )
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), ""))

    atom_facts: list[str | TetrahedralAtom] = []
    for index, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() not in (6, 8):
            raise ValueError(
                f"South Star 2 preparation does not yet spell atom {index} "
                f"with atomic number {atom.GetAtomicNum()}"
            )
        if atom.GetIsAromatic():
            raise ValueError(f"South Star 2 preparation does not yet spell aromatic atom {index}")
        formal_charge = atom.GetFormalCharge()
        if formal_charge != 0:
            raise ValueError(
                f"Cannot prepare atom {index} with {formal_charge = }; expected a neutral atom"
            )
        isotope = atom.GetIsotope()
        if isotope != 0:
            raise ValueError(
                f"Cannot prepare atom {index} with {isotope = }; isotope spelling is unsupported"
            )
        atom_map_num = atom.GetAtomMapNum()
        if atom_map_num != 0:
            raise ValueError(
                f"Cannot prepare atom {index} with {atom_map_num = }; atom-map spelling is unsupported"
            )
        radical_electrons = atom.GetNumRadicalElectrons()
        if radical_electrons != 0:
            raise ValueError(
                f"Cannot prepare atom {index} with {radical_electrons = }; radical spelling is unsupported"
            )
        chiral_tag = atom.GetChiralTag()
        if chiral_tag not in (
            Chem.ChiralType.CHI_UNSPECIFIED,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        ):
            raise ValueError(
                f"Cannot prepare atom {index} with {chiral_tag = !r}; stereo spelling is unsupported"
            )
        tetrahedral = chiral_tag != Chem.ChiralType.CHI_UNSPECIFIED
        if atom.GetNumExplicitHs() and not tetrahedral:
            raise ValueError(
                "South Star 2 preparation does not yet spell explicit hydrogen state "
                f"on atom {index}"
            )
        if atom.GetNoImplicit() and not tetrahedral:
            raise ValueError(
                f"South Star 2 preparation does not yet spell "
                f"no-implicit-hydrogen state on atom {index}"
            )

        valence = 4 if atom.GetAtomicNum() == 6 else 2
        graph_implicit_hydrogens = valence - atom.GetDegree()
        rdkit_implicit_hydrogens = atom.GetNumImplicitHs()
        non_graph_hydrogens = atom.GetTotalNumHs()
        if (
            graph_implicit_hydrogens < 0
            or rdkit_implicit_hydrogens + atom.GetNumExplicitHs() != graph_implicit_hydrogens
            or non_graph_hydrogens != graph_implicit_hydrogens
        ):
            raise ValueError(
                f"Cannot prepare atom {index}: the supported single-bond graph implies "
                f"{graph_implicit_hydrogens} non-graph hydrogens, but RDKit reports "
                f"{non_graph_hydrogens} total non-graph hydrogens"
            )
        if tetrahedral:
            incident_bonds = tuple(bond.GetIdx() for bond in atom.GetBonds())
            if atom.GetAtomicNum() != 6 or len(incident_bonds) != 3 or non_graph_hydrogens != 1:
                raise ValueError(
                    f"Cannot prepare tetrahedral atom {index}: expected degree-three carbon "
                    "with exactly one non-graph hydrogen"
                )
            # RDKit's tetrahedral tag is relative to its incident-bond order;
            # the non-graph hydrogen occupies the final reference position.
            order: tuple[TetrahedralLigandReference, ...] = (
                *incident_bonds,
                VirtualHydrogen.INSTANCE,
            )
            spellings: tuple[str, str] = (
                ("[C@H]", "[C@@H]")
                if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW
                else ("[C@@H]", "[C@H]")
            )
            atom_facts.append(TetrahedralAtom(order, spellings))
        else:
            atom_facts.append("C" if atom.GetAtomicNum() == 6 else "O")

    return PreparedMolecule(atoms=tuple(atom_facts), bonds=tuple(bonds))
