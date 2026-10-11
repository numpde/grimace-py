"""Prepare the first bounded set of real RDKit molecules for South Star 2."""

from __future__ import annotations

from dataclasses import dataclass
from importlib import import_module


@dataclass(frozen=True, slots=True)
class PreparedMolecule:
    """Immutable atom spelling and indexed bonds for the Rust writer."""

    atoms: tuple[str, ...]
    bonds: tuple[tuple[int, int, str], ...]

    def __post_init__(self) -> None:
        if any(type(value) is not tuple for value in (self.atoms, self.bonds)):
            raise TypeError("prepared molecule facts must use immutable tuples")
        if any(type(value) is not str for value in self.atoms):
            raise TypeError("prepared atom spellings must be strings")
        if any(value not in ("C", "O") for value in self.atoms):
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

        return core._SouthStar2PreparedWriter(list(self.atoms), list(self.bonds))


def prepare_molecule(mol: object) -> PreparedMolecule:
    """Prepare neutral, non-stereo aliphatic C/O graphs with single bonds.

    In this slice, the neutral C/O single-bond graph determines implicit
    hydrogens by standard valence. This checks RDKit's count against that rule
    and hands only the graph and fixed atom text to Rust. Atom spelling does
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

    atom_texts: list[str] = []
    for index, atom in enumerate(mol.GetAtoms()):
        if atom.GetAtomicNum() not in (6, 8):
            raise ValueError(
                f"South Star 2 preparation does not yet spell atom {index} "
                f"with atomic number {atom.GetAtomicNum()}"
            )
        if atom.GetIsAromatic():
            raise ValueError(f"South Star 2 preparation does not yet spell aromatic atom {index}")
        if atom.GetFormalCharge() != 0:
            raise ValueError(f"South Star 2 preparation requires neutral atom {index}")
        if atom.GetIsotope() != 0:
            raise ValueError(
                f"South Star 2 preparation does not yet preserve isotope on atom {index}"
            )
        if atom.GetAtomMapNum() != 0:
            raise ValueError(
                f"South Star 2 preparation does not yet preserve atom map on atom {index}"
            )
        if atom.GetNumRadicalElectrons() != 0:
            raise ValueError(
                f"South Star 2 preparation does not yet preserve radicals on atom {index}"
            )
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            raise ValueError(
                f"South Star 2 preparation does not yet represent configured "
                f"stereo on atom {index}"
            )
        if atom.GetNumExplicitHs() != 0:
            raise ValueError(
                f"South Star 2 preparation does not yet spell explicit hydrogen "
                f"state on atom {index}"
            )
        if atom.GetNoImplicit():
            raise ValueError(
                f"South Star 2 preparation does not yet spell "
                f"no-implicit-hydrogen state on atom {index}"
            )

        valence = 4 if atom.GetAtomicNum() == 6 else 2
        graph_implicit_hydrogens = valence - atom.GetDegree()
        rdkit_implicit_hydrogens = atom.GetNumImplicitHs()
        if graph_implicit_hydrogens < 0 or rdkit_implicit_hydrogens != graph_implicit_hydrogens:
            raise ValueError(
                f"Cannot prepare atom {index}: the supported single-bond graph implies "
                f"{graph_implicit_hydrogens} hydrogens, but RDKit reports "
                f"{rdkit_implicit_hydrogens} implicit hydrogens"
            )
        atom_texts.append("C" if atom.GetAtomicNum() == 6 else "O")

    return PreparedMolecule(atoms=tuple(atom_texts), bonds=tuple(bonds))
