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
class DirectionalDoubleBond:
    """A configured double bond and its two reference-ligand carrier bonds."""

    bond: int
    left_endpoint: int
    left_carrier: int
    right_endpoint: int
    right_carrier: int
    side_phase_xor: bool

    def __post_init__(self) -> None:
        if any(type(value) is not int for value in (
            self.bond, self.left_endpoint, self.left_carrier,
            self.right_endpoint, self.right_carrier,
        )) or type(self.side_phase_xor) is not bool:
            raise TypeError("directional double-bond facts must contain primitive immutable values")


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
    directional: tuple[DirectionalDoubleBond, ...] = ()

    def __post_init__(self) -> None:
        if any(type(value) is not tuple for value in (self.atoms, self.bonds, self.directional)):
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
            if bond[2] not in ("", "="):
                raise ValueError("prepared bonds must use an elided single-bond token or a double-bond token")
        for relation in self.directional:
            if type(relation) is not DirectionalDoubleBond:
                raise TypeError("prepared directional facts must be DirectionalDoubleBond values")
            if not (
                0 <= relation.bond < len(self.bonds)
                and 0 <= relation.left_endpoint < len(self.atoms)
                and 0 <= relation.right_endpoint < len(self.atoms)
                and 0 <= relation.left_carrier < len(self.bonds)
                and 0 <= relation.right_carrier < len(self.bonds)
            ):
                raise ValueError("prepared directional facts must reference existing atoms and bonds")
            if self.bonds[relation.bond] != (
                relation.left_endpoint, relation.right_endpoint, "="
            ):
                raise ValueError("prepared directional endpoint order must match its double-bond orientation")
            for endpoint, carrier_id in (
                (relation.left_endpoint, relation.left_carrier),
                (relation.right_endpoint, relation.right_carrier),
            ):
                a, b, token = self.bonds[carrier_id]
                if token != "" or endpoint not in (a, b):
                    raise ValueError("prepared directional carrier must be an incident single-bond edge")

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
        directional_facts = tuple(
            (
                relation.bond,
                relation.left_endpoint,
                relation.left_carrier,
                relation.right_endpoint,
                relation.right_carrier,
                relation.side_phase_xor,
            )
            for relation in self.directional
        )
        return core._SouthStar2PreparedWriter(
            atom_facts, self.bonds, "rdkit_random_noncanonical", directional_facts
        )


def prepare_molecule(mol: object) -> PreparedMolecule:
    """Prepare neutral nonaromatic aliphatic C/O graphs with single/double bonds.

    Tetrahedral facts cover degree-three carbon with one hydrogen. E/Z double
    bonds require one adjacent single-bond reference carrier per end. The
    native writer retains its narrower anchoring admission. Bond-order valence
    determines implicit hydrogens, which preparation checks against RDKit's
    count. Atom spelling does not come from an RDKit SMILES writer.
    """

    from rdkit import Chem

    if not isinstance(mol, Chem.Mol):
        raise TypeError("mol must be an RDKit Mol")
    if mol.NeedsUpdatePropertyCache():
        raise ValueError("South Star 2 preparation requires a current RDKit property cache")

    bonds: list[tuple[int, int, str]] = []
    directional: list[DirectionalDoubleBond] = []
    source_direction_bonds: set[int] = set()
    reference_carrier_bonds: set[int] = set()
    for index, bond in enumerate(mol.GetBonds()):
        bond_type = bond.GetBondType()
        is_aromatic = bond.GetIsAromatic()
        if bond_type not in (Chem.BondType.SINGLE, Chem.BondType.DOUBLE) or is_aromatic:
            raise ValueError(
                f"Cannot prepare South Star 2 molecule: bond {index} has "
                f"{bond_type = !r}, {is_aromatic = }; expected a nonaromatic SINGLE or DOUBLE bond."
            )
        stereo = bond.GetStereo()
        bond_dir = bond.GetBondDir()
        configured = stereo in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOZ,
                                Chem.BondStereo.STEREOTRANS, Chem.BondStereo.STEREOCIS)
        if bond_type == Chem.BondType.SINGLE and stereo != Chem.BondStereo.STEREONONE:
            raise ValueError(
                f"Cannot prepare South Star 2 molecule: bond {index} has "
                f"{stereo = !r}; a single bond cannot carry configured double-bond stereo."
            )
        if bond_type == Chem.BondType.DOUBLE and bond_dir != Chem.BondDir.NONE:
            raise ValueError(
                f"Cannot prepare double bond {index} with {bond_dir = !r}; expected no source direction."
            )
        if bond_dir not in (Chem.BondDir.NONE, Chem.BondDir.ENDUPRIGHT, Chem.BondDir.ENDDOWNRIGHT):
            raise ValueError(f"Cannot prepare bond {index} with unsupported source direction {bond_dir!r}")
        if bond_dir != Chem.BondDir.NONE:
            source_direction_bonds.add(index)
        if bond_type == Chem.BondType.DOUBLE and stereo != Chem.BondStereo.STEREONONE and not configured:
            raise ValueError(f"Cannot prepare double bond {index} with unsupported {stereo = !r}")
        token = "" if bond_type == Chem.BondType.SINGLE else "="
        bonds.append((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), token))
        if bond_type == Chem.BondType.DOUBLE and stereo != Chem.BondStereo.STEREONONE:
            references = tuple(bond.GetStereoAtoms())
            if len(references) != 2:
                raise ValueError(f"Cannot prepare configured double bond {index}: expected two stereo reference atoms")
            left, right = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            carriers = []
            for endpoint, reference in zip((left, right), references, strict=True):
                eligible = tuple(
                    adjacent.GetIdx() for adjacent in mol.GetAtomWithIdx(endpoint).GetBonds()
                    if adjacent.GetIdx() != index and adjacent.GetBondType() == Chem.BondType.SINGLE
                    and not adjacent.GetIsAromatic()
                )
                reference_bond = tuple(
                    adjacent.GetIdx() for adjacent in mol.GetAtomWithIdx(endpoint).GetBonds()
                    if adjacent.GetOtherAtomIdx(endpoint) == reference
                )
                if len(eligible) != 1 or reference_bond != eligible:
                    raise ValueError(
                        f"Cannot prepare configured double bond {index} at endpoint {endpoint}: "
                        "expected one single-bond carrier to the stereo reference atom"
                    )
                carriers.append(eligible[0])
                reference_carrier_bonds.add(eligible[0])
            phase = stereo in (Chem.BondStereo.STEREOE, Chem.BondStereo.STEREOTRANS)
            directional.append(DirectionalDoubleBond(index, left, carriers[0], right, carriers[1], phase))

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
        bond_valence = sum(int(bond.GetBondTypeAsDouble()) for bond in atom.GetBonds())
        graph_implicit_hydrogens = valence - bond_valence
        rdkit_implicit_hydrogens = atom.GetNumImplicitHs()
        non_graph_hydrogens = atom.GetTotalNumHs()
        if (
            graph_implicit_hydrogens < 0
            or rdkit_implicit_hydrogens + atom.GetNumExplicitHs() != graph_implicit_hydrogens
            or non_graph_hydrogens != graph_implicit_hydrogens
        ):
            raise ValueError(
                f"Cannot prepare atom {index}: supported bond-order valence implies "
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

    unpaired_source_direction_bonds = sorted(
        source_direction_bonds - reference_carrier_bonds
    )
    if unpaired_source_direction_bonds:
        raise ValueError(
            "Cannot prepare source directional marks on bonds "
            f"{unpaired_source_direction_bonds}: each must be a configured "
            "double-bond reference carrier"
        )
    return PreparedMolecule(atoms=tuple(atom_facts), bonds=tuple(bonds), directional=tuple(directional))
