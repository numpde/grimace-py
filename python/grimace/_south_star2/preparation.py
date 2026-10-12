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
    """Native bond identities followed by the atom's non-graph ligand facts."""

    facts: AtomFacts
    reference_order: tuple[TetrahedralLigandReference, ...]
    at_for_even_parity: bool

    def __post_init__(self) -> None:
        if type(self.reference_order) is not tuple or type(self.at_for_even_parity) is not bool:
            raise TypeError("tetrahedral atom facts must use immutable tuples")
        if type(self.facts) is not AtomFacts:
            raise TypeError("tetrahedral atom requires immutable AtomFacts")
        if any(
            type(value) is not int and type(value) is not VirtualHydrogen
            for value in self.reference_order
        ):
            raise TypeError("tetrahedral reference order must contain bond IDs or VirtualHydrogen")


@dataclass(frozen=True, slots=True)
class AtomFacts:
    """RDKit graph facts used by Rust to generate atom spellings."""

    element: str
    aromatic: bool
    isotope: int | None
    formal_charge: int
    non_graph_hydrogens: int
    atom_class: int
    aromatic_total_valence: int | None

    def __post_init__(self) -> None:
        if type(self.element) is not str or type(self.aromatic) is not bool:
            raise TypeError("atom facts require an element symbol and aromatic flag")
        if self.isotope is not None and type(self.isotope) is not int:
            raise TypeError("atom isotope must be an integer or None")
        if self.aromatic_total_valence is not None and type(self.aromatic_total_valence) is not int:
            raise TypeError("aromatic total valence must be an integer or None")
        if any(type(value) is not int for value in (
            self.formal_charge, self.non_graph_hydrogens, self.atom_class,
        )):
            raise TypeError("atom numeric facts must be integers")


@dataclass(frozen=True, slots=True)
class PreparedMolecule:
    """Immutable atom facts, source annotations, and indexed bonds for Rust."""

    atoms: tuple[AtomFacts | TetrahedralAtom, ...]
    bonds: tuple[tuple[int, int, str], ...]
    directional: tuple[DirectionalDoubleBond, ...] = ()
    source_atom_class_property_present: tuple[bool, ...] = ()

    def __post_init__(self) -> None:
        if any(type(value) is not tuple for value in (
            self.atoms, self.bonds, self.directional, self.source_atom_class_property_present,
        )):
            raise TypeError("prepared molecule facts must use immutable tuples")
        if not self.source_atom_class_property_present and self.atoms:
            object.__setattr__(
                self,
                "source_atom_class_property_present",
                (False,) * len(self.atoms),
            )
        if len(self.source_atom_class_property_present) != len(self.atoms) or any(
            type(value) is not bool for value in self.source_atom_class_property_present
        ):
            raise TypeError("source atom-class presence must be one immutable bool per atom")
        if any(
            type(value) is not AtomFacts and type(value) is not TetrahedralAtom
            for value in self.atoms
        ):
            raise TypeError("prepared atoms must be AtomFacts or TetrahedralAtom facts")
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
            if bond[2] not in ("", "-", ":", "=", "#"):
                raise ValueError(
                    "prepared bonds must use an elided, single, aromatic, double, or triple bond token"
                )
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

    def build_writer(self, *, style: str | None = None) -> object:
        """Construct the private Rust writer when its extension is available."""

        if style not in (None, "rdkit_random_noncanonical"):
            raise ValueError("style must be None or 'rdkit_random_noncanonical'")
        if style is not None:
            for index, atom in enumerate(self.atoms):
                facts = atom.facts if type(atom) is TetrahedralAtom else atom
                if (
                    facts.element not in ("C", "O")
                    or facts.aromatic
                    or facts.isotope is not None
                    or facts.formal_charge != 0
                    or facts.atom_class != 0
                    or self.source_atom_class_property_present[index]
                ):
                    raise ValueError(
                        "style='rdkit_random_noncanonical' is only qualified for neutral, "
                        "nonaromatic C/O atom facts without isotope or class fields"
                    )
            if any(token in (":", "#") for _, _, token in self.bonds):
                raise ValueError(
                    "style='rdkit_random_noncanonical' is not qualified for aromatic or triple bonds"
                )

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
            if type(atom) is TetrahedralAtom:
                facts = atom.facts
                reference_order = tuple(
                    "virtual_hydrogen" if ligand is VirtualHydrogen.INSTANCE else ligand
                    for ligand in atom.reference_order
                )
            else:
                facts = atom
                reference_order = ()
            atom_facts.append((
                facts.element,
                facts.aromatic,
                facts.isotope,
                facts.formal_charge,
                facts.non_graph_hydrogens,
                facts.atom_class,
                atom.at_for_even_parity if type(atom) is TetrahedralAtom else None,
                reference_order,
                self.source_atom_class_property_present[len(atom_facts)],
                facts.aromatic_total_valence,
            ))
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
            atom_facts,
            self.bonds,
            "generic_any_order" if style is None else style,
            directional_facts,
        )


def prepare_molecule(mol: object) -> PreparedMolecule:
    """Prepare finite graph facts supported by the native OpenSMILES writer.

    Tetrahedral facts cover degree-three carbon with one hydrogen. E/Z double
    bonds require one adjacent single-bond reference carrier per end. The
    native writer retains its narrower anchoring admission. Atom spelling does
    not come from an RDKit SMILES writer.
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
        if bond.HasQuery():
            raise ValueError(
                f"Cannot prepare South Star 2 molecule: bond {index} is a query bond"
            )
        if bond_type not in (
            Chem.BondType.SINGLE,
            Chem.BondType.DOUBLE,
            Chem.BondType.TRIPLE,
            Chem.BondType.AROMATIC,
        ):
            raise ValueError(
                f"Cannot prepare South Star 2 molecule: bond {index} has "
                f"{bond_type = !r}; expected a SINGLE, DOUBLE, TRIPLE, or AROMATIC bond."
            )
        if (bond_type == Chem.BondType.AROMATIC and not is_aromatic) or (
            is_aromatic and bond_type == Chem.BondType.TRIPLE
        ):
            raise ValueError(
                f"Cannot prepare South Star 2 molecule: bond {index} has inconsistent "
                f"{bond_type = !r} and {is_aromatic = }"
            )
        stereo = bond.GetStereo()
        bond_dir = bond.GetBondDir()
        begin_aromatic = mol.GetAtomWithIdx(bond.GetBeginAtomIdx()).GetIsAromatic()
        end_aromatic = mol.GetAtomWithIdx(bond.GetEndAtomIdx()).GetIsAromatic()
        if is_aromatic:
            if not (begin_aromatic and end_aromatic):
                raise ValueError(
                    f"Cannot prepare aromatic bond {index}: both endpoint atoms must be aromatic"
                )
            if stereo != Chem.BondStereo.STEREONONE or bond_dir != Chem.BondDir.NONE:
                raise ValueError(
                    f"Cannot prepare aromatic bond {index}: expected no source stereo or direction"
                )
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
        if is_aromatic:
            # RDKit Kekulize(clearAromaticFlags=False) retains source aromaticity
            # while replacing aromatic bond types with integer single/double types.
            token = ":"
        elif bond_type == Chem.BondType.SINGLE:
            token = "-" if begin_aromatic and end_aromatic else ""
        else:
            token = {
                Chem.BondType.DOUBLE: "=",
                Chem.BondType.TRIPLE: "#",
                Chem.BondType.AROMATIC: ":",
            }[bond_type]
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

    atom_facts: list[AtomFacts | TetrahedralAtom] = []
    source_atom_class_property_present: list[bool] = []
    for index, atom in enumerate(mol.GetAtoms()):
        if atom.HasQuery():
            raise ValueError(
                f"Cannot prepare South Star 2 molecule: atom {index} is a query atom"
            )
        formal_charge = atom.GetFormalCharge()
        isotope_number = atom.GetIsotope()
        isotope = isotope_number if isotope_number != 0 else None
        atom_map_num = atom.GetAtomMapNum()
        source_atom_class_property_present.append(
            bool(atom.HasProp("molAtomMapNumber"))
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
        non_graph_hydrogens = atom.GetTotalNumHs()
        facts = AtomFacts(
            element=atom.GetSymbol(),
            aromatic=atom.GetIsAromatic(),
            isotope=isotope,
            formal_charge=formal_charge,
            non_graph_hydrogens=non_graph_hydrogens,
            atom_class=atom_map_num,
            aromatic_total_valence=atom.GetTotalValence() if atom.GetIsAromatic() else None,
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
            atom_facts.append(TetrahedralAtom(
                facts,
                order,
                chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
            ))
        else:
            atom_facts.append(facts)

    unpaired_source_direction_bonds = sorted(
        source_direction_bonds - reference_carrier_bonds
    )
    if unpaired_source_direction_bonds:
        raise ValueError(
            "Cannot prepare source directional marks on bonds "
            f"{unpaired_source_direction_bonds}: each must be a configured "
            "double-bond reference carrier"
        )
    return PreparedMolecule(
        atoms=tuple(atom_facts),
        bonds=tuple(bonds),
        directional=tuple(directional),
        source_atom_class_property_present=tuple(source_atom_class_property_present),
    )
