from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any

from rdkit import Chem, rdBase

from grimace._reference.policy import ReferencePolicy
from grimace._reference.rdkit_random import identity_smiles


PREPARED_SMILES_GRAPH_SCHEMA_VERSION = 1
CONNECTED_NONSTEREO_SURFACE = "connected_nonstereo"
CONNECTED_STEREO_SURFACE = "connected_stereo"

SUPPORTED_BOND_TYPES = {
    Chem.BondType.SINGLE,
    Chem.BondType.DOUBLE,
    Chem.BondType.TRIPLE,
    Chem.BondType.AROMATIC,
}

SUPPORTED_STEREO_CHIRAL_TAGS = {
    Chem.ChiralType.CHI_UNSPECIFIED,
    Chem.ChiralType.CHI_TETRAHEDRAL,
    Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.ChiralType.CHI_TETRAHEDRAL_CW,
}

SUPPORTED_STEREO_BOND_STEREO = {
    Chem.BondStereo.STEREONONE,
    Chem.BondStereo.STEREOCIS,
    Chem.BondStereo.STEREOTRANS,
    Chem.BondStereo.STEREOE,
    Chem.BondStereo.STEREOZ,
}

SUPPORTED_STEREO_BOND_DIRS = {
    Chem.BondDir.NONE,
    Chem.BondDir.ENDUPRIGHT,
    Chem.BondDir.ENDDOWNRIGHT,
}

ORGANIC_SUBSET = {
    "B",
    "C",
    "N",
    "O",
    "P",
    "S",
    "F",
    "Cl",
    "Br",
    "I",
}

AROMATIC_SUBSET = {
    "b",
    "c",
    "n",
    "o",
    "p",
    "s",
    "se",
    "as",
}

UNBRACKETED_NEIGHBOR_ATOMIC_NUMS = {
    5,   # B
    6,   # C
    7,   # N
    8,   # O
    9,   # F
    14,  # Si
    15,  # P
    16,  # S
    17,  # Cl
    33,  # As
    34,  # Se
    35,  # Br
    52,  # Te
    53,  # I
}


def check_supported_smiles_graph_surface(
    mol: Chem.Mol,
    *,
    surface_kind: str = CONNECTED_NONSTEREO_SURFACE,
) -> None:
    if surface_kind not in {CONNECTED_NONSTEREO_SURFACE, CONNECTED_STEREO_SURFACE}:
        raise NotImplementedError(f"Unsupported prepared graph surface: {surface_kind}")

    if mol.GetNumAtoms() == 0:
        return
    if len(Chem.GetMolFrags(mol)) != 1:
        raise ValueError("Rooted enumeration currently requires a connected molecule")

    for atom in mol.GetAtoms():
        if atom.HasQuery():
            raise ValueError("Query atoms are not supported")
        if surface_kind == CONNECTED_NONSTEREO_SURFACE and atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED:
            raise ValueError("Atom stereochemistry is not supported yet")
        if surface_kind == CONNECTED_STEREO_SURFACE and atom.GetChiralTag() not in SUPPORTED_STEREO_CHIRAL_TAGS:
            raise ValueError(f"Unsupported atom chiral tag: {atom.GetChiralTag()}")

    for bond in mol.GetBonds():
        if bond.HasQuery():
            raise ValueError("Query bonds are not supported")
        if bond.GetBondType() not in SUPPORTED_BOND_TYPES:
            raise ValueError(f"Unsupported bond type: {bond.GetBondType()}")
        if surface_kind == CONNECTED_NONSTEREO_SURFACE and bond.GetStereo() != Chem.BondStereo.STEREONONE:
            raise ValueError("Bond stereochemistry is not supported yet")
        if surface_kind == CONNECTED_NONSTEREO_SURFACE and bond.GetBondDir() != Chem.BondDir.NONE:
            raise ValueError("Directional bond tokens are not supported yet")
        if surface_kind == CONNECTED_STEREO_SURFACE:
            if bond.GetStereo() not in SUPPORTED_STEREO_BOND_STEREO:
                raise ValueError(f"Unsupported bond stereo: {bond.GetStereo()}")
            if bond.GetBondDir() not in SUPPORTED_STEREO_BOND_DIRS:
                raise ValueError(f"Unsupported bond direction: {bond.GetBondDir()}")


def sampling_section(policy: ReferencePolicy) -> dict[str, object]:
    sampling = policy.data["sampling"]
    if not isinstance(sampling, dict):
        raise TypeError("sampling policy must be a JSON object")
    return sampling


def identity_section(policy: ReferencePolicy) -> dict[str, object]:
    identity = policy.data["identity_check"]
    if not isinstance(identity, dict):
        raise TypeError("identity_check policy must be a JSON object")
    return identity


def ring_label_text(label: int) -> str:
    if label < 10:
        return str(label)
    if label < 100:
        return f"%{label}"
    return f"%({label})"


def atom_symbol(atom: Chem.Atom, sampling: dict[str, object]) -> str:
    symbol = atom.GetSymbol()
    if atom.GetIsAromatic() and not bool(sampling["kekuleSmiles"]):
        lowered = symbol.lower()
        if lowered in AROMATIC_SUBSET:
            return lowered
    return symbol


def format_hydrogen_count(count: int) -> str:
    if count <= 0:
        return ""
    if count == 1:
        return "H"
    return f"H{count}"


def format_charge(charge: int) -> str:
    if charge == 0:
        return ""
    if charge == 1:
        return "+"
    if charge == -1:
        return "-"
    if charge > 0:
        return f"+{charge}"
    return f"-{abs(charge)}"


def atom_requires_brackets(atom: Chem.Atom, sampling: dict[str, object]) -> bool:
    if bool(sampling["allHsExplicit"]):
        return True

    atom_map_num = atom.GetAtomMapNum()
    if atom.GetIsotope() or atom_map_num or atom.GetFormalCharge() or atom.GetNumRadicalElectrons():
        return True

    symbol = atom_symbol(atom, sampling)
    if any(
        neighbor.GetAtomicNum() not in UNBRACKETED_NEIGHBOR_ATOMIC_NUMS
        for neighbor in atom.GetNeighbors()
    ):
        return True
    if symbol in ORGANIC_SUBSET:
        return False
    if symbol in AROMATIC_SUBSET:
        return atom.GetTotalNumHs() > 0 and symbol not in {"b", "c"}
    return True


def atom_token(atom: Chem.Atom, sampling: dict[str, object]) -> str:
    symbol = atom_symbol(atom, sampling)
    if not atom_requires_brackets(atom, sampling):
        return symbol

    atom_map_num = atom.GetAtomMapNum()
    parts = ["["]
    if atom.GetIsotope():
        parts.append(str(atom.GetIsotope()))
    parts.append(symbol)
    parts.append(format_hydrogen_count(atom.GetTotalNumHs()))
    parts.append(format_charge(atom.GetFormalCharge()))
    if atom_map_num:
        parts.append(f":{atom_map_num}")
    parts.append("]")
    return "".join(parts)


def bond_token(bond: Chem.Bond, sampling: dict[str, object]) -> str:
    begin = bond.GetBeginAtom()
    end = bond.GetEndAtom()
    bond_type = bond.GetBondType()
    all_bonds_explicit = bool(sampling["allBondsExplicit"])

    if bond_type == Chem.BondType.AROMATIC:
        if begin.GetIsAromatic() and end.GetIsAromatic():
            return ":" if all_bonds_explicit else ""
        return ":"
    if bond_type == Chem.BondType.SINGLE:
        return "-" if all_bonds_explicit else ""
    if bond_type == Chem.BondType.DOUBLE:
        return "="
    if bond_type == Chem.BondType.TRIPLE:
        return "#"
    raise ValueError(f"Unsupported bond type: {bond_type}")


def build_atom_tokens(mol: Chem.Mol, policy: ReferencePolicy) -> tuple[str, ...]:
    sampling = sampling_section(policy)
    return tuple(atom_token(mol.GetAtomWithIdx(atom_idx), sampling) for atom_idx in range(mol.GetNumAtoms()))


def _bond_kind_name(bond_type: Chem.BondType) -> str:
    if bond_type == Chem.BondType.SINGLE:
        return "SINGLE"
    if bond_type == Chem.BondType.DOUBLE:
        return "DOUBLE"
    if bond_type == Chem.BondType.TRIPLE:
        return "TRIPLE"
    if bond_type == Chem.BondType.AROMATIC:
        return "AROMATIC"
    return str(bond_type)


def _stereo_atom_pair(bond: Chem.Bond) -> tuple[int, int]:
    stereo_atoms = tuple(int(atom_idx) for atom_idx in bond.GetStereoAtoms())
    if not stereo_atoms:
        return (-1, -1)
    if len(stereo_atoms) != 2:
        raise ValueError(f"Unexpected stereo atom count for bond {bond.GetIdx()}: {stereo_atoms!r}")
    return stereo_atoms


def _jsonable(value: Any) -> Any:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    return value


def _tuple_of_int_pairs(value: Any, field_name: str) -> tuple[tuple[int, int], ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a JSON array")
    pairs: list[tuple[int, int]] = []
    for item in value:
        if not isinstance(item, list) or len(item) != 2:
            raise TypeError(f"{field_name} items must be 2-element JSON arrays")
        pairs.append((int(item[0]), int(item[1])))
    return tuple(pairs)


def _tuple_of_tuples(value: Any, *, inner_type: type, field_name: str) -> tuple[tuple[Any, ...], ...]:
    if not isinstance(value, list):
        raise TypeError(f"{field_name} must be a JSON array")
    outer: list[tuple[Any, ...]] = []
    for item in value:
        if not isinstance(item, list):
            raise TypeError(f"{field_name} items must be JSON arrays")
        outer.append(tuple(inner_type(element) for element in item))
    return tuple(outer)


@dataclass(frozen=True, slots=True)
class PreparedSmilesGraph:
    schema_version: int
    surface_kind: str
    policy_name: str
    policy_digest: str
    rdkit_version: str
    identity_smiles: str
    atom_count: int
    bond_count: int
    atom_atomic_numbers: tuple[int, ...]
    atom_is_aromatic: tuple[bool, ...]
    atom_isotopes: tuple[int, ...]
    atom_formal_charges: tuple[int, ...]
    atom_total_hs: tuple[int, ...]
    atom_radical_electrons: tuple[int, ...]
    atom_map_numbers: tuple[int, ...]
    atom_tokens: tuple[str, ...]
    neighbors: tuple[tuple[int, ...], ...]
    neighbor_bond_tokens: tuple[tuple[str, ...], ...]
    bond_pairs: tuple[tuple[int, int], ...]
    bond_kinds: tuple[str, ...]
    writer_do_isomeric_smiles: bool
    writer_kekule_smiles: bool
    writer_all_bonds_explicit: bool
    writer_all_hs_explicit: bool
    writer_ignore_atom_map_numbers: bool
    identity_parse_with_rdkit: bool
    identity_canonical: bool
    identity_do_isomeric_smiles: bool
    identity_kekule_smiles: bool
    identity_rooted_at_atom: int
    identity_all_bonds_explicit: bool
    identity_all_hs_explicit: bool
    identity_do_random: bool
    identity_ignore_atom_map_numbers: bool
    atom_chiral_tags: tuple[str, ...] = ()
    atom_stereo_neighbor_orders: tuple[tuple[int, ...], ...] = ()
    atom_explicit_h_counts: tuple[int, ...] = ()
    atom_implicit_h_counts: tuple[int, ...] = ()
    bond_stereo_kinds: tuple[str, ...] = ()
    bond_stereo_atoms: tuple[tuple[int, int], ...] = ()
    bond_dirs: tuple[str, ...] = ()
    bond_begin_atom_indices: tuple[int, ...] = ()
    bond_end_atom_indices: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.schema_version != PREPARED_SMILES_GRAPH_SCHEMA_VERSION:
            raise ValueError(
                "Unexpected PreparedSmilesGraph schema version: "
                f"{self.schema_version}"
            )
        if self.surface_kind not in {CONNECTED_NONSTEREO_SURFACE, CONNECTED_STEREO_SURFACE}:
            raise ValueError(f"Unsupported PreparedSmilesGraph surface: {self.surface_kind}")
        if self.atom_count < 0 or self.bond_count < 0:
            raise ValueError("atom_count and bond_count must be non-negative")
        self.validate()

    def validate(self) -> None:
        atom_fields = (
            self.atom_atomic_numbers,
            self.atom_is_aromatic,
            self.atom_isotopes,
            self.atom_formal_charges,
            self.atom_total_hs,
            self.atom_radical_electrons,
            self.atom_map_numbers,
            self.atom_tokens,
            self.neighbors,
            self.neighbor_bond_tokens,
        )
        for field in atom_fields:
            if len(field) != self.atom_count:
                raise ValueError("PreparedSmilesGraph atom field length mismatch")

        stereo_atom_fields = (
            self.atom_chiral_tags,
            self.atom_stereo_neighbor_orders,
            self.atom_explicit_h_counts,
            self.atom_implicit_h_counts,
        )
        for field in stereo_atom_fields:
            if field and len(field) != self.atom_count:
                raise ValueError("PreparedSmilesGraph stereo atom field length mismatch")

        if len(self.bond_pairs) != self.bond_count or len(self.bond_kinds) != self.bond_count:
            raise ValueError("PreparedSmilesGraph bond field length mismatch")
        stereo_bond_fields = (
            self.bond_stereo_kinds,
            self.bond_stereo_atoms,
            self.bond_dirs,
            self.bond_begin_atom_indices,
            self.bond_end_atom_indices,
        )
        for field in stereo_bond_fields:
            if field and len(field) != self.bond_count:
                raise ValueError("PreparedSmilesGraph stereo bond field length mismatch")

        bond_pairs_seen: set[tuple[int, int]] = set()
        neighbor_pairs_seen: set[tuple[int, int]] = set()

        for atom_idx, (neighbors, bond_tokens) in enumerate(
            zip(self.neighbors, self.neighbor_bond_tokens, strict=True)
        ):
            if len(neighbors) != len(bond_tokens):
                raise ValueError("PreparedSmilesGraph neighbor token row length mismatch")
            if tuple(sorted(neighbors)) != neighbors:
                raise ValueError("PreparedSmilesGraph neighbor rows must be sorted")
            if len(set(neighbors)) != len(neighbors):
                raise ValueError("PreparedSmilesGraph neighbor rows must be unique")
            for neighbor_idx in neighbors:
                if neighbor_idx < 0 or neighbor_idx >= self.atom_count:
                    raise ValueError("PreparedSmilesGraph neighbor index out of range")
                if neighbor_idx == atom_idx:
                    raise ValueError("PreparedSmilesGraph cannot contain self-loops")
                neighbor_pairs_seen.add(
                    (atom_idx, neighbor_idx)
                    if atom_idx < neighbor_idx
                    else (neighbor_idx, atom_idx)
                )

        for bond_pair in self.bond_pairs:
            begin_idx, end_idx = bond_pair
            if begin_idx < 0 or end_idx < 0 or begin_idx >= self.atom_count or end_idx >= self.atom_count:
                raise ValueError("PreparedSmilesGraph bond index out of range")
            if begin_idx >= end_idx:
                raise ValueError("PreparedSmilesGraph bond_pairs must be canonicalized")
            if bond_pair in bond_pairs_seen:
                raise ValueError("PreparedSmilesGraph bond_pairs must be unique")
            bond_pairs_seen.add(bond_pair)

            if end_idx not in self.neighbors[begin_idx] or begin_idx not in self.neighbors[end_idx]:
                raise ValueError("PreparedSmilesGraph bond_pairs must agree with neighbors")

        if bond_pairs_seen != neighbor_pairs_seen:
            raise ValueError("PreparedSmilesGraph neighbor graph and bond_pairs disagree")

        for begin_idx, neighbors in enumerate(self.neighbors):
            for offset, end_idx in enumerate(neighbors):
                if begin_idx not in self.neighbors[end_idx]:
                    raise ValueError("PreparedSmilesGraph neighbors must be symmetric")
                reverse_offset = self.neighbors[end_idx].index(begin_idx)
                if self.neighbor_bond_tokens[begin_idx][offset] != self.neighbor_bond_tokens[end_idx][reverse_offset]:
                    raise ValueError("PreparedSmilesGraph bond tokens must be symmetric")

        if self.atom_stereo_neighbor_orders:
            for atom_idx, stereo_neighbor_order in enumerate(self.atom_stereo_neighbor_orders):
                if len(stereo_neighbor_order) != len(self.neighbors[atom_idx]):
                    raise ValueError("PreparedSmilesGraph stereo neighbor order length mismatch")
                if set(stereo_neighbor_order) != set(self.neighbors[atom_idx]):
                    raise ValueError("PreparedSmilesGraph stereo neighbor order must match neighbors")

        if self.bond_stereo_atoms:
            for begin_idx, end_idx in self.bond_stereo_atoms:
                if (begin_idx, end_idx) == (-1, -1):
                    continue
                if begin_idx < 0 or end_idx < 0 or begin_idx >= self.atom_count or end_idx >= self.atom_count:
                    raise ValueError("PreparedSmilesGraph stereo atom index out of range")

    def neighbors_of(self, atom_idx: int) -> tuple[int, ...]:
        return self.neighbors[atom_idx]

    def bond_token(self, begin_idx: int, end_idx: int) -> str:
        for offset, neighbor_idx in enumerate(self.neighbors[begin_idx]):
            if neighbor_idx == end_idx:
                return self.neighbor_bond_tokens[begin_idx][offset]
        raise KeyError(f"No bond between atoms {begin_idx} and {end_idx}")

    def bond_index(self, begin_idx: int, end_idx: int) -> int:
        low_idx = begin_idx if begin_idx < end_idx else end_idx
        high_idx = end_idx if begin_idx < end_idx else begin_idx
        for bond_idx, bond_pair in enumerate(self.bond_pairs):
            if bond_pair == (low_idx, high_idx):
                return bond_idx
        raise KeyError(f"No bond between atoms {begin_idx} and {end_idx}")

    def directed_bond_token(self, begin_idx: int, end_idx: int) -> str:
        bond_idx = self.bond_index(begin_idx, end_idx)
        base_token = self.bond_token(begin_idx, end_idx)
        if not self.bond_dirs:
            return base_token

        bond_dir = self.bond_dirs[bond_idx]
        if bond_dir == "NONE":
            return base_token
        if bond_dir not in {"ENDUPRIGHT", "ENDDOWNRIGHT"}:
            raise NotImplementedError(f"Unsupported directional bond token: {bond_dir}")
        if not self.bond_begin_atom_indices or not self.bond_end_atom_indices:
            raise ValueError("PreparedSmilesGraph is missing bond begin/end orientation metadata")

        stored_begin_idx = self.bond_begin_atom_indices[bond_idx]
        stored_end_idx = self.bond_end_atom_indices[bond_idx]
        if (begin_idx, end_idx) == (stored_begin_idx, stored_end_idx):
            reverse = False
        elif (begin_idx, end_idx) == (stored_end_idx, stored_begin_idx):
            reverse = True
        else:
            raise KeyError(f"No directed bond between atoms {begin_idx} and {end_idx}")

        if bond_dir == "ENDUPRIGHT":
            token = "/"
        else:
            token = "\\"
        if reverse:
            return "\\" if token == "/" else "/"
        return token

    def validate_policy(self, policy: ReferencePolicy) -> None:
        if self.policy_name != policy.policy_name or self.policy_digest != policy.digest():
            raise ValueError("PreparedSmilesGraph does not match the provided policy")

    def identity_smiles_for(self, mol: Chem.Mol) -> str:
        if not self.identity_parse_with_rdkit:
            raise NotImplementedError("Only parse_with_rdkit=true identity checks are supported")
        return Chem.MolToSmiles(
            Chem.Mol(mol),
            canonical=self.identity_canonical,
            isomericSmiles=self.identity_do_isomeric_smiles,
            kekuleSmiles=self.identity_kekule_smiles,
            rootedAtAtom=self.identity_rooted_at_atom,
            allBondsExplicit=self.identity_all_bonds_explicit,
            allHsExplicit=self.identity_all_hs_explicit,
            doRandom=self.identity_do_random,
            ignoreAtomMapNumbers=self.identity_ignore_atom_map_numbers,
        )

    def to_dict(self) -> dict[str, Any]:
        data = {
            "schema_version": self.schema_version,
            "surface_kind": self.surface_kind,
            "policy_name": self.policy_name,
            "policy_digest": self.policy_digest,
            "rdkit_version": self.rdkit_version,
            "identity_smiles": self.identity_smiles,
            "atom_count": self.atom_count,
            "bond_count": self.bond_count,
            "atom_atomic_numbers": _jsonable(self.atom_atomic_numbers),
            "atom_is_aromatic": _jsonable(self.atom_is_aromatic),
            "atom_isotopes": _jsonable(self.atom_isotopes),
            "atom_formal_charges": _jsonable(self.atom_formal_charges),
            "atom_total_hs": _jsonable(self.atom_total_hs),
            "atom_radical_electrons": _jsonable(self.atom_radical_electrons),
            "atom_map_numbers": _jsonable(self.atom_map_numbers),
            "atom_tokens": _jsonable(self.atom_tokens),
            "neighbors": _jsonable(self.neighbors),
            "neighbor_bond_tokens": _jsonable(self.neighbor_bond_tokens),
            "bond_pairs": _jsonable(self.bond_pairs),
            "bond_kinds": _jsonable(self.bond_kinds),
            "writer_do_isomeric_smiles": self.writer_do_isomeric_smiles,
            "writer_kekule_smiles": self.writer_kekule_smiles,
            "writer_all_bonds_explicit": self.writer_all_bonds_explicit,
            "writer_all_hs_explicit": self.writer_all_hs_explicit,
            "writer_ignore_atom_map_numbers": self.writer_ignore_atom_map_numbers,
            "identity_parse_with_rdkit": self.identity_parse_with_rdkit,
            "identity_canonical": self.identity_canonical,
            "identity_do_isomeric_smiles": self.identity_do_isomeric_smiles,
            "identity_kekule_smiles": self.identity_kekule_smiles,
            "identity_rooted_at_atom": self.identity_rooted_at_atom,
            "identity_all_bonds_explicit": self.identity_all_bonds_explicit,
            "identity_all_hs_explicit": self.identity_all_hs_explicit,
            "identity_do_random": self.identity_do_random,
            "identity_ignore_atom_map_numbers": self.identity_ignore_atom_map_numbers,
        }
        if self.atom_chiral_tags:
            data["atom_chiral_tags"] = _jsonable(self.atom_chiral_tags)
            data["atom_stereo_neighbor_orders"] = _jsonable(self.atom_stereo_neighbor_orders)
            data["atom_explicit_h_counts"] = _jsonable(self.atom_explicit_h_counts)
            data["atom_implicit_h_counts"] = _jsonable(self.atom_implicit_h_counts)
        if self.bond_stereo_kinds:
            data["bond_stereo_kinds"] = _jsonable(self.bond_stereo_kinds)
            data["bond_stereo_atoms"] = _jsonable(self.bond_stereo_atoms)
            data["bond_dirs"] = _jsonable(self.bond_dirs)
            data["bond_begin_atom_indices"] = _jsonable(self.bond_begin_atom_indices)
            data["bond_end_atom_indices"] = _jsonable(self.bond_end_atom_indices)
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreparedSmilesGraph":
        if not isinstance(data, dict):
            raise TypeError("PreparedSmilesGraph data must be a JSON object")
        return cls(
            schema_version=int(data["schema_version"]),
            surface_kind=str(data["surface_kind"]),
            policy_name=str(data["policy_name"]),
            policy_digest=str(data["policy_digest"]),
            rdkit_version=str(data["rdkit_version"]),
            identity_smiles=str(data["identity_smiles"]),
            atom_count=int(data["atom_count"]),
            bond_count=int(data["bond_count"]),
            atom_atomic_numbers=tuple(int(value) for value in data["atom_atomic_numbers"]),
            atom_is_aromatic=tuple(bool(value) for value in data["atom_is_aromatic"]),
            atom_isotopes=tuple(int(value) for value in data["atom_isotopes"]),
            atom_formal_charges=tuple(int(value) for value in data["atom_formal_charges"]),
            atom_total_hs=tuple(int(value) for value in data["atom_total_hs"]),
            atom_radical_electrons=tuple(int(value) for value in data["atom_radical_electrons"]),
            atom_map_numbers=tuple(int(value) for value in data["atom_map_numbers"]),
            atom_tokens=tuple(str(value) for value in data["atom_tokens"]),
            neighbors=_tuple_of_tuples(data["neighbors"], inner_type=int, field_name="neighbors"),
            neighbor_bond_tokens=_tuple_of_tuples(
                data["neighbor_bond_tokens"],
                inner_type=str,
                field_name="neighbor_bond_tokens",
            ),
            bond_pairs=_tuple_of_int_pairs(data["bond_pairs"], "bond_pairs"),
            bond_kinds=tuple(str(value) for value in data["bond_kinds"]),
            writer_do_isomeric_smiles=bool(data["writer_do_isomeric_smiles"]),
            writer_kekule_smiles=bool(data["writer_kekule_smiles"]),
            writer_all_bonds_explicit=bool(data["writer_all_bonds_explicit"]),
            writer_all_hs_explicit=bool(data["writer_all_hs_explicit"]),
            writer_ignore_atom_map_numbers=bool(data["writer_ignore_atom_map_numbers"]),
            identity_parse_with_rdkit=bool(data["identity_parse_with_rdkit"]),
            identity_canonical=bool(data["identity_canonical"]),
            identity_do_isomeric_smiles=bool(data["identity_do_isomeric_smiles"]),
            identity_kekule_smiles=bool(data["identity_kekule_smiles"]),
            identity_rooted_at_atom=int(data["identity_rooted_at_atom"]),
            identity_all_bonds_explicit=bool(data["identity_all_bonds_explicit"]),
            identity_all_hs_explicit=bool(data["identity_all_hs_explicit"]),
            identity_do_random=bool(data["identity_do_random"]),
            identity_ignore_atom_map_numbers=bool(data["identity_ignore_atom_map_numbers"]),
            atom_chiral_tags=tuple(str(value) for value in data.get("atom_chiral_tags", [])),
            atom_stereo_neighbor_orders=_tuple_of_tuples(
                data.get("atom_stereo_neighbor_orders", []),
                inner_type=int,
                field_name="atom_stereo_neighbor_orders",
            ),
            atom_explicit_h_counts=tuple(int(value) for value in data.get("atom_explicit_h_counts", [])),
            atom_implicit_h_counts=tuple(int(value) for value in data.get("atom_implicit_h_counts", [])),
            bond_stereo_kinds=tuple(str(value) for value in data.get("bond_stereo_kinds", [])),
            bond_stereo_atoms=_tuple_of_int_pairs(data.get("bond_stereo_atoms", []), "bond_stereo_atoms"),
            bond_dirs=tuple(str(value) for value in data.get("bond_dirs", [])),
            bond_begin_atom_indices=tuple(int(value) for value in data.get("bond_begin_atom_indices", [])),
            bond_end_atom_indices=tuple(int(value) for value in data.get("bond_end_atom_indices", [])),
        )


def prepare_smiles_graph(
    mol: Chem.Mol,
    policy: ReferencePolicy,
    *,
    surface_kind: str = CONNECTED_NONSTEREO_SURFACE,
) -> PreparedSmilesGraph:
    working_mol = Chem.Mol(mol)
    check_supported_smiles_graph_surface(working_mol, surface_kind=surface_kind)
    sampling = sampling_section(policy)
    identity = identity_section(policy)
    return _prepare_smiles_graph_with_sections(
        working_mol,
        surface_kind=surface_kind,
        policy_name=policy.policy_name,
        policy_digest=policy.digest(),
        sampling=sampling,
        identity=identity,
    )


def _digest_runtime_descriptor(descriptor: dict[str, object]) -> str:
    payload = json.dumps(descriptor, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]


def prepare_smiles_graph_from_mol_to_smiles_kwargs(
    mol: Chem.Mol,
    *,
    surface_kind: str = CONNECTED_NONSTEREO_SURFACE,
    isomeric_smiles: bool = True,
    kekule_smiles: bool = False,
    all_bonds_explicit: bool = False,
    all_hs_explicit: bool = False,
    ignore_atom_map_numbers: bool = False,
) -> PreparedSmilesGraph:
    working_mol = Chem.Mol(mol)
    check_supported_smiles_graph_surface(working_mol, surface_kind=surface_kind)

    sampling = {
        "isomericSmiles": bool(isomeric_smiles),
        "kekuleSmiles": bool(kekule_smiles),
        "allBondsExplicit": bool(all_bonds_explicit),
        "allHsExplicit": bool(all_hs_explicit),
        "ignoreAtomMapNumbers": bool(ignore_atom_map_numbers),
    }
    identity = {
        "parse_with_rdkit": True,
        "canonical": True,
        "isomericSmiles": bool(isomeric_smiles),
        "kekuleSmiles": bool(kekule_smiles),
        "rootedAtAtom": -1,
        "allBondsExplicit": bool(all_bonds_explicit),
        "allHsExplicit": bool(all_hs_explicit),
        "doRandom": False,
        "ignoreAtomMapNumbers": bool(ignore_atom_map_numbers),
    }
    descriptor = {
        "surface_kind": surface_kind,
        "sampling": sampling,
        "identity_check": identity,
    }
    return _prepare_smiles_graph_with_sections(
        working_mol,
        surface_kind=surface_kind,
        policy_name="runtime_mol_to_smiles_support_v1",
        policy_digest=_digest_runtime_descriptor(descriptor),
        sampling=sampling,
        identity=identity,
    )


def _prepare_smiles_graph_with_sections(
    working_mol: Chem.Mol,
    *,
    surface_kind: str,
    policy_name: str,
    policy_digest: str,
    sampling: dict[str, object],
    identity: dict[str, object],
) -> PreparedSmilesGraph:
    if bool(sampling["kekuleSmiles"]):
        Chem.Kekulize(working_mol, clearAromaticFlags=True)

    neighbors: list[tuple[int, ...]] = []
    neighbor_bond_tokens: list[tuple[str, ...]] = []
    atom_atomic_numbers: list[int] = []
    atom_is_aromatic: list[bool] = []
    atom_isotopes: list[int] = []
    atom_formal_charges: list[int] = []
    atom_total_hs: list[int] = []
    atom_radical_electrons: list[int] = []
    atom_map_numbers: list[int] = []

    for atom in working_mol.GetAtoms():
        atom_atomic_numbers.append(atom.GetAtomicNum())
        atom_is_aromatic.append(atom.GetIsAromatic())
        atom_isotopes.append(atom.GetIsotope())
        atom_formal_charges.append(atom.GetFormalCharge())
        atom_total_hs.append(atom.GetTotalNumHs())
        atom_radical_electrons.append(atom.GetNumRadicalElectrons())
        atom_map_numbers.append(atom.GetAtomMapNum())

        atom_neighbors = tuple(sorted(neighbor.GetIdx() for neighbor in atom.GetNeighbors()))
        neighbors.append(atom_neighbors)
        neighbor_bond_tokens.append(
            tuple(
                bond_token(working_mol.GetBondBetweenAtoms(atom.GetIdx(), neighbor_idx), sampling)
                for neighbor_idx in atom_neighbors
            )
        )

    bond_pairs = tuple(
        sorted(
            tuple(sorted((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())))
            for bond in working_mol.GetBonds()
        )
    )
    bond_kinds = tuple(
        _bond_kind_name(working_mol.GetBondBetweenAtoms(begin_idx, end_idx).GetBondType())
        for begin_idx, end_idx in bond_pairs
    )
    atom_chiral_tags: tuple[str, ...] = ()
    atom_stereo_neighbor_orders: tuple[tuple[int, ...], ...] = ()
    atom_explicit_h_counts: tuple[int, ...] = ()
    atom_implicit_h_counts: tuple[int, ...] = ()
    bond_stereo_kinds: tuple[str, ...] = ()
    bond_stereo_atoms: tuple[tuple[int, int], ...] = ()
    bond_dirs: tuple[str, ...] = ()
    bond_begin_atom_indices: tuple[int, ...] = ()
    bond_end_atom_indices: tuple[int, ...] = ()

    if surface_kind == CONNECTED_STEREO_SURFACE:
        atom_chiral_tags = tuple(str(atom.GetChiralTag()) for atom in working_mol.GetAtoms())
        atom_stereo_neighbor_orders = tuple(
            tuple(bond.GetOtherAtomIdx(atom.GetIdx()) for bond in atom.GetBonds())
            for atom in working_mol.GetAtoms()
        )
        atom_explicit_h_counts = tuple(atom.GetNumExplicitHs() for atom in working_mol.GetAtoms())
        atom_implicit_h_counts = tuple(atom.GetNumImplicitHs() for atom in working_mol.GetAtoms())
        bond_stereo_kinds = tuple(
            str(working_mol.GetBondBetweenAtoms(begin_idx, end_idx).GetStereo())
            for begin_idx, end_idx in bond_pairs
        )
        bond_stereo_atoms = tuple(
            _stereo_atom_pair(working_mol.GetBondBetweenAtoms(begin_idx, end_idx))
            for begin_idx, end_idx in bond_pairs
        )
        bond_dirs = tuple(
            str(working_mol.GetBondBetweenAtoms(begin_idx, end_idx).GetBondDir())
            for begin_idx, end_idx in bond_pairs
        )
        bond_begin_atom_indices = tuple(
            working_mol.GetBondBetweenAtoms(begin_idx, end_idx).GetBeginAtomIdx()
            for begin_idx, end_idx in bond_pairs
        )
        bond_end_atom_indices = tuple(
            working_mol.GetBondBetweenAtoms(begin_idx, end_idx).GetEndAtomIdx()
            for begin_idx, end_idx in bond_pairs
        )

    return PreparedSmilesGraph(
        schema_version=PREPARED_SMILES_GRAPH_SCHEMA_VERSION,
        surface_kind=surface_kind,
        policy_name=policy_name,
        policy_digest=policy_digest,
        rdkit_version=rdBase.rdkitVersion,
        identity_smiles=Chem.MolToSmiles(
            Chem.Mol(working_mol),
            canonical=bool(identity["canonical"]),
            isomericSmiles=bool(identity["isomericSmiles"]),
            kekuleSmiles=bool(identity["kekuleSmiles"]),
            rootedAtAtom=int(identity["rootedAtAtom"]),
            allBondsExplicit=bool(identity["allBondsExplicit"]),
            allHsExplicit=bool(identity["allHsExplicit"]),
            doRandom=bool(identity["doRandom"]),
            ignoreAtomMapNumbers=bool(identity["ignoreAtomMapNumbers"]),
        ),
        atom_count=working_mol.GetNumAtoms(),
        bond_count=working_mol.GetNumBonds(),
        atom_atomic_numbers=tuple(atom_atomic_numbers),
        atom_is_aromatic=tuple(atom_is_aromatic),
        atom_isotopes=tuple(atom_isotopes),
        atom_formal_charges=tuple(atom_formal_charges),
        atom_total_hs=tuple(atom_total_hs),
        atom_radical_electrons=tuple(atom_radical_electrons),
        atom_map_numbers=tuple(atom_map_numbers),
        atom_tokens=tuple(atom_token(working_mol.GetAtomWithIdx(atom_idx), sampling) for atom_idx in range(working_mol.GetNumAtoms())),
        neighbors=tuple(neighbors),
        neighbor_bond_tokens=tuple(neighbor_bond_tokens),
        bond_pairs=bond_pairs,
        bond_kinds=bond_kinds,
        writer_do_isomeric_smiles=bool(sampling["isomericSmiles"]),
        writer_kekule_smiles=bool(sampling["kekuleSmiles"]),
        writer_all_bonds_explicit=bool(sampling["allBondsExplicit"]),
        writer_all_hs_explicit=bool(sampling["allHsExplicit"]),
        writer_ignore_atom_map_numbers=bool(sampling["ignoreAtomMapNumbers"]),
        identity_parse_with_rdkit=bool(identity["parse_with_rdkit"]),
        identity_canonical=bool(identity["canonical"]),
        identity_do_isomeric_smiles=bool(identity["isomericSmiles"]),
        identity_kekule_smiles=bool(identity["kekuleSmiles"]),
        identity_rooted_at_atom=int(identity["rootedAtAtom"]),
        identity_all_bonds_explicit=bool(identity["allBondsExplicit"]),
        identity_all_hs_explicit=bool(identity["allHsExplicit"]),
        identity_do_random=bool(identity["doRandom"]),
        identity_ignore_atom_map_numbers=bool(identity["ignoreAtomMapNumbers"]),
        atom_chiral_tags=atom_chiral_tags,
        atom_stereo_neighbor_orders=atom_stereo_neighbor_orders,
        atom_explicit_h_counts=atom_explicit_h_counts,
        atom_implicit_h_counts=atom_implicit_h_counts,
        bond_stereo_kinds=bond_stereo_kinds,
        bond_stereo_atoms=bond_stereo_atoms,
        bond_dirs=bond_dirs,
        bond_begin_atom_indices=bond_begin_atom_indices,
        bond_end_atom_indices=bond_end_atom_indices,
    )
