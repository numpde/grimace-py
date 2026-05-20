from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


SUPPORTED_ATOM_SYMBOLS: frozenset[str] = frozenset(
    {"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"}
)
SUPPORTED_BOND_TYPES: frozenset[Chem.BondType] = frozenset(
    {Chem.BondType.SINGLE, Chem.BondType.DOUBLE}
)
METAL_ATOMIC_NUMBERS: frozenset[int] = frozenset(
    {
        3,
        4,
        11,
        12,
        13,
        19,
        20,
        21,
        22,
        23,
        24,
        25,
        26,
        27,
        28,
        29,
        30,
        31,
        37,
        38,
        39,
        40,
        41,
        42,
        43,
        44,
        45,
        46,
        47,
        48,
        49,
        50,
        55,
        56,
        57,
        72,
        73,
        74,
        75,
        76,
        77,
        78,
        79,
        80,
        81,
        82,
        83,
        87,
        88,
        89,
    }
)


@dataclass(frozen=True, slots=True)
class SouthStarUnsupportedFeature:
    category: str
    atom_indices: tuple[int, ...]
    bond_indices: tuple[int, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class SouthStarSupportGateReport:
    unsupported_features: tuple[SouthStarUnsupportedFeature, ...]

    @property
    def supported(self) -> bool:
        return not self.unsupported_features

    @property
    def categories(self) -> frozenset[str]:
        return frozenset(feature.category for feature in self.unsupported_features)

    def fail_if_unsupported(self) -> None:
        if self.supported:
            return
        categories = ", ".join(sorted(self.categories))
        raise NotImplementedError(
            f"MolToSmilesEnumS unsupported South Star features: {categories}"
        )


def south_star_support_gate_report(mol: Chem.Mol) -> SouthStarSupportGateReport:
    unsupported: list[SouthStarUnsupportedFeature] = []
    unsupported.extend(_query_features(mol))
    unsupported.extend(_empty_molecule_features(mol))
    unsupported.extend(_unsupported_atom_text_features(mol))
    unsupported.extend(_atom_stereo_features(mol))
    unsupported.extend(_metal_features(mol))
    unsupported.extend(_bond_type_features(mol))
    unsupported.extend(_dative_bond_features(mol))
    unsupported.extend(_disconnected_features(mol))
    unsupported.extend(_ring_features(mol))
    unsupported.extend(_ring_stereo_features(mol))
    unsupported.extend(_aromatic_directional_features(mol))
    unsupported.extend(_unstated_component_equation_features(mol))
    return SouthStarSupportGateReport(unsupported_features=tuple(unsupported))


def _query_features(mol: Chem.Mol) -> tuple[SouthStarUnsupportedFeature, ...]:
    features: list[SouthStarUnsupportedFeature] = []
    for atom in mol.GetAtoms():
        if atom.HasQuery():
            features.append(
                SouthStarUnsupportedFeature(
                    category="query_atom",
                    atom_indices=(atom.GetIdx(),),
                    bond_indices=(),
                    reason="query atoms do not have a fixed semantic component model",
                )
            )
    for bond in mol.GetBonds():
        if bond.HasQuery():
            features.append(
                SouthStarUnsupportedFeature(
                    category="query_bond",
                    atom_indices=(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                    bond_indices=(bond.GetIdx(),),
                    reason="query bonds do not have a fixed semantic component model",
                )
            )
    return tuple(features)


def _empty_molecule_features(mol: Chem.Mol) -> tuple[SouthStarUnsupportedFeature, ...]:
    if mol.GetNumAtoms() != 0:
        return ()
    return (
        SouthStarUnsupportedFeature(
            category="empty_molecule",
            atom_indices=(),
            bond_indices=(),
            reason="South Star first scope requires at least one atom",
        ),
    )


def _unsupported_atom_text_features(
    mol: Chem.Mol,
) -> tuple[SouthStarUnsupportedFeature, ...]:
    return tuple(
        SouthStarUnsupportedFeature(
            category="unsupported_atom_text",
            atom_indices=(atom.GetIdx(),),
            bond_indices=(),
            reason=f"atom symbol {atom.GetSymbol()!r} is outside first South Star scope",
        )
        for atom in mol.GetAtoms()
        if atom.GetSymbol() not in SUPPORTED_ATOM_SYMBOLS
    )


def _atom_stereo_features(mol: Chem.Mol) -> tuple[SouthStarUnsupportedFeature, ...]:
    return tuple(
        SouthStarUnsupportedFeature(
            category="atom_stereo",
            atom_indices=(atom.GetIdx(),),
            bond_indices=(),
            reason="South Star first scope does not model atom stereo components",
        )
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
    )


def _metal_features(mol: Chem.Mol) -> tuple[SouthStarUnsupportedFeature, ...]:
    return tuple(
        SouthStarUnsupportedFeature(
            category="metal_atom",
            atom_indices=(atom.GetIdx(),),
            bond_indices=(),
            reason="metal-containing semantic stereo surfaces are not modeled yet",
        )
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS
    )


def _bond_type_features(mol: Chem.Mol) -> tuple[SouthStarUnsupportedFeature, ...]:
    return tuple(
        SouthStarUnsupportedFeature(
            category="unsupported_bond_type",
            atom_indices=(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            bond_indices=(bond.GetIdx(),),
            reason=f"bond type {bond.GetBondType()} is outside first South Star scope",
        )
        for bond in mol.GetBonds()
        if bond.GetBondType() not in SUPPORTED_BOND_TYPES
    )


def _dative_bond_features(mol: Chem.Mol) -> tuple[SouthStarUnsupportedFeature, ...]:
    return tuple(
        SouthStarUnsupportedFeature(
            category="dative_bond",
            atom_indices=(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            bond_indices=(bond.GetIdx(),),
            reason="dative-bond serializer quirks need separate semantic modeling",
        )
        for bond in mol.GetBonds()
        if bond.GetBondType() in (Chem.BondType.DATIVE, Chem.BondType.DATIVEL)
    )


def _disconnected_features(mol: Chem.Mol) -> tuple[SouthStarUnsupportedFeature, ...]:
    fragments = Chem.GetMolFrags(mol)
    if len(fragments) <= 1:
        return ()
    return (
        SouthStarUnsupportedFeature(
            category="disconnected_molecule",
            atom_indices=tuple(atom_idx for fragment in fragments for atom_idx in fragment),
            bond_indices=(),
            reason="disconnected traversal interactions are not modeled yet",
        ),
    )


def _ring_features(mol: Chem.Mol) -> tuple[SouthStarUnsupportedFeature, ...]:
    ring_atom_indices = tuple(
        atom.GetIdx() for atom in mol.GetAtoms() if atom.IsInRing()
    )
    ring_bond_indices = tuple(
        bond.GetIdx() for bond in mol.GetBonds() if bond.IsInRing()
    )
    if not ring_atom_indices and not ring_bond_indices:
        return ()
    return (
        SouthStarUnsupportedFeature(
            category="ring_molecule",
            atom_indices=ring_atom_indices,
            bond_indices=ring_bond_indices,
            reason="ring traversal and ring-closure marker bases are not modeled yet",
        ),
    )


def _ring_stereo_features(mol: Chem.Mol) -> tuple[SouthStarUnsupportedFeature, ...]:
    features: list[SouthStarUnsupportedFeature] = []
    for bond in mol.GetBonds():
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue
        if bond.IsInRing():
            features.append(
                SouthStarUnsupportedFeature(
                    category="ring_stereo",
                    atom_indices=(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                    bond_indices=(bond.GetIdx(),),
                    reason="ring-closure carrier basis is not modeled yet",
                )
            )
    return tuple(features)


def _aromatic_directional_features(
    mol: Chem.Mol,
) -> tuple[SouthStarUnsupportedFeature, ...]:
    return tuple(
        SouthStarUnsupportedFeature(
            category="aromatic_directional_surface",
            atom_indices=(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            bond_indices=(bond.GetIdx(),),
            reason="aromatic directional surfaces are not in first South Star scope",
        )
        for bond in mol.GetBonds()
        if bond.GetIsAromatic() and bond.GetBondDir() != Chem.BondDir.NONE
    )


def _unstated_component_equation_features(
    mol: Chem.Mol,
) -> tuple[SouthStarUnsupportedFeature, ...]:
    features: list[SouthStarUnsupportedFeature] = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue

        carrier_bonds = _directional_carrier_bonds_for_stereo_bond(mol, bond)
        if not carrier_bonds:
            features.append(
                SouthStarUnsupportedFeature(
                    category="unstated_component_equation",
                    atom_indices=(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
                    bond_indices=(bond.GetIdx(),),
                    reason=(
                        "stereo double bond has no directional carrier basis "
                        "for first-scope marker equations"
                    ),
                )
            )
            continue

        for carrier_bond in carrier_bonds:
            if carrier_bond.GetBondDir() == Chem.BondDir.NONE:
                features.append(
                    SouthStarUnsupportedFeature(
                        category="unstated_component_equation",
                        atom_indices=(
                            carrier_bond.GetBeginAtomIdx(),
                            carrier_bond.GetEndAtomIdx(),
                        ),
                        bond_indices=(carrier_bond.GetIdx(), bond.GetIdx()),
                        reason=(
                            "stereo double-bond carrier has no slash/backslash "
                            "basis for first-scope marker equations"
                        ),
                    )
                )
    return tuple(features)


def _directional_carrier_bonds_for_stereo_bond(
    mol: Chem.Mol,
    stereo_bond: Chem.Bond,
) -> tuple[Chem.Bond, ...]:
    carrier_bond_indices = []
    endpoints = (
        (stereo_bond.GetBeginAtomIdx(), stereo_bond.GetEndAtomIdx()),
        (stereo_bond.GetEndAtomIdx(), stereo_bond.GetBeginAtomIdx()),
    )
    for endpoint_atom_idx, excluded_atom_idx in endpoints:
        endpoint = mol.GetAtomWithIdx(endpoint_atom_idx)
        for neighbor in endpoint.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()
            if neighbor_idx == excluded_atom_idx:
                continue
            carrier_bond = mol.GetBondBetweenAtoms(endpoint_atom_idx, neighbor_idx)
            if carrier_bond is None:
                continue
            if carrier_bond.GetBondType() == Chem.BondType.SINGLE:
                carrier_bond_indices.append(carrier_bond.GetIdx())
    return tuple(
        mol.GetBondWithIdx(bond_idx)
        for bond_idx in dict.fromkeys(carrier_bond_indices)
    )
