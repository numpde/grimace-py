from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem


SUPPORTED_BOND_TYPES: frozenset[Chem.BondType] = frozenset(
    {
        Chem.BondType.SINGLE,
        Chem.BondType.DOUBLE,
        Chem.BondType.AROMATIC,
    }
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
    unsupported.extend(_atom_stereo_features(mol))
    unsupported.extend(_metal_features(mol))
    unsupported.extend(_bond_type_features(mol))
    unsupported.extend(_dative_bond_features(mol))
    unsupported.extend(_disconnected_features(mol))
    unsupported.extend(_ring_stereo_features(mol))
    unsupported.extend(_aromatic_directional_features(mol))
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
