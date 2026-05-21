from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.aromatic_policy import SOUTH_STAR_AROMATIC_TEXT_POLICY_CONTRACT
from grimace._south_star.atom_text import (
    SOUTH_STAR_AROMATIC_ATOM_TEXT_TOKENS,
    south_star_atom_text_fields,
    unsupported_atom_text_reasons,
)
from grimace._south_star.bond_text import SOUTH_STAR_SUPPORTED_BOND_TYPES
from grimace._south_star.tetrahedral import (
    extract_ring_tetrahedral_interaction_obligations,
    tetrahedral_atom_supported,
)


SUPPORTED_BOND_TYPES: frozenset[Chem.BondType] = SOUTH_STAR_SUPPORTED_BOND_TYPES
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
        raise SouthStarUnsupportedFeatureError(self.unsupported_features)


class SouthStarUnsupportedFeatureError(NotImplementedError):
    """Fail-fast boundary error that preserves support-gate evidence."""

    def __init__(
        self,
        unsupported_features: tuple[SouthStarUnsupportedFeature, ...],
    ) -> None:
        if not unsupported_features:
            raise ValueError("unsupported-feature error requires feature evidence")
        self.unsupported_features = unsupported_features
        categories = ", ".join(sorted(self.categories))
        super().__init__(
            f"MolToSmilesEnumS unsupported South Star features: {categories}"
        )

    @property
    def categories(self) -> frozenset[str]:
        return frozenset(feature.category for feature in self.unsupported_features)


def south_star_support_gate_report(mol: Chem.Mol) -> SouthStarSupportGateReport:
    unsupported: list[SouthStarUnsupportedFeature] = []
    unsupported.extend(_query_features(mol))
    unsupported.extend(_empty_molecule_features(mol))
    unsupported.extend(_unsupported_atom_text_policy_features(mol))
    unsupported.extend(_atom_stereo_features(mol))
    unsupported.extend(_metal_features(mol))
    unsupported.extend(_bond_type_features(mol))
    unsupported.extend(_dative_bond_features(mol))
    unsupported.extend(_disconnected_features(mol))
    unsupported.extend(_ring_features(mol))
    unsupported.extend(_polycyclic_ring_features(mol))
    unsupported.extend(_ring_tetrahedral_interaction_features(mol))
    unsupported.extend(_ring_stereo_features(mol))
    unsupported.extend(_aromatic_ring_features(mol))
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
            reason="South Star molecule facts require at least one atom",
        ),
    )


def _unsupported_atom_text_policy_features(
    mol: Chem.Mol,
) -> tuple[SouthStarUnsupportedFeature, ...]:
    features: list[SouthStarUnsupportedFeature] = []
    for atom in mol.GetAtoms():
        fields = south_star_atom_text_fields(atom)
        for reason in unsupported_atom_text_reasons(fields):
            features.append(
                SouthStarUnsupportedFeature(
                    category=reason.category,
                    atom_indices=(fields.atom_idx,),
                    bond_indices=(),
                    reason=reason.reason,
                )
            )
    return tuple(features)


def _atom_stereo_features(mol: Chem.Mol) -> tuple[SouthStarUnsupportedFeature, ...]:
    return tuple(
        SouthStarUnsupportedFeature(
            category="atom_stereo",
            atom_indices=(atom.GetIdx(),),
            bond_indices=(),
            reason=(
                "only tetrahedral centers with exactly four ligands are modeled "
                "in the current South Star atom-stereo scope"
            ),
        )
        for atom in mol.GetAtoms()
        if atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        and not tetrahedral_atom_supported(atom)
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
            reason=(
                f"bond type {bond.GetBondType()} is outside the current "
                "South Star bond-text scope"
            ),
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
    if disconnected_fragments_have_supported_independent_traversal(mol):
        return ()
    return (
        SouthStarUnsupportedFeature(
            category="disconnected_molecule",
            atom_indices=tuple(atom_idx for fragment in fragments for atom_idx in fragment),
            bond_indices=(),
            reason=(
                "this disconnected fragment combination is outside the current "
                "South Star composition scope"
            ),
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
    if disconnected_fragments_have_supported_independent_traversal(mol):
        return ()
    if is_supported_tetrahedral_monocycle_with_acyclic_branches(mol):
        return ()
    if is_supported_monocycle_with_acyclic_branches(mol):
        return ()
    if is_supported_nonstereo_polycyclic_skeleton(mol):
        return ()
    if is_supported_polycyclic_ring_stereo_skeleton(mol):
        return ()
    return (
        SouthStarUnsupportedFeature(
            category="ring_molecule",
            atom_indices=ring_atom_indices,
            bond_indices=ring_bond_indices,
            reason=(
                "this ring topology is outside the current South Star ring "
                "traversal scope"
            ),
        ),
    )


def _ring_stereo_features(mol: Chem.Mol) -> tuple[SouthStarUnsupportedFeature, ...]:
    if is_supported_ring_stereo_monocycle_with_acyclic_branches(
        mol
    ) or is_supported_polycyclic_ring_stereo_skeleton(mol):
        return ()

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
                    reason=(
                        "this ring-stereo carrier basis is outside the current "
                        "South Star marker-equation scope"
                    ),
                )
            )
    return tuple(features)


def _polycyclic_ring_features(mol: Chem.Mol) -> tuple[SouthStarUnsupportedFeature, ...]:
    if mol.GetNumAtoms() == 0:
        return ()
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() <= 1:
        return ()
    if is_supported_nonstereo_polycyclic_skeleton(
        mol
    ) or is_supported_polycyclic_ring_stereo_skeleton(mol):
        return ()
    ring_atom_indices = tuple(
        atom.GetIdx() for atom in mol.GetAtoms() if atom.IsInRing()
    )
    ring_bond_indices = tuple(
        bond.GetIdx() for bond in mol.GetBonds() if bond.IsInRing()
    )
    return (
        SouthStarUnsupportedFeature(
            category="fused_or_polycyclic_ring",
            atom_indices=ring_atom_indices,
            bond_indices=ring_bond_indices,
            reason=(
                "this fused or polycyclic traversal is outside the current "
                "South Star ring-system scope"
            ),
        ),
    )


def _ring_tetrahedral_interaction_features(
    mol: Chem.Mol,
) -> tuple[SouthStarUnsupportedFeature, ...]:
    if is_supported_tetrahedral_monocycle_with_acyclic_branches(
        mol
    ) or is_supported_tetrahedral_exocyclic_directional_monocycle(mol):
        return ()
    return tuple(
        SouthStarUnsupportedFeature(
            category="ring_tetrahedral_interaction",
            atom_indices=(obligation.center_atom_idx,),
            bond_indices=tuple(
                bond.GetIdx()
                for bond in mol.GetAtomWithIdx(obligation.center_atom_idx).GetBonds()
            ),
            reason=(
                "this ring-local tetrahedral ligand ordering is outside the "
                "current South Star ring/tetrahedral interaction scope"
            ),
        )
        for obligation in extract_ring_tetrahedral_interaction_obligations(mol)
    )


def _aromatic_ring_features(
    mol: Chem.Mol,
) -> tuple[SouthStarUnsupportedFeature, ...]:
    aromatic_bonds = tuple(bond for bond in mol.GetBonds() if bond.GetIsAromatic())
    if not aromatic_bonds:
        return ()
    if is_supported_aromatic_monocycle(mol):
        return ()
    contract = SOUTH_STAR_AROMATIC_TEXT_POLICY_CONTRACT
    return (
        SouthStarUnsupportedFeature(
            category="aromatic_ring_surface",
            atom_indices=tuple(
                dict.fromkeys(
                    atom_idx
                    for bond in aromatic_bonds
                    for atom_idx in (bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())
                )
            ),
            bond_indices=tuple(bond.GetIdx() for bond in aromatic_bonds),
            reason=(
                f"active South Star aromatic contract {contract.name!r} "
                "currently supports markerless aromatic monocycles only"
            ),
        ),
    )


def _aromatic_directional_features(
    mol: Chem.Mol,
) -> tuple[SouthStarUnsupportedFeature, ...]:
    contract = SOUTH_STAR_AROMATIC_TEXT_POLICY_CONTRACT
    return tuple(
        SouthStarUnsupportedFeature(
            category="aromatic_directional_surface",
            atom_indices=(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()),
            bond_indices=(bond.GetIdx(),),
            reason=(
                f"active South Star aromatic contract {contract.name!r} uses "
                f"{contract.directional_surface_policy}"
            ),
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


def is_saturated_monocycle_with_acyclic_branches(mol: Chem.Mol) -> bool:
    return (
        is_nonstereo_monocycle_with_acyclic_branches(mol)
        and all(bond.GetBondType() == Chem.BondType.SINGLE for bond in mol.GetBonds())
    )


def is_nonstereo_monocycle_with_acyclic_branches(mol: Chem.Mol) -> bool:
    if not _has_supported_monocycle_shape(mol):
        return False
    return all(bond.GetStereo() == Chem.BondStereo.STEREONONE for bond in mol.GetBonds())


def is_supported_ring_stereo_monocycle_with_acyclic_branches(mol: Chem.Mol) -> bool:
    if not _has_supported_monocycle_shape(mol):
        return False
    stereo_bonds = tuple(
        bond
        for bond in mol.GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    )
    return bool(stereo_bonds) and all(bond.IsInRing() for bond in stereo_bonds)


def is_supported_exocyclic_directional_monocycle_with_acyclic_branches(
    mol: Chem.Mol,
) -> bool:
    if not _has_supported_monocycle_shape(mol):
        return False
    stereo_bonds = tuple(
        bond
        for bond in mol.GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    )
    return len(stereo_bonds) == 1 and not stereo_bonds[0].IsInRing()


def is_supported_tetrahedral_exocyclic_directional_monocycle(
    mol: Chem.Mol,
) -> bool:
    if not extract_ring_tetrahedral_interaction_obligations(mol):
        return False
    if not _has_supported_monocycle_shape(mol, allow_tetrahedral_stereo=True):
        return False
    stereo_bonds = tuple(
        bond
        for bond in mol.GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    )
    return len(stereo_bonds) == 1 and not stereo_bonds[0].IsInRing()


def is_supported_monocycle_with_acyclic_branches(mol: Chem.Mol) -> bool:
    return is_nonstereo_monocycle_with_acyclic_branches(
        mol
    ) or is_supported_aromatic_monocycle(
        mol
    ) or is_supported_ring_stereo_monocycle_with_acyclic_branches(
        mol
    ) or is_supported_exocyclic_directional_monocycle_with_acyclic_branches(
        mol
    ) or is_supported_tetrahedral_exocyclic_directional_monocycle(
        mol
    ) or is_supported_tetrahedral_monocycle_with_acyclic_branches(mol)


def is_supported_aromatic_monocycle(mol: Chem.Mol) -> bool:
    if not _has_supported_monocycle_shape(mol):
        return False
    ring_info = mol.GetRingInfo()
    ring_atoms = set(ring_info.AtomRings()[0])
    ring_bonds = set(ring_info.BondRings()[0])
    return (
        len(ring_atoms) == mol.GetNumAtoms()
        and bool(ring_bonds)
        and all(
            _aromatic_atom_text_supported(mol.GetAtomWithIdx(atom_idx))
            for atom_idx in ring_atoms
        )
        and all(
            mol.GetBondWithIdx(bond_idx).GetIsAromatic()
            for bond_idx in ring_bonds
        )
    )


def _aromatic_atom_text_supported(atom: Chem.Atom) -> bool:
    fields = south_star_atom_text_fields(atom)
    return (
        fields.is_aromatic
        and fields.symbol.lower() in SOUTH_STAR_AROMATIC_ATOM_TEXT_TOKENS
        and fields.isotope == 0
        and fields.formal_charge == 0
        and fields.radical_electron_count == 0
        and fields.atom_map_number == 0
        and fields.explicit_hydrogen_count == 0
    )


def is_supported_tetrahedral_monocycle_with_acyclic_branches(
    mol: Chem.Mol,
) -> bool:
    if not extract_ring_tetrahedral_interaction_obligations(mol):
        return False
    if not _has_supported_monocycle_shape(mol, allow_tetrahedral_stereo=True):
        return False
    return all(bond.GetStereo() == Chem.BondStereo.STEREONONE for bond in mol.GetBonds())


def is_supported_nonstereo_polycyclic_skeleton(mol: Chem.Mol) -> bool:
    if not _has_supported_polycyclic_shape(mol):
        return False
    if any(
        bond.GetStereo() != Chem.BondStereo.STEREONONE
        for bond in mol.GetBonds()
    ):
        return False
    return True


def is_supported_polycyclic_ring_stereo_skeleton(mol: Chem.Mol) -> bool:
    if not _has_supported_polycyclic_shape(mol):
        return False
    stereo_bonds = tuple(
        bond
        for bond in mol.GetBonds()
        if bond.GetStereo() != Chem.BondStereo.STEREONONE
    )
    return bool(stereo_bonds) and all(bond.IsInRing() for bond in stereo_bonds)


def _has_supported_monocycle_shape(
    mol: Chem.Mol,
    *,
    allow_tetrahedral_stereo: bool = False,
) -> bool:
    if mol.GetNumAtoms() == 0:
        return False
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    if mol.GetNumBonds() != mol.GetNumAtoms():
        return False
    if not allow_tetrahedral_stereo and any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in mol.GetAtoms()
    ):
        return False
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() != 1:
        return False
    ring_atoms = set(ring_info.AtomRings()[0])
    ring_bonds = set(ring_info.BondRings()[0])
    if len(ring_atoms) != len(ring_bonds):
        return False
    return all(
        bond.GetBondType() in SUPPORTED_BOND_TYPES
        for bond in mol.GetBonds()
    )


def _has_supported_polycyclic_shape(mol: Chem.Mol) -> bool:
    if mol.GetNumAtoms() == 0:
        return False
    if len(Chem.GetMolFrags(mol)) != 1:
        return False
    ring_info = mol.GetRingInfo()
    if ring_info.NumRings() <= 1:
        return False
    if any(
        atom.GetChiralTag() != Chem.ChiralType.CHI_UNSPECIFIED
        for atom in mol.GetAtoms()
    ):
        return False
    if any(bond.GetBondType() not in SUPPORTED_BOND_TYPES for bond in mol.GetBonds()):
        return False
    return not any(atom.GetIsAromatic() for atom in mol.GetAtoms()) and not any(
        bond.GetIsAromatic() for bond in mol.GetBonds()
    )


def _has_ring_tetrahedral_interaction(mol: Chem.Mol) -> bool:
    return bool(extract_ring_tetrahedral_interaction_obligations(mol))


def disconnected_fragments_have_supported_independent_traversal(
    mol: Chem.Mol,
) -> bool:
    fragments = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=True)
    if len(fragments) <= 1:
        return False
    return all(
        south_star_support_gate_report(fragment).supported
        for fragment in fragments
    )
