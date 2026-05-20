from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from tests.helpers.south_star_annotation_policy import Edge, normalized_edge
from tests.helpers.south_star_support_gates import (
    SouthStarSupportGateReport,
    SouthStarUnsupportedFeature,
    south_star_support_gate_report,
)


@dataclass(frozen=True, slots=True)
class SouthStarSourceStereoFeature:
    feature_id: str
    central_bond: Edge
    rdkit_stereo: str
    left_carrier_edges: tuple[Edge, ...]
    right_carrier_edges: tuple[Edge, ...]

    @property
    def eligible_carrier_edges(self) -> tuple[Edge, ...]:
        return tuple(
            dict.fromkeys(self.left_carrier_edges + self.right_carrier_edges)
        )


@dataclass(frozen=True, slots=True)
class SouthStarSemanticStereoComponent:
    component_id: str
    source_features: tuple[SouthStarSourceStereoFeature, ...]
    eligible_carrier_edges: tuple[Edge, ...]
    unsupported_features: tuple[SouthStarUnsupportedFeature, ...] = ()


@dataclass(frozen=True, slots=True)
class SouthStarComponentExtraction:
    support_gate_report: SouthStarSupportGateReport
    components: tuple[SouthStarSemanticStereoComponent, ...]

    @property
    def supported(self) -> bool:
        return self.support_gate_report.supported

    @property
    def unsupported_features(self) -> tuple[SouthStarUnsupportedFeature, ...]:
        return self.support_gate_report.unsupported_features

    def fail_if_unsupported(self) -> None:
        self.support_gate_report.fail_if_unsupported()


def extract_south_star_components(mol: Chem.Mol) -> SouthStarComponentExtraction:
    gate_report = south_star_support_gate_report(mol)
    if not gate_report.supported:
        return SouthStarComponentExtraction(
            support_gate_report=gate_report,
            components=(),
        )

    features = _source_stereo_features(mol)
    components = _componentize_features(features)
    return SouthStarComponentExtraction(
        support_gate_report=gate_report,
        components=components,
    )


def _source_stereo_features(
    mol: Chem.Mol,
) -> tuple[SouthStarSourceStereoFeature, ...]:
    features = []
    for bond in mol.GetBonds():
        if bond.GetBondType() != Chem.BondType.DOUBLE:
            continue
        if bond.GetStereo() == Chem.BondStereo.STEREONONE:
            continue

        begin_atom_idx = bond.GetBeginAtomIdx()
        end_atom_idx = bond.GetEndAtomIdx()
        left_carriers = _carrier_edges_for_endpoint(
            mol,
            endpoint_atom_idx=begin_atom_idx,
            excluded_atom_idx=end_atom_idx,
        )
        right_carriers = _carrier_edges_for_endpoint(
            mol,
            endpoint_atom_idx=end_atom_idx,
            excluded_atom_idx=begin_atom_idx,
        )
        features.append(
            SouthStarSourceStereoFeature(
                feature_id=f"bond:{bond.GetIdx()}",
                central_bond=normalized_edge((begin_atom_idx, end_atom_idx)),
                rdkit_stereo=str(bond.GetStereo()),
                left_carrier_edges=left_carriers,
                right_carrier_edges=right_carriers,
            )
        )
    return tuple(features)


def _carrier_edges_for_endpoint(
    mol: Chem.Mol,
    *,
    endpoint_atom_idx: int,
    excluded_atom_idx: int,
) -> tuple[Edge, ...]:
    atom = mol.GetAtomWithIdx(endpoint_atom_idx)
    edges = []
    for neighbor in atom.GetNeighbors():
        neighbor_idx = neighbor.GetIdx()
        if neighbor_idx == excluded_atom_idx:
            continue
        bond = mol.GetBondBetweenAtoms(endpoint_atom_idx, neighbor_idx)
        if bond is None or bond.GetBondType() != Chem.BondType.SINGLE:
            continue
        edges.append(normalized_edge((endpoint_atom_idx, neighbor_idx)))
    return tuple(dict.fromkeys(edges))


def _componentize_features(
    features: tuple[SouthStarSourceStereoFeature, ...],
) -> tuple[SouthStarSemanticStereoComponent, ...]:
    if not features:
        return ()

    parent = list(range(len(features)))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(left: int, right: int) -> None:
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    carrier_owner: dict[Edge, int] = {}
    for feature_index, feature in enumerate(features):
        for edge in feature.eligible_carrier_edges:
            owner = carrier_owner.setdefault(edge, feature_index)
            union(owner, feature_index)

    groups: dict[int, list[SouthStarSourceStereoFeature]] = {}
    for feature_index, feature in enumerate(features):
        groups.setdefault(find(feature_index), []).append(feature)

    components = []
    for component_index, grouped_features in enumerate(groups.values()):
        carrier_edges = tuple(
            dict.fromkeys(
                edge
                for feature in grouped_features
                for edge in feature.eligible_carrier_edges
            )
        )
        components.append(
            SouthStarSemanticStereoComponent(
                component_id=f"component:{component_index}",
                source_features=tuple(grouped_features),
                eligible_carrier_edges=carrier_edges,
            )
        )
    return tuple(components)
