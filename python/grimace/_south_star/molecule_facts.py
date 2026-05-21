from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.annotation_policy import Edge
from grimace._south_star.annotation_policy import SemanticCarrierOpportunity
from grimace._south_star.annotation_policy import normalized_edge
from grimace._south_star.atom_text import SouthStarAtomTextFields
from grimace._south_star.atom_text import south_star_atom_text_fields
from grimace._south_star.components import (
    SouthStarComponentExtraction,
    SouthStarSemanticStereoComponent,
    extract_south_star_components,
)
from grimace._south_star.support_gates import (
    SouthStarSupportGateReport,
    south_star_support_gate_report,
)
from grimace._south_star.tetrahedral import (
    SouthStarTetrahedralCenterFact,
    extract_tetrahedral_center_facts,
)


SouthStarAtomTextFact = SouthStarAtomTextFields


@dataclass(frozen=True, slots=True)
class SouthStarBondTextFact:
    bond_idx: int
    edge: Edge
    bond_type: str
    bond_dir: str
    is_aromatic: bool
    is_in_ring: bool


@dataclass(frozen=True, slots=True)
class SouthStarRingSystemFacts:
    atom_rings: tuple[tuple[int, ...], ...]
    bond_rings: tuple[tuple[int, ...], ...]
    ring_atom_indices: tuple[int, ...]
    ring_bond_indices: tuple[int, ...]
    atom_ring_membership_counts: tuple[tuple[int, int], ...]
    bond_ring_membership_counts: tuple[tuple[int, int], ...]

    @property
    def ring_count(self) -> int:
        return len(self.bond_rings)

    @property
    def has_rings(self) -> bool:
        return self.ring_count > 0

    @property
    def simple_monocycle(self) -> bool:
        return (
            self.ring_count == 1
            and len(self.atom_rings[0]) == len(self.bond_rings[0])
        )

    @property
    def fused_or_polycyclic(self) -> bool:
        return self.ring_count > 1

    @property
    def shared_ring_atom_indices(self) -> tuple[int, ...]:
        return tuple(
            atom_idx
            for atom_idx, count in self.atom_ring_membership_counts
            if count > 1
        )

    @property
    def shared_ring_bond_indices(self) -> tuple[int, ...]:
        return tuple(
            bond_idx
            for bond_idx, count in self.bond_ring_membership_counts
            if count > 1
        )

    @property
    def spiro_like(self) -> bool:
        return (
            self.ring_count > 1
            and bool(self.shared_ring_atom_indices)
            and not self.shared_ring_bond_indices
        )


@dataclass(frozen=True, slots=True)
class SouthStarGraphTopologyFacts:
    atom_indices: tuple[int, ...]
    bond_edges: tuple[Edge, ...]
    fragment_atom_indices: tuple[tuple[int, ...], ...]
    ring_system: SouthStarRingSystemFacts

    @property
    def atom_count(self) -> int:
        return len(self.atom_indices)

    @property
    def bond_count(self) -> int:
        return len(self.bond_edges)

    @property
    def fragment_count(self) -> int:
        return len(self.fragment_atom_indices)

    @property
    def connected(self) -> bool:
        return self.fragment_count == 1

    @property
    def acyclic_connected_tree(self) -> bool:
        return self.connected and self.bond_count == self.atom_count - 1

    @property
    def ring_atom_indices(self) -> tuple[int, ...]:
        return self.ring_system.ring_atom_indices

    @property
    def ring_bond_indices(self) -> tuple[int, ...]:
        return self.ring_system.ring_bond_indices

    @property
    def ring_count(self) -> int:
        return self.ring_system.ring_count

    @property
    def cyclomatic_number(self) -> int:
        return self.bond_count - self.atom_count + self.fragment_count


@dataclass(frozen=True, slots=True)
class SouthStarMoleculeFacts:
    """Semantic fact boundary shared by South Star runtime and test oracles."""

    support_gate_report: SouthStarSupportGateReport
    atom_text_facts: tuple[SouthStarAtomTextFact, ...]
    bond_text_facts: tuple[SouthStarBondTextFact, ...]
    graph_topology: SouthStarGraphTopologyFacts
    components: tuple[SouthStarSemanticStereoComponent, ...]
    carrier_opportunities: tuple[SemanticCarrierOpportunity, ...]
    tetrahedral_center_facts: tuple[SouthStarTetrahedralCenterFact, ...]

    @classmethod
    def from_mol(cls, mol: Chem.Mol) -> SouthStarMoleculeFacts:
        support_gate_report = south_star_support_gate_report(mol)
        component_extraction = extract_south_star_components(
            mol,
            support_gate_report=support_gate_report,
        )
        _assert_shared_support_report(support_gate_report, component_extraction)
        return cls(
            support_gate_report=support_gate_report,
            atom_text_facts=_atom_text_facts(mol),
            bond_text_facts=_bond_text_facts(mol),
            graph_topology=_graph_topology_facts(mol),
            components=component_extraction.components,
            carrier_opportunities=_carrier_opportunities(
                component_extraction.components
            ),
            tetrahedral_center_facts=(
                extract_tetrahedral_center_facts(mol)
                if support_gate_report.supported
                else ()
            ),
        )

    @property
    def supported(self) -> bool:
        return self.support_gate_report.supported

    @property
    def unsupported_categories(self) -> frozenset[str]:
        return self.support_gate_report.categories

    def fail_if_unsupported(self) -> None:
        self.support_gate_report.fail_if_unsupported()


def _assert_shared_support_report(
    support_gate_report: SouthStarSupportGateReport,
    component_extraction: SouthStarComponentExtraction,
) -> None:
    if component_extraction.support_gate_report is not support_gate_report:
        raise AssertionError(
            "South Star molecule facts require component extraction to share "
            "the same support gate report"
        )


def _atom_text_facts(mol: Chem.Mol) -> tuple[SouthStarAtomTextFact, ...]:
    return tuple(south_star_atom_text_fields(atom) for atom in mol.GetAtoms())


def _bond_text_facts(mol: Chem.Mol) -> tuple[SouthStarBondTextFact, ...]:
    return tuple(
        SouthStarBondTextFact(
            bond_idx=bond.GetIdx(),
            edge=normalized_edge((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())),
            bond_type=str(bond.GetBondType()),
            bond_dir=str(bond.GetBondDir()),
            is_aromatic=bond.GetIsAromatic(),
            is_in_ring=bond.IsInRing(),
        )
        for bond in mol.GetBonds()
    )


def _graph_topology_facts(mol: Chem.Mol) -> SouthStarGraphTopologyFacts:
    return SouthStarGraphTopologyFacts(
        atom_indices=tuple(atom.GetIdx() for atom in mol.GetAtoms()),
        bond_edges=tuple(
            normalized_edge((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()))
            for bond in mol.GetBonds()
        ),
        fragment_atom_indices=tuple(
            tuple(fragment) for fragment in Chem.GetMolFrags(mol)
        ),
        ring_system=_ring_system_facts(mol),
    )


def _ring_system_facts(mol: Chem.Mol) -> SouthStarRingSystemFacts:
    ring_info = mol.GetRingInfo()
    atom_rings = tuple(tuple(ring) for ring in ring_info.AtomRings())
    bond_rings = tuple(tuple(ring) for ring in ring_info.BondRings())
    return SouthStarRingSystemFacts(
        atom_rings=atom_rings,
        bond_rings=bond_rings,
        ring_atom_indices=tuple(
            atom.GetIdx() for atom in mol.GetAtoms() if atom.IsInRing()
        ),
        ring_bond_indices=tuple(
            bond.GetIdx() for bond in mol.GetBonds() if bond.IsInRing()
        ),
        atom_ring_membership_counts=_membership_counts(atom_rings),
        bond_ring_membership_counts=_membership_counts(bond_rings),
    )


def _membership_counts(
    rings: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, int], ...]:
    counts = Counter(member for ring in rings for member in ring)
    return tuple(sorted(counts.items()))


def _carrier_opportunities(
    components: tuple[SouthStarSemanticStereoComponent, ...],
) -> tuple[SemanticCarrierOpportunity, ...]:
    carrier_edges = tuple(
        dict.fromkeys(
            edge
            for component in components
            for edge in component.eligible_carrier_edges
        )
    )
    return tuple(SemanticCarrierOpportunity(edge=edge) for edge in carrier_edges)
