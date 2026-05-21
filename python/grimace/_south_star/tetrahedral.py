from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.connected_traversal import connected_graph_plan_from_events
from grimace._south_star.reference_model import SouthStarConnectedGraphTraversalPlan
from grimace._south_star.reference_model import SouthStarTraversalEvent


TETRAHEDRAL_TOKENS: frozenset[str] = frozenset({"@", "@@"})
IMPLICIT_HYDROGEN_LIGAND = "implicit_hydrogen"
RING_TETRAHEDRAL_REQUIRED_FACT_AND_EVENT_FIELDS: tuple[str, ...] = (
    "center_atom_idx",
    "source_ligand_order",
    "source_token",
    "center_in_ring",
    "ring_ligand_atom_indices",
    "acyclic_ligand_atom_indices",
    "implicit_hydrogen_count",
    "parent_atom_idx",
    "child_atom_indices",
    "ring_closure_events",
    "ring_closure_ligand_atom_indices",
    "ring_closure_labels",
)


@dataclass(frozen=True, slots=True)
class SouthStarTetrahedralCenterFact:
    center_atom_idx: int
    chiral_tag: str
    source_token: str
    explicit_neighbor_atom_indices: tuple[int, ...]
    implicit_hydrogen_count: int
    source_ligand_order: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SouthStarRingTetrahedralInteractionObligation:
    center_atom_idx: int
    source_token: str
    source_ligand_order: tuple[str, ...]
    center_in_ring: bool
    ring_ligand_atom_indices: tuple[int, ...]
    acyclic_ligand_atom_indices: tuple[int, ...]
    implicit_hydrogen_count: int
    required_fact_and_event_fields: tuple[str, ...] = (
        RING_TETRAHEDRAL_REQUIRED_FACT_AND_EVENT_FIELDS
    )


@dataclass(frozen=True, slots=True)
class SouthStarTetrahedralTraversalObservation:
    center_atom_idx: int
    parent_atom_idx: int | None
    child_atom_indices: tuple[int, ...]
    ring_closure_ligand_atom_indices: tuple[int, ...]
    ring_closure_labels: tuple[str, ...]
    implicit_hydrogen_count: int


def extract_tetrahedral_center_facts(
    mol: Chem.Mol,
) -> tuple[SouthStarTetrahedralCenterFact, ...]:
    return tuple(
        _tetrahedral_center_fact(atom)
        for atom in mol.GetAtoms()
        if atom.GetChiralTag()
        in {
            Chem.ChiralType.CHI_TETRAHEDRAL_CW,
            Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
        }
    )


def tetrahedral_atom_supported(atom: Chem.Atom) -> bool:
    if atom.GetChiralTag() not in {
        Chem.ChiralType.CHI_TETRAHEDRAL_CW,
        Chem.ChiralType.CHI_TETRAHEDRAL_CCW,
    }:
        return False
    implicit_hydrogen_count = atom.GetTotalNumHs()
    if implicit_hydrogen_count > 1:
        return False
    return atom.GetDegree() + implicit_hydrogen_count == 4


def extract_ring_tetrahedral_interaction_obligations(
    mol: Chem.Mol,
) -> tuple[SouthStarRingTetrahedralInteractionObligation, ...]:
    return tuple(
        _ring_tetrahedral_interaction_obligation(mol, fact)
        for fact in (
            _tetrahedral_center_fact(atom)
            for atom in mol.GetAtoms()
            if tetrahedral_atom_supported(atom)
        )
        if _tetrahedral_fact_has_ring_ligand(mol, fact)
    )


def preserving_tetrahedral_token(
    *,
    source_token: str,
    source_ligand_order: Sequence[str],
    emitted_ligand_order: Sequence[str],
) -> str:
    _validate_tetrahedral_token(source_token)
    return (
        source_token
        if _permutation_is_even(source_ligand_order, emitted_ligand_order)
        else _flipped_tetrahedral_token(source_token)
    )


def tetrahedral_token_preserves_orientation(
    *,
    candidate_token: str,
    source_token: str,
    source_ligand_order: Sequence[str],
    emitted_ligand_order: Sequence[str],
) -> bool:
    _validate_tetrahedral_token(candidate_token)
    return candidate_token == preserving_tetrahedral_token(
        source_token=source_token,
        source_ligand_order=source_ligand_order,
        emitted_ligand_order=emitted_ligand_order,
    )


def emitted_tetrahedral_ligand_order_from_observation(
    observation: SouthStarTetrahedralTraversalObservation,
) -> tuple[str, ...]:
    emitted = []
    if observation.parent_atom_idx is None and observation.implicit_hydrogen_count:
        emitted.append(IMPLICIT_HYDROGEN_LIGAND)
    if observation.parent_atom_idx is not None:
        emitted.append(f"atom:{observation.parent_atom_idx}")
    emitted.extend(
        f"atom:{atom_idx}" for atom_idx in observation.ring_closure_ligand_atom_indices
    )
    emitted.extend(f"atom:{atom_idx}" for atom_idx in observation.child_atom_indices)
    if observation.parent_atom_idx is not None and observation.implicit_hydrogen_count:
        emitted.append(IMPLICIT_HYDROGEN_LIGAND)
    return tuple(emitted)


def tetrahedral_traversal_observation_from_events(
    events: Sequence[SouthStarTraversalEvent],
    *,
    center_atom_idx: int,
    implicit_hydrogen_count: int,
) -> SouthStarTetrahedralTraversalObservation:
    plan = connected_graph_plan_from_events(
        root_atom_idx=_root_atom_idx_from_events(events),
        events=tuple(events),
    )
    return tetrahedral_traversal_observation_from_connected_graph_plan(
        plan,
        center_atom_idx=center_atom_idx,
        implicit_hydrogen_count=implicit_hydrogen_count,
    )


def tetrahedral_traversal_observation_from_connected_graph_plan(
    plan: SouthStarConnectedGraphTraversalPlan,
    *,
    center_atom_idx: int,
    implicit_hydrogen_count: int,
) -> SouthStarTetrahedralTraversalObservation:
    if center_atom_idx not in plan.atom_order:
        raise ValueError(
            f"tetrahedral center {center_atom_idx} is not in traversal atom order"
        )

    parent_edges = tuple(
        edge for edge in plan.tree_edges if edge.end_atom_idx == center_atom_idx
    )
    if len(parent_edges) > 1:
        raise ValueError(
            f"tetrahedral center {center_atom_idx} has multiple traversal parents"
        )
    closure_endpoints = tuple(
        endpoint
        for endpoint in plan.closure_endpoints
        if endpoint.atom_idx == center_atom_idx
    )
    return SouthStarTetrahedralTraversalObservation(
        center_atom_idx=center_atom_idx,
        parent_atom_idx=(
            parent_edges[0].begin_atom_idx if parent_edges else None
        ),
        child_atom_indices=tuple(
            edge.end_atom_idx
            for edge in plan.tree_edges
            if edge.begin_atom_idx == center_atom_idx
        ),
        ring_closure_ligand_atom_indices=tuple(
            endpoint.partner_atom_idx for endpoint in closure_endpoints
        ),
        ring_closure_labels=tuple(endpoint.label for endpoint in closure_endpoints),
        implicit_hydrogen_count=implicit_hydrogen_count,
    )


def _root_atom_idx_from_events(events: Sequence[SouthStarTraversalEvent]) -> int:
    atom_event_positions = tuple(
        position
        for position, event in enumerate(events)
        if event.kind == "atom" and event.atom_idx is not None
    )
    if not atom_event_positions:
        raise ValueError("tetrahedral traversal observation requires atom events")
    first_atom = events[atom_event_positions[0]].atom_idx
    if first_atom is None:
        raise AssertionError("atom event position must carry an atom index")
    return first_atom


def _ring_tetrahedral_interaction_obligation(
    mol: Chem.Mol,
    fact: SouthStarTetrahedralCenterFact,
) -> SouthStarRingTetrahedralInteractionObligation:
    ring_ligands = tuple(
        atom_idx
        for atom_idx in fact.explicit_neighbor_atom_indices
        if mol.GetAtomWithIdx(atom_idx).IsInRing()
    )
    acyclic_ligands = tuple(
        atom_idx
        for atom_idx in fact.explicit_neighbor_atom_indices
        if not mol.GetAtomWithIdx(atom_idx).IsInRing()
    )
    return SouthStarRingTetrahedralInteractionObligation(
        center_atom_idx=fact.center_atom_idx,
        source_token=fact.source_token,
        source_ligand_order=fact.source_ligand_order,
        center_in_ring=mol.GetAtomWithIdx(fact.center_atom_idx).IsInRing(),
        ring_ligand_atom_indices=ring_ligands,
        acyclic_ligand_atom_indices=acyclic_ligands,
        implicit_hydrogen_count=fact.implicit_hydrogen_count,
    )


def _tetrahedral_fact_has_ring_ligand(
    mol: Chem.Mol,
    fact: SouthStarTetrahedralCenterFact,
) -> bool:
    center = mol.GetAtomWithIdx(fact.center_atom_idx)
    return center.IsInRing() or any(
        mol.GetAtomWithIdx(atom_idx).IsInRing()
        for atom_idx in fact.explicit_neighbor_atom_indices
    )


def _tetrahedral_center_fact(atom: Chem.Atom) -> SouthStarTetrahedralCenterFact:
    if not tetrahedral_atom_supported(atom):
        raise NotImplementedError(
            "South Star tetrahedral fact extraction requires a tetrahedral "
            "center with exactly four ligands"
        )
    implicit_hydrogen_count = atom.GetTotalNumHs()
    explicit_neighbors = tuple(neighbor.GetIdx() for neighbor in atom.GetNeighbors())
    source_ligand_order = tuple(f"atom:{atom_idx}" for atom_idx in explicit_neighbors)
    if implicit_hydrogen_count:
        source_ligand_order += (IMPLICIT_HYDROGEN_LIGAND,)
    return SouthStarTetrahedralCenterFact(
        center_atom_idx=atom.GetIdx(),
        chiral_tag=str(atom.GetChiralTag()),
        source_token=_source_token_for_chiral_tag(atom.GetChiralTag()),
        explicit_neighbor_atom_indices=explicit_neighbors,
        implicit_hydrogen_count=implicit_hydrogen_count,
        source_ligand_order=source_ligand_order,
    )


def _source_token_for_chiral_tag(chiral_tag: Chem.ChiralType) -> str:
    if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CCW:
        return "@"
    if chiral_tag == Chem.ChiralType.CHI_TETRAHEDRAL_CW:
        return "@@"
    raise ValueError(f"unsupported tetrahedral chiral tag {chiral_tag}")


def _permutation_is_even(
    source_ligand_order: Sequence[str],
    emitted_ligand_order: Sequence[str],
) -> bool:
    source = tuple(source_ligand_order)
    emitted = tuple(emitted_ligand_order)
    if len(source) != len(emitted):
        raise ValueError("tetrahedral ligand orders must have equal length")
    if len(set(source)) != len(source) or len(set(emitted)) != len(emitted):
        raise ValueError("tetrahedral ligand orders must contain unique ligands")
    if set(source) != set(emitted):
        raise ValueError("tetrahedral ligand orders must contain the same ligands")

    source_index = {ligand: idx for idx, ligand in enumerate(source)}
    permutation = tuple(source_index[ligand] for ligand in emitted)
    inversion_count = sum(
        1
        for idx, left in enumerate(permutation)
        for right in permutation[idx + 1 :]
        if left > right
    )
    return inversion_count % 2 == 0


def _flipped_tetrahedral_token(token: str) -> str:
    _validate_tetrahedral_token(token)
    return "@@" if token == "@" else "@"


def _validate_tetrahedral_token(token: str) -> None:
    if token not in TETRAHEDRAL_TOKENS:
        raise ValueError(f"tetrahedral token must be one of {TETRAHEDRAL_TOKENS}")

