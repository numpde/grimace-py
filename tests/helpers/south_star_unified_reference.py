from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.atom_text import SouthStarAtomTextFields
from grimace._south_star.atom_text import atom_text_modifier_obligations
from grimace._south_star.atom_text import atom_text_obligation_for_supported_fields
from grimace._south_star.enum_s import mol_to_smiles_enum_s_tree_traversals_for_case
from grimace._south_star.enum_s import render_south_star_tree_traversal
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.molecule_facts import SouthStarBondTextFact
from tests.helpers.south_star_semantic_oracle import parse_smiles


@dataclass(frozen=True, slots=True)
class SouthStarAtomTextRendererObligationSummary:
    atom_idx: int
    emitted_text: str
    token_family: str
    bracket_obligations: tuple[str, ...]
    modifier_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SouthStarBondTextRendererObligationSummary:
    bond_idx: int
    edge: tuple[int, int]
    bond_type: str
    emitted_text: str
    token_family: str


@dataclass(frozen=True, slots=True)
class SouthStarSingleAtomAtomTextSupport:
    emitted_text: str
    atom_text_obligations: tuple[SouthStarAtomTextRendererObligationSummary, ...]
    modifier_obligation_count: int
    bracket_obligation_count: int
    support: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SouthStarTwoAtomMarkerlessAtomTextSupportProof:
    atom_text_obligations: tuple[SouthStarAtomTextRendererObligationSummary, ...]
    bond_text_obligation: SouthStarBondTextRendererObligationSummary
    modifier_obligation_count: int
    bracket_obligation_count: int
    support: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class SouthStarMarkerlessAcyclicTreeSupportProof:
    case_id: str
    atom_count: int
    bond_count: int
    traversal_count: int
    atom_event_count: int
    bond_event_count: int
    branch_event_count: int
    atom_text_obligation_count: int
    atom_text_modifier_obligation_count: int
    atom_text_bracket_obligation_count: int
    bond_text_obligation_count: int
    bond_token_families: tuple[str, ...]
    raw_output_count: int
    output_count: int
    support: tuple[str, ...]
    expected_support_strings_used: bool


@dataclass(frozen=True, slots=True)
class SouthStarNonstereoMonocycleSupportProof:
    case_id: str
    atom_count: int
    bond_count: int
    ring_count: int
    traversal_count: int
    atom_event_count: int
    bond_event_count: int
    closure_event_count: int
    closure_open_bond_texts: tuple[str, ...]
    atom_text_obligation_count: int
    atom_text_modifier_obligation_count: int
    atom_text_bracket_obligation_count: int
    bond_text_obligation_count: int
    bond_token_families: tuple[str, ...]
    marker_slot_count: int
    renderer_input_count: int
    raw_output_count: int
    output_count: int
    support: tuple[str, ...]
    expected_support_strings_used: bool


@dataclass(frozen=True, slots=True)
class SouthStarNonstereoPolycyclicSupportProof:
    case_id: str
    atom_count: int
    bond_count: int
    ring_count: int
    cyclomatic_number: int
    traversal_count: int
    atom_event_count: int
    bond_event_count: int
    closure_edge_set_count: int
    closure_edge_count: int
    closure_event_count: int
    closure_label_count: int
    atom_text_obligation_count: int
    atom_text_modifier_obligation_count: int
    atom_text_bracket_obligation_count: int
    bond_text_obligation_count: int
    bond_token_families: tuple[str, ...]
    marker_slot_count: int
    renderer_input_count: int
    raw_output_count: int
    output_count: int
    support: tuple[str, ...]
    expected_support_strings_used: bool


def is_single_atom_atom_text_domain(facts: SouthStarMoleculeFacts) -> bool:
    topology = facts.graph_topology
    return (
        facts.supported
        and topology.connected
        and topology.atom_count == 1
        and topology.bond_count == 0
        and not topology.ring_system.has_rings
        and len(facts.atom_text_facts) == 1
        and not facts.components
        and not facts.carrier_opportunities
        and not facts.tetrahedral_center_facts
    )


def single_atom_atom_text_support_from_facts(
    facts: SouthStarMoleculeFacts,
) -> SouthStarSingleAtomAtomTextSupport:
    if not is_single_atom_atom_text_domain(facts):
        raise NotImplementedError(
            "single-atom atom-text unified reference requires one supported "
            "atom, zero bonds, one fragment, and no stereo constraints"
        )
    [atom_obligation] = _atom_text_obligation_summaries(facts)
    return SouthStarSingleAtomAtomTextSupport(
        emitted_text=atom_obligation.emitted_text,
        atom_text_obligations=(atom_obligation,),
        modifier_obligation_count=len(atom_obligation.modifier_names),
        bracket_obligation_count=len(atom_obligation.bracket_obligations),
        support=(atom_obligation.emitted_text,),
    )


def is_two_atom_markerless_atom_text_domain(facts: SouthStarMoleculeFacts) -> bool:
    topology = facts.graph_topology
    return (
        facts.supported
        and topology.connected
        and topology.atom_count == 2
        and topology.bond_count == 1
        and not topology.ring_system.has_rings
        and len(facts.atom_text_facts) == 2
        and len(facts.bond_text_facts) == 1
        and not facts.components
        and not facts.carrier_opportunities
        and not facts.tetrahedral_center_facts
    )


def two_atom_markerless_atom_text_support_from_facts(
    facts: SouthStarMoleculeFacts,
) -> tuple[str, ...]:
    return two_atom_markerless_atom_text_support_proof_from_facts(facts).support


def two_atom_markerless_atom_text_support_proof_from_facts(
    facts: SouthStarMoleculeFacts,
) -> SouthStarTwoAtomMarkerlessAtomTextSupportProof:
    if not is_two_atom_markerless_atom_text_domain(facts):
        raise NotImplementedError(
            "two-atom markerless atom-text unified reference requires two "
            "supported atoms, one bond, one fragment, and no stereo constraints"
        )
    [bond_fact] = facts.bond_text_facts
    bond_obligation = _bond_text_obligation_from_fact(bond_fact)
    atom_obligations = _atom_text_obligation_summaries(facts)
    atom_text_by_idx = {
        obligation.atom_idx: obligation.emitted_text
        for obligation in atom_obligations
    }
    begin_idx, end_idx = bond_fact.edge
    outputs = (
        f"{atom_text_by_idx[begin_idx]}{bond_obligation.emitted_text}"
        f"{atom_text_by_idx[end_idx]}",
        f"{atom_text_by_idx[end_idx]}{bond_obligation.emitted_text}"
        f"{atom_text_by_idx[begin_idx]}",
    )
    return SouthStarTwoAtomMarkerlessAtomTextSupportProof(
        atom_text_obligations=atom_obligations,
        bond_text_obligation=bond_obligation,
        modifier_obligation_count=sum(
            len(obligation.modifier_names) for obligation in atom_obligations
        ),
        bracket_obligation_count=sum(
            len(obligation.bracket_obligations) for obligation in atom_obligations
        ),
        support=tuple(dict.fromkeys(outputs)),
    )


def is_markerless_acyclic_tree_domain(facts: SouthStarMoleculeFacts) -> bool:
    topology = facts.graph_topology
    return (
        facts.supported
        and topology.connected
        and topology.acyclic_connected_tree
        and topology.atom_count >= 3
        and len(facts.atom_text_facts) == topology.atom_count
        and len(facts.bond_text_facts) == topology.bond_count
        and not topology.ring_system.has_rings
        and not facts.components
        and not facts.carrier_opportunities
        and not facts.tetrahedral_center_facts
    )


def is_nonstereo_monocycle_ring_traversal_domain(
    facts: SouthStarMoleculeFacts,
) -> bool:
    topology = facts.graph_topology
    return (
        facts.supported
        and topology.connected
        and topology.ring_system.simple_monocycle
        and len(facts.atom_text_facts) == topology.atom_count
        and len(facts.bond_text_facts) == topology.bond_count
        and all(
            bond.bond_type in {"SINGLE", "DOUBLE"}
            and bond.bond_dir == "NONE"
            and not bond.is_aromatic
            for bond in facts.bond_text_facts
        )
        and not facts.components
        and not facts.carrier_opportunities
        and not facts.tetrahedral_center_facts
    )


def is_nonstereo_polycyclic_ring_traversal_domain(
    facts: SouthStarMoleculeFacts,
) -> bool:
    topology = facts.graph_topology
    return (
        facts.supported
        and topology.connected
        and topology.ring_system.fused_or_polycyclic
        and topology.cyclomatic_number > 1
        and len(facts.atom_text_facts) == topology.atom_count
        and len(facts.bond_text_facts) == topology.bond_count
        and all(
            bond.bond_type in {"SINGLE", "DOUBLE"}
            and bond.bond_dir == "NONE"
            and not bond.is_aromatic
            for bond in facts.bond_text_facts
        )
        and not facts.components
        and not facts.carrier_opportunities
        and not facts.tetrahedral_center_facts
    )


def markerless_acyclic_tree_support_from_shared_spine(
    case: object,
) -> SouthStarMarkerlessAcyclicTreeSupportProof:
    mol = parse_smiles(case.source_smiles)
    facts = SouthStarMoleculeFacts.from_mol(mol)
    if not is_markerless_acyclic_tree_domain(facts):
        raise NotImplementedError(
            "markerless acyclic-tree unified reference requires one connected "
            "acyclic molecule, at least three atoms, supported atom/bond text, "
            "and no stereo constraints"
        )
    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
    if any(
        event.marker_slot is not None or event.renderer_input is not None
        for traversal in traversals
        for event in traversal.events
    ):
        raise AssertionError(
            "markerless acyclic-tree proof must not use stereo marker slots or "
            "renderer-input obligations"
        )
    raw_outputs = tuple(
        render_south_star_tree_traversal(traversal) for traversal in traversals
    )
    support = tuple(dict.fromkeys(raw_outputs))
    atom_obligations = _atom_text_obligation_summaries(facts)
    bond_obligations = _bond_text_obligation_summaries(facts)
    return SouthStarMarkerlessAcyclicTreeSupportProof(
        case_id=case.case_id,
        atom_count=facts.graph_topology.atom_count,
        bond_count=facts.graph_topology.bond_count,
        traversal_count=len(traversals),
        atom_event_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.kind == "atom"
        ),
        bond_event_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.kind == "bond"
        ),
        branch_event_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.kind in {"branch_open", "branch_close"}
        ),
        atom_text_obligation_count=len(atom_obligations),
        atom_text_modifier_obligation_count=sum(
            len(obligation.modifier_names) for obligation in atom_obligations
        ),
        atom_text_bracket_obligation_count=sum(
            len(obligation.bracket_obligations) for obligation in atom_obligations
        ),
        bond_text_obligation_count=len(bond_obligations),
        bond_token_families=tuple(
            sorted({obligation.token_family for obligation in bond_obligations})
        ),
        raw_output_count=len(raw_outputs),
        output_count=len(support),
        support=support,
        expected_support_strings_used=False,
    )


def nonstereo_monocycle_support_from_shared_spine(
    case: object,
) -> SouthStarNonstereoMonocycleSupportProof:
    mol = parse_smiles(case.source_smiles)
    facts = SouthStarMoleculeFacts.from_mol(mol)
    if not is_nonstereo_monocycle_ring_traversal_domain(facts):
        raise NotImplementedError(
            "nonstereo-monocycle unified reference requires one connected "
            "simple monocycle, supported atom text, single/double nonaromatic "
            "bond text, and no stereo constraints"
        )

    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
    closure_events = tuple(
        event
        for traversal in traversals
        for event in traversal.events
        if event.ring_closure is not None
    )
    raw_outputs = tuple(
        render_south_star_tree_traversal(traversal) for traversal in traversals
    )
    support = tuple(dict.fromkeys(raw_outputs))
    atom_obligations = _atom_text_obligation_summaries(facts)
    bond_obligations = _bond_text_obligation_summaries(facts)
    return SouthStarNonstereoMonocycleSupportProof(
        case_id=case.case_id,
        atom_count=facts.graph_topology.atom_count,
        bond_count=facts.graph_topology.bond_count,
        ring_count=facts.graph_topology.ring_count,
        traversal_count=len(traversals),
        atom_event_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.kind == "atom"
        ),
        bond_event_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.kind == "bond"
        ),
        closure_event_count=len(closure_events),
        closure_open_bond_texts=tuple(
            event.text for event in closure_events if event.kind == "ring_open"
        ),
        atom_text_obligation_count=len(atom_obligations),
        atom_text_modifier_obligation_count=sum(
            len(obligation.modifier_names) for obligation in atom_obligations
        ),
        atom_text_bracket_obligation_count=sum(
            len(obligation.bracket_obligations) for obligation in atom_obligations
        ),
        bond_text_obligation_count=len(bond_obligations),
        bond_token_families=tuple(
            sorted({obligation.token_family for obligation in bond_obligations})
        ),
        marker_slot_count=sum(
            1 for traversal in traversals for event in traversal.events
            if event.marker_slot is not None
        ),
        renderer_input_count=sum(
            1 for traversal in traversals for event in traversal.events
            if event.renderer_input is not None
        ),
        raw_output_count=len(raw_outputs),
        output_count=len(support),
        support=support,
        expected_support_strings_used=False,
    )


def nonstereo_polycyclic_support_from_shared_spine(
    case: object,
) -> SouthStarNonstereoPolycyclicSupportProof:
    mol = parse_smiles(case.source_smiles)
    facts = SouthStarMoleculeFacts.from_mol(mol)
    if not is_nonstereo_polycyclic_ring_traversal_domain(facts):
        raise NotImplementedError(
            "nonstereo-polycyclic unified reference requires one connected "
            "polycyclic molecule, supported atom text, single/double "
            "nonaromatic bond text, and no stereo constraints"
        )

    traversals = mol_to_smiles_enum_s_tree_traversals_for_case(case)
    closure_events = tuple(
        event
        for traversal in traversals
        for event in traversal.events
        if event.ring_closure is not None
    )
    closure_edge_sets = tuple(
        tuple(edge.edge for edge in traversal.connected_graph_plan.closure_edges)
        for traversal in traversals
        if traversal.connected_graph_plan is not None
    )
    closure_labels = tuple(
        label
        for traversal in traversals
        if traversal.connected_graph_plan is not None
        for label in (
            edge.label for edge in traversal.connected_graph_plan.closure_edges
        )
    )
    raw_outputs = tuple(
        render_south_star_tree_traversal(traversal) for traversal in traversals
    )
    support = tuple(dict.fromkeys(raw_outputs))
    closure_edge_count = facts.graph_topology.cyclomatic_number
    if any(len(edge_set) != closure_edge_count for edge_set in closure_edge_sets):
        raise AssertionError("polycyclic proof requires complete closure-edge sets")
    atom_obligations = _atom_text_obligation_summaries(facts)
    bond_obligations = _bond_text_obligation_summaries(facts)
    return SouthStarNonstereoPolycyclicSupportProof(
        case_id=case.case_id,
        atom_count=facts.graph_topology.atom_count,
        bond_count=facts.graph_topology.bond_count,
        ring_count=facts.graph_topology.ring_count,
        cyclomatic_number=facts.graph_topology.cyclomatic_number,
        traversal_count=len(traversals),
        atom_event_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.kind == "atom"
        ),
        bond_event_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.kind == "bond"
        ),
        closure_edge_set_count=len(set(frozenset(edges) for edges in closure_edge_sets)),
        closure_edge_count=closure_edge_count,
        closure_event_count=len(closure_events),
        closure_label_count=len(set(closure_labels)),
        atom_text_obligation_count=len(atom_obligations),
        atom_text_modifier_obligation_count=sum(
            len(obligation.modifier_names) for obligation in atom_obligations
        ),
        atom_text_bracket_obligation_count=sum(
            len(obligation.bracket_obligations) for obligation in atom_obligations
        ),
        bond_text_obligation_count=len(bond_obligations),
        bond_token_families=tuple(
            sorted({obligation.token_family for obligation in bond_obligations})
        ),
        marker_slot_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.marker_slot is not None
        ),
        renderer_input_count=sum(
            1
            for traversal in traversals
            for event in traversal.events
            if event.renderer_input is not None
        ),
        raw_output_count=len(raw_outputs),
        output_count=len(support),
        support=support,
        expected_support_strings_used=False,
    )


def _atom_text_obligation_summaries(
    facts: SouthStarMoleculeFacts,
) -> tuple[SouthStarAtomTextRendererObligationSummary, ...]:
    return tuple(
        _atom_text_obligation_summary(fields)
        for fields in facts.atom_text_facts
    )


def _atom_text_obligation_summary(
    fields: SouthStarAtomTextFields,
) -> SouthStarAtomTextRendererObligationSummary:
    obligation = atom_text_obligation_for_supported_fields(fields)
    return SouthStarAtomTextRendererObligationSummary(
        atom_idx=obligation.atom_idx,
        emitted_text=obligation.emitted_text,
        token_family=obligation.token_family,
        bracket_obligations=obligation.bracket_obligations,
        modifier_names=tuple(
            modifier.modifier_name
            for modifier in atom_text_modifier_obligations(fields)
        ),
    )


def _bond_text_obligation_from_fact(
    fact: SouthStarBondTextFact,
) -> SouthStarBondTextRendererObligationSummary:
    if fact.bond_type == "SINGLE":
        return SouthStarBondTextRendererObligationSummary(
            bond_idx=fact.bond_idx,
            edge=fact.edge,
            bond_type=fact.bond_type,
            emitted_text="",
            token_family="elided_single_bond",
        )
    if fact.bond_type == "DOUBLE":
        return SouthStarBondTextRendererObligationSummary(
            bond_idx=fact.bond_idx,
            edge=fact.edge,
            bond_type=fact.bond_type,
            emitted_text="=",
            token_family="explicit_double_bond",
        )
    if fact.bond_type == "TRIPLE":
        return SouthStarBondTextRendererObligationSummary(
            bond_idx=fact.bond_idx,
            edge=fact.edge,
            bond_type=fact.bond_type,
            emitted_text="#",
            token_family="explicit_triple_bond",
        )
    raise NotImplementedError(
        f"unsupported markerless bond type {fact.bond_type!r}"
    )


def _bond_text_obligation_summaries(
    facts: SouthStarMoleculeFacts,
) -> tuple[SouthStarBondTextRendererObligationSummary, ...]:
    return tuple(
        _bond_text_obligation_from_fact(fact)
        for fact in facts.bond_text_facts
    )
