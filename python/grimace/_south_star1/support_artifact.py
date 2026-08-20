"""Proof-carrying finite support artifacts for South Star 1."""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import Literal
from typing import Mapping

from .facts import MoleculeFacts
from .policy import SmilesPolicy
from .proof_terms import csp_key
from .proof_terms import prefix_key
from .proof_terms import sequence_hash
from .proof_terms import skeleton_key
from .proof_terms import slot_key
from .proof_terms import stereo_solution_key
from .semantics import ParserSemantics

if TYPE_CHECKING:
    from .skeleton import TraversalSkeleton
    from .support_enumeration import TracedCertifiedSupportImage


SUPPORT_ARTIFACT_SCHEMA_VERSION = 1


@dataclass(frozen=True, slots=True)
class SupportArtifactHeader:
    schema_version: int
    dialect: str
    compiler: str
    facts_hash: str
    policy_hash: str
    semantics_hash: str
    created_by: str | None = None


@dataclass(frozen=True, slots=True)
class ArtifactNode:
    kind: Literal[
        "root",
        "skeleton",
        "slot_bundle",
        "prefix",
        "csp",
        "stereo_solution",
        "selected_solution",
        "witness",
        "support_string",
    ]
    key: tuple[object, ...]


@dataclass(frozen=True, slots=True)
class ArtifactEdge:
    parent: ArtifactNode
    child: ArtifactNode
    label: str


@dataclass(frozen=True, slots=True)
class ArtifactDomain:
    name: str
    owner: ArtifactNode
    values: tuple[tuple[object, ...], ...]
    value_hash: str


@dataclass(frozen=True, slots=True)
class ArtifactRelation:
    name: str
    subject: str
    scope: tuple[str, ...]
    allowed_rows: tuple[tuple[object, ...], ...]
    row_hash: str


@dataclass(frozen=True, slots=True)
class ArtifactRenderProgram:
    witness_node: ArtifactNode
    rendered: str
    pieces: tuple[tuple[str, tuple[object, ...]], ...]


@dataclass(frozen=True, slots=True)
class TraversalDecisionCertificate:
    skeleton_key: tuple[object, ...]
    roots: tuple[int, ...]
    parent_items: tuple[tuple[int, int | None], ...]
    tree_bonds: tuple[int, ...]
    ring_bonds: tuple[int, ...]
    local_event_orders: tuple[tuple[int, tuple[object, ...]], ...]

    atoms_covered: tuple[int, ...]
    bonds_covered: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class TraversalSpaceCertificate:
    component_root_domains: tuple[tuple[int, tuple[int, ...]], ...]
    spanning_tree_edge_sets: tuple[tuple[int, ...], ...]
    local_order_keys_by_skeleton: tuple[
        tuple[tuple[object, ...], tuple[object, ...]], ...
    ]
    skeleton_keys: tuple[tuple[object, ...], ...]


@dataclass(frozen=True, slots=True)
class PrefixSpaceCertificate:
    skeleton_key: tuple[object, ...]
    atom_text_domains: tuple[tuple[int, tuple[str, ...]], ...]
    bond_text_domains: tuple[tuple[int, tuple[str, ...]], ...]
    ring_label_assignments: tuple[tuple[tuple[int, int], ...], ...]
    prefix_keys: tuple[tuple[object, ...], ...]


@dataclass(frozen=True, slots=True)
class CSPSolutionSpaceCertificate:
    csp_key: tuple[object, ...]
    tetra_domains: tuple[tuple[str, tuple[object, ...]], ...]
    direction_domains: tuple[tuple[str, tuple[object, ...]], ...]
    relation_names: tuple[str, ...]
    annotation_mode: str
    feasible_solution_keys: tuple[tuple[object, ...], ...]
    selected_solution_keys: tuple[tuple[object, ...], ...]
    rejected_solution_keys: tuple[tuple[object, ...], ...]


@dataclass(frozen=True, slots=True)
class SupportArtifact:
    header: SupportArtifactHeader

    facts_json: dict[str, object]
    policy_json: dict[str, object]
    semantics_name: str

    nodes: tuple[ArtifactNode, ...]
    edges: tuple[ArtifactEdge, ...]

    domains: tuple[ArtifactDomain, ...]
    relations: tuple[ArtifactRelation, ...]

    render_programs: tuple[ArtifactRenderProgram, ...]

    traversal_space: TraversalSpaceCertificate
    traversal_decisions: tuple[TraversalDecisionCertificate, ...]
    prefix_spaces: tuple[PrefixSpaceCertificate, ...]
    csp_solution_spaces: tuple[CSPSolutionSpaceCertificate, ...]

    traced_support: TracedCertifiedSupportImage


def compile_support_artifact(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    skeletons: Iterable[TraversalSkeleton] | None = None,
) -> SupportArtifact:
    from .graph_index import build_graph_index
    from .skeleton import enumerate_traversal_skeletons
    from .slots import allocate_traversal_slots
    from .stereo_csp import build_stereo_csp
    from .stereo_csp import select_stereo_solutions_with_certificates
    from .stereo_csp import solve_stereo_csp
    from .stereo_witness import enumerate_presentation_prefixes
    from .support_enumeration import enumerate_exhaustive_traced_certified_stereo_support

    facts.validate()
    policy.validate_for_facts(facts)
    facts_json = molecule_facts_to_canonical_json(facts)
    policy_json = policy_to_canonical_json(policy, facts)
    semantics_name = semantics_identity(semantics)

    if skeletons is None:
        skeleton_tuple = enumerate_traversal_skeletons(
            facts=facts,
            index=build_graph_index(facts),
            policy=policy,
        )
    else:
        skeleton_tuple = tuple(skeletons)

    traced = enumerate_exhaustive_traced_certified_stereo_support(
        facts=facts,
        policy=policy,
        semantics=semantics,
        skeletons=skeleton_tuple,
    )

    root = ArtifactNode(kind="root", key=("root",))
    nodes: list[ArtifactNode] = [root]
    edges: list[ArtifactEdge] = []
    domains: list[ArtifactDomain] = []
    relations: list[ArtifactRelation] = []
    render_programs: list[ArtifactRenderProgram] = []
    traversal_decisions: list[TraversalDecisionCertificate] = []
    prefix_spaces: list[PrefixSpaceCertificate] = []
    csp_solution_spaces: list[CSPSolutionSpaceCertificate] = []
    seen_nodes: set[ArtifactNode] = {root}

    def add_node(node: ArtifactNode) -> ArtifactNode:
        if node not in seen_nodes:
            seen_nodes.add(node)
            nodes.append(node)
        return node

    def add_edge(parent: ArtifactNode, child: ArtifactNode, label: str) -> None:
        edges.append(ArtifactEdge(parent=parent, child=child, label=label))

    skeleton_keys: list[tuple[object, ...]] = []
    for skeleton in skeleton_tuple:
        sk_key = skeleton_key(skeleton)
        skeleton_keys.append(sk_key)
        skeleton_node = add_node(ArtifactNode(kind="skeleton", key=sk_key))
        add_edge(root, skeleton_node, "skeleton")

        slots = allocate_traversal_slots(facts, skeleton)
        slot_node = add_node(ArtifactNode(kind="slot_bundle", key=(sk_key, slot_key(slots))))
        add_edge(skeleton_node, slot_node, "slots")
        traversal_decisions.append(
            _traversal_decision_certificate(facts=facts, skeleton=skeleton)
        )

        prefixes = tuple(
            enumerate_presentation_prefixes(
                facts=facts,
                slots=slots,
                policy=policy,
            )
        )
        prefix_spaces.append(
            _prefix_space_certificate(
                facts=facts,
                policy=policy,
                skeleton=skeleton,
                slots=slots,
                prefixes=prefixes,
            )
        )

        for prefix in prefixes:
            pr_key = prefix_key(prefix)
            prefix_node = add_node(ArtifactNode(kind="prefix", key=(sk_key, pr_key)))
            add_edge(slot_node, prefix_node, "prefix")

            csp = build_stereo_csp(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                prefix=prefix,
                policy=policy,
                semantics=semantics,
            )
            csp_node = add_node(ArtifactNode(kind="csp", key=csp_key(skeleton, prefix)))
            add_edge(prefix_node, csp_node, "csp")
            domains.extend(_artifact_domains_for_csp(csp_node, csp))
            csp_relations = _artifact_relations_for_csp(csp_node, csp)
            relations.extend(csp_relations)

            feasible = tuple(solve_stereo_csp(csp))
            selected = select_stereo_solutions_with_certificates(
                csp=csp,
                solutions=feasible,
                mode=policy.annotation_mode,
            )
            selected_keys = tuple(stereo_solution_key(item.solution) for item in selected)
            selected_key_set = set(selected_keys)
            feasible_keys = tuple(stereo_solution_key(solution) for solution in feasible)
            rejected_keys = tuple(
                key for key in feasible_keys if key not in selected_key_set
            )
            csp_solution_spaces.append(
                CSPSolutionSpaceCertificate(
                    csp_key=csp_node.key,
                    tetra_domains=_domain_table(csp.tetra_domains, "tetra"),
                    direction_domains=_domain_table(csp.direction_domains, "direction"),
                    relation_names=tuple(relation.subject for relation in csp_relations),
                    annotation_mode=policy.annotation_mode.value,
                    feasible_solution_keys=feasible_keys,
                    selected_solution_keys=selected_keys,
                    rejected_solution_keys=rejected_keys,
                )
            )

            selected_by_key = {
                stereo_solution_key(item.solution): item
                for item in selected
            }
            for solution in feasible:
                solution_key = stereo_solution_key(solution)
                solution_node = add_node(
                    ArtifactNode(kind="stereo_solution", key=(csp_node.key, solution_key))
                )
                add_edge(csp_node, solution_node, "feasible_solution")

                selected_solution = selected_by_key.get(solution_key)
                if selected_solution is None:
                    continue
                selected_node = add_node(
                    ArtifactNode(kind="selected_solution", key=(csp_node.key, solution_key))
                )
                add_edge(solution_node, selected_node, "selected")
                witness = _certified_for_key(
                    traced=traced,
                    csp_key_value=csp_node.key,
                    solution_key=solution_key,
                )
                witness_node = add_node(
                    ArtifactNode(kind="witness", key=(witness.witness.id,))
                )
                add_edge(selected_node, witness_node, "witness")
                support_node = add_node(
                    ArtifactNode(
                        kind="support_string",
                        key=(witness.witness.rendered,),
                    )
                )
                add_edge(witness_node, support_node, "renders_to")
                render_programs.append(
                    _render_program_for_witness(
                        witness_node=witness_node,
                        skeleton=skeleton,
                        assignment=witness.assignment,
                        slots=slots,
                        rendered=witness.witness.rendered,
                    )
                )

    for certificate in traced.trace.accepted:
        add_node(ArtifactNode(kind=certificate.node.kind, key=certificate.node.key))
    for certificate in traced.trace.rejected:
        add_node(ArtifactNode(kind=certificate.node.kind, key=certificate.node.key))

    traversal_space = TraversalSpaceCertificate(
        component_root_domains=tuple(
            (int(component.id), tuple(int(atom) for atom in component.atoms))
            for component in facts.components
        ),
        spanning_tree_edge_sets=tuple(
            sorted(
                {
                    tuple(sorted(int(bond) for bond in skeleton.tree_bonds))
                    for skeleton in skeleton_tuple
                }
            )
        ),
        local_order_keys_by_skeleton=tuple(
            (
                skeleton_key(skeleton),
                tuple(
                    sorted(
                        (
                            int(atom),
                            tuple(_event_key(event) for event in events),
                        )
                        for atom, events in skeleton.events_at.items()
                    )
                ),
            )
            for skeleton in skeleton_tuple
        ),
        skeleton_keys=tuple(skeleton_keys),
    )
    header = SupportArtifactHeader(
        schema_version=SUPPORT_ARTIFACT_SCHEMA_VERSION,
        dialect="south-star1:ordinary-bounded",
        compiler="south-star-support-artifact-v1",
        facts_hash=facts_hash(facts),
        policy_hash=policy_hash(policy, facts),
        semantics_hash=sequence_hash((semantics_name,)),
    )
    return SupportArtifact(
        header=header,
        facts_json=facts_json,
        policy_json=policy_json,
        semantics_name=semantics_name,
        nodes=tuple(nodes),
        edges=tuple(edges),
        domains=tuple(domains),
        relations=tuple(relations),
        render_programs=tuple(render_programs),
        traversal_space=traversal_space,
        traversal_decisions=tuple(traversal_decisions),
        prefix_spaces=tuple(prefix_spaces),
        csp_solution_spaces=tuple(csp_solution_spaces),
        traced_support=traced,
    )


def molecule_facts_to_canonical_json(facts: MoleculeFacts) -> dict[str, object]:
    return {
        "atoms": [
            {
                "id": int(atom.id),
                "atomic_num": atom.atomic_num,
                "symbol": atom.symbol,
                "isotope": atom.isotope,
                "formal_charge": atom.formal_charge,
                "is_aromatic": atom.is_aromatic,
                "explicit_h_count": atom.explicit_h_count,
                "implicit_h_count": atom.implicit_h_count,
                "no_implicit": atom.no_implicit,
            }
            for atom in facts.atoms
        ],
        "bonds": [
            {
                "id": int(bond.id),
                "a": int(bond.a),
                "b": int(bond.b),
                "order": bond.order.value,
                "is_aromatic": bond.is_aromatic,
                "is_conjugated": bond.is_conjugated,
            }
            for bond in facts.bonds
        ],
        "components": [
            {
                "id": int(component.id),
                "atoms": [int(atom) for atom in component.atoms],
                "bonds": [int(bond) for bond in component.bonds],
            }
            for component in facts.components
        ],
        "stereo": {
            "tetrahedral": [
                {
                    "id": int(site.id),
                    "center": int(site.center),
                    "status": site.status.value,
                    "target": site.target.value,
                    "ligand_occurrences": [
                        int(occurrence) for occurrence in site.ligand_occurrences
                    ],
                    "reference_order": [
                        int(occurrence) for occurrence in site.reference_order
                    ],
                }
                for site in facts.stereo.tetrahedral
            ],
            "directional": [
                {
                    "id": int(site.id),
                    "center_bond": int(site.center_bond),
                    "left_endpoint": int(site.left_endpoint),
                    "right_endpoint": int(site.right_endpoint),
                    "status": site.status.value,
                    "target": site.target.value,
                    "left_ligands": [int(item) for item in site.left_ligands],
                    "right_ligands": [int(item) for item in site.right_ligands],
                    "reference_pair": None
                    if site.reference_pair is None
                    else [int(item) for item in site.reference_pair],
                }
                for site in facts.stereo.directional
            ],
        },
        "ligand_occurrences": [
            {
                "id": int(occurrence.id),
                "site": int(occurrence.site),
                "kind": occurrence.kind.value,
                "atom": None if occurrence.atom is None else int(occurrence.atom),
                "bond": None if occurrence.bond is None else int(occurrence.bond),
                "ordinal": occurrence.ordinal,
            }
            for occurrence in facts.ligand_occurrences
        ],
    }


def policy_to_canonical_json(
    policy: SmilesPolicy,
    facts: MoleculeFacts,
) -> dict[str, object]:
    return {
        "ring_labels": [label.value for label in policy.ring_labels],
        "annotation_mode": policy.annotation_mode.value,
        "least_free_ring_labels": policy.least_free_ring_labels,
        "atom_text_domains": [
            {
                "atom": int(domain.atom),
                "choices": [
                    {
                        "name": choice.name,
                        "text_by_tetra": [
                            [token.value, text]
                            for token, text in choice.text_by_tetra
                        ],
                    }
                    for choice in domain.choices
                ],
            }
            for domain in policy.atom_text_domains
        ],
        "bond_text_domains": [
            {
                "bond": int(domain.bond),
                "slot_kind": domain.slot_kind,
                "choices": [
                    {
                        "name": choice.name,
                        "base_text": choice.base_text,
                        "permits_direction": choice.permits_direction,
                    }
                    for choice in domain.choices
                ],
            }
            for domain in policy.bond_text_domains
        ],
    }


def semantics_identity(semantics: ParserSemantics) -> str:
    name = semantics.__class__.__name__
    if name == "OrdinarySmilesSemantics":
        return "OrdinarySmilesSemantics:south-star1:v0"
    raise ValueError(f"unsupported semantics object for artifact: {name!r}")


def facts_hash(facts: MoleculeFacts) -> str:
    return _json_hash(molecule_facts_to_canonical_json(facts))


def policy_hash(policy: SmilesPolicy, facts: MoleculeFacts) -> str:
    return _json_hash(policy_to_canonical_json(policy, facts))


def support_artifact_digest(artifact: SupportArtifact) -> str:
    return _json_hash(support_artifact_to_jsonable(artifact))


def support_artifact_to_jsonable(artifact: SupportArtifact) -> dict[str, object]:
    from .support_enumeration import traced_certified_support_to_jsonable

    return {
        "header": _header_to_jsonable(artifact.header),
        "facts_json": artifact.facts_json,
        "policy_json": artifact.policy_json,
        "semantics_name": artifact.semantics_name,
        "nodes": [_node_to_jsonable(node) for node in artifact.nodes],
        "edges": [_edge_to_jsonable(edge) for edge in artifact.edges],
        "domains": [_domain_to_jsonable(domain) for domain in artifact.domains],
        "relations": [
            _relation_to_jsonable(relation)
            for relation in artifact.relations
        ],
        "render_programs": [
            _render_program_to_jsonable(program)
            for program in artifact.render_programs
        ],
        "traversal_space": _traversal_space_to_jsonable(artifact.traversal_space),
        "traversal_decisions": [
            _traversal_decision_to_jsonable(decision)
            for decision in artifact.traversal_decisions
        ],
        "prefix_spaces": [
            _prefix_space_to_jsonable(space)
            for space in artifact.prefix_spaces
        ],
        "csp_solution_spaces": [
            _csp_solution_space_to_jsonable(space)
            for space in artifact.csp_solution_spaces
        ],
        "traced_support": traced_certified_support_to_jsonable(
            artifact.traced_support,
        ),
    }


def support_artifact_from_jsonable(data: Mapping[str, object]) -> SupportArtifact:
    from .support_enumeration import traced_certified_support_from_jsonable

    return SupportArtifact(
        header=_header_from_jsonable(_require_mapping(data["header"])),
        facts_json=dict(_require_mapping(data["facts_json"])),
        policy_json=dict(_require_mapping(data["policy_json"])),
        semantics_name=str(data["semantics_name"]),
        nodes=tuple(_node_from_jsonable(item) for item in _require_list(data["nodes"])),
        edges=tuple(_edge_from_jsonable(item) for item in _require_list(data["edges"])),
        domains=tuple(
            _domain_from_jsonable(item)
            for item in _require_list(data["domains"])
        ),
        relations=tuple(
            _relation_from_jsonable(item)
            for item in _require_list(data["relations"])
        ),
        render_programs=tuple(
            _render_program_from_jsonable(item)
            for item in _require_list(data["render_programs"])
        ),
        traversal_space=_traversal_space_from_jsonable(
            _require_mapping(data["traversal_space"])
        ),
        traversal_decisions=tuple(
            _traversal_decision_from_jsonable(item)
            for item in _require_list(data["traversal_decisions"])
        ),
        prefix_spaces=tuple(
            _prefix_space_from_jsonable(item)
            for item in _require_list(data["prefix_spaces"])
        ),
        csp_solution_spaces=tuple(
            _csp_solution_space_from_jsonable(item)
            for item in _require_list(data["csp_solution_spaces"])
        ),
        traced_support=traced_certified_support_from_jsonable(
            _require_mapping(data["traced_support"])
        ),
    )


def support_artifact_from_jsonable_checked(
    data: Mapping[str, object],
) -> SupportArtifact:
    from .support_artifact_schema import validate_support_artifact_jsonable
    from .support_artifact_schema import validate_support_artifact_schema

    validate_support_artifact_jsonable(data)
    artifact = support_artifact_from_jsonable(data)
    validate_support_artifact_schema(artifact)
    return artifact


def _certified_for_key(
    *,
    traced: TracedCertifiedSupportImage,
    csp_key_value: tuple[object, ...],
    solution_key: tuple[object, ...],
):
    for certified in traced.certified_witnesses:
        cert = certified.certificate.stereo_solution.annotation_certificate
        if cert.selected_solution_key != solution_key:
            continue
        if certified.certificate.skeleton_key != csp_key_value[0]:
            continue
        if certified.certificate.prefix_key != csp_key_value[1]:
            continue
        return certified
    raise ValueError("selected solution has no certified witness")


def _traversal_decision_certificate(
    *,
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
) -> TraversalDecisionCertificate:
    atom_ids = tuple(int(atom.id) for atom in facts.atoms)
    bond_ids = tuple(int(bond.id) for bond in facts.bonds)
    return TraversalDecisionCertificate(
        skeleton_key=skeleton_key(skeleton),
        roots=tuple(int(root) for root in skeleton.roots),
        parent_items=tuple(
            sorted(
                (int(atom), None if parent is None else int(parent))
                for atom, parent in skeleton.parent.items()
            )
        ),
        tree_bonds=tuple(sorted(int(bond) for bond in skeleton.tree_bonds)),
        ring_bonds=tuple(sorted(int(bond) for bond in skeleton.ring_bonds)),
        local_event_orders=tuple(
            sorted(
                (
                    int(atom),
                    tuple(_event_key(event) for event in events),
                )
                for atom, events in skeleton.events_at.items()
            )
        ),
        atoms_covered=atom_ids,
        bonds_covered=bond_ids,
    )


def _event_key(event: object) -> tuple[object, ...]:
    if event.__class__.__name__ == "ChildEvent":
        return (
            "child",
            int(event.bond),
            int(event.parent),
            int(event.child),
            event.role.value,
        )
    if event.__class__.__name__ == "RingEvent":
        return (
            "ring",
            int(event.bond),
            int(event.atom),
            int(event.other_atom),
        )
    raise TypeError(event)


def _render_program_for_witness(
    *,
    witness_node: ArtifactNode,
    skeleton,
    slots,
    assignment,
    rendered: str,
) -> ArtifactRenderProgram:
    tree_slot_by_bond = {
        slot.bond: slot
        for slot in slots.bond_slots
        if slot.kind.value == "tree"
    }
    ring_slot_by_endpoint = {
        (slot.bond, slot.written_from): slot
        for slot in slots.bond_slots
        if slot.kind.value == "ring_endpoint"
    }
    endpoint_by_id = {endpoint.id: endpoint for endpoint in slots.ring_endpoints}
    carrier_by_bond_slot = {slot.bond_slot: slot for slot in slots.carrier_slots}
    pieces: list[tuple[str, tuple[object, ...]]] = []

    def append_atom(atom) -> None:
        pieces.append(
            (
                "atom",
                (
                    int(atom),
                    assignment.atom_text[atom].name,
                    assignment.tetra_tokens[atom].value,
                ),
            )
        )
        for event in skeleton.events_at[atom]:
            if event.__class__.__name__ == "RingEvent":
                slot = ring_slot_by_endpoint[(event.bond, event.atom)]
                if slot.ring_endpoint is None:
                    raise ValueError(f"ring slot lacks endpoint id: {slot.id!r}")
                pieces.append(_bond_piece(slot, assignment, carrier_by_bond_slot))
                pieces.append(
                    (
                        "ring_label",
                        (
                            int(slot.ring_endpoint),
                            assignment.ring_labels[
                                endpoint_by_id[slot.ring_endpoint].id
                            ].value,
                        ),
                    )
                )
                continue

            slot = tree_slot_by_bond[event.bond]
            if event.role.value == "branch":
                pieces.append(("branch_open", ()))
                pieces.append(_bond_piece(slot, assignment, carrier_by_bond_slot))
                append_atom(event.child)
                pieces.append(("branch_close", ()))
                continue

            pieces.append(_bond_piece(slot, assignment, carrier_by_bond_slot))
            append_atom(event.child)

    for index, root in enumerate(skeleton.roots):
        if index:
            pieces.append(("dot", ()))
        append_atom(root)

    return ArtifactRenderProgram(
        witness_node=witness_node,
        rendered=rendered,
        pieces=tuple(pieces),
    )


def _bond_piece(slot, assignment, carrier_by_bond_slot) -> tuple[str, tuple[object, ...]]:
    carrier = carrier_by_bond_slot[slot.id]
    return (
        "bond",
        (
            int(slot.id),
            int(slot.bond),
            slot.kind.value,
            assignment.bond_text[slot.id].name,
            assignment.direction_marks[carrier.id].value,
        ),
    )


def _prefix_space_certificate(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    skeleton: TraversalSkeleton,
    slots,
    prefixes,
) -> PrefixSpaceCertificate:
    return PrefixSpaceCertificate(
        skeleton_key=skeleton_key(skeleton),
        atom_text_domains=tuple(
            (int(atom.id), tuple(choice.name for choice in policy.atom_text_domain(facts, atom.id)))
            for atom in facts.atoms
        ),
        bond_text_domains=tuple(
            (
                int(slot.id),
                tuple(
                    choice.name
                    for choice in policy.bond_text_domain(
                        facts,
                        slot.bond,
                        slot_kind=slot.kind.value,
                    )
                ),
            )
            for slot in slots.bond_slots
        ),
        ring_label_assignments=tuple(
            tuple(
                sorted((int(endpoint), label.value) for endpoint, label in prefix.ring_labels.items())
            )
            for prefix in prefixes
        ),
        prefix_keys=tuple(prefix_key(prefix) for prefix in prefixes),
    )


def _artifact_domains_for_csp(
    csp_node: ArtifactNode,
    csp,
) -> tuple[ArtifactDomain, ...]:
    out: list[ArtifactDomain] = []
    for atom, values in sorted(csp.tetra_domains.items(), key=lambda item: int(item[0])):
        domain_values = tuple((token.value,) for token in values)
        out.append(
            ArtifactDomain(
                name=f"tetra:{int(atom)}",
                owner=csp_node,
                values=domain_values,
                value_hash=_rows_hash(domain_values),
            )
        )
    for carrier, values in sorted(
        csp.direction_domains.items(),
        key=lambda item: int(item[0]),
    ):
        domain_values = tuple((mark.value,) for mark in values)
        out.append(
            ArtifactDomain(
                name=f"direction:{int(carrier)}",
                owner=csp_node,
                values=domain_values,
                value_hash=_rows_hash(domain_values),
            )
        )
    return tuple(out)


def _artifact_relations_for_csp(
    csp_node: ArtifactNode,
    csp,
) -> tuple[ArtifactRelation, ...]:
    out: list[ArtifactRelation] = []
    for relation in csp.tetra_relations:
        rows = tuple((token.value,) for token in relation.allowed_tokens)
        out.append(
            ArtifactRelation(
                name="tetra_site",
                subject=f"{repr(csp_node.key)}::site:{int(relation.site)}",
                scope=(f"tetra:{int(relation.center)}",),
                allowed_rows=rows,
                row_hash=_rows_hash(rows),
            )
        )
    for relation in csp.mark_relations():
        rows = tuple(
            tuple(mark.value for mark in row)
            for row in sorted(relation.allowed_rows, key=repr)
        )
        out.append(
            ArtifactRelation(
                name=relation.name,
                subject=f"{repr(csp_node.key)}::{relation.subject}",
                scope=tuple(f"direction:{int(carrier)}" for carrier in relation.scope),
                allowed_rows=rows,
                row_hash=_rows_hash(rows),
            )
        )
    return tuple(out)


def _domain_table(domains: Mapping[object, tuple[object, ...]], prefix: str):
    return tuple(
        (
            f"{prefix}:{int(key)}",
            tuple(_enum_value(value) for value in values),
        )
        for key, values in sorted(domains.items(), key=lambda item: int(item[0]))
    )


def _enum_value(value: object) -> object:
    return getattr(value, "value", value)


def _rows_hash(rows: Iterable[tuple[object, ...]]) -> str:
    return sequence_hash(repr(tuple(rows_)) for rows_ in rows)


def _json_hash(data: object) -> str:
    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return sequence_hash((payload,))


def _header_to_jsonable(header: SupportArtifactHeader) -> dict[str, object]:
    return {
        "schema_version": header.schema_version,
        "dialect": header.dialect,
        "compiler": header.compiler,
        "facts_hash": header.facts_hash,
        "policy_hash": header.policy_hash,
        "semantics_hash": header.semantics_hash,
        "created_by": header.created_by,
    }


def _header_from_jsonable(data: Mapping[str, object]) -> SupportArtifactHeader:
    return SupportArtifactHeader(
        schema_version=int(data["schema_version"]),
        dialect=str(data["dialect"]),
        compiler=str(data["compiler"]),
        facts_hash=str(data["facts_hash"]),
        policy_hash=str(data["policy_hash"]),
        semantics_hash=str(data["semantics_hash"]),
        created_by=None if data["created_by"] is None else str(data["created_by"]),
    )


def _node_to_jsonable(node: ArtifactNode) -> dict[str, object]:
    return {"kind": node.kind, "key": _jsonable(node.key)}


def _node_from_jsonable(data: object) -> ArtifactNode:
    mapping = _require_mapping(data)
    return ArtifactNode(
        kind=str(mapping["kind"]),  # type: ignore[arg-type]
        key=_tuple_from_jsonable(mapping["key"]),
    )


def _edge_to_jsonable(edge: ArtifactEdge) -> dict[str, object]:
    return {
        "parent": _node_to_jsonable(edge.parent),
        "child": _node_to_jsonable(edge.child),
        "label": edge.label,
    }


def _edge_from_jsonable(data: object) -> ArtifactEdge:
    mapping = _require_mapping(data)
    return ArtifactEdge(
        parent=_node_from_jsonable(mapping["parent"]),
        child=_node_from_jsonable(mapping["child"]),
        label=str(mapping["label"]),
    )


def _domain_to_jsonable(domain: ArtifactDomain) -> dict[str, object]:
    return {
        "name": domain.name,
        "owner": _node_to_jsonable(domain.owner),
        "values": _jsonable(domain.values),
        "value_hash": domain.value_hash,
    }


def _domain_from_jsonable(data: object) -> ArtifactDomain:
    mapping = _require_mapping(data)
    return ArtifactDomain(
        name=str(mapping["name"]),
        owner=_node_from_jsonable(mapping["owner"]),
        values=tuple(_tuple_from_jsonable(item) for item in _require_list(mapping["values"])),
        value_hash=str(mapping["value_hash"]),
    )


def _relation_to_jsonable(relation: ArtifactRelation) -> dict[str, object]:
    return {
        "name": relation.name,
        "subject": relation.subject,
        "scope": list(relation.scope),
        "allowed_rows": _jsonable(relation.allowed_rows),
        "row_hash": relation.row_hash,
    }


def _relation_from_jsonable(data: object) -> ArtifactRelation:
    mapping = _require_mapping(data)
    return ArtifactRelation(
        name=str(mapping["name"]),
        subject=str(mapping["subject"]),
        scope=tuple(str(item) for item in _require_list(mapping["scope"])),
        allowed_rows=tuple(
            _tuple_from_jsonable(item)
            for item in _require_list(mapping["allowed_rows"])
        ),
        row_hash=str(mapping["row_hash"]),
    )


def _render_program_to_jsonable(program: ArtifactRenderProgram) -> dict[str, object]:
    return {
        "witness_node": _node_to_jsonable(program.witness_node),
        "rendered": program.rendered,
        "pieces": _jsonable(program.pieces),
    }


def _render_program_from_jsonable(data: object) -> ArtifactRenderProgram:
    mapping = _require_mapping(data)
    return ArtifactRenderProgram(
        witness_node=_node_from_jsonable(mapping["witness_node"]),
        rendered=str(mapping["rendered"]),
        pieces=tuple(
            (str(item[0]), _tuple_from_jsonable(item[1]))
            for item in _require_list(mapping["pieces"])
        ),
    )


def _traversal_space_to_jsonable(cert: TraversalSpaceCertificate) -> dict[str, object]:
    return _dataclass_json(cert)


def _traversal_space_from_jsonable(data: Mapping[str, object]) -> TraversalSpaceCertificate:
    return TraversalSpaceCertificate(
        component_root_domains=tuple(
            (int(item[0]), tuple(int(atom) for atom in item[1]))
            for item in _require_list(data["component_root_domains"])
        ),
        spanning_tree_edge_sets=tuple(
            tuple(int(bond) for bond in item)
            for item in _require_list(data["spanning_tree_edge_sets"])
        ),
        local_order_keys_by_skeleton=tuple(
            (_tuple_from_jsonable(item[0]), _tuple_from_jsonable(item[1]))
            for item in _require_list(data["local_order_keys_by_skeleton"])
        ),
        skeleton_keys=tuple(
            _tuple_from_jsonable(item)
            for item in _require_list(data["skeleton_keys"])
        ),
    )


def _traversal_decision_to_jsonable(cert: TraversalDecisionCertificate) -> dict[str, object]:
    return _dataclass_json(cert)


def _traversal_decision_from_jsonable(data: object) -> TraversalDecisionCertificate:
    mapping = _require_mapping(data)
    return TraversalDecisionCertificate(
        skeleton_key=_tuple_from_jsonable(mapping["skeleton_key"]),
        roots=tuple(int(item) for item in _require_list(mapping["roots"])),
        parent_items=tuple(
            (int(item[0]), None if item[1] is None else int(item[1]))
            for item in _require_list(mapping["parent_items"])
        ),
        tree_bonds=tuple(int(item) for item in _require_list(mapping["tree_bonds"])),
        ring_bonds=tuple(int(item) for item in _require_list(mapping["ring_bonds"])),
        local_event_orders=tuple(
            (int(item[0]), _tuple_from_jsonable(item[1]))
            for item in _require_list(mapping["local_event_orders"])
        ),
        atoms_covered=tuple(int(item) for item in _require_list(mapping["atoms_covered"])),
        bonds_covered=tuple(int(item) for item in _require_list(mapping["bonds_covered"])),
    )


def _prefix_space_to_jsonable(cert: PrefixSpaceCertificate) -> dict[str, object]:
    return _dataclass_json(cert)


def _prefix_space_from_jsonable(data: object) -> PrefixSpaceCertificate:
    mapping = _require_mapping(data)
    return PrefixSpaceCertificate(
        skeleton_key=_tuple_from_jsonable(mapping["skeleton_key"]),
        atom_text_domains=tuple(
            (int(item[0]), tuple(str(value) for value in item[1]))
            for item in _require_list(mapping["atom_text_domains"])
        ),
        bond_text_domains=tuple(
            (int(item[0]), tuple(str(value) for value in item[1]))
            for item in _require_list(mapping["bond_text_domains"])
        ),
        ring_label_assignments=tuple(
            tuple((int(pair[0]), int(pair[1])) for pair in item)
            for item in _require_list(mapping["ring_label_assignments"])
        ),
        prefix_keys=tuple(
            _tuple_from_jsonable(item)
            for item in _require_list(mapping["prefix_keys"])
        ),
    )


def _csp_solution_space_to_jsonable(cert: CSPSolutionSpaceCertificate) -> dict[str, object]:
    return _dataclass_json(cert)


def _csp_solution_space_from_jsonable(data: object) -> CSPSolutionSpaceCertificate:
    mapping = _require_mapping(data)
    return CSPSolutionSpaceCertificate(
        csp_key=_tuple_from_jsonable(mapping["csp_key"]),
        tetra_domains=tuple(
            (str(item[0]), _tuple_from_jsonable(item[1]))
            for item in _require_list(mapping["tetra_domains"])
        ),
        direction_domains=tuple(
            (str(item[0]), _tuple_from_jsonable(item[1]))
            for item in _require_list(mapping["direction_domains"])
        ),
        relation_names=tuple(str(item) for item in _require_list(mapping["relation_names"])),
        annotation_mode=str(mapping["annotation_mode"]),
        feasible_solution_keys=tuple(
            _tuple_from_jsonable(item)
            for item in _require_list(mapping["feasible_solution_keys"])
        ),
        selected_solution_keys=tuple(
            _tuple_from_jsonable(item)
            for item in _require_list(mapping["selected_solution_keys"])
        ),
        rejected_solution_keys=tuple(
            _tuple_from_jsonable(item)
            for item in _require_list(mapping["rejected_solution_keys"])
        ),
    )


def _dataclass_json(value: object) -> dict[str, object]:
    return {
        key: _jsonable(getattr(value, key))
        for key in value.__dataclass_fields__  # type: ignore[attr-defined]
    }


def _jsonable(value: object) -> object:
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    return value


def _tuple_from_jsonable(value: object) -> tuple[object, ...]:
    if not isinstance(value, list):
        raise TypeError(f"expected list for tuple: {value!r}")
    return tuple(
        _tuple_from_jsonable(item) if isinstance(item, list) else item
        for item in value
    )


def _require_list(value: object) -> list:
    if not isinstance(value, list):
        raise TypeError(f"expected list: {value!r}")
    return value


def _require_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping: {value!r}")
    return value


__all__ = (
    "ArtifactDomain",
    "ArtifactEdge",
    "ArtifactNode",
    "ArtifactRelation",
    "ArtifactRenderProgram",
    "CSPSolutionSpaceCertificate",
    "PrefixSpaceCertificate",
    "SUPPORT_ARTIFACT_SCHEMA_VERSION",
    "SupportArtifact",
    "SupportArtifactHeader",
    "TraversalDecisionCertificate",
    "TraversalSpaceCertificate",
    "compile_support_artifact",
    "facts_hash",
    "molecule_facts_to_canonical_json",
    "policy_hash",
    "policy_to_canonical_json",
    "semantics_identity",
    "support_artifact_digest",
    "support_artifact_from_jsonable",
    "support_artifact_from_jsonable_checked",
    "support_artifact_to_jsonable",
)
