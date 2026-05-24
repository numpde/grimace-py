"""Producer-free checks for South Star support artifacts."""

from __future__ import annotations

from itertools import product

from .enumeration_trace import TRACE_SCHEMA_VERSION
from .enumeration_trace import build_trace_index
from .proof_terms import sequence_hash
from .support_artifact import SUPPORT_ARTIFACT_SCHEMA_VERSION
from .support_artifact import ArtifactNode
from .support_artifact import SupportArtifact


_LEGAL_EDGES = {
    ("root", "skeleton"),
    ("skeleton", "slot_bundle"),
    ("slot_bundle", "prefix"),
    ("prefix", "csp"),
    ("csp", "stereo_solution"),
    ("stereo_solution", "selected_solution"),
    ("selected_solution", "witness"),
    ("witness", "support_string"),
}


def check_support_artifact(artifact: SupportArtifact) -> None:
    """Check a compiled support artifact without calling producer code."""

    if artifact.header.schema_version != SUPPORT_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unknown support artifact schema version")
    if artifact.traced_support.trace.schema_version != TRACE_SCHEMA_VERSION:
        raise ValueError("unknown trace schema version")
    if artifact.header.facts_hash != facts_hash_from_json(artifact.facts_json):
        raise ValueError("facts hash mismatch")
    if artifact.header.policy_hash != policy_hash_from_json(artifact.policy_json):
        raise ValueError("policy hash mismatch")
    if artifact.header.semantics_hash != sequence_hash((artifact.semantics_name,)):
        raise ValueError("semantics hash mismatch")

    nodes = _node_map(artifact.nodes)
    _check_edges(nodes, artifact.edges)
    _check_domains(nodes, artifact)
    _check_relations(artifact)
    _check_traversal_space(nodes, artifact)
    _check_prefix_spaces(nodes, artifact)
    _check_csp_solution_spaces(nodes, artifact)
    _check_render_programs(nodes, artifact)
    _check_trace_and_manifest(nodes, artifact)

def facts_hash_from_json(data: dict[str, object]) -> str:
    return _json_hash(data)


def policy_hash_from_json(data: dict[str, object]) -> str:
    return _json_hash(data)


def _node_map(nodes: tuple[ArtifactNode, ...]) -> dict[ArtifactNode, ArtifactNode]:
    out: dict[ArtifactNode, ArtifactNode] = {}
    for node in nodes:
        if node in out:
            raise ValueError("duplicate artifact node")
        out[node] = node
    if ArtifactNode(kind="root", key=("root",)) not in out:
        raise ValueError("artifact lacks root node")
    return out


def _check_edges(nodes: dict[ArtifactNode, ArtifactNode], edges) -> None:
    for edge in edges:
        if edge.parent not in nodes:
            raise ValueError("edge references unknown parent node")
        if edge.child not in nodes:
            raise ValueError("edge references unknown child node")
        if (edge.parent.kind, edge.child.kind) not in _LEGAL_EDGES:
            raise ValueError("illegal artifact edge kind")
        if not edge.label:
            raise ValueError("artifact edge label must be nonempty")


def _check_domains(nodes: dict[ArtifactNode, ArtifactNode], artifact: SupportArtifact) -> None:
    seen: set[tuple[ArtifactNode, str]] = set()
    for domain in artifact.domains:
        if domain.owner not in nodes:
            raise ValueError("domain references unknown owner node")
        key = (domain.owner, domain.name)
        if key in seen:
            raise ValueError("duplicate artifact domain")
        seen.add(key)
        if domain.value_hash != _rows_hash(domain.values):
            raise ValueError("domain value hash mismatch")


def _check_relations(artifact: SupportArtifact) -> None:
    seen: set[str] = set()
    for relation in artifact.relations:
        if relation.subject in seen:
            raise ValueError("duplicate artifact relation subject")
        seen.add(relation.subject)
        if relation.row_hash != _rows_hash(relation.allowed_rows):
            raise ValueError("relation row hash mismatch")
        for row in relation.allowed_rows:
            if len(row) != len(relation.scope):
                raise ValueError("relation row arity mismatch")


def _check_traversal_space(
    nodes: dict[ArtifactNode, ArtifactNode],
    artifact: SupportArtifact,
) -> None:
    skeleton_nodes = {
        node.key for node in nodes if node.kind == "skeleton"
    }
    listed = set(artifact.traversal_space.skeleton_keys)
    if skeleton_nodes != listed:
        raise ValueError("skeleton node set disagrees with traversal space")
    decision_keys = {decision.skeleton_key for decision in artifact.traversal_decisions}
    if decision_keys != listed:
        raise ValueError("traversal decision coverage mismatch")
    atom_ids = {
        int(atom["id"]) for atom in artifact.facts_json["atoms"]  # type: ignore[index]
    }
    bond_ids = {
        int(bond["id"]) for bond in artifact.facts_json["bonds"]  # type: ignore[index]
    }
    for decision in artifact.traversal_decisions:
        if not decision.parent_tree_check:
            raise ValueError("traversal parent tree check failed")
        if not decision.edge_partition_check:
            raise ValueError("traversal edge partition check failed")
        if set(decision.atoms_covered) != atom_ids:
            raise ValueError("traversal atom coverage mismatch")
        if set(decision.bonds_covered) != bond_ids:
            raise ValueError("traversal bond coverage mismatch")


def _check_prefix_spaces(
    nodes: dict[ArtifactNode, ArtifactNode],
    artifact: SupportArtifact,
) -> None:
    prefix_nodes_by_skeleton: dict[tuple[object, ...], set[tuple[object, ...]]] = {}
    for node in nodes:
        if node.kind != "prefix":
            continue
        skeleton_key = node.key[0]
        prefix_nodes_by_skeleton.setdefault(skeleton_key, set()).add(node.key[1])

    for space in artifact.prefix_spaces:
        expected = _prefix_cartesian_keys(space)
        listed = set(space.prefix_keys)
        if listed != expected:
            raise ValueError("prefix space is not the Cartesian product")
        if prefix_nodes_by_skeleton.get(space.skeleton_key, set()) != listed:
            raise ValueError("prefix node coverage mismatch")


def _check_csp_solution_spaces(
    nodes: dict[ArtifactNode, ArtifactNode],
    artifact: SupportArtifact,
) -> None:
    relation_by_subject = {relation.subject: relation for relation in artifact.relations}
    solution_nodes = {
        node.key for node in nodes if node.kind == "stereo_solution"
    }
    selected_nodes = {
        node.key for node in nodes if node.kind == "selected_solution"
    }
    for space in artifact.csp_solution_spaces:
        feasible = _feasible_solution_keys(space, relation_by_subject)
        if tuple(sorted(feasible, key=repr)) != tuple(sorted(space.feasible_solution_keys, key=repr)):
            raise ValueError("feasible solution coverage mismatch")
        selected = _selected_solution_keys(space, feasible)
        if tuple(sorted(selected, key=repr)) != tuple(sorted(space.selected_solution_keys, key=repr)):
            raise ValueError("selected solution coverage mismatch")
        rejected = tuple(key for key in feasible if key not in set(selected))
        if tuple(sorted(rejected, key=repr)) != tuple(sorted(space.rejected_solution_keys, key=repr)):
            raise ValueError("rejected solution coverage mismatch")
        for key in feasible:
            if (space.csp_key, key) not in solution_nodes:
                raise ValueError("missing feasible solution node")
        for key in selected:
            if (space.csp_key, key) not in selected_nodes:
                raise ValueError("missing selected solution node")


def _check_render_programs(
    nodes: dict[ArtifactNode, ArtifactNode],
    artifact: SupportArtifact,
) -> None:
    witness_nodes = {node for node in nodes if node.kind == "witness"}
    programs_by_node = {}
    for program in artifact.render_programs:
        if program.witness_node not in witness_nodes:
            raise ValueError("render program references unknown witness node")
        if program.witness_node in programs_by_node:
            raise ValueError("duplicate render program for witness")
        rendered = _render_program_text(program.pieces)
        if rendered != program.rendered:
            raise ValueError("render program mismatch")
        programs_by_node[program.witness_node] = program

    accepted_nodes = {
        ArtifactNode(kind=certificate.node.kind, key=certificate.node.key)
        for certificate in artifact.traced_support.trace.accepted
    }
    if set(programs_by_node) != accepted_nodes:
        raise ValueError("render program coverage mismatch")


def _check_trace_and_manifest(
    nodes: dict[ArtifactNode, ArtifactNode],
    artifact: SupportArtifact,
) -> None:
    trace = artifact.traced_support.trace
    build_trace_index(trace)
    for certificate in trace.accepted:
        if ArtifactNode(kind=certificate.node.kind, key=certificate.node.key) not in nodes:
            raise ValueError("trace acceptance references unreachable node")
    for certificate in trace.rejected:
        if ArtifactNode(kind=certificate.node.kind, key=certificate.node.key) not in nodes:
            raise ValueError("trace rejection references unreachable node")

    accepted_rendered = tuple(certificate.rendered for certificate in trace.accepted)
    support_strings = tuple(dict.fromkeys(accepted_rendered))
    support = artifact.traced_support.support
    manifest = artifact.traced_support.manifest
    if support.strings != support_strings:
        raise ValueError("support strings do not match accepted witness image")
    if manifest.support_hash != sequence_hash(support_strings):
        raise ValueError("support hash mismatch")
    if manifest.witness_hash != sequence_hash(
        certificate.witness_id for certificate in trace.accepted
    ):
        raise ValueError("witness hash mismatch")
    if manifest.support_count != len(support_strings):
        raise ValueError("manifest support count mismatch")
    if manifest.witness_count != len(trace.accepted):
        raise ValueError("manifest witness count mismatch")


def _prefix_cartesian_keys(space) -> set[tuple[object, ...]]:
    atom_keys = tuple(atom for atom, _ in space.atom_text_domains)
    atom_domains = tuple(values for _, values in space.atom_text_domains)
    bond_keys = tuple(slot for slot, _ in space.bond_text_domains)
    bond_domains = tuple(values for _, values in space.bond_text_domains)
    out: set[tuple[object, ...]] = set()
    for atom_values in product(*atom_domains):
        atom_part = tuple(sorted(zip(atom_keys, atom_values, strict=True)))
        for bond_values in product(*bond_domains):
            bond_part = tuple(sorted(zip(bond_keys, bond_values, strict=True)))
            for ring_labels in space.ring_label_assignments:
                out.add((atom_part, bond_part, ring_labels))
    return out


def _feasible_solution_keys(space, relation_by_subject) -> tuple[tuple[object, ...], ...]:
    variable_names = tuple(name for name, _ in space.tetra_domains + space.direction_domains)
    variable_domains = tuple(values for _, values in space.tetra_domains + space.direction_domains)
    relation_subjects = set(space.relation_names)
    relations = [
        relation_by_subject[subject]
        for subject in space.relation_names
    ]
    out: list[tuple[object, ...]] = []
    for values in product(*variable_domains):
        row_by_name = dict(zip(variable_names, values, strict=True))
        if not _satisfies_relations(row_by_name, relations):
            continue
        out.append(_solution_key_from_row(row_by_name))
    return tuple(out)


def _satisfies_relations(row_by_name, relations) -> bool:
    for relation in relations:
        row = tuple(row_by_name[name] for name in relation.scope)
        if row not in relation.allowed_rows:
            return False
    return True


def _solution_key_from_row(row_by_name) -> tuple[object, ...]:
    tetra = tuple(
        sorted(
            (
                int(name.split(":", 1)[1]),
                value,
            )
            for name, value in row_by_name.items()
            if name.startswith("tetra:")
        )
    )
    direction = tuple(
        sorted(
            (
                int(name.split(":", 1)[1]),
                value,
            )
            for name, value in row_by_name.items()
            if name.startswith("direction:")
        )
    )
    support = tuple(carrier for carrier, mark in direction if mark != 0)
    return (-len(support), support, tetra, direction)


def _selected_solution_keys(space, feasible) -> tuple[tuple[object, ...], ...]:
    if space.annotation_mode == "hard":
        return tuple(feasible)
    if space.annotation_mode == "support_maximal":
        return tuple(
            candidate
            for candidate in feasible
            if not any(_support(candidate) < _support(other) for other in feasible)
        )
    if space.annotation_mode == "cardinality_maximal":
        if not feasible:
            return ()
        max_size = max(len(_support(candidate)) for candidate in feasible)
        return tuple(candidate for candidate in feasible if len(_support(candidate)) == max_size)
    if space.annotation_mode == "canonical":
        maximal = _selected_solution_keys(
            _replace_mode(space, "support_maximal"),
            feasible,
        )
        return () if not maximal else (min(maximal),)
    raise ValueError(f"unknown annotation mode: {space.annotation_mode!r}")


def _replace_mode(space, mode: str):
    from dataclasses import replace

    return replace(space, annotation_mode=mode)


def _support(solution_key: tuple[object, ...]) -> set[int]:
    return set(solution_key[1])


def _render_program_text(pieces: tuple[tuple[str, tuple[object, ...]], ...]) -> str:
    out: list[str] = []
    for kind, args in pieces:
        if kind != "literal":
            raise ValueError(f"unsupported render-program piece: {kind!r}")
        if len(args) != 1:
            raise ValueError("literal render-program piece has wrong arity")
        out.append(str(args[0]))
    return "".join(out)


def _rows_hash(rows) -> str:
    return sequence_hash(repr(tuple(row)) for row in rows)


def _json_hash(data: object) -> str:
    import json

    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return sequence_hash((payload,))


__all__ = ("check_support_artifact",)
