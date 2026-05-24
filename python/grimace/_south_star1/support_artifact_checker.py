"""Producer-free checks for South Star support artifacts."""

from __future__ import annotations

from collections import Counter
from itertools import product
from typing import Mapping

from .enumeration_trace import TRACE_SCHEMA_VERSION
from .enumeration_trace import build_trace_index
from .finite_space_checker import expected_prefix_keys_from_policy_product
from .finite_space_checker import expected_skeleton_keys_from_traversal_grammar
from .finite_space_checker import expected_slot_bundle_key_from_traversal_decision
from .finite_space_checker import enumerate_ring_label_assignments_from_slots
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
    _check_witness_relation_certificates(artifact)
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
    expected = expected_skeleton_keys_from_traversal_grammar(
        facts_json=artifact.facts_json,
        policy_json=artifact.policy_json,
    )
    if frozenset(listed) != expected:
        raise ValueError("traversal grammar skeleton coverage mismatch")
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
        _check_traversal_decision(decision, artifact)
        if set(decision.atoms_covered) != atom_ids:
            raise ValueError("traversal atom coverage mismatch")
        if set(decision.bonds_covered) != bond_ids:
            raise ValueError("traversal bond coverage mismatch")
        if tuple(decision.tree_bonds) not in artifact.traversal_space.spanning_tree_edge_sets:
            raise ValueError("traversal tree set is outside spanning-tree table")
    local_order_by_skeleton = dict(artifact.traversal_space.local_order_keys_by_skeleton)
    for decision in artifact.traversal_decisions:
        if local_order_by_skeleton.get(decision.skeleton_key) != decision.local_event_orders:
            raise ValueError("traversal local-order table mismatch")


def _check_traversal_decision(decision, artifact: SupportArtifact) -> None:
    atom_ids = {
        int(atom["id"]) for atom in artifact.facts_json["atoms"]  # type: ignore[index]
    }
    bond_by_id = _bond_by_id(artifact.facts_json)
    bond_ids = set(bond_by_id)
    components = tuple(_require_mapping(component) for component in artifact.facts_json["components"])  # type: ignore[index]
    parent = dict(decision.parent_items)

    if set(parent) != atom_ids:
        raise ValueError("traversal parent map atom coverage mismatch")
    if set(decision.roots) != {atom for atom, item in parent.items() if item is None}:
        raise ValueError("traversal roots disagree with parent map")
    if len(decision.roots) != len(components):
        raise ValueError("traversal roots are not one per component")
    for component in components:
        atoms = {int(atom) for atom in _require_list(component["atoms"])}
        roots = [root for root in decision.roots if root in atoms]
        if len(roots) != 1:
            raise ValueError("traversal component root coverage mismatch")

    parent_bonds: set[int] = set()
    for atom, parent_atom in parent.items():
        if parent_atom is None:
            continue
        bond = _bond_between(bond_by_id, atom, parent_atom)
        if bond is None:
            raise ValueError("traversal parent edge is not a graph edge")
        parent_bonds.add(bond)
        _check_parent_reaches_root(parent, atom)

    tree_bonds = set(decision.tree_bonds)
    ring_bonds = set(decision.ring_bonds)
    if tree_bonds != parent_bonds:
        raise ValueError("traversal tree bonds do not match parent edges")
    if tree_bonds & ring_bonds:
        raise ValueError("traversal tree/ring partition overlaps")
    if tree_bonds | ring_bonds != bond_ids:
        raise ValueError("traversal tree/ring partition misses graph bonds")

    _check_local_events(
        decision=decision,
        parent=parent,
        bond_by_id=bond_by_id,
        tree_bonds=tree_bonds,
        ring_bonds=ring_bonds,
    )


def _check_parent_reaches_root(
    parent: Mapping[int, int | None],
    atom: int,
) -> None:
    seen: set[int] = set()
    cursor: int | None = atom
    while cursor is not None:
        if cursor in seen:
            raise ValueError("traversal parent map contains a cycle")
        seen.add(cursor)
        cursor = parent[cursor]


def _check_local_events(
    *,
    decision,
    parent: Mapping[int, int | None],
    bond_by_id: Mapping[int, tuple[int, int]],
    tree_bonds: set[int],
    ring_bonds: set[int],
) -> None:
    child_events: list[tuple[int, int, int]] = []
    ring_events: list[tuple[int, int, int]] = []
    for atom, events in decision.local_event_orders:
        if atom not in parent:
            raise ValueError("traversal local event references unknown atom")
        for event in events:
            if not isinstance(event, tuple) or not event:
                raise ValueError("traversal local event is malformed")
            if event[0] == "child":
                _, bond, event_parent, child, role = event
                bond = int(bond)
                event_parent = int(event_parent)
                child = int(child)
                if role not in {"branch", "continuation"}:
                    raise ValueError("traversal child event has unknown role")
                if atom != event_parent:
                    raise ValueError("traversal child event is stored at wrong atom")
                if bond not in tree_bonds:
                    raise ValueError("traversal child event uses non-tree bond")
                if _sorted_pair(bond_by_id[bond]) != _sorted_pair((event_parent, child)):
                    raise ValueError("traversal child event uses nonincident bond")
                if parent.get(child) != event_parent:
                    raise ValueError("traversal child event disagrees with parent map")
                child_events.append((bond, event_parent, child))
                continue
            if event[0] == "ring":
                _, bond, event_atom, other_atom = event
                bond = int(bond)
                event_atom = int(event_atom)
                other_atom = int(other_atom)
                if atom != event_atom:
                    raise ValueError("traversal ring event is stored at wrong atom")
                if bond not in ring_bonds:
                    raise ValueError("traversal ring event uses non-ring bond")
                if _sorted_pair(bond_by_id[bond]) != _sorted_pair((event_atom, other_atom)):
                    raise ValueError("traversal ring event uses nonincident bond")
                ring_events.append((bond, event_atom, other_atom))
                continue
            raise ValueError("traversal local event has unknown kind")

    expected_children = {
        (bond, event_parent, child)
        for child, event_parent in parent.items()
        if event_parent is not None
        for bond in (_bond_between(bond_by_id, child, event_parent),)
        if bond is not None
    }
    if Counter(child_events) != Counter(expected_children):
        raise ValueError("traversal child event coverage mismatch")
    expected_ring_events = []
    for bond in ring_bonds:
        left, right = bond_by_id[bond]
        expected_ring_events.append((bond, left, right))
        expected_ring_events.append((bond, right, left))
    if Counter(ring_events) != Counter(expected_ring_events):
        raise ValueError("traversal ring endpoint coverage mismatch")


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
    slot_bundle_by_skeleton = _slot_bundle_by_skeleton(nodes)
    decision_by_key = {
        decision.skeleton_key: decision
        for decision in artifact.traversal_decisions
    }

    for space in artifact.prefix_spaces:
        slot_bundle_key = slot_bundle_by_skeleton.get(space.skeleton_key)
        if slot_bundle_key is None:
            raise ValueError("prefix space lacks slot bundle")
        decision = decision_by_key.get(space.skeleton_key)
        if decision is None:
            raise ValueError("prefix space lacks traversal decision")
        expected_slot_key = expected_slot_bundle_key_from_traversal_decision(
            facts_json=artifact.facts_json,
            roots=decision.roots,
            local_event_orders=decision.local_event_orders,
        )
        if slot_bundle_key != expected_slot_key:
            raise ValueError("slot bundle is not induced by skeleton")
        _check_prefix_domains_against_policy(space, artifact)
        expected_ring_labels = enumerate_ring_label_assignments_from_slots(
            ring_endpoints=tuple(slot_bundle_key[2]),
            ring_labels=tuple(
                int(label)
                for label in _require_list(artifact.policy_json["ring_labels"])
            ),
            least_free=bool(artifact.policy_json["least_free_ring_labels"]),
        )
        if set(space.ring_label_assignments) != set(expected_ring_labels):
            raise ValueError("ring label assignment space mismatch")
        expected = expected_prefix_keys_from_policy_product(
            facts_json=artifact.facts_json,
            policy_json=artifact.policy_json,
            skeleton_key=space.skeleton_key,
            slot_bundle_key=slot_bundle_key,
        )
        listed = set(space.prefix_keys)
        if listed != expected:
            raise ValueError("prefix space is not the policy product")
        if prefix_nodes_by_skeleton.get(space.skeleton_key, set()) != listed:
            raise ValueError("prefix node coverage mismatch")


def _slot_bundle_by_skeleton(
    nodes: dict[ArtifactNode, ArtifactNode],
) -> dict[tuple[object, ...], tuple[object, ...]]:
    out: dict[tuple[object, ...], tuple[object, ...]] = {}
    for node in nodes:
        if node.kind != "slot_bundle":
            continue
        skeleton_key = node.key[0]
        if skeleton_key in out:
            raise ValueError("duplicate slot bundle for skeleton")
        out[skeleton_key] = node.key[1]
    return out


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
        rendered = _render_program_text(
            program.pieces,
            policy_json=artifact.policy_json,
        )
        if rendered != program.rendered:
            raise ValueError("render program mismatch")
        programs_by_node[program.witness_node] = program

    accepted_nodes = {
        ArtifactNode(kind=certificate.node.kind, key=certificate.node.key)
        for certificate in artifact.traced_support.trace.accepted
    }
    if set(programs_by_node) != accepted_nodes:
        raise ValueError("render program coverage mismatch")


def _check_witness_relation_certificates(artifact: SupportArtifact) -> None:
    relation_by_subject = {relation.subject: relation for relation in artifact.relations}
    csp_spaces = {space.csp_key: space for space in artifact.csp_solution_spaces}
    for certified in artifact.traced_support.certified_witnesses:
        witness_cert = certified.certificate
        csp_key = (witness_cert.skeleton_key, witness_cert.prefix_key)
        space = csp_spaces.get(csp_key)
        if space is None:
            raise ValueError("witness certificate references unknown CSP")
        selected_key = (
            witness_cert
            .stereo_solution
            .annotation_certificate
            .selected_solution_key
        )
        if selected_key not in space.selected_solution_keys:
            raise ValueError("witness certificate solution is not selected")

        expected_subjects = tuple(
            _artifact_relation_suffix(subject)
            for subject in space.relation_names
        )
        actual_subjects = tuple(
            relation.subject
            for relation in witness_cert.stereo_solution.relation_certificates
        )
        if actual_subjects != expected_subjects:
            raise ValueError("witness relation certificate coverage mismatch")

        for cert_relation, subject in zip(
            witness_cert.stereo_solution.relation_certificates,
            space.relation_names,
            strict=True,
        ):
            artifact_relation = relation_by_subject[subject]
            if _certificate_relation_row(cert_relation) not in artifact_relation.allowed_rows:
                raise ValueError("witness relation row is outside artifact relation")


def _artifact_relation_suffix(subject: str) -> str:
    try:
        _, suffix = subject.split("::", 1)
    except ValueError as exc:
        raise ValueError("artifact relation subject lacks CSP prefix") from exc
    return suffix


def _certificate_relation_row(relation) -> tuple[object, ...]:
    detail = relation.detail
    if "row" in detail:
        row_index = detail.index("row") + 1
        return tuple(detail[row_index])
    if "token" in detail:
        token_index = detail.index("token") + 1
        return (detail[token_index],)
    raise ValueError("witness relation certificate lacks row detail")


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


def _render_program_text(
    pieces: tuple[tuple[str, tuple[object, ...]], ...],
    *,
    policy_json: dict[str, object],
) -> str:
    out: list[str] = []
    for kind, args in pieces:
        if kind == "atom":
            if len(args) != 3:
                raise ValueError("atom render-program piece has wrong arity")
            out.append(_render_atom_piece(policy_json, args))
            continue
        if kind == "bond":
            if len(args) != 5:
                raise ValueError("bond render-program piece has wrong arity")
            out.append(_render_bond_piece(policy_json, args))
            continue
        if kind == "ring_label":
            if len(args) != 2:
                raise ValueError("ring label render-program piece has wrong arity")
            out.append(_ring_label_text(int(args[1])))
            continue
        if kind == "branch_open":
            if args:
                raise ValueError("branch_open render-program piece has arguments")
            out.append("(")
            continue
        if kind == "branch_close":
            if args:
                raise ValueError("branch_close render-program piece has arguments")
            out.append(")")
            continue
        if kind == "dot":
            if args:
                raise ValueError("dot render-program piece has arguments")
            out.append(".")
            continue
        raise ValueError(f"unsupported render-program piece: {kind!r}")
    return "".join(out)


def _render_atom_piece(policy_json: dict[str, object], args: tuple[object, ...]) -> str:
    atom, choice_name, tetra_token = int(args[0]), str(args[1]), str(args[2])
    for domain in _require_list(policy_json["atom_text_domains"]):
        mapping = _require_mapping(domain)
        if int(mapping["atom"]) != atom:
            continue
        for choice in _require_list(mapping["choices"]):
            choice_mapping = _require_mapping(choice)
            if choice_mapping["name"] != choice_name:
                continue
            for token, text in _require_list(choice_mapping["text_by_tetra"]):
                if str(token) == tetra_token:
                    return str(text)
    raise ValueError("atom render-program piece is outside policy")


def _render_bond_piece(policy_json: dict[str, object], args: tuple[object, ...]) -> str:
    _, bond, slot_kind, choice_name, mark = args
    bond = int(bond)
    slot_kind = str(slot_kind)
    choice_name = str(choice_name)
    mark = int(mark)
    for domain in _require_list(policy_json["bond_text_domains"]):
        mapping = _require_mapping(domain)
        if int(mapping["bond"]) != bond or str(mapping["slot_kind"]) != slot_kind:
            continue
        for choice in _require_list(mapping["choices"]):
            choice_mapping = _require_mapping(choice)
            if choice_mapping["name"] != choice_name:
                continue
            if mark == 0:
                return str(choice_mapping["base_text"])
            if not bool(choice_mapping["permits_direction"]):
                raise ValueError("direction mark is outside bond render policy")
            if mark == 1:
                return "/"
            if mark == -1:
                return "\\"
            raise ValueError("unknown direction mark in render program")
    raise ValueError("bond render-program piece is outside policy")


def _ring_label_text(value: int) -> str:
    if value <= 0:
        raise ValueError("ring label values must be positive")
    if value <= 9:
        return str(value)
    return f"%{value:02d}"


def _check_prefix_domains_against_policy(space, artifact: SupportArtifact) -> None:
    policy_atom_domains = {
        int(domain["atom"]): tuple(str(choice["name"]) for choice in domain["choices"])
        for domain in artifact.policy_json["atom_text_domains"]  # type: ignore[index]
    }
    if dict(space.atom_text_domains) != policy_atom_domains:
        raise ValueError("prefix atom domains disagree with policy")
    policy_bond_domains = {}
    for domain in artifact.policy_json["bond_text_domains"]:  # type: ignore[index]
        mapping = _require_mapping(domain)
        policy_bond_domains[
            (int(mapping["bond"]), str(mapping["slot_kind"]))
        ] = tuple(
            str(choice["name"]) for choice in _require_list(mapping["choices"])
        )

    slot_kind_by_slot: dict[int, tuple[int, str]] = {}
    skeleton_key = space.skeleton_key
    for node in artifact.nodes:
        if node.kind == "slot_bundle" and node.key[0] == skeleton_key:
            for slot in node.key[1][1]:
                slot_kind_by_slot[int(slot[0])] = (int(slot[1]), str(slot[2]))
            break
    expected_bond_domains = {
        slot: policy_bond_domains[(bond, slot_kind)]
        for slot, (bond, slot_kind) in slot_kind_by_slot.items()
    }
    if dict(space.bond_text_domains) != expected_bond_domains:
        raise ValueError("prefix bond domains disagree with policy")
    _check_ring_label_assignments(space, artifact, slot_kind_by_slot)


def _check_ring_label_assignments(space, artifact: SupportArtifact, slot_kind_by_slot) -> None:
    del slot_kind_by_slot
    labels = {int(label) for label in artifact.policy_json["ring_labels"]}  # type: ignore[index]
    least_free = bool(artifact.policy_json["least_free_ring_labels"])
    endpoint_info: dict[int, tuple[int, int]] = {}
    for node in artifact.nodes:
        if node.kind == "slot_bundle" and node.key[0] == space.skeleton_key:
            for endpoint in node.key[1][2]:
                endpoint_info[int(endpoint[0])] = (int(endpoint[1]), int(endpoint[5]))
            break
    endpoint_ids = set(endpoint_info)
    for assignment in space.ring_label_assignments:
        assigned = dict(assignment)
        if set(assigned) != endpoint_ids:
            raise ValueError("ring label endpoint coverage mismatch")
        if any(label not in labels for label in assigned.values()):
            raise ValueError("ring label is outside policy")
        intervals = _ring_intervals(endpoint_info, assigned)
        _check_ring_label_reuse(intervals)
        if least_free:
            _check_least_free_labels(intervals, labels)


def _ring_intervals(endpoint_info, assigned):
    by_bond: dict[int, list[tuple[int, int]]] = {}
    for endpoint, (bond, position) in endpoint_info.items():
        by_bond.setdefault(bond, []).append((endpoint, position))
    intervals = []
    for bond, endpoints in by_bond.items():
        if len(endpoints) != 2:
            raise ValueError("ring bond does not have exactly two endpoints")
        (endpoint_1, position_1), (endpoint_2, position_2) = endpoints
        if assigned[endpoint_1] != assigned[endpoint_2]:
            raise ValueError("ring bond endpoints use different labels")
        start, end = sorted((position_1, position_2))
        intervals.append((bond, assigned[endpoint_1], start, end))
    return tuple(sorted(intervals))


def _check_ring_label_reuse(intervals) -> None:
    for index, left in enumerate(intervals):
        for right in intervals[index + 1 :]:
            if left[1] != right[1]:
                continue
            if left[2] < right[3] and right[2] < left[3]:
                raise ValueError("ring label intervals overlap")


def _check_least_free_labels(intervals, labels: set[int]) -> None:
    for bond, label, start, _ in intervals:
        active = {
            other_label
            for _, other_label, other_start, other_end in intervals
            if other_start < start < other_end
        }
        candidates = labels - active
        if not candidates:
            raise ValueError("no free ring label")
        if label != min(candidates):
            raise ValueError("ring label violates least-free policy")


def _bond_by_id(facts_json: dict[str, object]) -> dict[int, tuple[int, int]]:
    return {
        int(bond["id"]): (int(bond["a"]), int(bond["b"]))
        for bond in facts_json["bonds"]  # type: ignore[index]
    }


def _bond_between(
    bond_by_id: Mapping[int, tuple[int, int]],
    left: int,
    right: int,
) -> int | None:
    pair = _sorted_pair((left, right))
    for bond, endpoints in bond_by_id.items():
        if _sorted_pair(endpoints) == pair:
            return bond
    return None


def _sorted_pair(pair: tuple[int, int]) -> tuple[int, int]:
    return tuple(sorted(pair))  # type: ignore[return-value]


def _require_list(value: object) -> list:
    if not isinstance(value, list):
        raise TypeError(f"expected list: {value!r}")
    return value


def _require_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping: {value!r}")
    return value


def _rows_hash(rows) -> str:
    return sequence_hash(repr(tuple(row)) for row in rows)


def _json_hash(data: object) -> str:
    import json

    payload = json.dumps(data, sort_keys=True, separators=(",", ":"))
    return sequence_hash((payload,))


__all__ = ("check_support_artifact",)
