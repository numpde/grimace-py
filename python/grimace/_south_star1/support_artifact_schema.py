"""Closed schema validation for South Star support artifacts."""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping
from typing import Any

from .enumeration_trace import TRACE_SCHEMA_VERSION
from .proof_terms import sequence_hash
from .support_artifact import SUPPORT_ARTIFACT_SCHEMA_VERSION
from .support_artifact import ArtifactNode
from .support_artifact import SupportArtifact


_TOP_LEVEL_FIELDS = frozenset(
    {
        "header",
        "facts_json",
        "policy_json",
        "semantics_name",
        "nodes",
        "edges",
        "domains",
        "relations",
        "render_programs",
        "traversal_space",
        "traversal_decisions",
        "prefix_spaces",
        "csp_solution_spaces",
        "traced_support",
    }
)
_HEADER_FIELDS = frozenset(
    {
        "schema_version",
        "dialect",
        "compiler",
        "facts_hash",
        "policy_hash",
        "semantics_hash",
        "created_by",
    }
)
_NODE_FIELDS = frozenset({"kind", "key"})
_EDGE_FIELDS = frozenset({"parent", "child", "label"})
_DOMAIN_FIELDS = frozenset({"name", "owner", "values", "value_hash"})
_RELATION_FIELDS = frozenset({"name", "subject", "scope", "allowed_rows", "row_hash"})
_RENDER_PROGRAM_FIELDS = frozenset({"witness_node", "rendered", "pieces"})
_TRAVERSAL_SPACE_FIELDS = frozenset(
    {
        "component_root_domains",
        "spanning_tree_edge_sets",
        "local_order_keys_by_skeleton",
        "skeleton_keys",
    }
)
_TRAVERSAL_DECISION_FIELDS = frozenset(
    {
        "skeleton_key",
        "roots",
        "parent_items",
        "tree_bonds",
        "ring_bonds",
        "local_event_orders",
        "atoms_covered",
        "bonds_covered",
    }
)
_PREFIX_SPACE_FIELDS = frozenset(
    {
        "skeleton_key",
        "atom_text_domains",
        "bond_text_domains",
        "ring_label_assignments",
        "prefix_keys",
    }
)
_CSP_SPACE_FIELDS = frozenset(
    {
        "csp_key",
        "tetra_domains",
        "direction_domains",
        "relation_names",
        "annotation_mode",
        "feasible_solution_keys",
        "selected_solution_keys",
        "rejected_solution_keys",
    }
)

_NODE_KINDS = frozenset(
    {
        "root",
        "skeleton",
        "slot_bundle",
        "prefix",
        "csp",
        "stereo_solution",
        "selected_solution",
        "witness",
        "support_string",
    }
)
_RELATION_NAMES = frozenset(
    {
        "tetra_site",
        "tree_bond_decode",
        "ring_pair_decode",
        "directional_site",
        "no_accidental_stereo",
    }
)
_RENDER_PIECES = frozenset(
    {
        "atom",
        "bond",
        "ring_label",
        "branch_open",
        "branch_close",
        "dot",
    }
)
_ANNOTATION_MODES = frozenset(
    {"hard", "support_maximal", "cardinality_maximal", "canonical"}
)
_BOND_ORDERS = frozenset({"single", "double", "triple", "aromatic"})
_SITE_STATUSES = frozenset({"specified", "unspecified"})
_TETRA_TARGETS = frozenset({"none", "plus", "minus"})
_DIRECTIONAL_TARGETS = frozenset({"none", "together", "opposite"})
_TETRA_TOKENS = frozenset({"", "@", "@@"})
_SLOT_KINDS = frozenset({"tree", "ring_endpoint"})


def validate_support_artifact_jsonable(data: Mapping[str, object]) -> None:
    """Validate the closed JSONable artifact language before decoding."""

    _require_exact_fields(data, _TOP_LEVEL_FIELDS, "support artifact")
    header = _require_mapping(data["header"], "header")
    _require_exact_fields(header, _HEADER_FIELDS, "support artifact header")
    if not isinstance(header["schema_version"], int):
        raise ValueError("schema version must be an integer")
    if header["schema_version"] != SUPPORT_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unknown support artifact schema version")
    for key in ("dialect", "compiler", "facts_hash", "policy_hash", "semantics_hash"):
        if not isinstance(header[key], str):
            raise ValueError(f"header field {key!r} must be a string")
    if header["created_by"] is not None and not isinstance(header["created_by"], str):
        raise ValueError("header created_by must be null or a string")

    _validate_facts_json_schema(_require_mapping(data["facts_json"], "facts_json"))
    _validate_policy_json_schema(_require_mapping(data["policy_json"], "policy_json"))
    if not isinstance(data["semantics_name"], str):
        raise ValueError("semantics_name must be a string")

    for item in _require_list(data["nodes"], "nodes"):
        mapping = _require_mapping(item, "node")
        _require_exact_fields(mapping, _NODE_FIELDS, "artifact node")
        if mapping["kind"] not in _NODE_KINDS:
            raise ValueError("unknown artifact node kind")
        _require_list(mapping["key"], "node key")
    for item in _require_list(data["edges"], "edges"):
        mapping = _require_mapping(item, "edge")
        _require_exact_fields(mapping, _EDGE_FIELDS, "artifact edge")
        _validate_jsonable_node(mapping["parent"], "edge parent")
        _validate_jsonable_node(mapping["child"], "edge child")
        if not isinstance(mapping["label"], str):
            raise ValueError("edge label must be a string")
    for item in _require_list(data["domains"], "domains"):
        mapping = _require_mapping(item, "domain")
        _require_exact_fields(mapping, _DOMAIN_FIELDS, "artifact domain")
        if not isinstance(mapping["name"], str):
            raise ValueError("domain name must be a string")
        _validate_jsonable_node(mapping["owner"], "domain owner")
        _require_list(mapping["values"], "domain values")
        if not isinstance(mapping["value_hash"], str):
            raise ValueError("domain value_hash must be a string")
    for item in _require_list(data["relations"], "relations"):
        mapping = _require_mapping(item, "relation")
        _require_exact_fields(mapping, _RELATION_FIELDS, "artifact relation")
        if mapping["name"] not in _RELATION_NAMES:
            raise ValueError("unknown artifact relation name")
        if not isinstance(mapping["subject"], str):
            raise ValueError("relation subject must be a string")
        _require_list(mapping["scope"], "relation scope")
        _require_list(mapping["allowed_rows"], "relation allowed rows")
        if not isinstance(mapping["row_hash"], str):
            raise ValueError("relation row_hash must be a string")
    for item in _require_list(data["render_programs"], "render_programs"):
        mapping = _require_mapping(item, "render program")
        _require_exact_fields(mapping, _RENDER_PROGRAM_FIELDS, "render program")
        _validate_jsonable_node(mapping["witness_node"], "render-program witness")
        if not isinstance(mapping["rendered"], str):
            raise ValueError("rendered text must be a string")
        _validate_jsonable_render_pieces(mapping["pieces"])

    _require_exact_fields(
        _require_mapping(data["traversal_space"], "traversal_space"),
        _TRAVERSAL_SPACE_FIELDS,
        "traversal space",
    )
    for item in _require_list(data["traversal_decisions"], "traversal_decisions"):
        _require_exact_fields(
            _require_mapping(item, "traversal decision"),
            _TRAVERSAL_DECISION_FIELDS,
            "traversal decision",
        )
    for item in _require_list(data["prefix_spaces"], "prefix_spaces"):
        _require_exact_fields(
            _require_mapping(item, "prefix space"),
            _PREFIX_SPACE_FIELDS,
            "prefix space",
        )
    for item in _require_list(data["csp_solution_spaces"], "csp_solution_spaces"):
        mapping = _require_mapping(item, "CSP solution space")
        _require_exact_fields(mapping, _CSP_SPACE_FIELDS, "CSP solution space")
        if mapping["annotation_mode"] not in _ANNOTATION_MODES:
            raise ValueError("unknown annotation mode")

    traced = _require_mapping(data["traced_support"], "traced_support")
    if "trace" in traced:
        trace = _require_mapping(traced["trace"], "traced_support.trace")
        if trace.get("schema_version") != TRACE_SCHEMA_VERSION:
            raise ValueError("unknown trace schema version")


def validate_support_artifact_schema(artifact: SupportArtifact) -> None:
    """Validate dataclass-level schema, cross references, and canonical data."""

    _validate_header(artifact)
    _validate_facts_json_schema(artifact.facts_json)
    _validate_policy_json_schema(artifact.policy_json)
    node_set = _validate_nodes(artifact)
    _validate_edges(artifact, node_set)
    _validate_domains(artifact, node_set)
    relation_subjects = _validate_relations(artifact)
    _validate_render_programs(artifact, node_set)
    _validate_traversal_certificates(artifact, node_set)
    _validate_prefix_spaces(artifact, node_set)
    _validate_csp_solution_spaces(artifact, node_set, relation_subjects)
    _validate_trace_and_witness_cross_references(artifact, node_set, relation_subjects)


def _validate_header(artifact: SupportArtifact) -> None:
    if artifact.header.schema_version != SUPPORT_ARTIFACT_SCHEMA_VERSION:
        raise ValueError("unknown support artifact schema version")
    if artifact.traced_support.trace.schema_version != TRACE_SCHEMA_VERSION:
        raise ValueError("unknown trace schema version")
    if not artifact.header.dialect:
        raise ValueError("artifact header dialect must be nonempty")
    if not artifact.header.compiler:
        raise ValueError("artifact header compiler must be nonempty")
    if not artifact.semantics_name:
        raise ValueError("artifact semantics name must be nonempty")


def _validate_nodes(artifact: SupportArtifact) -> set[ArtifactNode]:
    seen: set[ArtifactNode] = set()
    for node in artifact.nodes:
        if node.kind not in _NODE_KINDS:
            raise ValueError("unknown artifact node kind")
        if node in seen:
            raise ValueError("duplicate artifact node")
        seen.add(node)
    if ArtifactNode(kind="root", key=("root",)) not in seen:
        raise ValueError("artifact lacks root node")
    return seen


def _validate_edges(artifact: SupportArtifact, nodes: set[ArtifactNode]) -> None:
    seen = set()
    for edge in artifact.edges:
        if edge in seen:
            raise ValueError("duplicate artifact edge")
        seen.add(edge)
        if edge.parent not in nodes:
            raise ValueError("edge references unknown parent node")
        if edge.child not in nodes:
            raise ValueError("edge references unknown child node")
        if not edge.label:
            raise ValueError("artifact edge label must be nonempty")


def _validate_domains(artifact: SupportArtifact, nodes: set[ArtifactNode]) -> None:
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


def _validate_relations(artifact: SupportArtifact) -> set[str]:
    seen: set[tuple[str, str, tuple[str, ...]]] = set()
    subjects: set[str] = set()
    for relation in artifact.relations:
        if relation.name not in _RELATION_NAMES:
            raise ValueError("unknown artifact relation name")
        key = (relation.name, relation.subject, relation.scope)
        if key in seen:
            raise ValueError("duplicate artifact relation")
        seen.add(key)
        if relation.subject in subjects:
            raise ValueError("duplicate artifact relation subject")
        subjects.add(relation.subject)
        if tuple(sorted(relation.allowed_rows, key=_canonical_row_key)) != relation.allowed_rows:
            raise ValueError("noncanonical relation row order")
        if relation.row_hash != _rows_hash(relation.allowed_rows):
            raise ValueError("relation row hash mismatch")
        for row in relation.allowed_rows:
            if len(row) != len(relation.scope):
                raise ValueError("relation row arity mismatch")
    return subjects


def _validate_render_programs(
    artifact: SupportArtifact,
    nodes: set[ArtifactNode],
) -> None:
    seen: set[ArtifactNode] = set()
    for program in artifact.render_programs:
        if program.witness_node not in nodes:
            raise ValueError("render program references unknown witness node")
        if program.witness_node.kind != "witness":
            raise ValueError("render program witness node has wrong kind")
        if program.witness_node in seen:
            raise ValueError("duplicate render program for witness")
        seen.add(program.witness_node)
        for kind, args in program.pieces:
            if kind not in _RENDER_PIECES:
                raise ValueError("unsupported render-program piece")
            _check_render_piece_arity(kind, args)


def _validate_traversal_certificates(
    artifact: SupportArtifact,
    nodes: set[ArtifactNode],
) -> None:
    skeleton_nodes = {node.key for node in nodes if node.kind == "skeleton"}
    if _duplicates(artifact.traversal_space.skeleton_keys):
        raise ValueError("duplicate traversal skeleton key")
    if set(artifact.traversal_space.skeleton_keys) != skeleton_nodes:
        raise ValueError("skeleton node set disagrees with traversal skeleton keys")
    _reject_duplicate_by(
        artifact.traversal_decisions,
        lambda item: item.skeleton_key,
        "duplicate traversal certificate",
    )
    for decision in artifact.traversal_decisions:
        if decision.skeleton_key not in skeleton_nodes:
            raise ValueError("traversal certificate references unknown skeleton")


def _validate_prefix_spaces(
    artifact: SupportArtifact,
    nodes: set[ArtifactNode],
) -> None:
    skeleton_nodes = {node.key for node in nodes if node.kind == "skeleton"}
    prefix_nodes = {node.key for node in nodes if node.kind == "prefix"}
    _reject_duplicate_by(
        artifact.prefix_spaces,
        lambda item: item.skeleton_key,
        "duplicate prefix-space certificate",
    )
    for space in artifact.prefix_spaces:
        if space.skeleton_key not in skeleton_nodes:
            raise ValueError("prefix-space certificate references unknown skeleton")
        for key in space.prefix_keys:
            if (space.skeleton_key, key) not in prefix_nodes:
                raise ValueError("prefix-space key lacks prefix node")
        for atom, choices in space.atom_text_domains:
            if not isinstance(atom, int) or not choices:
                raise ValueError("malformed prefix atom domain")
        for slot, choices in space.bond_text_domains:
            if not isinstance(slot, int) or not choices:
                raise ValueError("malformed prefix bond domain")


def _validate_csp_solution_spaces(
    artifact: SupportArtifact,
    nodes: set[ArtifactNode],
    relation_subjects: set[str],
) -> None:
    csp_nodes = {node.key for node in nodes if node.kind == "csp"}
    solution_nodes = {node.key for node in nodes if node.kind == "stereo_solution"}
    _reject_duplicate_by(
        artifact.csp_solution_spaces,
        lambda item: item.csp_key,
        "duplicate CSP solution-space certificate",
    )
    for space in artifact.csp_solution_spaces:
        if space.csp_key not in csp_nodes:
            raise ValueError("CSP solution-space certificate references unknown CSP")
        if space.annotation_mode not in _ANNOTATION_MODES:
            raise ValueError("unknown annotation mode")
        if _duplicates(space.relation_names):
            raise ValueError("duplicate CSP relation reference")
        for subject in space.relation_names:
            if subject not in relation_subjects:
                raise ValueError("CSP relation reference is unknown")
        feasible = set(space.feasible_solution_keys)
        selected = set(space.selected_solution_keys)
        rejected = set(space.rejected_solution_keys)
        if rejected - feasible:
            raise ValueError("rejected CSP solution is not feasible")
        for key in feasible:
            if (space.csp_key, key) not in solution_nodes:
                raise ValueError("CSP feasible solution lacks node")


def _validate_trace_and_witness_cross_references(
    artifact: SupportArtifact,
    nodes: set[ArtifactNode],
    relation_subjects: set[str],
) -> None:
    accepted_nodes = set()
    for certificate in artifact.traced_support.trace.accepted:
        node = ArtifactNode(kind=certificate.node.kind, key=certificate.node.key)
        if node not in nodes:
            raise ValueError("trace acceptance references unknown node")
        accepted_nodes.add(node)
    for certificate in artifact.traced_support.trace.rejected:
        node = ArtifactNode(kind=certificate.node.kind, key=certificate.node.key)
        if node not in nodes:
            raise ValueError("trace rejection references unknown node")

    witness_ids: set[str] = set()
    for certified in artifact.traced_support.certified_witnesses:
        witness_id = certified.witness.id
        if witness_id in witness_ids:
            raise ValueError("duplicate witness certificate id")
        witness_ids.add(witness_id)
        if certified.certificate.witness_id != witness_id:
            raise ValueError("witness certificate id mismatch")
        if ArtifactNode(kind="witness", key=(witness_id,)) not in nodes:
            raise ValueError("witness certificate references unknown witness node")
    accepted_witness_ids = {
        certificate.witness_id
        for certificate in artifact.traced_support.trace.accepted
    }
    if not accepted_witness_ids <= witness_ids:
        raise ValueError("manifest witness ids lack certificates")


def _validate_facts_json_schema(data: Mapping[str, object]) -> None:
    _require_exact_fields(
        data,
        frozenset({"atoms", "bonds", "components", "stereo", "ligand_occurrences"}),
        "facts JSON",
    )
    atoms = tuple(_require_mapping(item, "atom") for item in _require_list(data["atoms"], "atoms"))
    bonds = tuple(_require_mapping(item, "bond") for item in _require_list(data["bonds"], "bonds"))
    atom_ids = tuple(_require_int(atom["id"], "atom id") for atom in atoms)
    bond_ids = tuple(_require_int(bond["id"], "bond id") for bond in bonds)
    if _duplicates(atom_ids):
        raise ValueError("duplicate atom id in facts JSON")
    if _duplicates(bond_ids):
        raise ValueError("duplicate bond id in facts JSON")
    atom_set = set(atom_ids)
    bond_set = set(bond_ids)
    for atom in atoms:
        _require_exact_fields(
            atom,
            frozenset(
                {
                    "id",
                    "atomic_num",
                    "symbol",
                    "isotope",
                    "formal_charge",
                    "is_aromatic",
                    "explicit_h_count",
                    "implicit_h_count",
                    "no_implicit",
                }
            ),
            "atom",
        )
        if not isinstance(atom["symbol"], str):
            raise ValueError("atom symbol must be a string")
    for bond in bonds:
        _require_exact_fields(
            bond,
            frozenset({"id", "a", "b", "order", "is_aromatic", "is_conjugated"}),
            "bond",
        )
        if _require_int(bond["a"], "bond endpoint") not in atom_set:
            raise ValueError("bond endpoint references unknown atom")
        if _require_int(bond["b"], "bond endpoint") not in atom_set:
            raise ValueError("bond endpoint references unknown atom")
        if bond["order"] not in _BOND_ORDERS:
            raise ValueError("unknown bond order")

    components = tuple(
        _require_mapping(item, "component")
        for item in _require_list(data["components"], "components")
    )
    component_atoms: list[int] = []
    component_bonds: list[int] = []
    component_ids: list[int] = []
    for component in components:
        _require_exact_fields(
            component,
            frozenset({"id", "atoms", "bonds"}),
            "component",
        )
        component_ids.append(_require_int(component["id"], "component id"))
        component_atoms.extend(
            _require_int(item, "component atom")
            for item in _require_list(component["atoms"], "component atoms")
        )
        component_bonds.extend(
            _require_int(item, "component bond")
            for item in _require_list(component["bonds"], "component bonds")
        )
    if _duplicates(component_ids):
        raise ValueError("duplicate component id")
    if Counter(component_atoms) != Counter(atom_ids):
        raise ValueError("components do not partition atom ids")
    if Counter(component_bonds) != Counter(bond_ids):
        raise ValueError("components do not partition bond ids")

    stereo = _require_mapping(data["stereo"], "stereo")
    _require_exact_fields(stereo, frozenset({"tetrahedral", "directional"}), "stereo")
    occurrences = tuple(
        _require_mapping(item, "ligand occurrence")
        for item in _require_list(data["ligand_occurrences"], "ligand occurrences")
    )
    occurrence_by_id = _validate_ligand_occurrences(occurrences, atom_set, bond_set)
    _validate_tetra_sites(
        tuple(
            _require_mapping(item, "tetrahedral site")
            for item in _require_list(stereo["tetrahedral"], "tetrahedral sites")
        ),
        atom_set,
        occurrence_by_id,
    )
    _validate_directional_sites(
        tuple(
            _require_mapping(item, "directional site")
            for item in _require_list(stereo["directional"], "directional sites")
        ),
        atom_set,
        bond_set,
        occurrence_by_id,
    )


def _validate_policy_json_schema(data: Mapping[str, object]) -> None:
    _require_exact_fields(
        data,
        frozenset(
            {
                "ring_labels",
                "annotation_mode",
                "least_free_ring_labels",
                "atom_text_domains",
                "bond_text_domains",
            }
        ),
        "policy JSON",
    )
    labels = tuple(
        _require_int(item, "ring label")
        for item in _require_list(data["ring_labels"], "ring labels")
    )
    if not labels or any(label <= 0 for label in labels) or _duplicates(labels):
        raise ValueError("ring labels must be finite positive unique values")
    if data["annotation_mode"] not in _ANNOTATION_MODES:
        raise ValueError("unknown annotation mode")
    if not isinstance(data["least_free_ring_labels"], bool):
        raise ValueError("least_free_ring_labels must be a boolean")

    atom_domains = tuple(
        _require_mapping(item, "atom text domain")
        for item in _require_list(data["atom_text_domains"], "atom text domains")
    )
    atom_ids: list[int] = []
    for domain in atom_domains:
        _require_exact_fields(domain, frozenset({"atom", "choices"}), "atom text domain")
        atom_ids.append(_require_int(domain["atom"], "atom text domain atom"))
        choices = tuple(
            _require_mapping(item, "atom text choice")
            for item in _require_list(domain["choices"], "atom text choices")
        )
        if not choices:
            raise ValueError("atom text domain must be nonempty")
        _validate_choice_names(choices, "atom text choice")
        for choice in choices:
            _require_exact_fields(
                choice,
                frozenset({"name", "text_by_tetra"}),
                "atom text choice",
            )
            token_rows = _require_list(choice["text_by_tetra"], "text_by_tetra")
            tokens = []
            for row in token_rows:
                row_list = _require_list(row, "text_by_tetra row")
                if len(row_list) != 2:
                    raise ValueError("text_by_tetra rows must have arity two")
                if row_list[0] not in _TETRA_TOKENS:
                    raise ValueError("unknown tetra token in atom text choice")
                if not isinstance(row_list[1], str):
                    raise ValueError("atom text spelling must be a string")
                tokens.append(row_list[0])
            if _duplicates(tokens):
                raise ValueError("duplicate tetra token in atom text choice")
    if _duplicates(atom_ids):
        raise ValueError("duplicate atom text domain")

    bond_domains = tuple(
        _require_mapping(item, "bond text domain")
        for item in _require_list(data["bond_text_domains"], "bond text domains")
    )
    bond_keys: list[tuple[int, str]] = []
    for domain in bond_domains:
        _require_exact_fields(
            domain,
            frozenset({"bond", "slot_kind", "choices"}),
            "bond text domain",
        )
        slot_kind = str(domain["slot_kind"])
        if slot_kind not in _SLOT_KINDS:
            raise ValueError("unknown bond slot kind")
        bond_keys.append((_require_int(domain["bond"], "bond text domain bond"), slot_kind))
        choices = tuple(
            _require_mapping(item, "bond text choice")
            for item in _require_list(domain["choices"], "bond text choices")
        )
        if not choices:
            raise ValueError("bond text domain must be nonempty")
        _validate_choice_names(choices, "bond text choice")
        for choice in choices:
            _require_exact_fields(
                choice,
                frozenset({"name", "base_text", "permits_direction"}),
                "bond text choice",
            )
            if not isinstance(choice["base_text"], str):
                raise ValueError("bond base_text must be a string")
            if not isinstance(choice["permits_direction"], bool):
                raise ValueError("bond permits_direction must be a boolean")
    if _duplicates(bond_keys):
        raise ValueError("duplicate bond text domain")


def _validate_ligand_occurrences(
    occurrences: tuple[Mapping[str, object], ...],
    atom_set: set[int],
    bond_set: set[int],
) -> dict[int, Mapping[str, object]]:
    occurrence_ids: list[int] = []
    out: dict[int, Mapping[str, object]] = {}
    for occurrence in occurrences:
        _require_exact_fields(
            occurrence,
            frozenset({"id", "site", "kind", "atom", "bond", "ordinal"}),
            "ligand occurrence",
        )
        occurrence_id = _require_int(occurrence["id"], "ligand occurrence id")
        occurrence_ids.append(occurrence_id)
        kind = occurrence["kind"]
        if kind not in {"neighbor_atom", "implicit_h"}:
            raise ValueError("unknown ligand occurrence kind")
        atom = occurrence["atom"]
        bond = occurrence["bond"]
        if kind == "neighbor_atom":
            if _require_int(atom, "ligand atom") not in atom_set:
                raise ValueError("ligand occurrence atom references unknown atom")
            if _require_int(bond, "ligand bond") not in bond_set:
                raise ValueError("ligand occurrence bond references unknown bond")
        else:
            if _require_int(atom, "implicit-H ligand atom") not in atom_set:
                raise ValueError("implicit-H ligand occurrence atom references unknown atom")
            if bond is not None:
                raise ValueError("implicit-H ligand occurrence must not reference a bond")
        out[occurrence_id] = occurrence
    if _duplicates(occurrence_ids):
        raise ValueError("duplicate ligand occurrence id")
    return out


def _validate_tetra_sites(
    sites: tuple[Mapping[str, object], ...],
    atom_set: set[int],
    occurrence_by_id: Mapping[int, Mapping[str, object]],
) -> None:
    site_ids: list[int] = []
    for site in sites:
        _require_exact_fields(
            site,
            frozenset(
                {
                    "id",
                    "center",
                    "status",
                    "target",
                    "ligand_occurrences",
                    "reference_order",
                }
            ),
            "tetrahedral site",
        )
        site_id = _require_int(site["id"], "tetrahedral site id")
        site_ids.append(site_id)
        if _require_int(site["center"], "tetrahedral center") not in atom_set:
            raise ValueError("tetrahedral center references unknown atom")
        _validate_site_status_and_target(
            status=site["status"],
            target=site["target"],
            targets=_TETRA_TARGETS,
            site_kind="tetrahedral",
        )
        ligands = tuple(
            _require_int(item, "tetrahedral ligand occurrence")
            for item in _require_list(site["ligand_occurrences"], "tetrahedral ligands")
        )
        reference = tuple(
            _require_int(item, "tetrahedral reference occurrence")
            for item in _require_list(site["reference_order"], "tetrahedral reference order")
        )
        if set(ligands) != set(reference) or len(reference) != len(set(reference)):
            raise ValueError("bad tetra reference order")
        if any(item not in occurrence_by_id for item in ligands):
            raise ValueError("tetrahedral site references unknown ligand occurrence")
        for occurrence in ligands:
            if _require_int(occurrence_by_id[occurrence]["site"], "occurrence site") != site_id:
                raise ValueError("tetrahedral ligand occurrence site mismatch")
    if _duplicates(site_ids):
        raise ValueError("duplicate tetrahedral site id")


def _validate_directional_sites(
    sites: tuple[Mapping[str, object], ...],
    atom_set: set[int],
    bond_set: set[int],
    occurrence_by_id: Mapping[int, Mapping[str, object]],
) -> None:
    site_ids: list[int] = []
    for site in sites:
        _require_exact_fields(
            site,
            frozenset(
                {
                    "id",
                    "center_bond",
                    "left_endpoint",
                    "right_endpoint",
                    "status",
                    "target",
                    "left_ligands",
                    "right_ligands",
                    "reference_pair",
                }
            ),
            "directional site",
        )
        site_id = _require_int(site["id"], "directional site id")
        site_ids.append(site_id)
        if _require_int(site["center_bond"], "directional center bond") not in bond_set:
            raise ValueError("directional center bond references unknown bond")
        if _require_int(site["left_endpoint"], "directional endpoint") not in atom_set:
            raise ValueError("directional endpoint references unknown atom")
        if _require_int(site["right_endpoint"], "directional endpoint") not in atom_set:
            raise ValueError("directional endpoint references unknown atom")
        _validate_site_status_and_target(
            status=site["status"],
            target=site["target"],
            targets=_DIRECTIONAL_TARGETS,
            site_kind="directional",
        )
        left = tuple(
            _require_int(item, "directional left ligand")
            for item in _require_list(site["left_ligands"], "directional left ligands")
        )
        right = tuple(
            _require_int(item, "directional right ligand")
            for item in _require_list(site["right_ligands"], "directional right ligands")
        )
        for occurrence in left + right:
            if occurrence not in occurrence_by_id:
                raise ValueError("directional site references unknown ligand occurrence")
            if _require_int(occurrence_by_id[occurrence]["site"], "occurrence site") != site_id:
                raise ValueError("directional ligand occurrence site mismatch")
        reference = site["reference_pair"]
        if reference is not None:
            pair = tuple(
                _require_int(item, "directional reference occurrence")
                for item in _require_list(reference, "directional reference pair")
            )
            if len(pair) != 2 or pair[0] not in left or pair[1] not in right:
                raise ValueError("directional reference pair is not endpoint-side ligands")
    if _duplicates(site_ids):
        raise ValueError("duplicate directional site id")


def _validate_site_status_and_target(
    *,
    status: object,
    target: object,
    targets: frozenset[str],
    site_kind: str,
) -> None:
    if status not in _SITE_STATUSES:
        raise ValueError(f"unknown {site_kind} site status")
    if target not in targets:
        raise ValueError(f"unknown {site_kind} target")
    if status == "specified" and target == "none":
        raise ValueError(f"specified {site_kind} site has NONE target")
    if status == "unspecified" and target != "none":
        raise ValueError(f"unspecified {site_kind} site has non-NONE target")


def _validate_jsonable_node(value: object, label: str) -> None:
    mapping = _require_mapping(value, label)
    _require_exact_fields(mapping, _NODE_FIELDS, label)
    if mapping["kind"] not in _NODE_KINDS:
        raise ValueError("unknown artifact node kind")
    _require_list(mapping["key"], f"{label} key")


def _validate_jsonable_render_pieces(value: object) -> None:
    for item in _require_list(value, "render pieces"):
        piece = _require_list(item, "render piece")
        if len(piece) != 2:
            raise ValueError("render piece must have arity two")
        if piece[0] not in _RENDER_PIECES:
            raise ValueError("unsupported render-program piece")
        args = tuple(_require_list(piece[1], "render piece arguments"))
        _check_render_piece_arity(str(piece[0]), args)


def _check_render_piece_arity(kind: str, args: tuple[object, ...]) -> None:
    expected = {
        "atom": 3,
        "bond": 5,
        "ring_label": 2,
        "branch_open": 0,
        "branch_close": 0,
        "dot": 0,
    }[kind]
    if len(args) != expected:
        raise ValueError(f"{kind} render-program piece has wrong arity")


def _canonical_row_key(row: tuple[object, ...]) -> tuple[tuple[int, object], ...]:
    return tuple(_canonical_value_key(value) for value in row)


def _canonical_value_key(value: object) -> tuple[int, object]:
    if value == "":
        return (0, 0)
    if value == "@":
        return (0, 1)
    if value == "@@":
        return (0, 2)
    if value == 0:
        return (1, 0)
    if value == 1:
        return (1, 1)
    if value == -1:
        return (1, 2)
    return (2, repr(value))


def _validate_choice_names(
    choices: tuple[Mapping[str, object], ...],
    label: str,
) -> None:
    names = []
    for choice in choices:
        name = choice.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{label} name must be a nonempty string")
        names.append(name)
    if _duplicates(names):
        raise ValueError(f"duplicate {label} name")


def _require_exact_fields(
    data: Mapping[str, object],
    fields: frozenset[str],
    label: str,
) -> None:
    actual = set(data)
    missing = fields - actual
    extra = actual - fields
    if missing:
        raise ValueError(f"{label} missing required field: {sorted(missing)[0]}")
    if extra:
        raise ValueError(f"{label} has unknown field: {sorted(extra)[0]}")


def _require_mapping(value: object, label: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be an object")
    return value


def _require_list(value: object, label: str) -> list:
    if not isinstance(value, list):
        raise ValueError(f"{label} must be a list")
    return value


def _require_int(value: object, label: str) -> int:
    if not isinstance(value, int):
        raise ValueError(f"{label} must be an integer")
    return value


def _reject_duplicate_by(items, key, message: str) -> None:
    seen = set()
    for item in items:
        value = key(item)
        if value in seen:
            raise ValueError(message)
        seen.add(value)


def _duplicates(values) -> bool:
    values_tuple = tuple(values)
    return len(set(values_tuple)) != len(values_tuple)


def _rows_hash(rows) -> str:
    return sequence_hash(repr(tuple(row)) for row in rows)


__all__ = (
    "validate_support_artifact_jsonable",
    "validate_support_artifact_schema",
)
