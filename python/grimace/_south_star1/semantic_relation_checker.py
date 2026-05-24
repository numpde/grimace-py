"""Producer-free semantic relation table checks for support artifacts."""

from __future__ import annotations

from itertools import product
from typing import Mapping

from .support_artifact import SupportArtifact


def check_semantic_relation_tables(artifact: SupportArtifact) -> None:
    domains_by_owner = _domains_by_owner(artifact)
    relations_by_subject = {relation.subject: relation for relation in artifact.relations}
    slot_bundle_by_skeleton = _slot_bundle_by_skeleton(artifact)
    traversal_by_key = {
        decision.skeleton_key: decision
        for decision in artifact.traversal_decisions
    }

    for space in artifact.csp_solution_spaces:
        skeleton_key, prefix_key = space.csp_key
        slot_bundle = slot_bundle_by_skeleton[skeleton_key]
        traversal = traversal_by_key[skeleton_key]
        prefix = _prefix_from_key(prefix_key)

        expected_tetra_domains = expected_tetra_domains_from_artifact(
            facts_json=artifact.facts_json,
            policy_json=artifact.policy_json,
            prefix=prefix,
        )
        expected_direction_domains = expected_direction_domains_from_artifact(
            facts_json=artifact.facts_json,
            policy_json=artifact.policy_json,
            slot_bundle=slot_bundle,
            prefix=prefix,
        )
        _check_domains(
            space=space,
            artifact_domains=domains_by_owner[space.csp_key],
            expected_tetra=expected_tetra_domains,
            expected_direction=expected_direction_domains,
        )

        expected_relations = {}
        expected_relations.update(
            expected_tetra_relation_rows_from_artifact(
                facts_json=artifact.facts_json,
                traversal=traversal,
                slot_bundle=slot_bundle,
                prefix_key=prefix_key,
                csp_key=space.csp_key,
                tetra_domains=expected_tetra_domains,
            )
        )
        expected_relations.update(
            expected_tree_bond_mark_relation_rows_from_artifact(
                facts_json=artifact.facts_json,
                policy_json=artifact.policy_json,
                slot_bundle=slot_bundle,
                prefix=prefix,
                csp_key=space.csp_key,
                direction_domains=expected_direction_domains,
            )
        )
        expected_relations.update(
            expected_ring_pair_relation_rows_from_artifact(
                facts_json=artifact.facts_json,
                policy_json=artifact.policy_json,
                slot_bundle=slot_bundle,
                prefix=prefix,
                csp_key=space.csp_key,
                direction_domains=expected_direction_domains,
            )
        )
        expected_relations.update(
            expected_directional_relation_rows_from_artifact(
                facts_json=artifact.facts_json,
                slot_bundle=slot_bundle,
                csp_key=space.csp_key,
                direction_domains=expected_direction_domains,
            )
        )
        _check_relations(
            relation_subjects=space.relation_names,
            relations_by_subject=relations_by_subject,
            expected=expected_relations,
        )


def expected_tetra_domains_from_artifact(
    *,
    facts_json: Mapping[str, object],
    policy_json: Mapping[str, object],
    prefix,
) -> dict[str, tuple[object, ...]]:
    tetra_by_center = {
        int(site["center"]): site
        for site in _require_list(_require_mapping(facts_json["stereo"])["tetrahedral"])
    }
    out = {}
    for atom in _require_list(facts_json["atoms"]):
        atom_id = int(atom["id"])
        site = tetra_by_center.get(atom_id)
        if site is None:
            raw = ("",)
        elif site["status"] == "specified":
            raw = ("@", "@@")
        else:
            raw = ("",)
        choice = _atom_choice(policy_json, atom_id, prefix.atom_text[atom_id])
        allowed_tokens = {
            str(token)
            for token, _ in _require_list(choice["text_by_tetra"])
        }
        out[f"tetra:{atom_id}"] = tuple(token for token in raw if token in allowed_tokens)
    return out


def expected_direction_domains_from_artifact(
    *,
    facts_json: Mapping[str, object],
    policy_json: Mapping[str, object],
    slot_bundle,
    prefix,
) -> dict[str, tuple[object, ...]]:
    eligible = set()
    for site in _directional_sites(facts_json):
        if site["status"] != "specified":
            continue
        eligible.update(
            expected_directional_scope_from_artifact(
                facts_json=facts_json,
                slot_bundle=slot_bundle,
                site_id=int(site["id"]),
            )
        )
    out = {}
    for carrier in _carrier_slots(slot_bundle):
        carrier_id, bond_slot_id, bond_id, _, _ = carrier
        bond_slot = _bond_slot_by_id(slot_bundle)[bond_slot_id]
        choice_name = prefix.bond_text[bond_slot_id]
        choice = _bond_choice(policy_json, bond_id, bond_slot[2], choice_name)
        if carrier_id in eligible and bool(choice["permits_direction"]):
            out[f"direction:{carrier_id}"] = (0, 1, -1)
        else:
            out[f"direction:{carrier_id}"] = (0,)
    return out


def expected_tetra_relation_rows_from_artifact(
    *,
    facts_json: Mapping[str, object],
    traversal,
    slot_bundle,
    prefix_key: tuple[object, ...],
    csp_key: tuple[object, ...],
    tetra_domains: Mapping[str, tuple[object, ...]],
) -> dict[str, tuple[str, tuple[str, ...], tuple[tuple[object, ...], ...]]]:
    del slot_bundle, prefix_key
    out = {}
    for site in _tetra_sites(facts_json):
        site_id = int(site["id"])
        center = int(site["center"])
        rows = tuple(
            (token,)
            for token in tetra_domains[f"tetra:{center}"]
            if _tetra_value(
                facts_json=facts_json,
                traversal=traversal,
                site=site,
                token=str(token),
            )
            == site["target"]
        )
        subject = f"{repr(csp_key)}::site:{site_id}"
        out[subject] = ("tetra_site", (f"tetra:{center}",), rows)
    return out


def expected_directional_scope_from_artifact(
    *,
    facts_json: Mapping[str, object],
    slot_bundle,
    site_id: int,
) -> tuple[int, ...]:
    site = _directional_site_by_id(facts_json)[site_id]
    substituent_bonds = _directional_substituent_bonds(facts_json, site)
    return tuple(
        int(carrier[0])
        for carrier in _carrier_slots(slot_bundle)
        if int(carrier[2]) in substituent_bonds
        and int(carrier[2]) != int(site["center_bond"])
    )


def expected_directional_relation_rows_from_artifact(
    *,
    facts_json: Mapping[str, object],
    slot_bundle,
    csp_key: tuple[object, ...],
    direction_domains: Mapping[str, tuple[object, ...]],
) -> dict[str, tuple[str, tuple[str, ...], tuple[tuple[object, ...], ...]]]:
    out = {}
    for site in _directional_sites(facts_json):
        site_id = int(site["id"])
        scope = expected_directional_scope_from_artifact(
            facts_json=facts_json,
            slot_bundle=slot_bundle,
            site_id=site_id,
        )
        domains = tuple(direction_domains[f"direction:{carrier}"] for carrier in scope)
        rows = []
        for row in product(*domains):
            value = _directional_value(
                facts_json=facts_json,
                slot_bundle=slot_bundle,
                site=site,
                marks=dict(zip(scope, row, strict=True)),
            )
            if value == site["target"]:
                rows.append(tuple(row))
        relation_name = (
            "directional_site"
            if site["status"] == "specified"
            else "no_accidental_stereo"
        )
        subject = f"{repr(csp_key)}::site:{site_id}"
        out[subject] = (
            relation_name,
            tuple(f"direction:{carrier}" for carrier in scope),
            tuple(sorted(rows, key=repr)),
        )
    return out


def expected_tree_bond_mark_relation_rows_from_artifact(
    *,
    facts_json: Mapping[str, object],
    policy_json: Mapping[str, object],
    slot_bundle,
    prefix,
    csp_key: tuple[object, ...],
    direction_domains: Mapping[str, tuple[object, ...]],
) -> dict[str, tuple[str, tuple[str, ...], tuple[tuple[object, ...], ...]]]:
    out = {}
    carrier_by_bond_slot = _carrier_by_bond_slot(slot_bundle)
    for slot in _bond_slots(slot_bundle):
        slot_id, bond_id, slot_kind, _, _, _, _ = slot
        if slot_kind != "tree":
            continue
        carrier = carrier_by_bond_slot[slot_id]
        choice = _bond_choice(
            policy_json,
            bond_id,
            slot_kind,
            prefix.bond_text[slot_id],
        )
        rows = tuple(
            (mark,)
            for mark in direction_domains[f"direction:{carrier[0]}"]
            if _bond_decode_ok(facts_json, bond_id, choice, int(mark))
        )
        subject = f"{repr(csp_key)}::bond_slot:{slot_id}"
        out[subject] = (
            "tree_bond_decode",
            (f"direction:{carrier[0]}",),
            rows,
        )
    return out


def expected_ring_pair_relation_rows_from_artifact(
    *,
    facts_json: Mapping[str, object],
    policy_json: Mapping[str, object],
    slot_bundle,
    prefix,
    csp_key: tuple[object, ...],
    direction_domains: Mapping[str, tuple[object, ...]],
) -> dict[str, tuple[str, tuple[str, ...], tuple[tuple[object, ...], ...]]]:
    out = {}
    carrier_by_bond_slot = _carrier_by_bond_slot(slot_bundle)
    for bond_id, slots in _ring_bond_slots_by_bond(slot_bundle).items():
        left_slot, right_slot = slots
        left_carrier = carrier_by_bond_slot[left_slot[0]]
        right_carrier = carrier_by_bond_slot[right_slot[0]]
        left_choice = _bond_choice(
            policy_json,
            bond_id,
            left_slot[2],
            prefix.bond_text[left_slot[0]],
        )
        right_choice = _bond_choice(
            policy_json,
            bond_id,
            right_slot[2],
            prefix.bond_text[right_slot[0]],
        )
        rows = []
        for mark_1, mark_2 in product(
            direction_domains[f"direction:{left_carrier[0]}"],
            direction_domains[f"direction:{right_carrier[0]}"],
        ):
            if _ring_pair_decode_ok(
                facts_json,
                bond_id,
                left_choice,
                int(mark_1),
                right_choice,
                int(mark_2),
            ):
                rows.append((mark_1, mark_2))
        subject = f"{repr(csp_key)}::bond:{bond_id}"
        out[subject] = (
            "ring_pair_decode",
            (f"direction:{left_carrier[0]}", f"direction:{right_carrier[0]}"),
            tuple(sorted(rows, key=repr)),
        )
    return out


def _check_domains(
    *,
    space,
    artifact_domains,
    expected_tetra,
    expected_direction,
) -> None:
    actual = {
        domain.name: tuple(row[0] for row in domain.values)
        for domain in artifact_domains
    }
    expected = {**expected_tetra, **expected_direction}
    if actual != expected:
        raise ValueError("semantic variable domain mismatch")
    if dict(space.tetra_domains) != expected_tetra:
        raise ValueError("semantic tetra domain mismatch")
    if dict(space.direction_domains) != expected_direction:
        raise ValueError("semantic direction domain mismatch")


def _check_relations(
    *,
    relation_subjects,
    relations_by_subject,
    expected,
) -> None:
    if set(relation_subjects) != set(expected):
        raise ValueError("semantic relation coverage mismatch")
    for subject in relation_subjects:
        relation = relations_by_subject[subject]
        expected_name, expected_scope, expected_rows = expected[subject]
        if relation.name != expected_name:
            raise ValueError("semantic relation name mismatch")
        if relation.scope != expected_scope:
            raise ValueError("semantic relation scope mismatch")
        if tuple(sorted(relation.allowed_rows, key=repr)) != tuple(
            sorted(expected_rows, key=repr)
        ):
            raise ValueError("semantic relation row mismatch")


def _tetra_value(*, facts_json, traversal, site, token: str) -> str | None:
    if token == "":
        return "none"
    reference = tuple(int(item) for item in site["reference_order"])
    local = _local_tetra_order(facts_json, traversal, site)
    if set(local) != set(reference) or len(local) != len(reference):
        return None
    indices = tuple(reference.index(item) for item in local)
    even = _is_even_permutation(indices)
    if token == "@":
        return "plus" if even else "minus"
    if token == "@@":
        return "minus" if even else "plus"
    return None


def _local_tetra_order(facts_json, traversal, site) -> tuple[int, ...]:
    center = int(site["center"])
    occurrence_by_atom = {}
    implicit_h = []
    occurrence_ids = {int(item) for item in site["ligand_occurrences"]}
    for occurrence in _ligand_occurrences(facts_json):
        occurrence_id = int(occurrence["id"])
        if occurrence_id not in occurrence_ids:
            continue
        if occurrence["kind"] == "neighbor_atom":
            occurrence_by_atom[int(occurrence["atom"])] = occurrence_id
        elif occurrence["kind"] == "implicit_h":
            implicit_h.append(occurrence_id)
    parent = dict(traversal.parent_items)[center]
    out = []
    if parent is not None and parent in occurrence_by_atom:
        out.append(occurrence_by_atom[parent])
    events = dict(traversal.local_event_orders).get(center, ())
    for event in events:
        if event[0] == "child":
            atom = int(event[3])
        elif event[0] == "ring":
            atom = int(event[3])
        else:
            raise ValueError("unknown local event kind")
        if atom in occurrence_by_atom:
            out.append(occurrence_by_atom[atom])
    out.extend(sorted(implicit_h))
    return tuple(out)


def _directional_value(*, facts_json, slot_bundle, site, marks) -> str | None:
    carrier_by_id = {int(carrier[0]): carrier for carrier in _carrier_slots(slot_bundle)}
    models = _directional_carrier_models(facts_json, site, slot_bundle)
    left_values = []
    right_values = []
    for carrier_id, mark in marks.items():
        mark = int(mark)
        if mark == 0:
            continue
        carrier = carrier_by_id[carrier_id]
        if int(carrier[2]) == int(site["center_bond"]):
            return None
        model = models.get(carrier_id)
        if model is None:
            return None
        raw = _signed_direction(mark, carrier, endpoint=model["endpoint"])
        normalized = raw * model["ligand_factor"]
        if model["side"] == "left":
            left_values.append(normalized)
        else:
            right_values.append(normalized)
    if not left_values and not right_values:
        return "none"
    if not left_values or not right_values:
        return "none"
    if len(set(left_values)) != 1 or len(set(right_values)) != 1:
        return None
    return "together" if left_values[0] == right_values[0] else "opposite"


def _directional_carrier_models(facts_json, site, slot_bundle):
    occurrences = {
        int(occurrence["id"]): occurrence
        for occurrence in _ligand_occurrences(facts_json)
    }
    left_reference, right_reference = _directional_reference_pair(site)
    left_by_bond = _neighbor_ligands_by_bond(occurrences, site["left_ligands"])
    right_by_bond = _neighbor_ligands_by_bond(occurrences, site["right_ligands"])
    out = {}
    for carrier in _carrier_slots(slot_bundle):
        carrier_id = int(carrier[0])
        bond_id = int(carrier[2])
        if bond_id == int(site["center_bond"]):
            continue
        if bond_id in left_by_bond:
            occurrence = left_by_bond[bond_id]
            out[carrier_id] = {
                "side": "left",
                "endpoint": int(site["left_endpoint"]),
                "ligand_factor": _ligand_factor(
                    occurrence,
                    left_reference,
                    tuple(int(item) for item in site["left_ligands"]),
                ),
            }
            continue
        if bond_id in right_by_bond:
            occurrence = right_by_bond[bond_id]
            out[carrier_id] = {
                "side": "right",
                "endpoint": int(site["right_endpoint"]),
                "ligand_factor": _ligand_factor(
                    occurrence,
                    right_reference,
                    tuple(int(item) for item in site["right_ligands"]),
                ),
            }
    return out


def _directional_reference_pair(site) -> tuple[int, int]:
    if site["reference_pair"] is not None:
        left, right = site["reference_pair"]
        return int(left), int(right)
    if site["status"] == "specified":
        raise ValueError("specified directional site lacks reference pair")
    return min(int(item) for item in site["left_ligands"]), min(
        int(item) for item in site["right_ligands"]
    )


def _neighbor_ligands_by_bond(occurrences, ligand_ids) -> dict[int, int]:
    out = {}
    for ligand_id in ligand_ids:
        occurrence = occurrences[int(ligand_id)]
        if occurrence["kind"] != "neighbor_atom":
            continue
        bond = int(occurrence["bond"])
        if bond in out:
            raise ValueError("directional side has multiple ligands on bond")
        out[bond] = int(ligand_id)
    return out


def _ligand_factor(occurrence: int, reference: int, side_ligands: tuple[int, ...]) -> int:
    if len(side_ligands) > 2:
        raise ValueError("ordinary directional endpoint has too many ligands")
    if occurrence == reference:
        return 1
    if occurrence not in side_ligands:
        raise ValueError("occurrence is not on directional side")
    return -1


def _signed_direction(mark: int, carrier, *, endpoint: int) -> int:
    if int(carrier[3]) == endpoint:
        orientation = 1
    elif carrier[4] is not None and int(carrier[4]) == endpoint:
        orientation = -1
    else:
        raise ValueError("carrier is not incident to directional endpoint")
    if mark == 1:
        return orientation
    if mark == -1:
        return -orientation
    raise ValueError("absent mark has no sign")


def _bond_decode_ok(facts_json, bond_id: int, choice, mark: int) -> bool:
    bond = _bond_by_id(facts_json)[bond_id]
    if mark != 0:
        if not bool(choice["permits_direction"]):
            return False
        if bond["order"] != "single":
            return False
    return _bond_text_matches_order(str(choice["base_text"]), str(bond["order"]))


def _ring_pair_decode_ok(
    facts_json,
    bond_id: int,
    choice_1,
    mark_1: int,
    choice_2,
    mark_2: int,
) -> bool:
    bond = _bond_by_id(facts_json)[bond_id]
    order = str(bond["order"])
    text_1 = str(choice_1["base_text"])
    text_2 = str(choice_2["base_text"])
    if order == "double":
        return mark_1 == 0 and mark_2 == 0 and (text_1, text_2) in {
            ("=", ""),
            ("", "="),
        }
    if order == "triple":
        return mark_1 == 0 and mark_2 == 0 and (text_1, text_2) in {
            ("#", ""),
            ("", "#"),
        }
    return _bond_decode_ok(facts_json, bond_id, choice_1, mark_1) and _bond_decode_ok(
        facts_json,
        bond_id,
        choice_2,
        mark_2,
    )


def _bond_text_matches_order(base_text: str, order: str) -> bool:
    if order == "single":
        return base_text in {"", "-"}
    if order == "double":
        return base_text == "="
    if order == "triple":
        return base_text == "#"
    if order == "aromatic":
        return base_text in {"", ":"}
    return False


def _directional_substituent_bonds(facts_json, site) -> frozenset[int]:
    occurrences = {
        int(occurrence["id"]): occurrence
        for occurrence in _ligand_occurrences(facts_json)
    }
    out = set()
    for ligand_id in tuple(site["left_ligands"]) + tuple(site["right_ligands"]):
        occurrence = occurrences[int(ligand_id)]
        if occurrence["kind"] == "neighbor_atom":
            out.add(int(occurrence["bond"]))
    return frozenset(out)


def _prefix_from_key(prefix_key: tuple[object, ...]):
    return _Prefix(
        atom_text={int(atom): str(choice) for atom, choice in prefix_key[0]},
        bond_text={int(slot): str(choice) for slot, choice in prefix_key[1]},
        ring_labels={int(endpoint): int(label) for endpoint, label in prefix_key[2]},
    )


class _Prefix:
    def __init__(self, *, atom_text, bond_text, ring_labels) -> None:
        self.atom_text = atom_text
        self.bond_text = bond_text
        self.ring_labels = ring_labels


def _domains_by_owner(artifact: SupportArtifact):
    out = {}
    for domain in artifact.domains:
        out.setdefault(domain.owner.key, []).append(domain)
    return {key: tuple(value) for key, value in out.items()}


def _slot_bundle_by_skeleton(artifact: SupportArtifact):
    out = {}
    for node in artifact.nodes:
        if node.kind == "slot_bundle":
            out[node.key[0]] = node.key[1]
    return out


def _atom_choice(policy_json, atom_id: int, name: str):
    for domain in _require_list(policy_json["atom_text_domains"]):
        if int(domain["atom"]) != atom_id:
            continue
        for choice in _require_list(domain["choices"]):
            if str(choice["name"]) == name:
                return choice
    raise ValueError("unknown atom text choice")


def _bond_choice(policy_json, bond_id: int, slot_kind: str, name: str):
    for domain in _require_list(policy_json["bond_text_domains"]):
        if int(domain["bond"]) != bond_id or str(domain["slot_kind"]) != slot_kind:
            continue
        for choice in _require_list(domain["choices"]):
            if str(choice["name"]) == name:
                return choice
    raise ValueError("unknown bond text choice")


def _bond_slots(slot_bundle):
    return tuple(slot_bundle[1])


def _carrier_slots(slot_bundle):
    return tuple(slot_bundle[3])


def _bond_slot_by_id(slot_bundle):
    return {int(slot[0]): slot for slot in _bond_slots(slot_bundle)}


def _carrier_by_bond_slot(slot_bundle):
    return {int(carrier[1]): carrier for carrier in _carrier_slots(slot_bundle)}


def _ring_bond_slots_by_bond(slot_bundle):
    out = {}
    for slot in _bond_slots(slot_bundle):
        if slot[2] == "ring_endpoint":
            out.setdefault(int(slot[1]), []).append(slot)
    return {
        bond: tuple(sorted(slots, key=lambda item: int(item[5])))
        for bond, slots in out.items()
        if len(slots) == 2
    }


def _tetra_sites(facts_json):
    return tuple(_require_list(_require_mapping(facts_json["stereo"])["tetrahedral"]))


def _directional_sites(facts_json):
    return tuple(_require_list(_require_mapping(facts_json["stereo"])["directional"]))


def _directional_site_by_id(facts_json):
    return {int(site["id"]): site for site in _directional_sites(facts_json)}


def _ligand_occurrences(facts_json):
    return tuple(_require_list(facts_json["ligand_occurrences"]))


def _bond_by_id(facts_json):
    return {int(bond["id"]): bond for bond in _require_list(facts_json["bonds"])}


def _is_even_permutation(indices: tuple[int, ...]) -> bool:
    inversions = 0
    for left, value in enumerate(indices):
        for other in indices[left + 1 :]:
            if value > other:
                inversions += 1
    return inversions % 2 == 0


def _require_list(value: object) -> list:
    if not isinstance(value, list):
        raise TypeError(f"expected list: {value!r}")
    return value


def _require_mapping(value: object) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise TypeError(f"expected mapping: {value!r}")
    return value


__all__ = (
    "check_semantic_relation_tables",
    "expected_direction_domains_from_artifact",
    "expected_directional_relation_rows_from_artifact",
    "expected_directional_scope_from_artifact",
    "expected_ring_pair_relation_rows_from_artifact",
    "expected_tetra_domains_from_artifact",
    "expected_tetra_relation_rows_from_artifact",
    "expected_tree_bond_mark_relation_rows_from_artifact",
)
