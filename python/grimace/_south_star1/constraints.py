"""Named finite-constraint records for the private proof kernel."""

from __future__ import annotations

from dataclasses import dataclass
from dataclasses import field

from .facts import MoleculeFacts
from .ids import AtomId
from .ids import BondSlotId
from .ids import RingEndpointId
from .policy import AtomTextChoice
from .policy import BondTextChoice
from .policy import DirectionMark
from .policy import RingLabel
from .policy import SmilesPolicy
from .policy import TetraToken
from .ring_labels import validate_bounded_ring_labels
from .semantics import ParserSemantics
from .skeleton import TraversalSkeleton
from .slots import BondSlotKind
from .slots import SlotBundle
from .slots import ring_bond_slots_by_bond


@dataclass(frozen=True, slots=True)
class NonStereoTreeAssignment:
    """Finite choices for the first RDKit-free, non-stereo traversal fragment."""

    atom_text: dict[AtomId, AtomTextChoice]
    tetra_tokens: dict[AtomId, TetraToken]
    bond_text: dict[BondSlotId, BondTextChoice]
    ring_labels: dict[RingEndpointId, RingLabel]
    direction_marks: dict[BondSlotId, DirectionMark] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class NamedConstraint:
    name: str
    subject: str


def validate_nonstereo_tree_witness(
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    slots: SlotBundle,
    assignment: NonStereoTreeAssignment,
) -> tuple[NamedConstraint, ...]:
    """Validate the currently supported proof-kernel witness fragment.

    The function intentionally accepts only non-stereo, ringless tree witnesses.
    Later slices should broaden this by adding new named constraint families,
    not by weakening this boundary silently.
    """

    facts.validate()
    if facts.stereo.tetrahedral or facts.stereo.directional:
        raise NotImplementedError("non-stereo tree witnesses reject stereo facts")
    if skeleton.ring_bonds:
        raise NotImplementedError("non-stereo tree witnesses reject ring bonds")
    if assignment.ring_labels:
        raise ValueError("ringless witness must not assign ring labels")

    atom_ids = {atom.id for atom in facts.atoms}
    _require_exact_keys("atom text", set(assignment.atom_text), atom_ids)
    _require_exact_keys("tetra tokens", set(assignment.tetra_tokens), atom_ids)

    non_none_tetra = {
        atom
        for atom, token in assignment.tetra_tokens.items()
        if token is not TetraToken.NONE
    }
    if non_none_tetra:
        raise ValueError(
            "non-stereo witness must use only empty tetra tokens: "
            f"{non_none_tetra!r}"
        )

    slot_atom_ids = {slot.atom for slot in slots.atom_slots}
    _require_exact_keys("atom slots", slot_atom_ids, atom_ids)

    tree_slots = tuple(
        slot for slot in slots.bond_slots if slot.kind is BondSlotKind.TREE
    )
    tree_slot_ids = {slot.id for slot in tree_slots}
    _require_exact_keys("tree bond text", set(assignment.bond_text), tree_slot_ids)

    tree_slot_bonds = {slot.bond for slot in tree_slots}
    _require_exact_keys("tree bond slots", tree_slot_bonds, set(skeleton.tree_bonds))

    return (
        NamedConstraint("nonstereo_only", "molecule"),
        NamedConstraint("ringless_tree_only", "skeleton"),
        NamedConstraint("atom_text_slot_coverage", "atom_slots"),
        NamedConstraint("tree_bond_slot_coverage", "bond_slots"),
        NamedConstraint("empty_tetra_token_coverage", "atom_slots"),
    )


def validate_nonstereo_traversal_witness(
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    slots: SlotBundle,
    assignment: NonStereoTreeAssignment,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
) -> tuple[NamedConstraint, ...]:
    facts.validate()
    policy.validate_for_facts(facts)
    if facts.stereo.tetrahedral or facts.stereo.directional:
        raise NotImplementedError("non-stereo traversal witnesses reject stereo facts")

    atom_ids = {atom.id for atom in facts.atoms}
    _require_exact_keys("atom text", set(assignment.atom_text), atom_ids)
    _require_exact_keys("tetra tokens", set(assignment.tetra_tokens), atom_ids)
    _require_exact_keys(
        "bond text",
        set(assignment.bond_text),
        {slot.id for slot in slots.bond_slots},
    )
    _require_exact_keys(
        "direction marks",
        set(assignment.direction_marks),
        {slot.id for slot in slots.bond_slots},
    )

    non_none_tetra = {
        atom
        for atom, token in assignment.tetra_tokens.items()
        if token is not TetraToken.NONE
    }
    if non_none_tetra:
        raise ValueError(
            "non-stereo witness must use only empty tetra tokens: "
            f"{non_none_tetra!r}"
        )

    slot_atom_ids = {slot.atom for slot in slots.atom_slots}
    _require_exact_keys("atom slots", slot_atom_ids, atom_ids)

    tree_slots = tuple(
        slot for slot in slots.bond_slots if slot.kind is BondSlotKind.TREE
    )
    _require_exact_keys(
        "tree bond slots",
        {slot.bond for slot in tree_slots},
        set(skeleton.tree_bonds),
    )
    _require_exact_keys(
        "ring endpoint bonds",
        {endpoint.bond for endpoint in slots.ring_endpoints},
        set(skeleton.ring_bonds),
    )

    validate_bounded_ring_labels(policy, slots, assignment.ring_labels)
    _validate_atom_decode_relations(facts, slots, assignment, semantics)
    _validate_bond_decode_relations(facts, slots, assignment, semantics)

    return (
        NamedConstraint("nonstereo_only", "molecule"),
        NamedConstraint("bounded_ring_labels", "ring_endpoints"),
        NamedConstraint("atom_decode_relations", "atom_slots"),
        NamedConstraint("bond_decode_relations", "bond_slots"),
        NamedConstraint("ring_pair_decode_relations", "ring_bonds"),
    )


def _validate_atom_decode_relations(
    facts: MoleculeFacts,
    slots: SlotBundle,
    assignment: NonStereoTreeAssignment,
    semantics: ParserSemantics,
) -> None:
    incident_texts: dict[AtomId, list[BondTextChoice]] = {
        atom.id: [] for atom in facts.atoms
    }
    for slot in slots.bond_slots:
        incident_texts[slot.written_from].append(assignment.bond_text[slot.id])
        if slot.written_to is not None:
            incident_texts[slot.written_to].append(assignment.bond_text[slot.id])

    for atom in facts.atoms:
        if not semantics.atom_decode_ok(
            facts,
            atom.id,
            assignment.atom_text[atom.id],
            assignment.tetra_tokens[atom.id],
            tuple(incident_texts[atom.id]),
        ):
            raise ValueError(f"atom decode relation rejected atom {atom.id!r}")


def _validate_bond_decode_relations(
    facts: MoleculeFacts,
    slots: SlotBundle,
    assignment: NonStereoTreeAssignment,
    semantics: ParserSemantics,
) -> None:
    ring_slots_by_bond = ring_bond_slots_by_bond(slots)
    for slot in slots.bond_slots:
        mark = assignment.direction_marks[slot.id]
        if slot.kind is BondSlotKind.TREE:
            if not semantics.bond_decode_ok(
                facts,
                slot.bond,
                assignment.bond_text[slot.id],
                mark,
            ):
                raise ValueError(f"bond decode relation rejected slot {slot.id!r}")
            continue
        if slot.kind is not BondSlotKind.RING_ENDPOINT:
            raise ValueError(f"unknown bond slot kind {slot.kind!r}")

    for bond, (endpoint_1, endpoint_2) in ring_slots_by_bond.items():
        if not semantics.ring_pair_decode_ok(
            facts,
            bond,
            assignment.bond_text[endpoint_1.id],
            assignment.direction_marks[endpoint_1.id],
            assignment.bond_text[endpoint_2.id],
            assignment.direction_marks[endpoint_2.id],
        ):
            raise ValueError(f"ring-pair decode relation rejected bond {bond!r}")


def _require_exact_keys(label: str, actual: set[object], expected: set[object]) -> None:
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise ValueError(
            f"{label} coverage mismatch: missing={missing!r}, extra={extra!r}"
        )


__all__ = (
    "NamedConstraint",
    "NonStereoTreeAssignment",
    "validate_nonstereo_tree_witness",
    "validate_nonstereo_traversal_witness",
)
