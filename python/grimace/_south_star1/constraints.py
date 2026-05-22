"""Named finite-constraint records for the private proof kernel."""

from __future__ import annotations

from dataclasses import dataclass

from .facts import MoleculeFacts
from .ids import AtomId
from .ids import BondSlotId
from .ids import RingEndpointId
from .policy import AtomTextChoice
from .policy import BondTextChoice
from .policy import RingLabel
from .policy import TetraToken
from .skeleton import TraversalSkeleton
from .slots import BondSlotKind
from .slots import SlotBundle


@dataclass(frozen=True, slots=True)
class NonStereoTreeAssignment:
    """Finite choices for the first RDKit-free, non-stereo tree fragment."""

    atom_text: dict[AtomId, AtomTextChoice]
    tetra_tokens: dict[AtomId, TetraToken]
    bond_text: dict[BondSlotId, BondTextChoice]
    ring_labels: dict[RingEndpointId, RingLabel]


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
)
