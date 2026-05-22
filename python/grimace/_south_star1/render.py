"""Pure rendering boundary for satisfying South Star 1 assignments."""

from __future__ import annotations

from .constraints import NonStereoTreeAssignment
from .constraints import validate_nonstereo_tree_witness
from .facts import MoleculeFacts
from .ids import AtomId
from .skeleton import ChildRole
from .skeleton import TraversalSkeleton
from .slots import SlotBundle
from .slots import tree_bond_slot_by_bond


def render_nonstereo_tree(
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    slots: SlotBundle,
    assignment: NonStereoTreeAssignment,
) -> str:
    validate_nonstereo_tree_witness(facts, skeleton, slots, assignment)
    tree_slot_by_bond = tree_bond_slot_by_bond(slots)

    def render_atom(atom: AtomId) -> str:
        text = assignment.atom_text[atom].render(assignment.tetra_tokens[atom])
        for event in skeleton.events_at[atom]:
            slot = tree_slot_by_bond[event.bond]
            bond_text = assignment.bond_text[slot.id].base_text
            child_text = bond_text + render_atom(event.child)
            if event.role is ChildRole.BRANCH:
                text += f"({child_text})"
            else:
                text += child_text
        return text

    return ".".join(render_atom(root) for root in skeleton.roots)


__all__ = ("render_nonstereo_tree",)
