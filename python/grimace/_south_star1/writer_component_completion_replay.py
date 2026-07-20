"""Producer-free graph checks shared by component boundaries and terminalization."""

from __future__ import annotations


class WriterComponentCompletionReplayError(ValueError):
    """A writer state does not prove the declared completed component prefix."""


def replay_completed_component_prefix(
    *,
    facts,
    state,
    completed_component_index: int,
    require_final: bool,
    rooted_at_atom: int,
    allowed_ring_label_values: tuple[int, ...] | None = None,
    ring_endpoint_choices: dict[int, tuple[tuple[str, object], ...]] | None = None,
) -> None:
    components = facts.components
    if (
        completed_component_index < 0
        or completed_component_index >= len(components)
        or require_final != (completed_component_index == len(components) - 1)
    ):
        _fail("component_boundary_index_mismatch")

    roots = state.component_cursor.component_roots
    if len(roots) != len(components) or any(
        root not in component.atoms
        for root, component in zip(roots, components, strict=True)
    ):
        _fail("component_boundary_root_vector_mismatch")
    if rooted_at_atom >= 0:
        rooted_components = tuple(
            index
            for index, component in enumerate(components)
            if rooted_at_atom in component.atoms
        )
        if len(rooted_components) != 1 or roots[rooted_components[0]] != rooted_at_atom:
            _fail("component_boundary_root_vector_mismatch")

    completed = components[: completed_component_index + 1]
    future = components[completed_component_index + 1 :]
    completed_atoms = {atom for component in completed for atom in component.atoms}
    completed_bonds = {bond for component in completed for bond in component.bonds}
    future_atoms = {atom for component in future for atom in component.atoms}
    future_bonds = {bond for component in future for bond in component.bonds}
    closure_bonds = tuple(item.bond for item in state.ring_state.closed_closures)
    reusable_labels = state.ring_state.label_state.reusable
    if len(reusable_labels) != len({label.value for label in reusable_labels}):
        _fail("component_boundary_ring_state_mismatch")
    for label in reusable_labels:
        expected_text = str(label.value) if label.value < 10 else f"%{label.value}"
        if (
            label.text != expected_text
            or (
                allowed_ring_label_values is not None
                and label.value not in allowed_ring_label_values
            )
        ):
            _fail("component_boundary_ring_policy_mismatch")

    if set(state.visited_atoms) != completed_atoms:
        if set(state.visited_atoms) & future_atoms:
            _fail("component_boundary_future_component_touched")
        _fail("component_boundary_current_component_incomplete")
    if set(state.written_bonds) & set(closure_bonds):
        _fail("component_boundary_current_component_incomplete")
    if len(closure_bonds) != len(set(closure_bonds)):
        _fail("component_boundary_current_component_incomplete")
    if set(state.written_bonds) | set(closure_bonds) != completed_bonds:
        if set(state.written_bonds) & future_bonds or set(closure_bonds) & future_bonds:
            _fail("component_boundary_future_component_touched")
        _fail("component_boundary_current_component_incomplete")

    bond_by_id = {bond.id: bond for bond in facts.bonds}
    component_by_atom = {
        atom: index for index, component in enumerate(components) for atom in component.atoms
    }
    for closure in state.ring_state.closed_closures:
        bond = bond_by_id.get(closure.bond)
        if bond is None or {closure.first_atom, closure.second_atom} != {bond.a, bond.b}:
            _fail("component_boundary_ring_state_mismatch")
        if component_by_atom[bond.a] != component_by_atom[bond.b]:
            _fail("component_boundary_ring_state_mismatch")
        expected_label_text = (
            str(closure.label.value)
            if closure.label.value < 10
            else f"%{closure.label.value}"
        )
        if (
            closure.label.value <= 0
            or (
                allowed_ring_label_values is not None
                and closure.label.value not in allowed_ring_label_values
            )
            or closure.label.text != expected_label_text
            or closure.first_endpoint_text != expected_label_text
            or closure.second_endpoint_text != expected_label_text
        ):
            _fail("component_boundary_ring_state_mismatch")
        markers = (
            closure.first_endpoint_bond_text,
            closure.second_endpoint_bond_text,
        )
        required = (
            "=" if bond.order.value == "double"
            else "#" if bond.order.value == "triple"
            else ""
        )
        if required:
            if markers.count(required) != 1 or any(
                marker not in ("", required) for marker in markers
            ):
                _fail("component_boundary_ring_state_mismatch")
            if (
                closure.first_endpoint_direction_mark.value
                or closure.second_endpoint_direction_mark.value
            ):
                _fail("component_boundary_ring_state_mismatch")
        elif ring_endpoint_choices is None:
            allowed = ("", ":") if bond.order.value == "aromatic" else ("", "-")
            if any(marker not in allowed for marker in markers):
                _fail("component_boundary_ring_state_mismatch")
        if ring_endpoint_choices is not None:
            choices = ring_endpoint_choices.get(int(closure.bond))
            endpoint_choices = (
                (
                    closure.first_endpoint_bond_text,
                    closure.first_endpoint_direction_mark,
                ),
                (
                    closure.second_endpoint_bond_text,
                    closure.second_endpoint_direction_mark,
                ),
            )
            if choices is None or any(
                choice not in choices for choice in endpoint_choices
            ):
                _fail("component_boundary_ring_policy_mismatch")

    if (
        state.obligations.pending_entry is not None
        or state.branch_stack
        or state.ring_state.open_endpoints
        or state.ring_state.label_state.allocated
        or state.active.atom not in components[completed_component_index].atoms
        or state.active.atom not in state.visited_atoms
        or not state.active.atom_emitted
    ):
        _fail("component_boundary_current_component_incomplete")
    if state.active.parent is None:
        if state.active.incoming_bond is not None:
            _fail("component_boundary_current_component_incomplete")
    else:
        incoming = bond_by_id.get(state.active.incoming_bond)
        if (
            incoming is None
            or {incoming.a, incoming.b}
            != {state.active.parent, state.active.atom}
        ):
            _fail("component_boundary_current_component_incomplete")

    atom_text = dict(state.policy_state.atom_text)
    bond_text = dict(state.policy_state.bond_text)
    if set(atom_text) & future_atoms or set(bond_text) & future_bonds:
        _fail("component_boundary_future_component_touched")
    stereo = state.stereo_state
    if any(item.atom in future_atoms for item in stereo.atom_occurrences):
        _fail("component_boundary_future_component_touched")
    if any(item.bond in future_bonds for item in stereo.bond_occurrences):
        _fail("component_boundary_future_component_touched")
    if any(item.atom in future_atoms for item in stereo.local_orders):
        _fail("component_boundary_future_component_touched")
    for record in stereo.local_orders:
        component_index = component_by_atom.get(record.atom)
        if component_index is None:
            _fail("component_boundary_local_order_state_mismatch")
        if component_index < completed_component_index and not record.closed:
            _fail("component_boundary_local_order_state_mismatch")
        if (
            component_index == completed_component_index
            and record.atom != state.active.atom
            and not record.closed
        ):
            _fail("component_boundary_local_order_state_mismatch")


def _fail(reason: str) -> None:
    raise WriterComponentCompletionReplayError(reason)


__all__ = (
    "WriterComponentCompletionReplayError",
    "replay_completed_component_prefix",
)
