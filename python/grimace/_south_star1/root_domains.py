"""Shared traversal root-domain helpers."""

from __future__ import annotations

from collections.abc import Mapping

from .facts import MoleculeFacts
from .ids import AtomId
from .ids import ComponentId


def component_root_domains_for_facts(
    facts: MoleculeFacts,
    rooted_at_atom: AtomId | None,
) -> tuple[tuple[ComponentId, tuple[AtomId, ...]], ...]:
    facts.validate()
    if rooted_at_atom is None:
        return tuple((component.id, component.atoms) for component in facts.components)
    domains: list[tuple[ComponentId, tuple[AtomId, ...]]] = []
    found = False
    for component in facts.components:
        if rooted_at_atom in component.atoms:
            domains.append((component.id, (rooted_at_atom,)))
            found = True
        else:
            domains.append((component.id, component.atoms))
    if not found:
        raise ValueError(f"rooted atom is not present in any component: {rooted_at_atom!r}")
    return tuple(domains)


def component_root_domains_from_metadata(
    *,
    all_root_domains: tuple[tuple[ComponentId, tuple[AtomId, ...]], ...],
    atom_component_map: Mapping[AtomId, ComponentId],
    rooted_at_atom: AtomId | None,
) -> tuple[tuple[ComponentId, tuple[AtomId, ...]], ...]:
    if rooted_at_atom is None:
        return all_root_domains
    component_id = atom_component_map.get(rooted_at_atom)
    if component_id is None:
        raise ValueError(f"rooted atom is not present in any component: {rooted_at_atom!r}")
    return tuple(
        (
            current_component_id,
            (rooted_at_atom,) if current_component_id == component_id else atoms,
        )
        for current_component_id, atoms in all_root_domains
    )


__all__ = (
    "component_root_domains_for_facts",
    "component_root_domains_from_metadata",
)
