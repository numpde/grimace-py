"""Shared traversal root-domain helpers."""

from __future__ import annotations

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


__all__ = ("component_root_domains_for_facts",)
