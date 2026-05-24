"""Finite policy factory for the bounded ordinary South Star dialect."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from .errors import SouthStarError
from .errors import SouthStarErrorKind
from .facts import AtomFacts
from .facts import BondFacts
from .facts import BondOrder
from .facts import LigandKind
from .facts import MoleculeFacts
from .ids import AtomId
from .ids import BondId
from .policy import AnnotationMode
from .policy import AtomTextChoice
from .policy import AtomTextDomain
from .policy import BondTextChoice
from .policy import BondTextDomain
from .policy import RingLabel
from .policy import SmilesPolicy
from .policy import TetraToken
from .slots import BondSlotKind


@dataclass(frozen=True, slots=True)
class OrdinaryPolicyOptions:
    """Options for the first bounded ordinary SMILES presentation policy."""

    ring_label_values: tuple[int, ...] = tuple(range(1, 10))
    annotation_mode: AnnotationMode = AnnotationMode.SUPPORT_MAXIMAL

    single_bond_mode: Literal["elide", "explicit", "both"] = "elide"
    aromatic_bond_mode: Literal["elide", "explicit", "both"] = "elide"
    non_single_ring_closures: Literal["unsupported", "joint"] = "unsupported"

    bracket_all_atoms: bool = False
    allow_aromatic_atoms: bool = True


def ordinary_policy_for_facts(
    facts: MoleculeFacts,
    options: OrdinaryPolicyOptions = OrdinaryPolicyOptions(),
) -> SmilesPolicy:
    """Construct the finite spelling policy for the current ordinary dialect.

    This is deliberately narrow.  Unsupported chemistry or presentation
    variants raise rather than silently broadening the declared South Star
    language.
    """

    facts.validate()
    _validate_options(options)
    _reject_unsupported_ring_closures(facts, options)

    tetra_centers = {site.center for site in facts.stereo.tetrahedral}
    implicit_h_by_tetra_center = _implicit_h_by_tetra_center(facts)

    policy = SmilesPolicy(
        ring_labels=tuple(RingLabel(value) for value in options.ring_label_values),
        annotation_mode=options.annotation_mode,
        atom_text_domains=tuple(
            AtomTextDomain(
                atom=atom.id,
                choices=(
                    _atom_text_choice(
                        atom,
                        tetra_center=atom.id in tetra_centers,
                        implicit_h_count=implicit_h_by_tetra_center.get(atom.id, 0),
                        options=options,
                    ),
                ),
            )
            for atom in facts.atoms
        ),
        bond_text_domains=tuple(_bond_text_domains(facts, options)),
    )
    policy.validate_for_facts(facts)
    return policy


def _validate_options(options: OrdinaryPolicyOptions) -> None:
    if not options.ring_label_values:
        raise ValueError("ordinary policy requires at least one ring label")
    if len(set(options.ring_label_values)) != len(options.ring_label_values):
        raise ValueError("ordinary policy ring labels contain duplicates")
    if any(value < 1 for value in options.ring_label_values):
        raise ValueError("ordinary policy ring labels must be positive")
    if options.bracket_all_atoms:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "bracket_all_atoms is not implemented yet",
        )
    if options.non_single_ring_closures == "joint":
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_POLICY,
            "joint non-single ring closures are not implemented",
        )


def _atom_text_choice(
    atom: AtomFacts,
    *,
    tetra_center: bool,
    implicit_h_count: int,
    options: OrdinaryPolicyOptions,
) -> AtomTextChoice:
    _require_plain_neutral_atom(atom)
    if atom.is_aromatic:
        return _aromatic_atom_choice(atom, options)
    if tetra_center:
        return _tetra_atom_choice(atom, implicit_h_count=implicit_h_count)
    return _organic_atom_choice(atom)


def _require_plain_neutral_atom(atom: AtomFacts) -> None:
    if atom.isotope is not None:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_ATOM,
            f"isotopic atoms are unsupported: {atom.id!r}",
        )
    if atom.formal_charge != 0:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_ATOM,
            f"charged atoms are unsupported: {atom.id!r}",
        )
    if atom.explicit_h_count != 0:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_ATOM,
            f"explicit atom hydrogens are unsupported: {atom.id!r}",
        )
    if atom.no_implicit:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_ATOM,
            f"no-implicit atoms are unsupported: {atom.id!r}",
        )


def _organic_atom_choice(atom: AtomFacts) -> AtomTextChoice:
    if atom.symbol not in _ORGANIC_SUBSET:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_ATOM,
            "ordinary organic-subset spelling is unsupported for "
            f"{atom.symbol!r}",
        )
    return AtomTextChoice(
        name=f"organic_{atom.symbol}",
        text_by_tetra=((TetraToken.NONE, atom.symbol),),
    )


def _aromatic_atom_choice(
    atom: AtomFacts,
    options: OrdinaryPolicyOptions,
) -> AtomTextChoice:
    if not options.allow_aromatic_atoms:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_ATOM,
            f"aromatic atoms are disabled: {atom.id!r}",
        )
    symbol = _AROMATIC_SYMBOLS.get(atom.symbol)
    if symbol is None:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_ATOM,
            f"ordinary aromatic spelling is unsupported for {atom.symbol!r}",
        )
    return AtomTextChoice(
        name=f"aromatic_{symbol}",
        text_by_tetra=((TetraToken.NONE, symbol),),
    )


def _tetra_atom_choice(
    atom: AtomFacts,
    *,
    implicit_h_count: int,
) -> AtomTextChoice:
    if atom.is_aromatic:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_ATOM,
            f"aromatic tetrahedral centers are unsupported: {atom.id!r}",
        )
    if atom.symbol not in _ORGANIC_SUBSET:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_ATOM,
            f"tetrahedral spelling is unsupported for {atom.symbol!r}",
        )
    if implicit_h_count not in {0, 1}:
        raise SouthStarError(
            SouthStarErrorKind.UNSUPPORTED_ATOM,
            f"tetrahedral center {atom.id!r} has unsupported implicit-H count "
            f"{implicit_h_count}",
        )

    hydrogen = "H" if implicit_h_count == 1 else ""
    return AtomTextChoice(
        name=f"tetra_{atom.symbol}",
        text_by_tetra=(
            (TetraToken.NONE, atom.symbol),
            (TetraToken.AT, f"[{atom.symbol}@{hydrogen}]"),
            (TetraToken.ATAT, f"[{atom.symbol}@@{hydrogen}]"),
        ),
    )


def _bond_text_domains(
    facts: MoleculeFacts,
    options: OrdinaryPolicyOptions,
) -> tuple[BondTextDomain, ...]:
    domains: list[BondTextDomain] = []
    for bond in facts.bonds:
        tree_choices = _tree_bond_choices(bond, options)
        domains.append(
            BondTextDomain(
                bond=bond.id,
                slot_kind=BondSlotKind.TREE.value,
                choices=tree_choices,
            )
        )

        ring_choices = _ring_endpoint_bond_choices(bond, options)
        if ring_choices is not None:
            domains.append(
                BondTextDomain(
                    bond=bond.id,
                    slot_kind=BondSlotKind.RING_ENDPOINT.value,
                    choices=ring_choices,
                )
            )

    return tuple(domains)


def _tree_bond_choices(
    bond: BondFacts,
    options: OrdinaryPolicyOptions,
) -> tuple[BondTextChoice, ...]:
    if bond.order is BondOrder.SINGLE:
        return _single_bond_choices(options)
    if bond.order is BondOrder.DOUBLE:
        return (BondTextChoice(name="double", base_text="=", permits_direction=False),)
    if bond.order is BondOrder.TRIPLE:
        return (BondTextChoice(name="triple", base_text="#", permits_direction=False),)
    if bond.order is BondOrder.AROMATIC:
        return _aromatic_bond_choices(options)
    raise SouthStarError(
        SouthStarErrorKind.UNSUPPORTED_BOND,
        f"unsupported bond order: {bond.order!r}",
    )


def _ring_endpoint_bond_choices(
    bond: BondFacts,
    options: OrdinaryPolicyOptions,
) -> tuple[BondTextChoice, ...] | None:
    if bond.order is BondOrder.SINGLE:
        return _single_bond_choices(options)
    if bond.order is BondOrder.AROMATIC:
        return _aromatic_bond_choices(options)
    if options.non_single_ring_closures == "unsupported":
        return None
    raise SouthStarError(
        SouthStarErrorKind.UNSUPPORTED_POLICY,
        "joint non-single ring closures are not implemented",
    )


def _single_bond_choices(
    options: OrdinaryPolicyOptions,
) -> tuple[BondTextChoice, ...]:
    choices = []
    if options.single_bond_mode in {"elide", "both"}:
        choices.append(
            BondTextChoice(
                name="single_elided_or_directional",
                base_text="",
                permits_direction=True,
            )
        )
    if options.single_bond_mode in {"explicit", "both"}:
        choices.append(
            BondTextChoice(
                name="single_explicit_or_directional",
                base_text="-",
                permits_direction=True,
            )
        )
    return tuple(choices)


def _aromatic_bond_choices(
    options: OrdinaryPolicyOptions,
) -> tuple[BondTextChoice, ...]:
    choices = []
    if options.aromatic_bond_mode in {"elide", "both"}:
        choices.append(
            BondTextChoice(
                name="aromatic_elided",
                base_text="",
                permits_direction=False,
            )
        )
    if options.aromatic_bond_mode in {"explicit", "both"}:
        choices.append(
            BondTextChoice(
                name="aromatic_explicit",
                base_text=":",
                permits_direction=False,
            )
        )
    return tuple(choices)


def _reject_unsupported_ring_closures(
    facts: MoleculeFacts,
    options: OrdinaryPolicyOptions,
) -> None:
    if options.non_single_ring_closures != "unsupported":
        return

    for bond in facts.bonds:
        if bond.order in {BondOrder.SINGLE, BondOrder.AROMATIC}:
            continue
        if _bond_can_be_ring_closure(facts, bond):
            raise SouthStarError(
                SouthStarErrorKind.UNSUPPORTED_POLICY,
                "ordinary policy does not yet support non-single ring closures "
                f"for bond {bond.id!r}",
            )


def _bond_can_be_ring_closure(
    facts: MoleculeFacts,
    target: BondFacts,
) -> bool:
    adjacency: dict[AtomId, set[AtomId]] = {atom.id: set() for atom in facts.atoms}
    for bond in facts.bonds:
        if bond.id == target.id:
            continue
        adjacency[bond.a].add(bond.b)
        adjacency[bond.b].add(bond.a)

    seen = {target.a}
    stack = [target.a]
    while stack:
        atom = stack.pop()
        for neighbor in adjacency[atom]:
            if neighbor == target.b:
                return True
            if neighbor in seen:
                continue
            seen.add(neighbor)
            stack.append(neighbor)
    return False


def _implicit_h_by_tetra_center(facts: MoleculeFacts) -> dict[AtomId, int]:
    occurrences = {occurrence.id: occurrence for occurrence in facts.ligand_occurrences}
    counts: dict[AtomId, int] = {}
    for site in facts.stereo.tetrahedral:
        counts[site.center] = sum(
            1
            for occurrence_id in site.ligand_occurrences
            if occurrences[occurrence_id].kind is LigandKind.IMPLICIT_H
        )
    return counts


_ORGANIC_SUBSET = frozenset({"B", "C", "N", "O", "P", "S", "F", "Cl", "Br", "I"})
_AROMATIC_SYMBOLS = {
    "B": "b",
    "C": "c",
    "N": "n",
    "O": "o",
    "P": "p",
    "S": "s",
}


__all__ = ("OrdinaryPolicyOptions", "ordinary_policy_for_facts")
