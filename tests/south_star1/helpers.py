"""Shared hand-built fixtures for South Star 1 proof-kernel tests."""

from __future__ import annotations

from grimace._south_star1.facts import AtomFacts
from grimace._south_star1.facts import BondFacts
from grimace._south_star1.facts import BondOrder
from grimace._south_star1.facts import ComponentFacts
from grimace._south_star1.facts import MoleculeFacts
from grimace._south_star1.ids import AtomId
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import ComponentId
from grimace._south_star1.policy import AnnotationMode
from grimace._south_star1.policy import AtomTextChoice
from grimace._south_star1.policy import AtomTextDomain
from grimace._south_star1.policy import BondTextChoice
from grimace._south_star1.policy import RingLabel
from grimace._south_star1.policy import SmilesPolicy
from grimace._south_star1.policy import TetraToken


def cco_facts() -> MoleculeFacts:
    return MoleculeFacts(
        atoms=(atom(0, "C"), atom(1, "C"), atom(2, "O")),
        bonds=(
            single_bond(0, 0, 1),
            single_bond(1, 1, 2),
        ),
        components=(
            ComponentFacts(
                id=ComponentId(0),
                atoms=(AtomId(0), AtomId(1), AtomId(2)),
                bonds=(BondId(0), BondId(1)),
            ),
        ),
    )


def atom(idx: int, symbol: str) -> AtomFacts:
    return AtomFacts(
        id=AtomId(idx),
        atomic_num={"C": 6, "O": 8, "F": 9, "Cl": 17, "Br": 35}[symbol],
        symbol=symbol,
        isotope=None,
        formal_charge=0,
        is_aromatic=False,
        explicit_h_count=0,
        implicit_h_count=0,
        no_implicit=False,
    )


def single_bond(idx: int, a: int, b: int) -> BondFacts:
    return bond(idx, a, b, BondOrder.SINGLE)


def bond(idx: int, a: int, b: int, order: BondOrder) -> BondFacts:
    return BondFacts(
        id=BondId(idx),
        a=AtomId(a),
        b=AtomId(b),
        order=order,
        is_aromatic=False,
        is_conjugated=False,
    )


def organic_subset_policy(facts: MoleculeFacts) -> SmilesPolicy:
    return SmilesPolicy(
        ring_labels=(RingLabel(1), RingLabel(2)),
        annotation_mode=AnnotationMode.HARD,
        atom_text_domains=tuple(
            AtomTextDomain(
                atom=atom_facts.id,
                choices=(organic_atom_choice(atom_facts.symbol),),
            )
            for atom_facts in facts.atoms
        ),
        bond_text_domains=(),
    )


def organic_atom_choice(symbol: str) -> AtomTextChoice:
    return AtomTextChoice(
        name=f"organic_{symbol}",
        text_by_tetra=((TetraToken.NONE, symbol),),
    )


def empty_bond_choice() -> BondTextChoice:
    return BondTextChoice(name="elided_single", base_text="", permits_direction=False)
