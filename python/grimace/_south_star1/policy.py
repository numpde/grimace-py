"""Finite presentation-policy records for the private proof kernel."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from dataclasses import replace
from enum import Enum

from .facts import MoleculeFacts
from .ids import AtomId
from .ids import BondId


class TetraToken(Enum):
    NONE = ""
    AT = "@"
    ATAT = "@@"


class DirectionMark(Enum):
    ABSENT = 0
    FWD = 1
    REV = -1


class AnnotationMode(Enum):
    HARD = "hard"
    SUPPORT_MAXIMAL = "support_maximal"
    CARDINALITY_MAXIMAL = "cardinality_maximal"
    CANONICAL = "canonical"


class BranchPresentationMode(Enum):
    EXHAUSTIVE = "exhaustive"
    WRITER_SHAPED = "writer_shaped"


@dataclass(frozen=True, slots=True)
class RingLabel:
    value: int

    def __post_init__(self) -> None:
        if self.value < 1:
            raise ValueError("ring label values must be positive")

    def text(self) -> str:
        if self.value <= 9:
            return str(self.value)
        return f"%{self.value:02d}"


@dataclass(frozen=True, slots=True)
class AtomTextChoice:
    name: str
    text_by_tetra: tuple[tuple[TetraToken, str], ...]

    def __post_init__(self) -> None:
        _require_nonempty_tuple("atom text_by_tetra", self.text_by_tetra)
        tokens = [token for token, _ in self.text_by_tetra]
        if len(set(tokens)) != len(tokens):
            raise ValueError(f"atom text choice {self.name!r} repeats tetra tokens")

    def permits(self, token: TetraToken) -> bool:
        return any(candidate == token for candidate, _ in self.text_by_tetra)

    def render(self, token: TetraToken) -> str:
        for candidate, text in self.text_by_tetra:
            if candidate == token:
                return text
        raise KeyError(token)


@dataclass(frozen=True, slots=True)
class BondTextChoice:
    name: str
    base_text: str
    permits_direction: bool


@dataclass(frozen=True, slots=True)
class AtomTextDomain:
    atom: AtomId
    choices: tuple[AtomTextChoice, ...]

    def __post_init__(self) -> None:
        _require_nonempty_tuple("atom text choices", self.choices)


@dataclass(frozen=True, slots=True)
class BondTextDomain:
    bond: BondId
    slot_kind: str
    choices: tuple[BondTextChoice, ...]

    def __post_init__(self) -> None:
        if not self.slot_kind:
            raise ValueError("bond text domain slot_kind must be nonempty")
        _require_nonempty_tuple("bond text choices", self.choices)


@dataclass(frozen=True, slots=True)
class SmilesPolicy:
    ring_labels: tuple[RingLabel, ...]
    annotation_mode: AnnotationMode
    atom_text_domains: tuple[AtomTextDomain, ...]
    bond_text_domains: tuple[BondTextDomain, ...]
    least_free_ring_labels: bool = True
    branch_presentation_mode: BranchPresentationMode = BranchPresentationMode.EXHAUSTIVE

    def validate_for_facts(self, facts: MoleculeFacts) -> None:
        facts.validate()
        _require_nonempty_tuple("ring labels", self.ring_labels)
        if not isinstance(self.branch_presentation_mode, BranchPresentationMode):
            raise ValueError(
                "branch_presentation_mode must be a BranchPresentationMode"
            )
        if len(set(self.ring_labels)) != len(self.ring_labels):
            raise ValueError("ring label domain contains duplicates")

        atom_ids = {atom.id for atom in facts.atoms}
        atom_domain_ids = _unique_domain_ids(
            "atom text domain",
            (domain.atom for domain in self.atom_text_domains),
        )
        if atom_domain_ids != atom_ids:
            missing = atom_ids - atom_domain_ids
            extra = atom_domain_ids - atom_ids
            raise ValueError(
                "atom text domain coverage mismatch: "
                f"missing={missing!r}, extra={extra!r}"
            )

        bond_ids = {bond.id for bond in facts.bonds}
        seen_bond_domains: set[tuple[BondId, str]] = set()
        for domain in self.bond_text_domains:
            if domain.bond not in bond_ids:
                raise ValueError(f"bond text domain has unknown bond {domain.bond!r}")
            key = (domain.bond, domain.slot_kind)
            if key in seen_bond_domains:
                raise ValueError(f"duplicate bond text domain {key!r}")
            seen_bond_domains.add(key)

    def atom_text_domain(
        self,
        facts: MoleculeFacts,
        atom: AtomId,
    ) -> tuple[AtomTextChoice, ...]:
        facts.validate()
        return self.atom_text_domain_unchecked(atom)

    def atom_text_domain_unchecked(
        self,
        atom: AtomId,
    ) -> tuple[AtomTextChoice, ...]:
        for domain in self.atom_text_domains:
            if domain.atom == atom:
                return domain.choices
        raise KeyError(atom)

    def bond_text_domain(
        self,
        facts: MoleculeFacts,
        bond: BondId,
        *,
        slot_kind: str,
    ) -> tuple[BondTextChoice, ...]:
        facts.validate()
        return self.bond_text_domain_unchecked(bond, slot_kind=slot_kind)

    def bond_text_domain_unchecked(
        self,
        bond: BondId,
        *,
        slot_kind: str,
    ) -> tuple[BondTextChoice, ...]:
        for domain in self.bond_text_domains:
            if domain.bond == bond and domain.slot_kind == slot_kind:
                return domain.choices
        raise KeyError((bond, slot_kind))


def with_branch_presentation_mode(
    policy: SmilesPolicy,
    mode: BranchPresentationMode,
) -> SmilesPolicy:
    return replace(policy, branch_presentation_mode=mode)


def _require_nonempty_tuple(label: str, value: tuple[object, ...]) -> None:
    if not isinstance(value, tuple):
        raise TypeError(f"{label} must be a tuple")
    if not value:
        raise ValueError(f"{label} must be nonempty")


def _unique_domain_ids(label: str, values: Iterable[object]) -> set[object]:
    seen: set[object] = set()
    for value in values:
        if value in seen:
            raise ValueError(f"duplicate {label}: {value!r}")
        seen.add(value)
    return seen


__all__ = (
    "AnnotationMode",
    "AtomTextChoice",
    "AtomTextDomain",
    "BondTextChoice",
    "BondTextDomain",
    "BranchPresentationMode",
    "DirectionMark",
    "RingLabel",
    "SmilesPolicy",
    "TetraToken",
    "with_branch_presentation_mode",
)
