"""Finite stereo CSP construction and solving for the South Star 1 kernel.

This module is RDKit-free.

It builds the stereo part of the finite constraint problem induced by:

    MoleculeFacts
    TraversalSkeleton
    SlotBundle
    PresentationPrefix
    SmilesPolicy
    ParserSemantics

The module does not parse, sanitize, canonicalize, or repair rendered strings.
It enumerates finite stereo assignments over:

    - tetrahedral tokens: NONE, @, @@
    - directional carrier marks: ABSENT, /, \\

and constrains them by:

    - atom decode relations involving tetra tokens;
    - tree-bond decode relations involving directional marks;
    - ring-pair decode relations involving directional marks;
    - tetrahedral site relations;
    - directional site relations, including potential-but-unspecified sites;
    - annotation-policy selection, including support-wise maximality.

The output of this module is still assignment-level, not string-level.  Rendering
and rendered-support deduplication remain separate layers.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from itertools import product
from typing import Any

from .constraints import TraversalAssignment
from .facts import (
    DirectionalValue,
    MoleculeFacts,
    SiteStatus,
    TetraValue,
)
from .ids import AtomId
from .ids import BondSlotId
from .ids import CarrierSlotId
from .ids import RingEndpointId
from .ids import SiteId
from .policy import AnnotationMode
from .policy import AtomTextChoice
from .policy import BondTextChoice
from .policy import DirectionMark
from .policy import RingLabel
from .policy import SmilesPolicy
from .policy import TetraToken
from .ring_labels import validate_bounded_ring_labels
from .semantics import INVALID
from .semantics import Invalid
from .semantics import ParserSemantics
from .skeleton import TraversalSkeleton
from .slots import BondSlotKind
from .slots import SlotBundle
from .slots import carrier_slot_by_bond_slot
from .slots import ring_bond_slots_by_bond


@dataclass(frozen=True, slots=True)
class PresentationPrefix:
    """Syntax choices fixed before stereo-variable solving.

    A presentation prefix fixes atom text, non-directional bond text, and ring
    labels.  It deliberately does not fix tetrahedral tokens or directional
    carrier marks.
    """

    atom_text: dict[AtomId, AtomTextChoice]
    bond_text: dict[BondSlotId, BondTextChoice]
    ring_labels: dict[RingEndpointId, RingLabel]


@dataclass(frozen=True, slots=True)
class TetraSiteRelation:
    """Unary finite relation for one tetrahedral site."""

    site: SiteId
    center: AtomId
    target: TetraValue
    allowed_tokens: frozenset[TetraToken]


@dataclass(frozen=True, slots=True)
class MarkRelation:
    """Finite relation over directional carrier marks.

    The relation is generic enough to represent:

      - unary tree-bond decode constraints;
      - binary ring-pair decode constraints;
      - hyperedge directional-stereo site constraints.

    The row order is the order of ``scope``.
    """

    name: str
    subject: str
    scope: tuple[CarrierSlotId, ...]
    allowed_rows: frozenset[tuple[DirectionMark, ...]]

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("mark relation name must be nonempty")
        if not self.subject:
            raise ValueError("mark relation subject must be nonempty")
        if len(set(self.scope)) != len(self.scope):
            raise ValueError(
                f"mark relation {self.name!r}/{self.subject!r} repeats carriers"
            )
        bad_rows = {
            row for row in self.allowed_rows if len(row) != len(self.scope)
        }
        if bad_rows:
            raise ValueError(
                f"mark relation {self.name!r}/{self.subject!r} has rows "
                f"with incorrect arity: {bad_rows!r}"
            )


@dataclass(frozen=True, slots=True)
class StereoCSP:
    """The finite stereo CSP for one skeleton and one presentation prefix."""

    tetra_domains: dict[AtomId, tuple[TetraToken, ...]]
    direction_domains: dict[CarrierSlotId, tuple[DirectionMark, ...]]

    tetra_relations: tuple[TetraSiteRelation, ...]
    tree_bond_mark_relations: tuple[MarkRelation, ...]
    ring_pair_mark_relations: tuple[MarkRelation, ...]
    directional_site_relations: tuple[MarkRelation, ...]

    eligible_marker_carriers: frozenset[CarrierSlotId]

    def mark_relations(self) -> tuple[MarkRelation, ...]:
        """All directional-mark relations in one tuple."""

        return (
            self.tree_bond_mark_relations
            + self.ring_pair_mark_relations
            + self.directional_site_relations
        )


@dataclass(frozen=True, slots=True)
class StereoSolution:
    """A complete stereo-variable assignment for one StereoCSP."""

    tetra_tokens: dict[AtomId, TetraToken]
    direction_marks: dict[CarrierSlotId, DirectionMark]

    @property
    def marker_support(self) -> frozenset[CarrierSlotId]:
        """Carrier slots whose directional marker is not omitted."""

        return frozenset(
            carrier
            for carrier, mark in self.direction_marks.items()
            if mark is not DirectionMark.ABSENT
        )


def build_stereo_csp(
    *,
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    slots: SlotBundle,
    prefix: PresentationPrefix,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    eligible_marker_carriers: frozenset[CarrierSlotId] | None = None,
    allow_global_directional_scope: bool = False,
) -> StereoCSP:
    """Build the finite stereo CSP for one traversal skeleton and prefix.

    ``eligible_marker_carriers`` may be supplied by a presentation policy.  If it
    is omitted, eligibility is the union of carrier scopes of specified
    directional sites.

    For a rigorous South Star semantics, the concrete semantics object should
    provide::

        directional_scope(facts, skeleton, slots, site) -> tuple[CarrierSlotId, ...]

    If no such method is present and directional sites exist, this function
    raises by default.  Setting ``allow_global_directional_scope=True`` uses all
    carrier slots as every directional site's scope.  That fallback is finite and
    useful for diagnostics, but it is usually too coarse for maximal annotation
    policy.
    """

    facts.validate()
    policy.validate_for_facts(facts)
    _validate_prefix(facts, slots, prefix, policy)

    directional_scopes = _directional_scopes(
        facts=facts,
        skeleton=skeleton,
        slots=slots,
        semantics=semantics,
        allow_global_directional_scope=allow_global_directional_scope,
    )

    if eligible_marker_carriers is None:
        eligible_marker_carriers = _default_eligible_marker_carriers(
            facts=facts,
            directional_scopes=directional_scopes,
        )
    else:
        _validate_carrier_subset(
            "eligible marker carriers",
            eligible_marker_carriers,
            slots,
        )

    tetra_domains = _tetra_token_domains(
        facts=facts,
        slots=slots,
        prefix=prefix,
        semantics=semantics,
    )

    direction_domains = _direction_mark_domains(
        slots=slots,
        prefix=prefix,
        eligible_marker_carriers=eligible_marker_carriers,
    )

    tetra_relations = _tetra_site_relations(
        facts=facts,
        skeleton=skeleton,
        slots=slots,
        tetra_domains=tetra_domains,
        semantics=semantics,
    )

    tree_bond_relations = _tree_bond_mark_relations(
        facts=facts,
        slots=slots,
        prefix=prefix,
        direction_domains=direction_domains,
        semantics=semantics,
    )

    ring_pair_relations = _ring_pair_mark_relations(
        facts=facts,
        slots=slots,
        prefix=prefix,
        direction_domains=direction_domains,
        semantics=semantics,
    )

    directional_relations = _directional_site_relations(
        facts=facts,
        skeleton=skeleton,
        slots=slots,
        direction_domains=direction_domains,
        directional_scopes=directional_scopes,
        semantics=semantics,
    )

    return StereoCSP(
        tetra_domains=tetra_domains,
        direction_domains=direction_domains,
        tetra_relations=tetra_relations,
        tree_bond_mark_relations=tree_bond_relations,
        ring_pair_mark_relations=ring_pair_relations,
        directional_site_relations=directional_relations,
        eligible_marker_carriers=eligible_marker_carriers,
    )


def solve_stereo_csp(csp: StereoCSP) -> Iterator[StereoSolution]:
    """Enumerate all satisfying assignments of a finite StereoCSP."""

    for tetra_tokens in _solve_tetra_part(csp):
        for direction_marks in _solve_direction_part(csp):
            yield StereoSolution(
                tetra_tokens=tetra_tokens,
                direction_marks=direction_marks,
            )


def select_stereo_solutions(
    *,
    csp: StereoCSP,
    solutions: tuple[StereoSolution, ...],
    mode: AnnotationMode,
) -> tuple[StereoSolution, ...]:
    """Apply the annotation policy to already-valid stereo solutions.

    Selection is performed inside one CSP instance: same molecule facts, same
    skeleton, same slots, same atom text, same bond text, and same ring labels.
    Carrier-slot identities should not be compared across different skeletons.

    ``SUPPORT_MAXIMAL`` keeps all inclusion-maximal marker supports.
    ``CARDINALITY_MAXIMAL`` keeps all largest-cardinality marker supports.
    ``CANONICAL`` first applies support-wise maximality and then chooses one
    deterministic representative.
    """

    if mode is AnnotationMode.HARD:
        return solutions

    if mode is AnnotationMode.SUPPORT_MAXIMAL:
        return _support_maximal_solutions(solutions)

    if mode is AnnotationMode.CARDINALITY_MAXIMAL:
        if not solutions:
            return ()
        max_size = max(len(solution.marker_support) for solution in solutions)
        return tuple(
            solution
            for solution in solutions
            if len(solution.marker_support) == max_size
        )

    if mode is AnnotationMode.CANONICAL:
        maximal = _support_maximal_solutions(solutions)
        if not maximal:
            return ()
        return (min(maximal, key=_canonical_stereo_solution_key),)

    raise ValueError(f"unknown annotation mode: {mode!r}")


def enumerate_stereo_assignments_for_prefix(
    *,
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    slots: SlotBundle,
    prefix: PresentationPrefix,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    eligible_marker_carriers: frozenset[CarrierSlotId] | None = None,
    allow_global_directional_scope: bool = False,
) -> Iterator[TraversalAssignment]:
    """Yield full TraversalAssignment objects for one presentation prefix.

    This is the most convenient API for the eventual witness-search layer.
    """

    csp = build_stereo_csp(
        facts=facts,
        skeleton=skeleton,
        slots=slots,
        prefix=prefix,
        policy=policy,
        semantics=semantics,
        eligible_marker_carriers=eligible_marker_carriers,
        allow_global_directional_scope=allow_global_directional_scope,
    )

    raw_solutions = tuple(solve_stereo_csp(csp))
    selected_solutions = select_stereo_solutions(
        csp=csp,
        solutions=raw_solutions,
        mode=policy.annotation_mode,
    )

    for solution in selected_solutions:
        yield assignment_from_prefix_solution(prefix, solution)


def assignment_from_prefix_solution(
    prefix: PresentationPrefix,
    solution: StereoSolution,
) -> TraversalAssignment:
    """Combine fixed presentation choices with a stereo solution."""

    return TraversalAssignment(
        atom_text=dict(prefix.atom_text),
        tetra_tokens=dict(solution.tetra_tokens),
        bond_text=dict(prefix.bond_text),
        ring_labels=dict(prefix.ring_labels),
        direction_marks=dict(solution.direction_marks),
    )


def _validate_prefix(
    facts: MoleculeFacts,
    slots: SlotBundle,
    prefix: PresentationPrefix,
    policy: SmilesPolicy,
) -> None:
    atom_ids = {atom.id for atom in facts.atoms}
    bond_slot_ids = {slot.id for slot in slots.bond_slots}

    _require_exact_keys("prefix atom text", set(prefix.atom_text), atom_ids)
    _require_exact_keys("prefix bond text", set(prefix.bond_text), bond_slot_ids)

    for atom in facts.atoms:
        domain = policy.atom_text_domain(facts, atom.id)
        choice = prefix.atom_text[atom.id]
        if choice not in domain:
            raise ValueError(
                f"atom text choice {choice!r} is outside policy domain "
                f"for atom {atom.id!r}"
            )

    for slot in slots.bond_slots:
        domain = policy.bond_text_domain(
            facts,
            slot.bond,
            slot_kind=slot.kind.value,
        )
        choice = prefix.bond_text[slot.id]
        if choice not in domain:
            raise ValueError(
                f"bond text choice {choice!r} is outside policy domain "
                f"for bond slot {slot.id!r}"
            )

    validate_bounded_ring_labels(policy, slots, prefix.ring_labels)


def _tetra_token_domains(
    *,
    facts: MoleculeFacts,
    slots: SlotBundle,
    prefix: PresentationPrefix,
    semantics: ParserSemantics,
) -> dict[AtomId, tuple[TetraToken, ...]]:
    tetra_sites_by_center = _tetra_sites_by_center(facts)
    incident_texts = _incident_bond_texts(facts, slots, prefix)

    domains: dict[AtomId, tuple[TetraToken, ...]] = {}

    for atom in facts.atoms:
        site = tetra_sites_by_center.get(atom.id)

        if site is None:
            raw_domain = (TetraToken.NONE,)
        elif site.status is SiteStatus.SPECIFIED:
            raw_domain = (TetraToken.AT, TetraToken.ATAT)
        else:
            raw_domain = (TetraToken.NONE,)

        atom_text = prefix.atom_text[atom.id]

        domains[atom.id] = tuple(
            token
            for token in raw_domain
            if atom_text.permits(token)
            and semantics.atom_decode_ok(
                facts,
                atom.id,
                atom_text,
                token,
                incident_texts[atom.id],
            )
        )

    return domains


def _tetra_sites_by_center(facts: MoleculeFacts) -> dict[AtomId, Any]:
    out: dict[AtomId, Any] = {}

    for site in facts.stereo.tetrahedral:
        if site.center in out:
            raise ValueError(
                f"multiple tetrahedral sites share center {site.center!r}"
            )
        out[site.center] = site

    return out


def _incident_bond_texts(
    facts: MoleculeFacts,
    slots: SlotBundle,
    prefix: PresentationPrefix,
) -> dict[AtomId, tuple[BondTextChoice, ...]]:
    incident: dict[AtomId, list[BondTextChoice]] = {
        atom.id: [] for atom in facts.atoms
    }

    for slot in slots.bond_slots:
        text = prefix.bond_text[slot.id]
        incident[slot.written_from].append(text)
        if slot.written_to is not None:
            incident[slot.written_to].append(text)

    return {
        atom_id: tuple(texts)
        for atom_id, texts in incident.items()
    }


def _directional_scopes(
    *,
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    slots: SlotBundle,
    semantics: ParserSemantics,
    allow_global_directional_scope: bool,
) -> dict[SiteId, tuple[CarrierSlotId, ...]]:
    scopes: dict[SiteId, tuple[CarrierSlotId, ...]] = {}

    if not facts.stereo.directional:
        return scopes

    scope_method = getattr(semantics, "directional_scope", None)

    if scope_method is None:
        if not allow_global_directional_scope:
            raise TypeError(
                "directional stereo requires semantics.directional_scope(...). "
                "Pass allow_global_directional_scope=True only for diagnostic "
                "fallback behavior."
            )

        global_scope = tuple(carrier.id for carrier in slots.carrier_slots)
        for site in facts.stereo.directional:
            scopes[site.id] = global_scope
        return scopes

    carrier_ids = {carrier.id for carrier in slots.carrier_slots}

    for site in facts.stereo.directional:
        scope = tuple(scope_method(facts, skeleton, slots, site.id))

        if len(set(scope)) != len(scope):
            raise ValueError(
                f"directional scope for site {site.id!r} repeats carriers"
            )

        unknown = set(scope) - carrier_ids
        if unknown:
            raise ValueError(
                f"directional scope for site {site.id!r} has unknown carriers "
                f"{unknown!r}"
            )

        scopes[site.id] = scope

    return scopes


def _default_eligible_marker_carriers(
    *,
    facts: MoleculeFacts,
    directional_scopes: Mapping[SiteId, tuple[CarrierSlotId, ...]],
) -> frozenset[CarrierSlotId]:
    eligible: set[CarrierSlotId] = set()

    for site in facts.stereo.directional:
        if site.status is SiteStatus.SPECIFIED:
            eligible.update(directional_scopes[site.id])

    return frozenset(eligible)


def _validate_carrier_subset(
    label: str,
    carriers: frozenset[CarrierSlotId],
    slots: SlotBundle,
) -> None:
    known = {carrier.id for carrier in slots.carrier_slots}
    unknown = set(carriers) - known
    if unknown:
        raise ValueError(f"{label} contain unknown carrier slots: {unknown!r}")


def _direction_mark_domains(
    *,
    slots: SlotBundle,
    prefix: PresentationPrefix,
    eligible_marker_carriers: frozenset[CarrierSlotId],
) -> dict[CarrierSlotId, tuple[DirectionMark, ...]]:
    bond_slot_by_id = {slot.id: slot for slot in slots.bond_slots}
    domains: dict[CarrierSlotId, tuple[DirectionMark, ...]] = {}

    for carrier in slots.carrier_slots:
        bond_slot = bond_slot_by_id[carrier.bond_slot]
        bond_text = prefix.bond_text[bond_slot.id]

        if (
            carrier.id in eligible_marker_carriers
            and bond_text.permits_direction
        ):
            domains[carrier.id] = (
                DirectionMark.ABSENT,
                DirectionMark.FWD,
                DirectionMark.REV,
            )
        else:
            domains[carrier.id] = (DirectionMark.ABSENT,)

    return domains


def _tetra_site_relations(
    *,
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    slots: SlotBundle,
    tetra_domains: Mapping[AtomId, tuple[TetraToken, ...]],
    semantics: ParserSemantics,
) -> tuple[TetraSiteRelation, ...]:
    relations: list[TetraSiteRelation] = []

    for site in facts.stereo.tetrahedral:
        local_order = semantics.local_tetra_order(
            facts,
            skeleton,
            slots,
            site.id,
        )

        allowed: set[TetraToken] = set()

        for token in tetra_domains[site.center]:
            value = semantics.tetra_value(
                facts,
                site.id,
                local_order,
                token,
            )

            if not _is_invalid(value) and value == site.target:
                allowed.add(token)

        relations.append(
            TetraSiteRelation(
                site=site.id,
                center=site.center,
                target=site.target,
                allowed_tokens=frozenset(allowed),
            )
        )

    return tuple(relations)


def _tree_bond_mark_relations(
    *,
    facts: MoleculeFacts,
    slots: SlotBundle,
    prefix: PresentationPrefix,
    direction_domains: Mapping[CarrierSlotId, tuple[DirectionMark, ...]],
    semantics: ParserSemantics,
) -> tuple[MarkRelation, ...]:
    carrier_by_bond_slot = carrier_slot_by_bond_slot(slots)
    relations: list[MarkRelation] = []

    for slot in slots.bond_slots:
        if slot.kind is not BondSlotKind.TREE:
            continue

        carrier = carrier_by_bond_slot[slot.id]
        bond_text = prefix.bond_text[slot.id]

        allowed_rows = frozenset(
            (mark,)
            for mark in direction_domains[carrier.id]
            if semantics.bond_decode_ok(
                facts,
                slot.bond,
                bond_text,
                mark,
            )
        )

        relations.append(
            MarkRelation(
                name="tree_bond_decode",
                subject=f"bond_slot:{int(slot.id)}",
                scope=(carrier.id,),
                allowed_rows=allowed_rows,
            )
        )

    return tuple(relations)


def _ring_pair_mark_relations(
    *,
    facts: MoleculeFacts,
    slots: SlotBundle,
    prefix: PresentationPrefix,
    direction_domains: Mapping[CarrierSlotId, tuple[DirectionMark, ...]],
    semantics: ParserSemantics,
) -> tuple[MarkRelation, ...]:
    carrier_by_bond_slot = carrier_slot_by_bond_slot(slots)
    ring_slots_by_bond = ring_bond_slots_by_bond(slots)
    relations: list[MarkRelation] = []

    for bond, (endpoint_1, endpoint_2) in ring_slots_by_bond.items():
        carrier_1 = carrier_by_bond_slot[endpoint_1.id]
        carrier_2 = carrier_by_bond_slot[endpoint_2.id]
        text_1 = prefix.bond_text[endpoint_1.id]
        text_2 = prefix.bond_text[endpoint_2.id]

        allowed: set[tuple[DirectionMark, DirectionMark]] = set()

        for mark_1, mark_2 in product(
            direction_domains[carrier_1.id],
            direction_domains[carrier_2.id],
        ):
            if semantics.ring_pair_decode_ok(
                facts,
                bond,
                text_1,
                mark_1,
                text_2,
                mark_2,
            ):
                allowed.add((mark_1, mark_2))

        relations.append(
            MarkRelation(
                name="ring_pair_decode",
                subject=f"bond:{int(bond)}",
                scope=(carrier_1.id, carrier_2.id),
                allowed_rows=frozenset(allowed),
            )
        )

    return tuple(relations)


def _directional_site_relations(
    *,
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    slots: SlotBundle,
    direction_domains: Mapping[CarrierSlotId, tuple[DirectionMark, ...]],
    directional_scopes: Mapping[SiteId, tuple[CarrierSlotId, ...]],
    semantics: ParserSemantics,
) -> tuple[MarkRelation, ...]:
    relations: list[MarkRelation] = []

    for site in facts.stereo.directional:
        scope = directional_scopes[site.id]
        allowed: set[tuple[DirectionMark, ...]] = set()
        scope_domains = tuple(direction_domains[carrier] for carrier in scope)

        for row in product(*scope_domains):
            marks = dict(zip(scope, row, strict=True))

            value = semantics.directional_value(
                facts,
                skeleton,
                slots,
                site.id,
                marks,
            )

            if not _is_invalid(value) and value == site.target:
                allowed.add(row)

        relations.append(
            MarkRelation(
                name="directional_site",
                subject=f"site:{int(site.id)}",
                scope=scope,
                allowed_rows=frozenset(allowed),
            )
        )

    return tuple(relations)


def _solve_tetra_part(csp: StereoCSP) -> Iterator[dict[AtomId, TetraToken]]:
    atoms = tuple(csp.tetra_domains)

    def rec(
        index: int,
        out: dict[AtomId, TetraToken],
    ) -> Iterator[dict[AtomId, TetraToken]]:
        if not _tetra_relations_still_possible(csp, out):
            return

        if index == len(atoms):
            if _tetra_relations_ok(csp, out):
                yield dict(out)
            return

        atom = atoms[index]

        for token in csp.tetra_domains[atom]:
            out[atom] = token
            yield from rec(index + 1, out)
            del out[atom]

    yield from rec(0, {})


def _tetra_relations_still_possible(
    csp: StereoCSP,
    partial: Mapping[AtomId, TetraToken],
) -> bool:
    for relation in csp.tetra_relations:
        if not relation.allowed_tokens:
            return False

        token = partial.get(relation.center)
        if token is not None and token not in relation.allowed_tokens:
            return False

    return True


def _tetra_relations_ok(
    csp: StereoCSP,
    assignment: Mapping[AtomId, TetraToken],
) -> bool:
    for atom, domain in csp.tetra_domains.items():
        if atom not in assignment:
            return False
        if assignment[atom] not in domain:
            return False

    for relation in csp.tetra_relations:
        if assignment[relation.center] not in relation.allowed_tokens:
            return False

    return True


def _solve_direction_part(
    csp: StereoCSP,
) -> Iterator[dict[CarrierSlotId, DirectionMark]]:
    carriers = _carrier_order(csp)

    def rec(
        index: int,
        out: dict[CarrierSlotId, DirectionMark],
    ) -> Iterator[dict[CarrierSlotId, DirectionMark]]:
        if not _mark_relations_still_possible(csp, out):
            return

        if index == len(carriers):
            if _mark_relations_ok(csp, out):
                yield dict(out)
            return

        carrier = carriers[index]

        for mark in csp.direction_domains[carrier]:
            out[carrier] = mark
            yield from rec(index + 1, out)
            del out[carrier]

    yield from rec(0, {})


def _carrier_order(csp: StereoCSP) -> tuple[CarrierSlotId, ...]:
    membership_count: dict[CarrierSlotId, int] = {
        carrier: 0 for carrier in csp.direction_domains
    }

    for relation in csp.mark_relations():
        for carrier in relation.scope:
            membership_count[carrier] = membership_count.get(carrier, 0) + 1

    return tuple(
        sorted(
            csp.direction_domains,
            key=lambda carrier: (
                -membership_count.get(carrier, 0),
                len(csp.direction_domains[carrier]),
                int(carrier),
            ),
        )
    )


def _mark_relations_still_possible(
    csp: StereoCSP,
    partial: Mapping[CarrierSlotId, DirectionMark],
) -> bool:
    for relation in csp.mark_relations():
        if not relation.allowed_rows:
            return False

        assigned_positions = tuple(
            (position, carrier)
            for position, carrier in enumerate(relation.scope)
            if carrier in partial
        )

        if not assigned_positions:
            continue

        has_extension = any(
            all(
                row[position] == partial[carrier]
                for position, carrier in assigned_positions
            )
            for row in relation.allowed_rows
        )

        if not has_extension:
            return False

    return True


def _mark_relations_ok(
    csp: StereoCSP,
    assignment: Mapping[CarrierSlotId, DirectionMark],
) -> bool:
    for carrier, domain in csp.direction_domains.items():
        if carrier not in assignment:
            return False
        if assignment[carrier] not in domain:
            return False

    for relation in csp.mark_relations():
        row = tuple(assignment[carrier] for carrier in relation.scope)
        if row not in relation.allowed_rows:
            return False

    return True


def _support_maximal_solutions(
    solutions: tuple[StereoSolution, ...],
) -> tuple[StereoSolution, ...]:
    return tuple(
        candidate
        for candidate in solutions
        if not any(
            candidate.marker_support < other.marker_support
            for other in solutions
        )
    )


def _canonical_stereo_solution_key(
    solution: StereoSolution,
) -> tuple[object, ...]:
    return (
        -len(solution.marker_support),
        tuple(sorted(int(carrier) for carrier in solution.marker_support)),
        tuple(
            sorted(
                (int(atom), token.value)
                for atom, token in solution.tetra_tokens.items()
            )
        ),
        tuple(
            sorted(
                (int(carrier), mark.value)
                for carrier, mark in solution.direction_marks.items()
            )
        ),
    )


def _is_invalid(value: object) -> bool:
    return value is INVALID or isinstance(value, Invalid)


def _require_exact_keys(
    label: str,
    actual: set[object],
    expected: set[object],
) -> None:
    if actual != expected:
        missing = expected - actual
        extra = actual - expected
        raise ValueError(
            f"{label} coverage mismatch: missing={missing!r}, extra={extra!r}"
        )


__all__ = (
    "MarkRelation",
    "PresentationPrefix",
    "StereoCSP",
    "StereoSolution",
    "TetraSiteRelation",
    "assignment_from_prefix_solution",
    "build_stereo_csp",
    "enumerate_stereo_assignments_for_prefix",
    "select_stereo_solutions",
    "solve_stereo_csp",
)
