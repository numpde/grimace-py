"""Finite witness search for the South Star 1 proof kernel.

This module is RDKit-free.

It turns the existing proof-kernel pieces into an executable non-stereo
enumerator:

    MoleculeFacts + SmilesPolicy + ParserSemantics
        -> TraversalSkeleton
        -> SlotBundle
        -> TraversalAssignment
        -> ValidWitness

It deliberately does not parse, sanitize, canonicalize, or repair rendered
strings.  Candidate rejection happens only through the declared finite
constraints and parser-semantics relations.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from hashlib import blake2b
from itertools import product
from typing import TypeVar

from .annotation import ValidWitness
from .constraints import TraversalAssignment
from .constraints import validate_nonstereo_traversal_witness
from .enumerate import SupportImage
from .enumerate import render_image_from_witnesses
from .facts import MoleculeFacts
from .graph_index import build_graph_index
from .ids import AtomId
from .ids import BondSlotId
from .ids import CarrierSlotId
from .ids import RingEndpointId
from .policy import AtomTextChoice
from .policy import BondTextChoice
from .policy import DirectionMark
from .policy import RingLabel
from .policy import SmilesPolicy
from .policy import TetraToken
from .render import render_nonstereo_traversal
from .semantics import ParserSemantics
from .skeleton import TraversalSkeleton
from .skeleton import enumerate_traversal_skeletons
from .slots import SlotBundle
from .slots import allocate_traversal_slots


K = TypeVar("K")
V = TypeVar("V")


@dataclass(frozen=True, slots=True)
class WitnessSearchStats:
    """Optional lightweight instrumentation for debugging search blowup."""

    skeleton_count: int
    candidate_assignment_count: int
    valid_witness_count: int


def enumerate_nonstereo_support(
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
) -> SupportImage:
    """Enumerate the rendered support image for the current non-stereo slice.

    This is the convenient API for tests and small examples.  The lower-level
    generator is ``enumerate_nonstereo_witnesses``.
    """

    witnesses = enumerate_nonstereo_witnesses(facts, policy, semantics)
    return render_image_from_witnesses(witnesses)


def enumerate_nonstereo_witnesses(
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
) -> Iterator[ValidWitness]:
    """Yield valid rendered witnesses for non-stereo molecule facts.

    The function intentionally supports the full traversal skeleton generator,
    including ring/cotree edges, but it rejects stereo facts through
    ``validate_nonstereo_traversal_witness``.

    Every yielded witness has passed:
      - fact validation;
      - policy validation;
      - bounded ring-label validation;
      - atom decode relations;
      - bond/ring-pair decode relations;
      - non-stereo token/marker invariants.
    """

    facts.validate()
    policy.validate_for_facts(facts)
    index = build_graph_index(facts)

    for skeleton in enumerate_traversal_skeletons(facts, index, policy):
        slots = allocate_traversal_slots(facts, skeleton)

        for assignment in enumerate_nonstereo_assignments(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            policy=policy,
        ):
            try:
                constraints = validate_nonstereo_traversal_witness(
                    facts=facts,
                    skeleton=skeleton,
                    slots=slots,
                    assignment=assignment,
                    policy=policy,
                    semantics=semantics,
                )
            except ValueError:
                # This is ordinary finite-CSP rejection, not post-hoc parsing.
                continue

            rendered = render_nonstereo_traversal(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                assignment=assignment,
                policy=policy,
                semantics=semantics,
            )

            yield ValidWitness(
                id=_witness_id(skeleton, slots, assignment, rendered),
                rendered=rendered,
                annotation_count=_annotation_count(assignment),
                constraints=constraints,
            )


def enumerate_nonstereo_assignments(
    *,
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    slots: SlotBundle,
    policy: SmilesPolicy,
) -> Iterator[TraversalAssignment]:
    """Generate finite non-stereo assignment candidates for one skeleton.

    This generator deliberately does not call ``semantics``.  It enumerates
    policy-domain candidates only.  Semantic rejection is performed by the
    validator in ``enumerate_nonstereo_witnesses``.
    """

    tetra_tokens = {
        atom.id: TetraToken.NONE
        for atom in facts.atoms
    }

    direction_marks = {
        carrier.id: DirectionMark.ABSENT
        for carrier in slots.carrier_slots
    }

    atom_domains = tuple(
        (
            atom.id,
            tuple(
                choice
                for choice in policy.atom_text_domain(facts, atom.id)
                if choice.permits(TetraToken.NONE)
            ),
        )
        for atom in facts.atoms
    )

    bond_domains = tuple(
        (
            slot.id,
            policy.bond_text_domain(
                facts,
                slot.bond,
                slot_kind=slot.kind.value,
            ),
        )
        for slot in slots.bond_slots
    )

    for atom_text in _dict_product(atom_domains):
        for bond_text in _dict_product(bond_domains):
            for ring_labels in enumerate_ring_label_assignments(policy, slots):
                yield TraversalAssignment(
                    atom_text=atom_text,
                    tetra_tokens=tetra_tokens,
                    bond_text=bond_text,
                    ring_labels=ring_labels,
                    direction_marks=direction_marks,
                )


def enumerate_ring_label_assignments(
    policy: SmilesPolicy,
    slots: SlotBundle,
) -> Iterator[dict[RingEndpointId, RingLabel]]:
    """Enumerate bounded ring-label assignments for a fixed slot bundle.

    The assignment domain is endpoint-indexed because the existing
    ``TraversalAssignment`` stores ``ring_labels`` as:

        dict[RingEndpointId, RingLabel]

    The generator assigns the same label to both endpoints of a ring bond and
    enforces non-overlap for reused labels. The ring-label validator remains
    the source of truth, including least-free normalization when requested;
    this generator mirrors that policy by construction.
    """

    if not slots.ring_endpoints:
        yield {}
        return

    intervals = _ring_intervals(slots)
    labels = policy.ring_labels
    out: dict[RingEndpointId, RingLabel] = {}
    chosen: list[tuple[int, int, RingLabel]] = []

    def active_labels_at(position: int) -> set[RingLabel]:
        return {
            label
            for start, end, label in chosen
            if start < position < end
        }

    def rec(i: int) -> Iterator[dict[RingEndpointId, RingLabel]]:
        if i == len(intervals):
            yield dict(out)
            return

        interval = intervals[i]
        active = active_labels_at(interval.start)
        candidates = tuple(label for label in labels if label not in active)

        if policy.least_free_ring_labels:
            if not candidates:
                return
            candidates = (min(candidates, key=lambda label: label.value),)

        for label in candidates:
            out[interval.endpoint_1] = label
            out[interval.endpoint_2] = label
            chosen.append((interval.start, interval.end, label))

            yield from rec(i + 1)

            chosen.pop()
            del out[interval.endpoint_1]
            del out[interval.endpoint_2]

    yield from rec(0)


@dataclass(frozen=True, slots=True)
class _RingInterval:
    endpoint_1: RingEndpointId
    endpoint_2: RingEndpointId
    start: int
    end: int


def _ring_intervals(slots: SlotBundle) -> tuple[_RingInterval, ...]:
    by_bond: dict[object, list[object]] = {}

    for endpoint in slots.ring_endpoints:
        by_bond.setdefault(endpoint.bond, []).append(endpoint)

    intervals: list[_RingInterval] = []

    for bond, endpoints in by_bond.items():
        if len(endpoints) != 2:
            raise ValueError(
                f"ring bond {bond!r} has {len(endpoints)} endpoints, not two"
            )

        left, right = sorted(
            endpoints,
            key=lambda endpoint: endpoint.syntax_position,
        )

        intervals.append(
            _RingInterval(
                endpoint_1=left.id,
                endpoint_2=right.id,
                start=left.syntax_position,
                end=right.syntax_position,
            )
        )

    return tuple(sorted(intervals, key=lambda interval: interval.start))


def _dict_product(
    domains: tuple[tuple[K, tuple[V, ...]], ...],
) -> Iterator[dict[K, V]]:
    """Cartesian product of finite keyed domains.

    Empty domain list yields one empty assignment.
    Any empty value domain yields no assignments.
    """

    if not domains:
        yield {}
        return

    keys = tuple(key for key, _ in domains)
    value_domains = tuple(values for _, values in domains)

    if any(not values for values in value_domains):
        return

    for values in product(*value_domains):
        yield dict(zip(keys, values, strict=True))


def _annotation_count(assignment: TraversalAssignment) -> int:
    return sum(
        mark is not DirectionMark.ABSENT
        for mark in assignment.direction_marks.values()
    )


def _witness_id(
    skeleton: TraversalSkeleton,
    slots: SlotBundle,
    assignment: TraversalAssignment,
    rendered: str,
) -> str:
    """Stable-ish witness id for debugging.

    This is not a canonical chemical identifier and should not be used as a
    support quotient. The rendered support image preserves witness multiplicity.
    """

    payload = repr(
        (
            skeleton.roots,
            tuple(sorted(skeleton.parent.items(), key=lambda item: int(item[0]))),
            tuple(sorted((int(k), v.name) for k, v in assignment.atom_text.items())),
            tuple(sorted((int(k), v.name) for k, v in assignment.bond_text.items())),
            tuple(sorted((int(k), v.value) for k, v in assignment.ring_labels.items())),
            tuple(sorted((int(k), v.value) for k, v in assignment.direction_marks.items())),
            rendered,
        )
    ).encode("utf8")

    digest = blake2b(payload, digest_size=12).hexdigest()
    return f"witness:{digest}"


__all__ = (
    "WitnessSearchStats",
    "enumerate_nonstereo_assignments",
    "enumerate_nonstereo_support",
    "enumerate_nonstereo_witnesses",
    "enumerate_ring_label_assignments",
)
