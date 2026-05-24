"""Stereo witness enumeration for the South Star 1 proof kernel.

This module is RDKit-free.

It is the first end-to-end stereo enumeration layer:

    TraversalSkeleton
        -> SlotBundle
        -> PresentationPrefix
        -> StereoCSP solutions
        -> TraversalAssignment
        -> rendered ValidWitness

It does not parse, sanitize, canonicalize, or repair generated strings.  It also
does not implement stereo semantics directly; it delegates the stereo constraint
problem to ``stereo_csp.py`` and delegates string construction to
``render_stereo_traversal``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from itertools import product
from typing import TypeVar

from .annotation import ValidWitness
from .certificates import WitnessCertificate
from .certificates import certify_traversal_assignment
from .constraints import TraversalAssignment
from .constraints import validate_stereo_traversal_witness
from .enumerate import SupportImage
from .enumerate import render_image_from_witnesses
from .facts import MoleculeFacts
from .ids import AtomId
from .ids import BondSlotId
from .ids import CarrierSlotId
from .policy import AtomTextChoice
from .policy import BondTextChoice
from .policy import DirectionMark
from .policy import SmilesPolicy
from .proof_terms import assignment_key
from .proof_terms import prefix_key
from .proof_terms import skeleton_key
from .proof_terms import witness_id
from .render import render_stereo_traversal
from .ring_labels import enumerate_ring_label_assignments
from .semantics import ParserSemantics
from .skeleton import TraversalSkeleton
from .slots import SlotBundle
from .slots import allocate_traversal_slots
from .stereo_csp import PresentationPrefix
from .stereo_csp import assignment_from_prefix_solution
from .stereo_csp import build_stereo_csp
from .stereo_csp import certify_stereo_solution
from .stereo_csp import enumerate_stereo_assignments_for_prefix
from .stereo_csp import select_stereo_solutions_with_certificates
from .stereo_csp import solve_stereo_csp


K = TypeVar("K")
V = TypeVar("V")


@dataclass(frozen=True, slots=True)
class StereoWitnessSearchStats:
    """Optional instrumentation for tests and debugging.

    The main generator is streaming, so it does not populate this automatically.
    Tests can use ``collect_stereo_witnesses_for_skeleton`` if counts are useful.
    """

    prefix_count: int
    witness_count: int


@dataclass(frozen=True, slots=True)
class CertifiedStereoWitnessSearchStats:
    prefix_count: int
    csp_count: int
    feasible_solution_count: int
    selected_solution_count: int
    witness_count: int


@dataclass(frozen=True, slots=True)
class CertifiedWitness:
    witness: ValidWitness
    assignment: TraversalAssignment
    certificate: WitnessCertificate


def enumerate_stereo_support_for_skeleton(
    *,
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    eligible_marker_carriers: frozenset[CarrierSlotId] | None = None,
    allow_global_directional_scope: bool = False,
) -> SupportImage:
    """Convenience wrapper returning the rendered support image for one skeleton."""

    witnesses = enumerate_stereo_witnesses_for_skeleton(
        facts=facts,
        skeleton=skeleton,
        policy=policy,
        semantics=semantics,
        eligible_marker_carriers=eligible_marker_carriers,
        allow_global_directional_scope=allow_global_directional_scope,
    )

    return render_image_from_witnesses(witnesses)


def enumerate_stereo_witnesses_for_skeleton(
    *,
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    slots: SlotBundle | None = None,
    eligible_marker_carriers: frozenset[CarrierSlotId] | None = None,
    allow_global_directional_scope: bool = False,
) -> Iterator[ValidWitness]:
    """Yield stereo-valid witnesses for one traversal skeleton.

    This is the intended next South Star layer.

    The function deliberately does not catch ``ValueError`` from the stereo
    validator or renderer.  If the CSP construction produced an assignment that
    later validation rejects, that is a bug in the declared finite model or in
    the implementation.  Silently treating that rejection as an ordinary filter
    would recreate the post-hoc validation pattern South Star is meant to avoid.
    """

    facts.validate()
    policy.validate_for_facts(facts)

    if slots is None:
        slots = allocate_traversal_slots(facts, skeleton)

    for prefix in enumerate_presentation_prefixes(
        facts=facts,
        slots=slots,
        policy=policy,
    ):
        for assignment in enumerate_stereo_assignments_for_prefix(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            prefix=prefix,
            policy=policy,
            semantics=semantics,
            eligible_marker_carriers=eligible_marker_carriers,
            allow_global_directional_scope=allow_global_directional_scope,
        ):
            constraints = validate_stereo_traversal_witness(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                assignment=assignment,
                policy=policy,
                semantics=semantics,
            )

            rendered = render_stereo_traversal(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                assignment=assignment,
                policy=policy,
                semantics=semantics,
                validate=False,
            )

            yield ValidWitness(
                id=witness_id(
                    skeleton=skeleton,
                    slots=slots,
                    assignment=assignment,
                    rendered=rendered,
                ),
                rendered=rendered,
                annotation_count=_annotation_count(assignment),
                constraints=constraints,
            )


def enumerate_certified_stereo_witnesses_for_skeleton(
    *,
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    slots: SlotBundle | None = None,
    eligible_marker_carriers: frozenset[CarrierSlotId] | None = None,
    allow_global_directional_scope: bool = False,
) -> Iterator[CertifiedWitness]:
    """Yield stereo witnesses with finite CSP proof objects."""

    facts.validate()
    policy.validate_for_facts(facts)

    if slots is None:
        slots = allocate_traversal_slots(facts, skeleton)

    for prefix in enumerate_presentation_prefixes(
        facts=facts,
        slots=slots,
        policy=policy,
    ):
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
        selected_solutions = select_stereo_solutions_with_certificates(
            csp=csp,
            solutions=raw_solutions,
            mode=policy.annotation_mode,
        )

        for selected in selected_solutions:
            yield build_certified_witness_from_selected_solution(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                prefix=prefix,
                policy=policy,
                semantics=semantics,
                csp=csp,
                selected=selected,
            )


def collect_certified_stereo_witnesses_for_skeleton(
    *,
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    slots: SlotBundle | None = None,
    eligible_marker_carriers: frozenset[CarrierSlotId] | None = None,
    allow_global_directional_scope: bool = False,
) -> tuple[tuple[CertifiedWitness, ...], CertifiedStereoWitnessSearchStats]:
    """Materialize certified witnesses and exact per-skeleton solver counts."""

    facts.validate()
    policy.validate_for_facts(facts)

    if slots is None:
        slots = allocate_traversal_slots(facts, skeleton)

    prefix_count = 0
    csp_count = 0
    feasible_solution_count = 0
    selected_solution_count = 0
    witnesses: list[CertifiedWitness] = []

    for prefix in enumerate_presentation_prefixes(
        facts=facts,
        slots=slots,
        policy=policy,
    ):
        prefix_count += 1
        csp_count += 1
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
        feasible_solution_count += len(raw_solutions)
        selected_solutions = select_stereo_solutions_with_certificates(
            csp=csp,
            solutions=raw_solutions,
            mode=policy.annotation_mode,
        )
        selected_solution_count += len(selected_solutions)

        for selected in selected_solutions:
            witnesses.append(
                build_certified_witness_from_selected_solution(
                    facts=facts,
                    skeleton=skeleton,
                    slots=slots,
                    prefix=prefix,
                    policy=policy,
                    semantics=semantics,
                    selected=selected,
                    csp=csp,
                )
            )

    return (
        tuple(witnesses),
        CertifiedStereoWitnessSearchStats(
            prefix_count=prefix_count,
            csp_count=csp_count,
            feasible_solution_count=feasible_solution_count,
            selected_solution_count=selected_solution_count,
            witness_count=len(witnesses),
        ),
    )


def collect_stereo_witnesses_for_skeleton(
    *,
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    slots: SlotBundle | None = None,
    eligible_marker_carriers: frozenset[CarrierSlotId] | None = None,
    allow_global_directional_scope: bool = False,
) -> tuple[tuple[ValidWitness, ...], StereoWitnessSearchStats]:
    """Materialize witnesses and return lightweight counts.

    This is intended for tests and debugging, not for large support enumeration.
    """

    if slots is None:
        slots = allocate_traversal_slots(facts, skeleton)

    prefix_count = 0
    witnesses: list[ValidWitness] = []

    for prefix in enumerate_presentation_prefixes(
        facts=facts,
        slots=slots,
        policy=policy,
    ):
        prefix_count += 1

        for assignment in enumerate_stereo_assignments_for_prefix(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            prefix=prefix,
            policy=policy,
            semantics=semantics,
            eligible_marker_carriers=eligible_marker_carriers,
            allow_global_directional_scope=allow_global_directional_scope,
        ):
            constraints = validate_stereo_traversal_witness(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                assignment=assignment,
                policy=policy,
                semantics=semantics,
            )

            rendered = render_stereo_traversal(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                assignment=assignment,
                policy=policy,
                semantics=semantics,
                validate=False,
            )

            witnesses.append(
                ValidWitness(
                    id=witness_id(
                        skeleton=skeleton,
                        slots=slots,
                        assignment=assignment,
                        rendered=rendered,
                    ),
                    rendered=rendered,
                    annotation_count=_annotation_count(assignment),
                    constraints=constraints,
                )
            )

    return (
        tuple(witnesses),
        StereoWitnessSearchStats(
            prefix_count=prefix_count,
            witness_count=len(witnesses),
        ),
    )


def enumerate_presentation_prefixes(
    *,
    facts: MoleculeFacts,
    slots: SlotBundle,
    policy: SmilesPolicy,
) -> Iterator[PresentationPrefix]:
    """Enumerate finite presentation prefixes for one slot bundle.

    A prefix fixes atom text, non-directional bond text, and ring labels.  It
    intentionally does not fix tetrahedral tokens or directional carrier marks.

    Semantic feasibility is not checked here.  For example, an atom text choice
    that permits no valid tetra token for a particular skeleton is still a
    presentation-domain candidate; the stereo CSP will reject it by having an
    empty relation or domain.
    """

    facts.validate()
    policy.validate_for_facts(facts)

    atom_domains = tuple(
        (atom.id, policy.atom_text_domain(facts, atom.id))
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

    for ring_labels in enumerate_ring_label_assignments(
        slots=slots,
        policy=policy,
    ):
        for atom_text in _dict_product(atom_domains):
            for bond_text in _dict_product(bond_domains):
                yield PresentationPrefix(
                    atom_text=atom_text,
                    bond_text=bond_text,
                    ring_labels=ring_labels,
                )


def _dict_product(
    domains: tuple[tuple[K, tuple[V, ...]], ...],
) -> Iterator[dict[K, V]]:
    """Cartesian product of finite keyed domains."""

    if not domains:
        yield {}
        return

    keys = tuple(key for key, _ in domains)
    value_domains = tuple(values for _, values in domains)

    if any(not values for values in value_domains):
        return

    for values in product(*value_domains):
            yield dict(zip(keys, values, strict=True))


def build_certified_witness_from_selected_solution(
    *,
    facts: MoleculeFacts,
    skeleton: TraversalSkeleton,
    slots: SlotBundle,
    prefix: PresentationPrefix,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    csp,
    selected,
) -> CertifiedWitness:
    assignment = assignment_from_prefix_solution(prefix, selected.solution)
    constraints = validate_stereo_traversal_witness(
        facts=facts,
        skeleton=skeleton,
        slots=slots,
        assignment=assignment,
        policy=policy,
        semantics=semantics,
    )
    rendered = render_stereo_traversal(
        facts=facts,
        skeleton=skeleton,
        slots=slots,
        assignment=assignment,
        policy=policy,
        semantics=semantics,
        validate=False,
    )
    witness = ValidWitness(
        id=witness_id(
            skeleton=skeleton,
            slots=slots,
            assignment=assignment,
            rendered=rendered,
        ),
        rendered=rendered,
        annotation_count=_annotation_count(assignment),
        constraints=constraints,
    )
    stereo_certificate = certify_stereo_solution(
        csp=csp,
        solution=selected.solution,
        annotation_certificate=selected.certificate,
    )
    traversal_certificate = certify_traversal_assignment(
        facts=facts,
        skeleton=skeleton,
        slots=slots,
        assignment=assignment,
        policy=policy,
        semantics=semantics,
    )
    return CertifiedWitness(
        witness=witness,
        assignment=assignment,
        certificate=WitnessCertificate(
            witness_id=witness.id,
            rendered=rendered,
            skeleton_key=skeleton_key(skeleton),
            prefix_key=prefix_key(prefix),
            assignment_key=assignment_key(assignment),
            traversal_relation_certificates=traversal_certificate,
            stereo_solution=stereo_certificate,
        ),
    )


def _annotation_count(assignment: TraversalAssignment) -> int:
    return sum(
        mark is not DirectionMark.ABSENT
        for mark in assignment.direction_marks.values()
    )


__all__ = (
    "CertifiedStereoWitnessSearchStats",
    "CertifiedWitness",
    "StereoWitnessSearchStats",
    "build_certified_witness_from_selected_solution",
    "collect_certified_stereo_witnesses_for_skeleton",
    "collect_stereo_witnesses_for_skeleton",
    "enumerate_certified_stereo_witnesses_for_skeleton",
    "enumerate_presentation_prefixes",
    "enumerate_stereo_support_for_skeleton",
    "enumerate_stereo_witnesses_for_skeleton",
)
