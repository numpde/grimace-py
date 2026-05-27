"""End-to-end stereo support enumeration for the South Star 1 kernel.

This module is RDKit-free.

It closes the current proof-kernel path:

    MoleculeFacts
        -> GraphIndex
        -> TraversalSkeletons
        -> SlotBundle
        -> PresentationPrefix
        -> StereoCSP
        -> TraversalAssignment
        -> render_stereo_traversal(...)
        -> ValidWitness
        -> SupportImage

The module does not parse, sanitize, canonicalize, or repair rendered strings.
Semantic rejection must happen inside the declared finite constraints used by
the lower layers.
"""

from __future__ import annotations

from collections.abc import Callable
from collections.abc import Iterable
from collections.abc import Iterator
from dataclasses import dataclass

from .annotation import ValidWitness
from .certificates import SupportEnumerationManifest
from .certificates import manifest_from_jsonable
from .certificates import manifest_to_jsonable
from .certificates import witness_certificate_from_jsonable
from .certificates import witness_certificate_to_jsonable
from .constraints import NamedConstraint
from .constraints import TraversalAssignment
from .enumerate import SupportImage
from .enumerate import render_image_from_witnesses
from .enumeration_trace import AcceptanceCertificate
from .enumeration_trace import EnumerationNodeId
from .enumeration_trace import EnumerationTrace
from .enumeration_trace import RejectionCertificate
from .enumeration_trace import enumeration_trace_from_jsonable
from .enumeration_trace import enumeration_trace_to_jsonable
from .enumeration_trace import rejection_annotation_not_selected
from .enumeration_trace import rejection_csp_unsatisfied
from .enumeration_trace import rejection_empty_direction_domain
from .enumeration_trace import rejection_empty_mark_relation
from .enumeration_trace import rejection_empty_tetra_domain
from .enumeration_trace import rejection_empty_tetra_relation
from .enumeration_trace import rejection_render_duplicate
from .facts import MoleculeFacts
from .graph_index import build_graph_index
from .ids import CarrierSlotId
from .ids import AtomId
from .ids import BondSlotId
from .ids import RingEndpointId
from .policy import SmilesPolicy
from .policy import AtomTextChoice
from .policy import BondTextChoice
from .policy import DirectionMark
from .policy import RingLabel
from .policy import TetraToken
from .proof_terms import csp_key
from .proof_terms import render_duplicate_node_id
from .proof_terms import sequence_hash
from .proof_terms import skeleton_key
from .proof_terms import stereo_solution_key
from .proof_terms import witness_node_id
from .semantics import ParserSemantics
from .skeleton import TraversalSkeleton
from .skeleton import enumerate_traversal_skeletons
from .slots import SlotBundle
from .slots import allocate_traversal_slots
from .stereo_witness import CertifiedWitness
from .stereo_witness import build_certified_witness_from_selected_solution
from .stereo_witness import collect_certified_stereo_witnesses_for_skeleton
from .stereo_witness import enumerate_certified_stereo_witnesses_for_skeleton
from .stereo_witness import enumerate_presentation_prefixes
from .stereo_witness import enumerate_stereo_witnesses_for_skeleton
from .stereo_csp import build_stereo_csp
from .stereo_csp import select_stereo_solutions_with_certificates
from .stereo_csp import solve_stereo_csp


EligibleMarkerCarrierSelector = Callable[
    [MoleculeFacts, TraversalSkeleton, SlotBundle],
    frozenset[CarrierSlotId] | None,
]


@dataclass(frozen=True, slots=True)
class StereoSupportStats:
    """Small materialized statistics for an end-to-end stereo enumeration."""

    skeleton_count: int
    witness_count: int
    distinct_count: int


@dataclass(frozen=True, slots=True)
class StereoSupportResult:
    """Support image plus all-skeleton enumeration statistics."""

    image: SupportImage
    stats: StereoSupportStats


@dataclass(frozen=True, slots=True)
class CertifiedSupportImage:
    support: SupportImage
    certified_witnesses: tuple[CertifiedWitness, ...]
    manifest: SupportEnumerationManifest


@dataclass(frozen=True, slots=True)
class TracedCertifiedSupportImage:
    support: SupportImage
    certified_witnesses: tuple[CertifiedWitness, ...]
    manifest: SupportEnumerationManifest
    trace: EnumerationTrace


def enumerate_stereo_witnesses(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    skeletons: Iterable[TraversalSkeleton] | None = None,
    eligible_marker_carriers: EligibleMarkerCarrierSelector | None = None,
    allow_global_directional_scope: bool = False,
    validate_inputs: bool = True,
) -> Iterator[ValidWitness]:
    """Yield stereo-valid witnesses over all traversal skeletons.

    This is the streaming end-to-end witness API.

    Parameters
    ----------
    facts:
        Immutable molecule facts.  RDKit is not consulted.

    policy:
        Finite presentation policy.

    semantics:
        Declared finite parser semantics.

    skeletons:
        Optional externally supplied traversal skeletons.  If omitted, the
        function enumerates all traversal skeletons using
        ``enumerate_traversal_skeletons``.

    eligible_marker_carriers:
        Optional per-skeleton/per-slot-bundle selector.  CarrierSlotId values
        are local to a SlotBundle, so this must be a callback rather than a
        global carrier set.

        If omitted, ``stereo_csp.build_stereo_csp`` uses its default eligibility
        policy: carriers in scopes of specified directional sites.

    allow_global_directional_scope:
        Diagnostic escape hatch passed through to the stereo CSP.  The rigorous
        path should provide concrete directional scopes through the semantics
        object instead of relying on global scopes.
    """

    if validate_inputs:
        facts.validate()
        policy.validate_for_facts(facts)

    if skeletons is None:
        index = build_graph_index(facts)
        skeleton_iterable = enumerate_traversal_skeletons(
            facts=facts,
            index=index,
            policy=policy,
            validate_inputs=validate_inputs,
        )
    else:
        skeleton_iterable = skeletons

    for skeleton in skeleton_iterable:
        slots = allocate_traversal_slots(
            facts,
            skeleton,
            validate_inputs=validate_inputs,
        )

        if eligible_marker_carriers is None:
            eligible = None
        else:
            eligible = eligible_marker_carriers(facts, skeleton, slots)

        yield from enumerate_stereo_witnesses_for_skeleton(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            policy=policy,
            semantics=semantics,
            eligible_marker_carriers=eligible,
            allow_global_directional_scope=allow_global_directional_scope,
            validate_inputs=validate_inputs,
        )


def enumerate_certified_stereo_witnesses(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    skeletons: Iterable[TraversalSkeleton] | None = None,
    eligible_marker_carriers: EligibleMarkerCarrierSelector | None = None,
    allow_global_directional_scope: bool = False,
) -> Iterator[CertifiedWitness]:
    """Yield certified stereo witnesses over all traversal skeletons."""

    facts.validate()
    policy.validate_for_facts(facts)

    if skeletons is None:
        index = build_graph_index(facts)
        skeleton_iterable = enumerate_traversal_skeletons(
            facts=facts,
            index=index,
            policy=policy,
        )
    else:
        skeleton_iterable = skeletons

    for skeleton in skeleton_iterable:
        slots = allocate_traversal_slots(facts, skeleton)

        if eligible_marker_carriers is None:
            eligible = None
        else:
            eligible = eligible_marker_carriers(facts, skeleton, slots)

        yield from enumerate_certified_stereo_witnesses_for_skeleton(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            policy=policy,
            semantics=semantics,
            eligible_marker_carriers=eligible,
            allow_global_directional_scope=allow_global_directional_scope,
        )


def enumerate_stereo_support(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    skeletons: Iterable[TraversalSkeleton] | None = None,
    eligible_marker_carriers: EligibleMarkerCarrierSelector | None = None,
    allow_global_directional_scope: bool = False,
    validate_inputs: bool = True,
) -> SupportImage:
    """Enumerate the rendered stereo support image over all skeletons.

    This is the convenient end-to-end API:

        facts -> skeletons -> slots -> prefixes -> CSP -> assignments
        -> render -> support image

    The returned ``SupportImage`` stores unique rendered strings. Use
    ``enumerate_stereo_witnesses`` if witness multiplicity is needed.
    """

    witnesses = enumerate_stereo_witnesses(
        facts=facts,
        policy=policy,
        semantics=semantics,
        skeletons=skeletons,
        eligible_marker_carriers=eligible_marker_carriers,
        allow_global_directional_scope=allow_global_directional_scope,
        validate_inputs=validate_inputs,
    )

    return render_image_from_witnesses(witnesses)


def enumerate_certified_stereo_support(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    skeletons: Iterable[TraversalSkeleton] | None = None,
    eligible_marker_carriers: EligibleMarkerCarrierSelector | None = None,
    allow_global_directional_scope: bool = False,
) -> CertifiedSupportImage:
    """Enumerate certified witnesses and their unique rendered support image."""

    facts.validate()
    policy.validate_for_facts(facts)

    if skeletons is None:
        index = build_graph_index(facts)
        skeleton_tuple = enumerate_traversal_skeletons(
            facts=facts,
            index=index,
            policy=policy,
        )
    else:
        skeleton_tuple = tuple(skeletons)

    certified_list: list[CertifiedWitness] = []
    prefix_count = 0
    csp_count = 0
    feasible_solution_count = 0
    selected_solution_count = 0
    for skeleton in skeleton_tuple:
        slots = allocate_traversal_slots(facts, skeleton)
        if eligible_marker_carriers is None:
            eligible = None
        else:
            eligible = eligible_marker_carriers(facts, skeleton, slots)
        witnesses, stats = collect_certified_stereo_witnesses_for_skeleton(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            policy=policy,
            semantics=semantics,
            eligible_marker_carriers=eligible,
            allow_global_directional_scope=allow_global_directional_scope,
        )
        certified_list.extend(witnesses)
        prefix_count += stats.prefix_count
        csp_count += stats.csp_count
        feasible_solution_count += stats.feasible_solution_count
        selected_solution_count += stats.selected_solution_count

    certified = tuple(certified_list)
    support = render_image_from_witnesses(
        certified_witness.witness
        for certified_witness in certified
    )
    return CertifiedSupportImage(
        support=support,
        certified_witnesses=certified,
        manifest=_support_manifest(
            skeleton_count=len(skeleton_tuple),
            prefix_count=prefix_count,
            csp_count=csp_count,
            feasible_solution_count=feasible_solution_count,
            selected_solution_count=selected_solution_count,
            support=support,
            certified_witnesses=certified,
        ),
    )


def enumerate_traced_certified_stereo_support(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    skeletons: Iterable[TraversalSkeleton] | None = None,
    eligible_marker_carriers: EligibleMarkerCarrierSelector | None = None,
    allow_global_directional_scope: bool = False,
) -> TracedCertifiedSupportImage:
    """Enumerate support with an explicit finite accept/reject ledger."""

    facts.validate()
    policy.validate_for_facts(facts)

    if skeletons is None:
        index = build_graph_index(facts)
        skeleton_tuple = enumerate_traversal_skeletons(
            facts=facts,
            index=index,
            policy=policy,
        )
    else:
        skeleton_tuple = tuple(skeletons)

    certified_list: list[CertifiedWitness] = []
    accepted: list[AcceptanceCertificate] = []
    rejected: list[RejectionCertificate] = []
    prefix_count = 0
    csp_count = 0
    feasible_solution_count = 0
    selected_solution_count = 0

    for skeleton in skeleton_tuple:
        slots = allocate_traversal_slots(facts, skeleton)
        if eligible_marker_carriers is None:
            eligible = None
        else:
            eligible = eligible_marker_carriers(facts, skeleton, slots)

        for prefix in enumerate_presentation_prefixes(
            facts=facts,
            slots=slots,
            policy=policy,
        ):
            prefix_count += 1
            csp_count += 1
            node_key = csp_key(skeleton, prefix)
            csp = build_stereo_csp(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                prefix=prefix,
                policy=policy,
                semantics=semantics,
                eligible_marker_carriers=eligible,
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

            if not raw_solutions:
                rejected.append(_empty_csp_rejection(csp, node_key))
                continue

            selected_keys = {
                stereo_solution_key(selected.solution)
                for selected in selected_solutions
            }
            for solution in raw_solutions:
                solution_node_key = stereo_solution_key(solution)
                if solution_node_key in selected_keys:
                    continue
                rejected.append(
                    rejection_annotation_not_selected(
                        node=EnumerationNodeId(
                            kind="stereo_solution",
                            key=(node_key, solution_node_key),
                        ),
                        support=solution.marker_support,
                        selected_supports=frozenset(
                            selected.solution.marker_support
                            for selected in selected_solutions
                        ),
                        mode=policy.annotation_mode,
                    )
                )

            for selected in selected_solutions:
                certified = build_certified_witness_from_selected_solution(
                    facts=facts,
                    skeleton=skeleton,
                    slots=slots,
                    prefix=prefix,
                    policy=policy,
                    semantics=semantics,
                    selected=selected,
                    csp=csp,
                )
                certified_list.append(certified)
                accepted.append(
                    AcceptanceCertificate(
                        node=witness_node_id(certified.witness.id),
                        witness_id=certified.witness.id,
                        rendered=certified.witness.rendered,
                    )
                )

    certified = tuple(certified_list)
    support = render_image_from_witnesses(
        certified_witness.witness
        for certified_witness in certified
    )
    rejected.extend(_render_duplicate_rejections(certified))
    manifest = _support_manifest(
        skeleton_count=len(skeleton_tuple),
        prefix_count=prefix_count,
        csp_count=csp_count,
        feasible_solution_count=feasible_solution_count,
        selected_solution_count=selected_solution_count,
        support=support,
        certified_witnesses=certified,
    )
    trace = EnumerationTrace(
        accepted=tuple(accepted),
        rejected=tuple(rejected),
        skeleton_count=len(skeleton_tuple),
        prefix_count=prefix_count,
        csp_count=csp_count,
        feasible_solution_count=feasible_solution_count,
        selected_solution_count=selected_solution_count,
        witness_count=len(certified),
        support_count=len(support.strings),
    )
    return TracedCertifiedSupportImage(
        support=support,
        certified_witnesses=certified,
        manifest=manifest,
        trace=trace,
    )


def traced_certified_support_to_jsonable(
    result: TracedCertifiedSupportImage,
) -> dict[str, object]:
    return {
        "support": {
            "witness_count": result.support.witness_count,
            "distinct_count": result.support.distinct_count,
            "strings": list(result.support.strings),
        },
        "manifest": manifest_to_jsonable(result.manifest),
        "trace": enumeration_trace_to_jsonable(result.trace),
        "certified_witnesses": [
            _certified_witness_to_jsonable(certified)
            for certified in result.certified_witnesses
        ],
    }


def traced_certified_support_from_jsonable(
    data: dict[str, object],
) -> TracedCertifiedSupportImage:
    support_data = _require_mapping(data["support"])
    return TracedCertifiedSupportImage(
        support=SupportImage(
            witness_count=int(support_data["witness_count"]),
            distinct_count=int(support_data["distinct_count"]),
            strings=tuple(str(item) for item in _require_list(support_data["strings"])),
        ),
        certified_witnesses=tuple(
            _certified_witness_from_jsonable(item)
            for item in _require_list(data["certified_witnesses"])
        ),
        manifest=manifest_from_jsonable(_require_mapping(data["manifest"])),
        trace=enumeration_trace_from_jsonable(_require_mapping(data["trace"])),
    )


def _certified_witness_to_jsonable(
    certified: CertifiedWitness,
) -> dict[str, object]:
    return {
        "witness": {
            "id": certified.witness.id,
            "rendered": certified.witness.rendered,
            "annotation_count": certified.witness.annotation_count,
            "constraints": [
                [constraint.name, constraint.subject]
                for constraint in certified.witness.constraints
            ],
        },
        "assignment": _assignment_to_jsonable(certified.assignment),
        "certificate": witness_certificate_to_jsonable(certified.certificate),
    }


def _certified_witness_from_jsonable(data: object) -> CertifiedWitness:
    mapping = _require_mapping(data)
    witness_data = _require_mapping(mapping["witness"])
    return CertifiedWitness(
        witness=ValidWitness(
            id=str(witness_data["id"]),
            rendered=str(witness_data["rendered"]),
            annotation_count=int(witness_data["annotation_count"]),
            constraints=tuple(
                NamedConstraint(str(item[0]), str(item[1]))
                for item in _require_list(witness_data["constraints"])
            ),
        ),
        assignment=_assignment_from_jsonable(mapping["assignment"]),
        certificate=witness_certificate_from_jsonable(
            _require_mapping(mapping["certificate"])
        ),
    )


def _assignment_to_jsonable(assignment: TraversalAssignment) -> dict[str, object]:
    return {
        "atom_text": [
            [int(atom), _atom_text_choice_to_jsonable(choice)]
            for atom, choice in sorted(
                assignment.atom_text.items(),
                key=lambda item: int(item[0]),
            )
        ],
        "tetra_tokens": [
            [int(atom), token.value]
            for atom, token in sorted(
                assignment.tetra_tokens.items(),
                key=lambda item: int(item[0]),
            )
        ],
        "bond_text": [
            [int(slot), _bond_text_choice_to_jsonable(choice)]
            for slot, choice in sorted(
                assignment.bond_text.items(),
                key=lambda item: int(item[0]),
            )
        ],
        "ring_labels": [
            [int(endpoint), label.value]
            for endpoint, label in sorted(
                assignment.ring_labels.items(),
                key=lambda item: int(item[0]),
            )
        ],
        "direction_marks": [
            [int(carrier), mark.value]
            for carrier, mark in sorted(
                assignment.direction_marks.items(),
                key=lambda item: int(item[0]),
            )
        ],
    }


def _assignment_from_jsonable(data: object) -> TraversalAssignment:
    mapping = _require_mapping(data)
    return TraversalAssignment(
        atom_text={
            AtomId(int(item[0])): _atom_text_choice_from_jsonable(item[1])
            for item in _require_list(mapping["atom_text"])
        },
        tetra_tokens={
            AtomId(int(item[0])): TetraToken(str(item[1]))
            for item in _require_list(mapping["tetra_tokens"])
        },
        bond_text={
            BondSlotId(int(item[0])): _bond_text_choice_from_jsonable(item[1])
            for item in _require_list(mapping["bond_text"])
        },
        ring_labels={
            RingEndpointId(int(item[0])): RingLabel(int(item[1]))
            for item in _require_list(mapping["ring_labels"])
        },
        direction_marks={
            CarrierSlotId(int(item[0])): DirectionMark(int(item[1]))
            for item in _require_list(mapping["direction_marks"])
        },
    )


def _atom_text_choice_to_jsonable(choice: AtomTextChoice) -> dict[str, object]:
    return {
        "name": choice.name,
        "text_by_tetra": [
            [token.value, text]
            for token, text in choice.text_by_tetra
        ],
    }


def _atom_text_choice_from_jsonable(data: object) -> AtomTextChoice:
    mapping = _require_mapping(data)
    return AtomTextChoice(
        name=str(mapping["name"]),
        text_by_tetra=tuple(
            (TetraToken(str(item[0])), str(item[1]))
            for item in _require_list(mapping["text_by_tetra"])
        ),
    )


def _bond_text_choice_to_jsonable(choice: BondTextChoice) -> dict[str, object]:
    return {
        "name": choice.name,
        "base_text": choice.base_text,
        "permits_direction": choice.permits_direction,
    }


def _bond_text_choice_from_jsonable(data: object) -> BondTextChoice:
    mapping = _require_mapping(data)
    return BondTextChoice(
        name=str(mapping["name"]),
        base_text=str(mapping["base_text"]),
        permits_direction=bool(mapping["permits_direction"]),
    )


def _require_list(value: object) -> list:
    if not isinstance(value, list):
        raise TypeError(f"expected list: {value!r}")
    return value


def _require_mapping(value: object) -> dict[str, object]:
    if not isinstance(value, dict):
        raise TypeError(f"expected mapping: {value!r}")
    return value


def _empty_csp_rejection(csp, csp_key: tuple[object, ...]) -> RejectionCertificate:
    empty_tetra = tuple(
        atom for atom, domain in csp.tetra_domains.items() if not domain
    )
    empty_direction = tuple(
        carrier
        for carrier, domain in csp.direction_domains.items()
        if not domain
    )
    empty_tetra_relations = tuple(
        relation.site
        for relation in csp.tetra_relations
        if not relation.allowed_tokens
    )
    empty_mark_relations = tuple(
        (relation.name, relation.subject)
        for relation in csp.mark_relations()
        if not relation.allowed_rows
    )

    if empty_tetra:
        return rejection_empty_tetra_domain(
            EnumerationNodeId(kind="csp", key=csp_key),
            empty_tetra,
        )
    elif empty_direction:
        return rejection_empty_direction_domain(
            EnumerationNodeId(kind="csp", key=csp_key),
            empty_direction,
        )
    elif empty_tetra_relations:
        return rejection_empty_tetra_relation(
            EnumerationNodeId(kind="csp", key=csp_key),
            empty_tetra_relations,
        )
    elif empty_mark_relations:
        return rejection_empty_mark_relation(
            EnumerationNodeId(kind="csp", key=csp_key),
            empty_mark_relations,
        )
    return rejection_csp_unsatisfied(EnumerationNodeId(kind="csp", key=csp_key))


def _render_duplicate_rejections(
    certified_witnesses: tuple[CertifiedWitness, ...],
) -> tuple[RejectionCertificate, ...]:
    first_witness_id_by_rendered: dict[str, str] = {}
    out: list[RejectionCertificate] = []
    for certified in certified_witnesses:
        rendered = certified.witness.rendered
        first_witness_id = first_witness_id_by_rendered.get(rendered)
        if first_witness_id is not None:
            out.append(
                rejection_render_duplicate(
                    render_duplicate_node_id(certified.witness.id),
                    rendered=rendered,
                    first_witness_id=first_witness_id,
                )
            )
        else:
            first_witness_id_by_rendered[rendered] = certified.witness.id
    return tuple(out)


def _support_manifest(
    *,
    skeleton_count: int,
    prefix_count: int,
    csp_count: int,
    feasible_solution_count: int,
    selected_solution_count: int,
    support: SupportImage,
    certified_witnesses: tuple[CertifiedWitness, ...],
) -> SupportEnumerationManifest:
    return SupportEnumerationManifest(
        skeleton_count=skeleton_count,
        prefix_count=prefix_count,
        csp_count=csp_count,
        feasible_solution_count=feasible_solution_count,
        selected_solution_count=selected_solution_count,
        witness_count=len(certified_witnesses),
        support_count=len(support.strings),
        support_hash=_hash_sequence(support.strings),
        witness_hash=_hash_sequence(
            witness.witness.id for witness in certified_witnesses
        ),
    )


def _hash_sequence(values: Iterable[str]) -> str:
    return sequence_hash(values)


def enumerate_stereo_support_with_stats(
    *,
    facts: MoleculeFacts,
    policy: SmilesPolicy,
    semantics: ParserSemantics,
    skeletons: Iterable[TraversalSkeleton] | None = None,
    eligible_marker_carriers: EligibleMarkerCarrierSelector | None = None,
    allow_global_directional_scope: bool = False,
) -> StereoSupportResult:
    """Enumerate stereo support and return small all-skeleton statistics.

    This function materializes skeletons if they are not supplied, so it is a
    test/debug convenience rather than the preferred large-enumeration API.
    """

    facts.validate()
    policy.validate_for_facts(facts)

    if skeletons is None:
        index = build_graph_index(facts)
        skeleton_tuple = enumerate_traversal_skeletons(
            facts=facts,
            index=index,
            policy=policy,
        )
    else:
        skeleton_tuple = tuple(skeletons)

    image = enumerate_stereo_support(
        facts=facts,
        policy=policy,
        semantics=semantics,
        skeletons=skeleton_tuple,
        eligible_marker_carriers=eligible_marker_carriers,
        allow_global_directional_scope=allow_global_directional_scope,
    )

    return StereoSupportResult(
        image=image,
        stats=StereoSupportStats(
            skeleton_count=len(skeleton_tuple),
            witness_count=image.witness_count,
            distinct_count=image.distinct_count,
        ),
    )


enumerate_exhaustive_stereo_witnesses = enumerate_stereo_witnesses
enumerate_exhaustive_stereo_support = enumerate_stereo_support
enumerate_exhaustive_certified_stereo_support = enumerate_certified_stereo_support
enumerate_exhaustive_traced_certified_stereo_support = (
    enumerate_traced_certified_stereo_support
)


__all__ = (
    "CertifiedSupportImage",
    "EligibleMarkerCarrierSelector",
    "StereoSupportResult",
    "StereoSupportStats",
    "TracedCertifiedSupportImage",
    "enumerate_certified_stereo_support",
    "enumerate_certified_stereo_witnesses",
    "enumerate_exhaustive_certified_stereo_support",
    "enumerate_exhaustive_stereo_support",
    "enumerate_exhaustive_stereo_witnesses",
    "enumerate_exhaustive_traced_certified_stereo_support",
    "enumerate_stereo_support",
    "enumerate_stereo_support_with_stats",
    "enumerate_stereo_witnesses",
    "enumerate_traced_certified_stereo_support",
    "traced_certified_support_from_jsonable",
    "traced_certified_support_to_jsonable",
)
