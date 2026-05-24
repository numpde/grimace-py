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
from hashlib import blake2b

from .annotation import ValidWitness
from .certificates import SupportEnumerationManifest
from .certificates import manifest_to_jsonable
from .certificates import witness_certificate_to_jsonable
from .enumerate import SupportImage
from .enumerate import render_image_from_witnesses
from .enumeration_trace import AcceptanceCertificate
from .enumeration_trace import EnumerationNodeId
from .enumeration_trace import EnumerationTrace
from .enumeration_trace import RejectionCertificate
from .enumeration_trace import enumeration_trace_to_jsonable
from .facts import MoleculeFacts
from .graph_index import build_graph_index
from .ids import CarrierSlotId
from .policy import SmilesPolicy
from .semantics import ParserSemantics
from .skeleton import TraversalSkeleton
from .skeleton import enumerate_traversal_skeletons
from .slots import SlotBundle
from .slots import allocate_traversal_slots
from .stereo_witness import CertifiedWitness
from .stereo_witness import collect_certified_stereo_witnesses_for_skeleton
from .stereo_witness import enumerate_certified_stereo_witnesses_for_skeleton
from .stereo_witness import enumerate_presentation_prefixes
from .stereo_witness import enumerate_stereo_witnesses_for_skeleton
from .stereo_witness import _certified_witness_from_selected_solution
from .stereo_witness import _prefix_key
from .stereo_witness import _skeleton_key
from .stereo_csp import build_stereo_csp
from .stereo_csp import select_stereo_solutions_with_certificates
from .stereo_csp import solve_stereo_csp
from .stereo_csp import stereo_solution_canonical_key


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

        yield from enumerate_stereo_witnesses_for_skeleton(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            policy=policy,
            semantics=semantics,
            eligible_marker_carriers=eligible,
            allow_global_directional_scope=allow_global_directional_scope,
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
        skeleton_key = _skeleton_key(skeleton)
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
            prefix_key = _prefix_key(prefix)
            csp_key = (skeleton_key, prefix_key)
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
                rejected.append(_empty_csp_rejection(csp, csp_key))
                continue

            selected_keys = {
                stereo_solution_canonical_key(selected.solution)
                for selected in selected_solutions
            }
            for solution in raw_solutions:
                solution_key = stereo_solution_canonical_key(solution)
                if solution_key in selected_keys:
                    continue
                rejected.append(
                    RejectionCertificate(
                        node=EnumerationNodeId(
                            kind="stereo_solution",
                            key=(csp_key, solution_key),
                        ),
                        reason="annotation_not_selected",
                        detail=(
                            "support",
                            tuple(sorted(int(c) for c in solution.marker_support)),
                            "selected_supports",
                            tuple(
                                sorted(
                                    tuple(sorted(int(c) for c in selected.solution.marker_support))
                                    for selected in selected_solutions
                                )
                            ),
                            "mode",
                            policy.annotation_mode.value,
                        ),
                    )
                )

            for selected in selected_solutions:
                certified = _certified_witness_from_selected_solution(
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
                        node=EnumerationNodeId(
                            kind="witness",
                            key=(certified.witness.id,),
                        ),
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
        "witness_certificates": [
            witness_certificate_to_jsonable(certified.certificate)
            for certified in result.certified_witnesses
        ],
    }


def _empty_csp_rejection(csp, csp_key: tuple[object, ...]) -> RejectionCertificate:
    detail: tuple[object, ...] = ()
    empty_tetra = tuple(
        sorted(int(atom) for atom, domain in csp.tetra_domains.items() if not domain)
    )
    empty_direction = tuple(
        sorted(
            int(carrier)
            for carrier, domain in csp.direction_domains.items()
            if not domain
        )
    )
    empty_tetra_relations = tuple(
        int(relation.site)
        for relation in csp.tetra_relations
        if not relation.allowed_tokens
    )
    empty_mark_relations = tuple(
        (relation.name, relation.subject)
        for relation in csp.mark_relations()
        if not relation.allowed_rows
    )

    if empty_tetra:
        reason = "empty_tetra_domain"
        detail = ("atoms", empty_tetra)
    elif empty_direction:
        reason = "empty_direction_domain"
        detail = ("carriers", empty_direction)
    elif empty_tetra_relations:
        reason = "empty_tetra_relation"
        detail = ("sites", empty_tetra_relations)
    elif empty_mark_relations:
        reason = "empty_mark_relation"
        detail = ("relations", empty_mark_relations)
    else:
        reason = "csp_unsatisfied"

    return RejectionCertificate(
        node=EnumerationNodeId(kind="csp", key=csp_key),
        reason=reason,
        detail=detail,
    )


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
                RejectionCertificate(
                    node=EnumerationNodeId(
                        kind="witness",
                        key=("render_duplicate", certified.witness.id),
                    ),
                    reason="render_duplicate",
                    detail=(
                        "rendered",
                        rendered,
                        "first_witness_id",
                        first_witness_id,
                    ),
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
    digest = blake2b(digest_size=16)
    for value in values:
        digest.update(value.encode("utf8"))
        digest.update(b"\0")
    return digest.hexdigest()


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


__all__ = (
    "CertifiedSupportImage",
    "EligibleMarkerCarrierSelector",
    "StereoSupportResult",
    "StereoSupportStats",
    "TracedCertifiedSupportImage",
    "enumerate_certified_stereo_support",
    "enumerate_certified_stereo_witnesses",
    "enumerate_stereo_support",
    "enumerate_stereo_support_with_stats",
    "enumerate_stereo_witnesses",
    "enumerate_traced_certified_stereo_support",
    "traced_certified_support_to_jsonable",
)
