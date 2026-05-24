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
from .enumerate import SupportImage
from .enumerate import render_image_from_witnesses
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
from .stereo_witness import enumerate_certified_stereo_witnesses_for_skeleton
from .stereo_witness import enumerate_stereo_witnesses_for_skeleton


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

    certified = tuple(
        enumerate_certified_stereo_witnesses(
            facts=facts,
            policy=policy,
            semantics=semantics,
            skeletons=skeletons,
            eligible_marker_carriers=eligible_marker_carriers,
            allow_global_directional_scope=allow_global_directional_scope,
        )
    )
    support = render_image_from_witnesses(
        certified_witness.witness
        for certified_witness in certified
    )
    return CertifiedSupportImage(
        support=support,
        certified_witnesses=certified,
    )


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
    "enumerate_certified_stereo_support",
    "enumerate_certified_stereo_witnesses",
    "enumerate_stereo_support",
    "enumerate_stereo_support_with_stats",
    "enumerate_stereo_witnesses",
)
