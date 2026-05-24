"""Replay checks for South Star support-completeness traces."""

from __future__ import annotations

from .certificate_checker import replay_witness_certificate
from .support_enumeration import TracedCertifiedSupportImage
from .support_enumeration import enumerate_traced_certified_stereo_support
from .stereo_witness import _skeleton_key
from .graph_index import build_graph_index
from .skeleton import enumerate_traversal_skeletons
from .slots import allocate_traversal_slots


def replay_support_completeness_certificate(
    *,
    facts,
    policy,
    semantics,
    result: TracedCertifiedSupportImage,
) -> None:
    """Replay a traced certified support image against the finite model."""

    expected = enumerate_traced_certified_stereo_support(
        facts=facts,
        policy=policy,
        semantics=semantics,
    )
    if result.support != expected.support:
        raise ValueError("support image mismatch")
    if result.manifest != expected.manifest:
        raise ValueError("support manifest mismatch")
    if result.trace != expected.trace:
        raise ValueError("enumeration trace mismatch")

    accepted_rendered = tuple(certificate.rendered for certificate in result.trace.accepted)
    unique_rendered = tuple(dict.fromkeys(accepted_rendered))
    if unique_rendered != result.support.strings:
        raise ValueError("support strings do not match accepted witness quotient")
    if len(result.trace.accepted) != len(result.certified_witnesses):
        raise ValueError("accepted trace count does not match certified witnesses")
    if result.trace.witness_count != len(result.certified_witnesses):
        raise ValueError("trace witness count mismatch")
    if result.trace.support_count != len(result.support.strings):
        raise ValueError("trace support count mismatch")

    skeletons = enumerate_traversal_skeletons(
        facts,
        build_graph_index(facts),
        policy,
    )
    skeleton_by_key = {
        _skeleton_key(skeleton): skeleton
        for skeleton in skeletons
    }
    for certified in result.certified_witnesses:
        skeleton = skeleton_by_key.get(certified.certificate.skeleton_key)
        if skeleton is None:
            raise ValueError("witness certificate references unknown skeleton")
        slots = allocate_traversal_slots(facts, skeleton)
        replay_witness_certificate(
            facts=facts,
            skeleton=skeleton,
            slots=slots,
            assignment=certified.assignment,
            policy=policy,
            semantics=semantics,
            certificate=certified.certificate,
        )


__all__ = ("replay_support_completeness_certificate",)
