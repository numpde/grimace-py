"""Structural replay checks for South Star support-completeness traces."""

from __future__ import annotations

from dataclasses import dataclass

from .certificate_checker import replay_witness_certificate
from .enumeration_trace import EnumerationNodeId
from .enumeration_trace import RejectionCertificate
from .enumeration_trace import TraceIndex
from .enumeration_trace import build_trace_index
from .graph_index import build_graph_index
from .skeleton import TraversalSkeleton
from .skeleton import enumerate_traversal_skeletons
from .slots import SlotBundle
from .slots import allocate_traversal_slots
from .stereo_csp import PresentationPrefix
from .stereo_csp import StereoCSP
from .stereo_csp import build_stereo_csp
from .stereo_csp import select_stereo_solutions_with_certificates
from .stereo_csp import solve_stereo_csp
from .stereo_csp import stereo_solution_canonical_key
from .stereo_witness import CertifiedWitness
from .stereo_witness import _certified_witness_from_selected_solution
from .stereo_witness import _prefix_key
from .stereo_witness import _skeleton_key
from .stereo_witness import enumerate_presentation_prefixes
from .support_enumeration import TracedCertifiedSupportImage
from .support_enumeration import _hash_sequence
from .support_enumeration import enumerate_traced_certified_stereo_support


@dataclass(frozen=True, slots=True)
class EnumerationReplayContext:
    skeleton_by_key: dict[tuple[object, ...], TraversalSkeleton]
    slots_by_skeleton_key: dict[tuple[object, ...], SlotBundle]
    prefixes_by_key: dict[tuple[object, ...], PresentationPrefix]
    csp_by_key: dict[tuple[object, ...], StereoCSP]


@dataclass(slots=True)
class _ReplayObserved:
    skeleton_count: int = 0
    prefix_count: int = 0
    csp_count: int = 0
    feasible_solution_count: int = 0
    selected_solution_count: int = 0
    witness_count: int = 0
    support_strings: tuple[str, ...] = ()
    accepted_nodes: set[EnumerationNodeId] | None = None
    rejected_nodes: set[EnumerationNodeId] | None = None

    def __post_init__(self) -> None:
        if self.accepted_nodes is None:
            self.accepted_nodes = set()
        if self.rejected_nodes is None:
            self.rejected_nodes = set()


def replay_support_completeness_certificate(
    *,
    facts,
    policy,
    semantics,
    result: TracedCertifiedSupportImage,
    compare_against_regeneration: bool = False,
) -> None:
    """Replay a support-completeness certificate against finite domains."""

    facts.validate()
    policy.validate_for_facts(facts)

    if compare_against_regeneration:
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

    trace_index = build_trace_index(result.trace)
    observed = _ReplayObserved()
    context = EnumerationReplayContext(
        skeleton_by_key={},
        slots_by_skeleton_key={},
        prefixes_by_key={},
        csp_by_key={},
    )
    certified_by_node = {
        EnumerationNodeId(
            kind="witness",
            key=(certified.witness.id,),
        ): certified
        for certified in result.certified_witnesses
    }
    _replay_domains(
        facts=facts,
        policy=policy,
        semantics=semantics,
        result=result,
        trace_index=trace_index,
        certified_by_node=certified_by_node,
        context=context,
        observed=observed,
    )
    _validate_no_uncovered_nodes(trace_index, observed)
    _validate_counts_and_hashes(result, observed)


def validate_rejection_certificate(
    *,
    facts,
    policy,
    semantics,
    node_context: EnumerationReplayContext,
    rejection: RejectionCertificate,
) -> None:
    """Validate one typed rejection statement against replay context."""

    reason = rejection.reason
    if reason == "render_duplicate":
        _validate_render_duplicate_rejection(rejection)
        return

    if reason == "annotation_not_selected":
        csp, solution_key = _csp_and_solution_key(node_context, rejection.node)
        solutions = tuple(solve_stereo_csp(csp))
        feasible_by_key = {
            stereo_solution_canonical_key(solution): solution
            for solution in solutions
        }
        solution = feasible_by_key.get(solution_key)
        if solution is None:
            raise ValueError("annotation rejection references infeasible solution")
        selected = select_stereo_solutions_with_certificates(
            csp=csp,
            solutions=solutions,
            mode=policy.annotation_mode,
        )
        selected_keys = {
            stereo_solution_canonical_key(item.solution)
            for item in selected
        }
        if solution_key in selected_keys:
            raise ValueError("annotation rejection references selected solution")
        if ("mode", policy.annotation_mode.value) != rejection.detail[-2:]:
            raise ValueError("annotation rejection mode detail mismatch")
        return

    if rejection.node.kind != "csp":
        raise ValueError(f"{reason} rejection must target a CSP node")
    csp = node_context.csp_by_key.get(rejection.node.key)
    if csp is None:
        raise ValueError("CSP rejection references unknown CSP")

    if reason == "empty_tetra_domain":
        empty = tuple(
            sorted(
                int(atom)
                for atom, domain in csp.tetra_domains.items()
                if not domain
            )
        )
        if not empty:
            raise ValueError("false empty_tetra_domain rejection")
        if rejection.detail != ("atoms", empty):
            raise ValueError("empty_tetra_domain detail mismatch")
        return

    if reason == "empty_direction_domain":
        empty = tuple(
            sorted(
                int(carrier)
                for carrier, domain in csp.direction_domains.items()
                if not domain
            )
        )
        if not empty:
            raise ValueError("false empty_direction_domain rejection")
        if rejection.detail != ("carriers", empty):
            raise ValueError("empty_direction_domain detail mismatch")
        return

    if reason == "empty_tetra_relation":
        empty = tuple(
            int(relation.site)
            for relation in csp.tetra_relations
            if not relation.allowed_tokens
        )
        if not empty:
            raise ValueError("false empty_tetra_relation rejection")
        if rejection.detail != ("sites", empty):
            raise ValueError("empty_tetra_relation detail mismatch")
        return

    if reason == "empty_mark_relation":
        empty = tuple(
            (relation.name, relation.subject)
            for relation in csp.mark_relations()
            if not relation.allowed_rows
        )
        if not empty:
            raise ValueError("false empty_mark_relation rejection")
        if rejection.detail != ("relations", empty):
            raise ValueError("empty_mark_relation detail mismatch")
        return

    if reason == "csp_unsatisfied":
        if tuple(solve_stereo_csp(csp)):
            raise ValueError("false csp_unsatisfied rejection")
        return

    if reason == "policy_rejected":
        raise ValueError("policy_rejected is not valid for replayed traces yet")

    if reason == "internal_invariant":
        raise ValueError("internal_invariant cannot appear in valid traces")

    raise ValueError(f"unsupported rejection reason: {reason!r}")


def _replay_domains(
    *,
    facts,
    policy,
    semantics,
    result: TracedCertifiedSupportImage,
    trace_index: TraceIndex,
    certified_by_node: dict[EnumerationNodeId, CertifiedWitness],
    context: EnumerationReplayContext,
    observed: _ReplayObserved,
) -> None:
    skeletons = enumerate_traversal_skeletons(
        facts,
        build_graph_index(facts),
        policy,
    )
    observed.skeleton_count = len(skeletons)

    seen_rendered: dict[str, str] = {}
    support_order: list[str] = []
    for skeleton in skeletons:
        skeleton_key = _skeleton_key(skeleton)
        context.skeleton_by_key[skeleton_key] = skeleton
        slots = allocate_traversal_slots(facts, skeleton)
        context.slots_by_skeleton_key[skeleton_key] = slots

        for prefix in enumerate_presentation_prefixes(
            facts=facts,
            slots=slots,
            policy=policy,
        ):
            observed.prefix_count += 1
            observed.csp_count += 1
            prefix_key = _prefix_key(prefix)
            csp_key = (skeleton_key, prefix_key)
            context.prefixes_by_key[csp_key] = prefix
            csp = build_stereo_csp(
                facts=facts,
                skeleton=skeleton,
                slots=slots,
                prefix=prefix,
                policy=policy,
                semantics=semantics,
            )
            context.csp_by_key[csp_key] = csp
            raw_solutions = tuple(solve_stereo_csp(csp))
            observed.feasible_solution_count += len(raw_solutions)
            selected = select_stereo_solutions_with_certificates(
                csp=csp,
                solutions=raw_solutions,
                mode=policy.annotation_mode,
            )
            observed.selected_solution_count += len(selected)

            if not raw_solutions:
                node = EnumerationNodeId(kind="csp", key=csp_key)
                rejection = trace_index.rejected_by_node.get(node)
                if rejection is None:
                    raise ValueError("missing CSP rejection certificate")
                validate_rejection_certificate(
                    facts=facts,
                    policy=policy,
                    semantics=semantics,
                    node_context=context,
                    rejection=rejection,
                )
                observed.rejected_nodes.add(node)  # type: ignore[union-attr]
                continue

            selected_by_key = {
                stereo_solution_canonical_key(item.solution): item
                for item in selected
            }
            for solution in raw_solutions:
                solution_key = stereo_solution_canonical_key(solution)
                if solution_key not in selected_by_key:
                    node = EnumerationNodeId(
                        kind="stereo_solution",
                        key=(csp_key, solution_key),
                    )
                    rejection = trace_index.rejected_by_node.get(node)
                    if rejection is None:
                        raise ValueError("missing annotation rejection certificate")
                    validate_rejection_certificate(
                        facts=facts,
                        policy=policy,
                        semantics=semantics,
                        node_context=context,
                        rejection=rejection,
                    )
                    observed.rejected_nodes.add(node)  # type: ignore[union-attr]

            for selected_solution in selected:
                certified = _certified_witness_from_selected_solution(
                    facts=facts,
                    skeleton=skeleton,
                    slots=slots,
                    prefix=prefix,
                    policy=policy,
                    semantics=semantics,
                    selected=selected_solution,
                    csp=csp,
                )
                node = EnumerationNodeId(
                    kind="witness",
                    key=(certified.witness.id,),
                )
                acceptance = trace_index.accepted_by_node.get(node)
                if acceptance is None:
                    raise ValueError("missing witness acceptance certificate")
                if acceptance.witness_id != certified.witness.id:
                    raise ValueError("accepted witness id mismatch")
                if acceptance.rendered != certified.witness.rendered:
                    raise ValueError("accepted rendered string mismatch")
                supplied = certified_by_node.get(node)
                if supplied is None:
                    raise ValueError("accepted witness lacks certified witness")
                if supplied.witness.id != certified.witness.id:
                    raise ValueError("certified witness id mismatch")
                if supplied.witness.rendered != certified.witness.rendered:
                    raise ValueError("certified witness rendered mismatch")
                replay_witness_certificate(
                    facts=facts,
                    skeleton=skeleton,
                    slots=slots,
                    assignment=supplied.assignment,
                    policy=policy,
                    semantics=semantics,
                    certificate=supplied.certificate,
                )
                observed.accepted_nodes.add(node)  # type: ignore[union-attr]
                observed.witness_count += 1

                first_witness_id = seen_rendered.get(certified.witness.rendered)
                if first_witness_id is None:
                    seen_rendered[certified.witness.rendered] = certified.witness.id
                    support_order.append(certified.witness.rendered)
                else:
                    duplicate_node = EnumerationNodeId(
                        kind="witness",
                        key=("render_duplicate", certified.witness.id),
                    )
                    rejection = trace_index.rejected_by_node.get(duplicate_node)
                    if rejection is None:
                        raise ValueError("missing render_duplicate rejection")
                    _validate_render_duplicate_rejection(
                        rejection,
                        rendered=certified.witness.rendered,
                        first_witness_id=first_witness_id,
                    )
                    observed.rejected_nodes.add(duplicate_node)  # type: ignore[union-attr]

    observed.support_strings = tuple(support_order)


def _validate_render_duplicate_rejection(
    rejection: RejectionCertificate,
    *,
    rendered: str | None = None,
    first_witness_id: str | None = None,
) -> None:
    if rejection.node.kind != "witness":
        raise ValueError("render_duplicate must target a witness node")
    if not rejection.node.key or rejection.node.key[0] != "render_duplicate":
        raise ValueError("render_duplicate node key must be quotient-scoped")
    detail = rejection.detail
    if len(detail) != 4 or detail[0] != "rendered" or detail[2] != "first_witness_id":
        raise ValueError("render_duplicate detail schema mismatch")
    if rendered is not None and detail[1] != rendered:
        raise ValueError("render_duplicate rendered detail mismatch")
    if first_witness_id is not None and detail[3] != first_witness_id:
        raise ValueError("render_duplicate first witness detail mismatch")


def _csp_and_solution_key(
    context: EnumerationReplayContext,
    node: EnumerationNodeId,
) -> tuple[StereoCSP, tuple[object, ...]]:
    if node.kind != "stereo_solution":
        raise ValueError("annotation_not_selected must target a stereo solution")
    if len(node.key) != 2:
        raise ValueError("stereo solution node key has wrong arity")
    csp_key = node.key[0]
    solution_key = node.key[1]
    if not isinstance(csp_key, tuple):
        raise ValueError("stereo solution node lacks CSP key")
    if not isinstance(solution_key, tuple):
        raise ValueError("stereo solution node lacks solution key")
    csp = context.csp_by_key.get(csp_key)
    if csp is None:
        raise ValueError("stereo solution rejection references unknown CSP")
    return csp, solution_key


def _validate_no_uncovered_nodes(
    trace_index: TraceIndex,
    observed: _ReplayObserved,
) -> None:
    accepted_nodes = observed.accepted_nodes or set()
    rejected_nodes = observed.rejected_nodes or set()
    extra_accepted = set(trace_index.accepted_by_node) - accepted_nodes
    if extra_accepted:
        raise ValueError("trace contains unreachable acceptance node")
    extra_rejected = set(trace_index.rejected_by_node) - rejected_nodes
    if extra_rejected:
        raise ValueError("trace contains unreachable rejection node")


def _validate_counts_and_hashes(
    result: TracedCertifiedSupportImage,
    observed: _ReplayObserved,
) -> None:
    if result.trace.skeleton_count != observed.skeleton_count:
        raise ValueError("trace skeleton count mismatch")
    if result.trace.prefix_count != observed.prefix_count:
        raise ValueError("trace prefix count mismatch")
    if result.trace.csp_count != observed.csp_count:
        raise ValueError("trace CSP count mismatch")
    if result.trace.feasible_solution_count != observed.feasible_solution_count:
        raise ValueError("trace feasible solution count mismatch")
    if result.trace.selected_solution_count != observed.selected_solution_count:
        raise ValueError("trace selected solution count mismatch")
    if result.trace.witness_count != observed.witness_count:
        raise ValueError("trace witness count mismatch")
    if result.trace.support_count != len(observed.support_strings):
        raise ValueError("trace support count mismatch")

    if result.manifest.skeleton_count != observed.skeleton_count:
        raise ValueError("manifest skeleton count mismatch")
    if result.manifest.prefix_count != observed.prefix_count:
        raise ValueError("manifest prefix count mismatch")
    if result.manifest.csp_count != observed.csp_count:
        raise ValueError("manifest CSP count mismatch")
    if result.manifest.feasible_solution_count != observed.feasible_solution_count:
        raise ValueError("manifest feasible solution count mismatch")
    if result.manifest.selected_solution_count != observed.selected_solution_count:
        raise ValueError("manifest selected solution count mismatch")
    if result.manifest.witness_count != observed.witness_count:
        raise ValueError("manifest witness count mismatch")
    if result.manifest.support_count != len(observed.support_strings):
        raise ValueError("manifest support count mismatch")
    if result.support.strings != observed.support_strings:
        raise ValueError("support strings do not match accepted witness quotient")
    if result.manifest.support_hash != _hash_sequence(observed.support_strings):
        raise ValueError("manifest support hash mismatch")
    witness_ids = tuple(certificate.witness_id for certificate in result.trace.accepted)
    if result.manifest.witness_hash != _hash_sequence(witness_ids):
        raise ValueError("manifest witness hash mismatch")


__all__ = (
    "EnumerationReplayContext",
    "replay_support_completeness_certificate",
    "validate_rejection_certificate",
)
