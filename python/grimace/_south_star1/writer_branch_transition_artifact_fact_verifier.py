"""Producer-free facts-bound verification for one branch transition artifact."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .errors import SouthStarError
from .facts import MoleculeFacts
from .policy import SmilesPolicy
from .prepared_runtime import SouthStarRuntimeOptions
from .writer_branch_transition_artifact_checker import verify_writer_branch_transition_artifact_consistency
from .writer_envelope_work import WriterEnvelopeWorkExceeded
from .writer_envelope_work import default_writer_envelope_work_budget
from .writer_envelope_work import writer_envelope_work_reason
from .writer_facts_replay_context import _WriterFactsReplayContext
from .writer_facts_replay_context import _writer_facts_replay_context
from .writer_support_artifact_fact_verifier import _check_prepared_identity
from .writer_support_artifact_fact_verifier import _check_source_snapshot_identity
from .writer_support_artifact_offline_verifier import _branch_ref_from_transition_artifact
from .writer_support_artifact_offline_verifier import verify_branch_obligations_offline
from .writer_support_artifact_offline_verifier import verify_transition_branch_projection_identity_offline
from .writer_support_artifact_offline_verifier import verify_graph_ring_branch_deltas_offline
from .writer_support_artifact_offline_verifier import verify_local_branch_successor_evidence_offline


@dataclass(frozen=True, slots=True)
class WriterBranchTransitionArtifactFactVerification:
    accepted: bool
    semantically_replayed_operations: tuple[str, ...] = ()
    checked_relation_families: tuple[str, ...] = ()
    checked_obligation_families: tuple[str, ...] = ()
    unchecked_obligation_families: tuple[str, ...] = ()
    reason: str | None = None


def verify_writer_branch_transition_artifact_for_facts(
    *,
    facts: MoleculeFacts,
    runtime_options: SouthStarRuntimeOptions,
    artifact,
    policy: SmilesPolicy | None = None,
    budget=None,
) -> WriterBranchTransitionArtifactFactVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        structural = verify_writer_branch_transition_artifact_consistency(artifact, budget=budget)
        if not structural.accepted:
            return WriterBranchTransitionArtifactFactVerification(accepted=False, reason=structural.reason)
        context = _writer_facts_replay_context(
            facts=facts,
            runtime_options=runtime_options,
            policy=policy,
            budget=budget,
        )
        return _verify_writer_branch_transition_artifact_for_facts_with_context(
            context=context,
            artifact=artifact,
            budget=budget,
        )
    except WriterEnvelopeWorkExceeded as exc:
        return WriterBranchTransitionArtifactFactVerification(accepted=False, reason=writer_envelope_work_reason(exc))
    except SouthStarError as exc:
        return WriterBranchTransitionArtifactFactVerification(accepted=False, reason=exc.args[-1] if exc.args else "verification_error")
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterBranchTransitionArtifactFactVerification(accepted=False, reason=f"malformed_branch_transition_artifact:{type(exc).__name__}")


def _verify_writer_branch_transition_artifact_for_facts_with_context(
    *, context: _WriterFactsReplayContext, artifact, budget=None
) -> WriterBranchTransitionArtifactFactVerification:
    try:
        budget = default_writer_envelope_work_budget(budget)
        if not isinstance(artifact, Mapping):
            raise TypeError("artifact must be a mapping")
        facts = context.facts
        runtime_options = context.runtime_options
        prepared = context.prepared
        expected_identity = context.expected_identity
        _check_prepared_identity(artifact, expected_identity)
        _check_source_snapshot_identity(
            artifact["source_snapshot"],
            expected_identity=expected_identity,
            runtime_options=runtime_options,
        )
        objects = {item["object_id"]: item for item in artifact["objects"]}
        branch_ref = _branch_ref_from_transition_artifact(
            artifact=artifact,
            objects=objects,
        )
        branch = objects[branch_ref]
        branch_refs = (branch_ref,)
        checks = (
            verify_transition_branch_projection_identity_offline(
                projection_ref=artifact["roots"]["text_projection_ref"],
                branch_ref=branch_ref,
                objects=objects,
            ),
            verify_graph_ring_branch_deltas_offline(
                facts=facts,
                artifact=artifact,
                objects=objects,
                budget=budget,
                branch_refs=branch_refs,
            ),
            verify_local_branch_successor_evidence_offline(
                facts=facts,
                artifact=artifact,
                objects=objects,
                budget=budget,
                branch_refs=branch_refs,
            ),
        )
        for check in checks:
            if not check.accepted:
                return WriterBranchTransitionArtifactFactVerification(accepted=False, reason=check.reason)
        obligations = verify_branch_obligations_offline(
            facts=facts,
            prepared=prepared,
            artifact=artifact,
            objects=objects,
            branch_ref=branch_ref,
        )
        if not obligations.accepted:
            return WriterBranchTransitionArtifactFactVerification(accepted=False, reason=obligations.reason)
        return WriterBranchTransitionArtifactFactVerification(
            accepted=True,
            semantically_replayed_operations=obligations.semantically_replayed_operations,
            checked_relation_families=(
                ("component_boundary_transition",)
                if checks[1].checked_component_boundary_steps == 1
                and branch["payload"]["digest"]
                in obligations.replayed_component_boundary_branch_digests
                else ()
            ),
            checked_obligation_families=obligations.checked_families,
            unchecked_obligation_families=obligations.unchecked_families,
        )
    except WriterEnvelopeWorkExceeded as exc:
        return WriterBranchTransitionArtifactFactVerification(accepted=False, reason=writer_envelope_work_reason(exc))
    except SouthStarError as exc:
        return WriterBranchTransitionArtifactFactVerification(accepted=False, reason=exc.args[-1] if exc.args else "verification_error")
    except (AssertionError, KeyError, TypeError, ValueError) as exc:
        return WriterBranchTransitionArtifactFactVerification(accepted=False, reason=f"malformed_branch_transition_artifact:{type(exc).__name__}")


__all__ = (
    "WriterBranchTransitionArtifactFactVerification",
    "_verify_writer_branch_transition_artifact_for_facts_with_context",
    "verify_writer_branch_transition_artifact_for_facts",
)
