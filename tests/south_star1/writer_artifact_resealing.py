"""Coherent test-only resealing for writer artifact forgeries."""

from collections.abc import MutableMapping
from grimace._south_star1.writer_branch_transition_artifact import branch_transition_artifact_manifest
from grimace._south_star1.writer_envelope_terms import _digest_terms_bounded
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_envelope_work import default_writer_envelope_work_budget
from grimace._south_star1.writer_support_artifact_checker import artifact_manifest
from grimace._south_star1.writer_support_artifact_checker import artifact_metrics
from grimace._south_star1.writer_support_artifact_checker import support_artifact_object_identity_term
from grimace._south_star1.writer_terminalization_artifact import terminalization_artifact_manifest
from tests.south_star1.writer_artifact_test_support import unique_artifact_object_by_kind

def _budget(budget):
    return default_writer_envelope_work_budget(budget)


def refresh_text_projection_payload_digest(
    payload: MutableMapping[str, object],
    *,
    operation: str,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> str:
    budget = _budget(budget)
    identity = {
        key: value
        for key, value in payload.items()
        if key not in {"digest", "branch_support_refs"}
    }
    digest = _identity_digest(identity, budget=budget, operation=operation)
    payload["digest"] = digest
    return digest


def _object_identity(obj, *, operation: str, budget=None) -> str:
    budget = _budget(budget)
    digest = _identity_digest(
        support_artifact_object_identity_term(obj["kind"], obj["payload"]),
        budget=budget,
        operation=operation,
    )
    obj["digest"] = digest
    obj["object_id"] = f"obj:{digest}"
    return obj["object_id"]


def reseal_branch_transition_artifact(
    artifact: MutableMapping[str, object],
    *,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> None:
    budget = _budget(budget)
    if artifact.get("schema_name") != "writer_branch_transition_artifact":
        raise AssertionError("wrong branch-transition artifact schema")
    objects = artifact.get("objects")
    if not isinstance(objects, list) or len(objects) != 3:
        raise AssertionError("branch-transition artifact requires exactly three objects")
    required = {"source_snapshot", "text_projection", "branch_support"}
    if {item.get("kind") for item in objects} != required:
        raise AssertionError("branch-transition artifact object kinds mismatch")
    source = unique_artifact_object_by_kind(artifact, "source_snapshot")
    projection = unique_artifact_object_by_kind(artifact, "text_projection")
    branch = unique_artifact_object_by_kind(artifact, "branch_support")
    _object_identity(source, operation="test.branch_transition.source_object", budget=budget)
    _object_identity(branch, operation="test.branch_transition.branch_object", budget=budget)
    projection_payload = projection["payload"]
    if not isinstance(projection_payload, MutableMapping):
        raise AssertionError("branch text projection payload must be mutable")
    projection_payload["branch_support_refs"] = [branch["object_id"]]
    refresh_text_projection_payload_digest(
        projection_payload,
        operation="test.branch_transition.projection_identity",
        budget=budget,
    )
    _object_identity(projection, operation="test.branch_transition.projection_object", budget=budget)
    artifact["roots"] = {
        "source_ref": source["object_id"],
        "text_projection_ref": projection["object_id"],
        "branch_support_ref": branch["object_id"],
    }
    artifact["objects"] = sorted(objects, key=lambda item: item["object_id"])
    artifact["metrics"] = {
        **artifact_metrics(artifact["objects"]),
        "reachable_object_count": 3,
        "unreferenced_object_count": 0,
    }
    artifact["digest"] = _digest_terms_bounded(
        branch_transition_artifact_manifest(artifact),
        budget=budget,
        operation="test.branch_transition.artifact",
    )


def reseal_terminalization_artifact(
    artifact: MutableMapping[str, object],
    *,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> None:
    budget = _budget(budget)
    if artifact.get("schema_name") != "writer_terminalization_artifact":
        raise AssertionError("wrong terminalization artifact schema")
    objects = artifact.get("objects")
    if not isinstance(objects, list) or len(objects) != 3:
        raise AssertionError("terminalization artifact requires exactly three objects")
    required = {"source_snapshot", "terminal_projection", "terminal_support"}
    if {item.get("kind") for item in objects} != required:
        raise AssertionError("terminalization artifact object kinds mismatch")
    source = unique_artifact_object_by_kind(artifact, "source_snapshot")
    projection = unique_artifact_object_by_kind(artifact, "terminal_projection")
    support = unique_artifact_object_by_kind(artifact, "terminal_support")
    support_payload = support["payload"]
    if not isinstance(support_payload, MutableMapping):
        raise AssertionError("terminal support payload must be mutable")
    support_payload["terminalization_term_digest"] = _identity_digest(
        support_payload["terminalization_term"],
        budget=budget,
        operation="test.terminalization.term",
    )
    _object_identity(source, operation="test.terminalization.source_object", budget=budget)
    _object_identity(support, operation="test.terminalization.support_object", budget=budget)
    projection_payload = projection["payload"]
    if not isinstance(projection_payload, MutableMapping):
        raise AssertionError("terminal projection payload must be mutable")
    projection_payload["terminal_support_refs"] = [support["object_id"]]
    identity = {key: value for key, value in projection_payload.items() if key != "digest"}
    projection_payload["digest"] = _identity_digest(
        identity,
        budget=budget,
        operation="test.terminalization.projection",
    )
    _object_identity(projection, operation="test.terminalization.projection_object", budget=budget)
    artifact["roots"] = {
        "source_ref": source["object_id"],
        "terminal_projection_ref": projection["object_id"],
        "terminal_support_ref": support["object_id"],
    }
    artifact["objects"] = sorted(objects, key=lambda item: item["object_id"])
    artifact["metrics"] = {
        **artifact_metrics(artifact["objects"]),
        "reachable_object_count": 3,
        "unreferenced_object_count": 0,
    }
    artifact["digest"] = _digest_terms_bounded(
        terminalization_artifact_manifest(artifact),
        budget=budget,
        operation="test.terminalization.artifact",
    )


def _replace_exact_refs(value, *, old_id: str, new_id: str) -> None:
    if isinstance(value, MutableMapping):
        for key, child in list(value.items()):
            if child == old_id:
                value[key] = new_id
            else:
                _replace_exact_refs(child, old_id=old_id, new_id=new_id)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            if child == old_id:
                value[index] = new_id
            else:
                _replace_exact_refs(child, old_id=old_id, new_id=new_id)


def reseal_support_artifact(
    artifact: MutableMapping[str, object],
    *,
    budget: WriterEnvelopeWorkBudget | None = None,
) -> None:
    budget = _budget(budget)
    if artifact.get("schema_name") != "writer_support_artifact":
        raise AssertionError("wrong support artifact schema")
    objects = artifact.get("objects")
    if not isinstance(objects, list):
        raise AssertionError("support artifact objects must be a list")
    seen_states = set()
    rounds = 0
    while True:
        rounds += 1
        if rounds > max(1, len(objects)) * 4:
            raise AssertionError("support artifact object graph did not converge")
        state = (
            tuple(item.get("object_id") for item in objects),
            repr(artifact.get("roots")),
        )
        if state in seen_states:
            raise AssertionError("support artifact object graph did not converge")
        seen_states.add(state)
        changed = False
        for item in objects:
            old_id = item.get("object_id")
            new_id = _object_identity(item, operation="test.support_artifact.object", budget=budget)
            if old_id != new_id:
                _replace_exact_refs(artifact, old_id=old_id, new_id=new_id)
                changed = True
        if not changed:
            break
    artifact["objects"] = sorted(objects, key=lambda item: item["object_id"])
    artifact["metrics"] = artifact_metrics(artifact["objects"], roots=artifact["roots"])
    artifact["digest"] = _digest_terms_bounded(
        artifact_manifest(artifact),
        budget=budget,
        operation="test.support_artifact.artifact",
    )


__all__ = (
    "refresh_text_projection_payload_digest",
    "reseal_branch_transition_artifact",
    "reseal_terminalization_artifact",
    "reseal_support_artifact",
)
