"""Shared navigation and verifier adapters for rich support artifacts."""

from __future__ import annotations

from typing import Mapping, MutableMapping

from grimace._south_star1.writer_support_artifact_checker import (
    verify_writer_support_artifact_consistency,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    classify_residual_stereo_obligations_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_branch_projection_identities_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_graph_ring_branch_deltas_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_local_branch_successor_evidence_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_support_image_coverage_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_support_string_replay_paths_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_terminal_support_identities_offline,
)


def support_artifact_object_index(
    artifact: Mapping[str, object],
) -> dict[str, MutableMapping[str, object]]:
    objects = artifact.get("objects")
    if not isinstance(objects, list):
        raise AssertionError("support artifact objects must be a list")
    result: dict[str, MutableMapping[str, object]] = {}
    for item in objects:
        if not isinstance(item, MutableMapping):
            raise AssertionError("support artifact object must be mutable mapping-shaped")
        object_id = item.get("object_id")
        if not isinstance(object_id, str):
            raise AssertionError("support artifact object_id must be a string")
        if object_id in result:
            raise AssertionError(f"duplicate support artifact object_id: {object_id}")
        result[object_id] = item
    return result


def _root_object(artifact, root_name: str):
    roots = artifact.get("roots")
    if not isinstance(roots, Mapping) or root_name not in roots:
        raise AssertionError(f"missing support artifact root: {root_name}")
    object_id = roots[root_name]
    index = support_artifact_object_index(artifact)
    try:
        return index[object_id]
    except KeyError as exc:
        raise AssertionError(f"root references missing object: {object_id}") from exc


def support_image_root_object(artifact):
    return _root_object(artifact, "support_image_root")


def support_strings(artifact) -> tuple[str, ...]:
    values = support_image_root_object(artifact)["payload"]["support_strings"]
    if not isinstance(values, list) or not all(isinstance(value, str) for value in values):
        raise AssertionError("support_strings must be a list of strings")
    return tuple(values)


def coverage_object(artifact):
    root = support_image_root_object(artifact)
    return support_artifact_object_index(artifact)[root["payload"]["coverage_ref"]]


def first_support_string_object(artifact):
    root = support_image_root_object(artifact)
    refs = root["payload"]["support_string_refs"]
    if not refs:
        raise AssertionError("support image has no support-string objects")
    return support_artifact_object_index(artifact)[refs[0]]


def first_text_projection_object(artifact):
    support = first_support_string_object(artifact)
    return support_artifact_object_index(artifact)[
        support["payload"]["text_projection_refs"][0]
    ]


def first_branch_support_object(artifact):
    projection = first_text_projection_object(artifact)
    return support_artifact_object_index(artifact)[
        projection["payload"]["branch_support_refs"][0]
    ]


def first_terminal_projection_object(artifact):
    support = first_support_string_object(artifact)
    return support_artifact_object_index(artifact)[
        support["payload"]["terminal_projection_ref"]
    ]


def first_terminal_support_object(artifact):
    support = first_support_string_object(artifact)
    return support_artifact_object_index(artifact)[
        support["payload"]["terminal_support_refs"][0]
    ]


def first_graph_ring_delta_branch(artifact, *, kind: str):
    for item in artifact["objects"]:
        if item["kind"] == "branch_support" and item["payload"]["graph_ring_delta"]["kind"] == kind:
            return item
    raise AssertionError(f"missing graph/ring delta kind: {kind}")


def first_residual_work_branch(artifact, *, operation: str):
    for item in artifact["objects"]:
        if item["kind"] != "branch_support":
            continue
        if any(
            manifest["operation"] == operation
            for manifest in item["payload"]["obligation_manifests"]["residual_work"]
        ):
            return item
    raise AssertionError(f"missing residual work operation: {operation}")


def first_local_evidence(artifact, *, kind: str):
    for item in artifact["objects"]:
        if item["kind"] == "branch_support" and item["payload"]["local_evidence"]["kind"] == kind:
            return item["payload"]["local_evidence"]
    raise AssertionError(f"missing local evidence kind: {kind}")


def _offline_objects(artifact):
    return support_artifact_object_index(artifact)


def verify_support_image_coverage_relation(artifact):
    return verify_support_image_coverage_offline(
        artifact=artifact,
        objects=_offline_objects(artifact),
    )


def verify_support_string_replay_relation(artifact):
    return verify_support_string_replay_paths_offline(
        artifact=artifact,
        objects=_offline_objects(artifact),
    )


def verify_branch_projection_relation(artifact):
    return verify_branch_projection_identities_offline(
        artifact=artifact,
        objects=_offline_objects(artifact),
    )


def verify_graph_ring_delta_relation(*, facts, artifact):
    return verify_graph_ring_branch_deltas_offline(
        facts=facts,
        artifact=artifact,
        objects=_offline_objects(artifact),
    )


def classify_obligation_replay(*, facts, artifact):
    return classify_residual_stereo_obligations_offline(
        facts=facts,
        artifact=artifact,
        objects=_offline_objects(artifact),
    )


def verify_local_branch_evidence_relation(*, facts, artifact):
    return verify_local_branch_successor_evidence_offline(
        facts=facts,
        artifact=artifact,
        objects=_offline_objects(artifact),
    )


def verify_terminal_identity_relation(artifact):
    return verify_terminal_support_identities_offline(
        artifact=artifact,
        objects=_offline_objects(artifact),
    )


def require_structurally_valid_support_artifact(artifact) -> None:
    verification = verify_writer_support_artifact_consistency(artifact)
    if not verification.accepted:
        raise AssertionError(verification.reason)
