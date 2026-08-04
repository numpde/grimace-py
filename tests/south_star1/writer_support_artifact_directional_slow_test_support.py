"""Non-test support for rich support-artifact relationships."""

from copy import deepcopy
from tests.south_star1.writer_artifact_resealing import (
    refresh_text_projection_payload_digest,
    reseal_support_artifact,
)
from tests.south_star1.writer_artifact_test_support import closed_term_digest
from tests.south_star1.writer_artifact_test_support import closed_term_field
from tests.south_star1.writer_artifact_test_support import set_closed_term_field
from tests.south_star1.writer_artifact_test_support import refresh_cursor_digest
from tests.south_star1.writer_artifact_test_support import refresh_closed_term_digest_field
from tests.south_star1.writer_artifact_test_support import refresh_kind_manifest_digest
from tests.south_star1.writer_support_artifact_transition_test_support import linked_tetra_lifecycle_manifest
from tests.south_star1.writer_support_artifact_transition_test_support import refresh_linked_raw_lifecycle_residual_digest
from tests.south_star1.writer_support_artifact_transition_test_support import propagate_text_projection_cursor_change





def ring_projection_branch_and_manifest(artifact, *, changed: bool | None = None):
    for branch in artifact["objects"]:
        if branch["kind"] != "branch_support":
            continue
        for manifest in branch["payload"]["obligation_manifests"]["residual_work"]:
            if manifest["operation"] != "directional ring endpoint projection":
                continue
            term = manifest["transition_term"]
            is_changed = closed_term_field(term, "source_snapshot") != closed_term_field(
                term, "successor_snapshot"
            )
            if changed is None or changed == is_changed:
                return branch, manifest
    raise AssertionError("missing directional ring projection transition")


def ring_pair_branch_and_manifest(artifact):
    for branch in artifact["objects"]:
        if branch["kind"] != "branch_support":
            continue
        for manifest in branch["payload"]["obligation_manifests"]["residual_work"]:
            if manifest["operation"] == "directional ring pair restriction":
                return branch, manifest
    raise AssertionError("missing directional ring pair transition")


def refresh_ring_pair_term(artifact, branch, manifest) -> None:
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def forge_ring_pair_missing_term(artifact) -> None:
    branch, manifest = ring_pair_branch_and_manifest(artifact)
    manifest["transition_term"] = None
    manifest["transition_digest"] = None
    reseal_support_artifact(artifact)


def forge_ring_pair_compatible_choices(artifact) -> None:
    branch, manifest = ring_pair_branch_and_manifest(artifact)
    del closed_term_field(manifest["transition_term"], "compatible_second_endpoint_choices")[-1]
    refresh_ring_pair_term(artifact, branch, manifest)


def forge_ring_pair_first_mark(artifact) -> None:
    branch, manifest = ring_pair_branch_and_manifest(artifact)
    mark = closed_term_field(manifest["transition_term"], "first_endpoint_direction_mark")
    mark["value"] = 1 if mark["value"] != 1 else -1
    refresh_ring_pair_term(artifact, branch, manifest)


def forge_ring_pair_second_mark(artifact) -> None:
    branch, manifest = ring_pair_branch_and_manifest(artifact)
    mark = closed_term_field(manifest["transition_term"], "second_endpoint_direction_mark")
    mark["value"] = -1 if mark["value"] != -1 else 1
    refresh_ring_pair_term(artifact, branch, manifest)


def forge_ring_pair_orientation(artifact) -> None:
    branch, manifest = ring_pair_branch_and_manifest(artifact)
    term = manifest["transition_term"]
    value = closed_term_field(term, "second_canonical_orientation")
    set_closed_term_field(term, "second_canonical_orientation", -value)
    refresh_ring_pair_term(artifact, branch, manifest)


def forge_ring_pair_carrier(artifact) -> None:
    branch, manifest = ring_pair_branch_and_manifest(artifact)
    model = closed_term_field(manifest["transition_term"], "carrier_models")[0]
    value = closed_term_field(model, "ligand_factor")
    set_closed_term_field(model, "ligand_factor", -value)
    refresh_ring_pair_term(artifact, branch, manifest)


def forge_ring_pair_restriction(artifact) -> None:
    branch, manifest = ring_pair_branch_and_manifest(artifact)
    sign = closed_term_field(manifest["transition_term"], "restrictions")[0][1]
    sign["value"] = "negative" if sign["value"] == "positive" else "positive"
    refresh_ring_pair_term(artifact, branch, manifest)


def forge_ring_pair_occurrence(artifact) -> None:
    branch, manifest = ring_pair_branch_and_manifest(artifact)
    term = manifest["transition_term"]
    parent = closed_term_field(term, "bond_occurrence_parent")
    child = closed_term_field(term, "bond_occurrence_child")
    set_closed_term_field(term, "bond_occurrence_parent", child)
    set_closed_term_field(term, "bond_occurrence_child", parent)
    refresh_ring_pair_term(artifact, branch, manifest)


def forge_ring_pair_discharge(artifact) -> None:
    branch, manifest = ring_pair_branch_and_manifest(artifact)
    del closed_term_field(manifest["transition_term"], "discharged_factor_keys")[-1]
    refresh_ring_pair_term(artifact, branch, manifest)


def forge_ring_pair_successor(artifact) -> None:
    branch, manifest = ring_pair_branch_and_manifest(artifact)
    term = manifest["transition_term"]
    successor = deepcopy(closed_term_field(term, "source_snapshot"))
    digest = closed_term_digest(successor, operation="test.transition.successor")
    set_closed_term_field(term, "successor_snapshot", successor)
    set_closed_term_field(term, "successor_snapshot_digest", digest)
    refresh_linked_raw_lifecycle_residual_digest(
        branch,
        manifest=manifest,
        field="successor_residual_snapshot_digest",
        digest=digest,
    )
    refresh_ring_pair_term(artifact, branch, manifest)


def refresh_ring_projection_term(artifact, branch, manifest) -> None:
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def forge_ring_compatible_seconds(artifact) -> None:
    branch, manifest = ring_projection_branch_and_manifest(artifact)
    choices = closed_term_field(manifest["transition_term"], "compatible_second_endpoint_choices")
    del choices[-1]
    refresh_ring_projection_term(artifact, branch, manifest)


def forge_ring_domain_intersection(artifact) -> None:
    branch, manifest = ring_projection_branch_and_manifest(artifact, changed=True)
    values = closed_term_field(manifest["transition_term"], "domain_intersections")[0][1]
    values[0]["value"] = "negative" if values[0]["value"] == "positive" else "positive"
    refresh_ring_projection_term(artifact, branch, manifest)


def forge_ring_carrier_orientation(artifact) -> None:
    branch, manifest = ring_projection_branch_and_manifest(artifact)
    model = closed_term_field(manifest["transition_term"], "carrier_model")
    orientation = closed_term_field(model, "endpoint_orientation_factor")
    set_closed_term_field(model, "endpoint_orientation_factor", -orientation)
    refresh_ring_projection_term(artifact, branch, manifest)


def forge_ring_term_mark(artifact) -> None:
    branch, manifest = ring_projection_branch_and_manifest(artifact)
    mark = closed_term_field(manifest["transition_term"], "direction_mark")
    mark["value"] = -1 if mark["value"] != -1 else 1
    refresh_ring_projection_term(artifact, branch, manifest)


def forge_ring_false_noop(artifact) -> None:
    branch, manifest = ring_projection_branch_and_manifest(artifact, changed=True)
    term = manifest["transition_term"]
    source = deepcopy(closed_term_field(term, "source_snapshot"))
    set_closed_term_field(term, "successor_snapshot", source)
    digest = closed_term_digest(source, operation="test.transition.source")
    set_closed_term_field(term, "successor_snapshot_digest", digest)
    refresh_linked_raw_lifecycle_residual_digest(
        branch,
        manifest=manifest,
        field="successor_residual_snapshot_digest",
        digest=digest,
    )
    refresh_ring_projection_term(artifact, branch, manifest)


def forge_ring_false_change(artifact) -> None:
    branch, manifest = ring_projection_branch_and_manifest(artifact, changed=False)
    _other_branch, other = ring_projection_branch_and_manifest(artifact, changed=True)
    term = manifest["transition_term"]
    successor = deepcopy(closed_term_field(other["transition_term"], "successor_snapshot"))
    set_closed_term_field(term, "successor_snapshot", successor)
    digest = closed_term_digest(successor, operation="test.transition.successor")
    set_closed_term_field(term, "successor_snapshot_digest", digest)
    refresh_linked_raw_lifecycle_residual_digest(
        branch,
        manifest=manifest,
        field="successor_residual_snapshot_digest",
        digest=digest,
    )
    refresh_ring_projection_term(artifact, branch, manifest)


def forge_ring_factor_discharge(artifact) -> None:
    branch, manifest = ring_projection_branch_and_manifest(artifact)
    term = manifest["transition_term"]
    source = closed_term_field(term, "source_snapshot")
    factor = closed_term_field(closed_term_field(source, "factors")[0], "key")
    set_closed_term_field(term, "discharged_factor_keys", [factor])
    refresh_ring_projection_term(artifact, branch, manifest)


def forge_ring_source_snapshot(artifact) -> None:
    branch, manifest = ring_projection_branch_and_manifest(artifact)
    term = manifest["transition_term"]
    source = closed_term_field(term, "source_snapshot")
    domains = closed_term_field(source, "domains")
    domains[:] = list(reversed(domains))
    digest = closed_term_digest(source, operation="test.transition.source")
    set_closed_term_field(term, "source_snapshot_digest", digest)
    refresh_linked_raw_lifecycle_residual_digest(
        branch,
        manifest=manifest,
        field="source_residual_snapshot_digest",
        digest=digest,
    )
    refresh_ring_projection_term(artifact, branch, manifest)


def forge_ring_missing_term(artifact) -> None:
    branch, manifest = ring_projection_branch_and_manifest(artifact)
    manifest["transition_term"] = None
    manifest["transition_digest"] = None
    reseal_support_artifact(artifact)


def forge_ring_lifecycle_operation(artifact) -> None:
    branch, manifest = ring_projection_branch_and_manifest(artifact)
    lifecycle = linked_tetra_lifecycle_manifest(
        branch=branch,
        manifest=manifest,
        lifecycle_kind="raw",
        certificate_kind="",
    )
    lifecycle["residual_work_operations"] = ["wrong"]
    reseal_support_artifact(artifact)


def forge_ring_successor_open_endpoint(artifact) -> None:
    branch, manifest = ring_projection_branch_and_manifest(artifact)
    projection = _text_projection_for_branch(artifact, branch)
    cursor = projection["payload"]["successor_cursor"]
    state = _cursor_state_by_digest(
        cursor,
        branch["payload"]["successor_state_digest"],
    )
    ring_state = closed_term_field(state, "ring_state")
    endpoint = next(
        endpoint
        for endpoint in closed_term_field(ring_state, "open_endpoints")
        if int(closed_term_field(endpoint, "bond")) == 3
    )
    set_closed_term_field(endpoint, "first_endpoint_text", "%01")
    refresh_ring_successor_cursor_change(
        artifact=artifact,
        branch=branch,
        manifest=manifest,
        projection=projection,
        cursor=cursor,
        state=state,
    )


def forge_ring_bond_occurrence_added(artifact) -> None:
    branch, manifest = ring_projection_branch_and_manifest(artifact)
    projection = _text_projection_for_branch(artifact, branch)
    cursor = projection["payload"]["successor_cursor"]
    state = _cursor_state_by_digest(
        cursor,
        branch["payload"]["successor_state_digest"],
    )
    stereo = closed_term_field(state, "stereo_state")
    closed_term_field(stereo, "bond_occurrences").append(
        {
            "__dataclass__": "grimace._south_star1.writer_stereo.WriterBondOccurrenceRecord",
            "fields": [
                ["bond", 3],
                ["parent", 0],
                ["child", 2],
                ["mark", {"__enum__": "grimace._south_star1.policy.DirectionMark", "value": 0}],
            ],
        }
    )
    refresh_ring_successor_cursor_change(
        artifact=artifact,
        branch=branch,
        manifest=manifest,
        projection=projection,
        cursor=cursor,
        state=state,
    )


def refresh_ring_successor_cursor_change(
    *, artifact, branch, manifest, projection, cursor, state
) -> None:
    old_cursor_digest = branch["payload"]["successor_cursor_digest"]
    old_state_digest = branch["payload"]["successor_state_digest"]
    refresh_cursor_digest(cursor, operation="test.cursor.digest")
    successor_state_digest = closed_term_digest(state, operation="test.directional.successor_bond_occurrence")
    propagate_text_projection_cursor_change(
        artifact,
        old_cursor_digest=old_cursor_digest,
        new_cursor=cursor,
        old_state_digest=old_state_digest,
        new_state_digest=successor_state_digest,
    )
    branch["payload"]["successor_state_digest"] = successor_state_digest
    branch["payload"]["graph_ring_delta"]["manifest"]["successor_state_digest"] = (
        successor_state_digest
    )
    manifest["successor_digest"] = successor_state_digest
    for lifecycle in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]:
        if manifest["evidence_digest"] in lifecycle["linked_residual_work_digests"]:
            lifecycle["successor_digest"] = successor_state_digest
    branch["payload"]["successor_cursor_digest"] = cursor["digest"]
    branch["payload"]["graph_ring_delta"]["manifest"]["successor_cursor_digest"] = (
        cursor["digest"]
    )
    refresh_kind_manifest_digest(branch["payload"]["graph_ring_delta"], operation="test.graph_ring_delta.digest")
    refresh_text_projection_payload_digest(
        projection["payload"],
        operation="test.text_projection.cursor_change",
    )
    reseal_support_artifact(artifact)

