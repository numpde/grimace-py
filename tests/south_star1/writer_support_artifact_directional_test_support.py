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
from tests.south_star1.writer_support_artifact_queries import text_projection_for_branch
from tests.south_star1.writer_support_artifact_queries import single_cursor_state
from tests.south_star1.writer_support_artifact_transition_test_support import linked_tetra_lifecycle_manifest
from tests.south_star1.writer_support_artifact_transition_test_support import refresh_linked_raw_lifecycle_residual_digest
from tests.south_star1.writer_support_artifact_transition_test_support import propagate_text_projection_cursor_change






def directional_transition_branch_and_manifest(artifact, *, bond: int):
    for branch in artifact["objects"]:
        if branch["kind"] != "branch_support":
            continue
        for manifest in branch["payload"]["obligation_manifests"]["residual_work"]:
            if manifest["operation"] != "directional carrier-mark restriction":
                continue
            if closed_term_field(manifest["transition_term"], "bond") == bond:
                return branch, manifest
    raise AssertionError(f"missing directional carrier transition for bond {bond}")


def directional_transition_manifest(artifact, *, bond: int):
    return directional_transition_branch_and_manifest(artifact, bond=bond)[1]


def directional_discharge_key_pairs(manifest):
    return tuple(
        (
            closed_term_field(key, "kind"),
            tuple(closed_term_field(key, "key")),
        )
        for key in closed_term_field(
            manifest["transition_term"],
            "discharged_factor_keys",
        )
    )


def bond_occurrence_terms_for_branch(
    artifact,
    branch,
    *,
    cursor_name: str,
    bond: int,
):
    projection = text_projection_for_branch(artifact, branch)
    state = single_cursor_state(projection["payload"][cursor_name])
    stereo = closed_term_field(state, "stereo_state")
    return tuple(
        occurrence
        for occurrence in closed_term_field(stereo, "bond_occurrences")
        if int(closed_term_field(occurrence, "bond")) == bond
    )


def mutate_directional_restriction_sign(artifact, *, bond: int) -> None:
    branch, manifest = directional_transition_branch_and_manifest(artifact, bond=bond)
    sign = closed_term_field(manifest["transition_term"], "restrictions")[0][1]
    sign["value"] = "negative" if sign["value"] == "positive" else "positive"
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def mutate_directional_canonical_orientation(artifact, *, bond: int) -> None:
    branch, manifest = directional_transition_branch_and_manifest(artifact, bond=bond)
    value = closed_term_field(manifest["transition_term"], "canonical_orientation")
    set_closed_term_field(manifest["transition_term"], "canonical_orientation", -value)
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def mutate_directional_model_field(
    artifact,
    *,
    bond: int,
    field: str,
    value,
    model_index: int = 0,
) -> None:
    branch, manifest = directional_transition_branch_and_manifest(artifact, bond=bond)
    model = closed_term_field(manifest["transition_term"], "carrier_models")[model_index]
    set_closed_term_field(model, field, value)
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def remove_directional_model(artifact, *, bond: int) -> None:
    branch, manifest = directional_transition_branch_and_manifest(artifact, bond=bond)
    models = closed_term_field(manifest["transition_term"], "carrier_models")
    del models[-1]
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def remove_directional_restriction(artifact, *, bond: int) -> None:
    branch, manifest = directional_transition_branch_and_manifest(artifact, bond=bond)
    restrictions = closed_term_field(manifest["transition_term"], "restrictions")
    del restrictions[-1]
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def duplicate_directional_model_site(artifact, *, bond: int) -> None:
    branch, manifest = directional_transition_branch_and_manifest(artifact, bond=bond)
    models = closed_term_field(manifest["transition_term"], "carrier_models")
    duplicate_site = closed_term_field(models[0], "site")
    set_closed_term_field(models[1], "site", duplicate_site)
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def mutate_directional_successor_snapshot(artifact, *, bond: int) -> None:
    branch, manifest = directional_transition_branch_and_manifest(artifact, bond=bond)
    successor = closed_term_field(manifest["transition_term"], "successor_snapshot")
    domains = closed_term_field(successor, "domains")
    domains[:] = list(reversed(domains))
    digest = closed_term_digest(successor, operation="test.directional.successor_snapshot")
    set_closed_term_field(manifest["transition_term"], "successor_snapshot_digest", digest)
    refresh_linked_raw_lifecycle_residual_digest(
        branch,
        manifest=manifest,
        field="successor_residual_snapshot_digest",
        digest=digest,
    )
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def set_directional_discharges(
    artifact,
    *,
    bond: int,
    kinds: tuple[str, ...],
) -> None:
    branch, manifest = directional_transition_branch_and_manifest(artifact, bond=bond)
    source = closed_term_field(manifest["transition_term"], "source_snapshot")
    factor_by_kind = {
        closed_term_field(closed_term_field(factor, "key"), "kind"): closed_term_field(factor, "key")
        for factor in closed_term_field(source, "factors")
    }
    set_closed_term_field(
        manifest["transition_term"],
        "discharged_factor_keys",
        [factor_by_kind[kind] for kind in kinds],
    )
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def set_directional_discharges_by_keys(
    artifact,
    *,
    bond: int,
    key_pairs: tuple[tuple[str, tuple[int, ...]], ...],
) -> None:
    branch, manifest = directional_transition_branch_and_manifest(artifact, bond=bond)
    source = closed_term_field(manifest["transition_term"], "source_snapshot")
    factor_by_pair = {
        (
            closed_term_field(closed_term_field(factor, "key"), "kind"),
            tuple(closed_term_field(closed_term_field(factor, "key"), "key")),
        ): closed_term_field(factor, "key")
        for factor in closed_term_field(source, "factors")
    }
    set_closed_term_field(
        manifest["transition_term"],
        "discharged_factor_keys",
        [factor_by_pair[key_pair] for key_pair in key_pairs],
    )
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def remove_raw_lifecycle_capability(
    artifact,
    *,
    bond: int,
    capability: str,
) -> None:
    branch, manifest = directional_transition_branch_and_manifest(artifact, bond=bond)
    lifecycle = linked_tetra_lifecycle_manifest(
        branch=branch,
        manifest=manifest,
        lifecycle_kind="raw",
        certificate_kind="directional_carrier_restricted",
    )
    capabilities = lifecycle["lifecycle_capabilities"]
    capabilities.remove(capability)
    reseal_support_artifact(artifact)


def mutate_directional_term_mark(artifact, *, bond: int, value: int) -> None:
    branch, manifest = directional_transition_branch_and_manifest(artifact, bond=bond)
    mark = closed_term_field(manifest["transition_term"], "direction_mark")
    mark["value"] = value if mark["value"] != value else -value
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def mutate_directional_term_bond(artifact, *, bond: int, value: int) -> None:
    branch, manifest = directional_transition_branch_and_manifest(artifact, bond=bond)
    set_closed_term_field(manifest["transition_term"], "bond", value)
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)


def remove_directional_successor_bond_occurrence(artifact, *, bond: int) -> None:
    branch, manifest = directional_transition_branch_and_manifest(artifact, bond=bond)
    projection = text_projection_for_branch(artifact, branch)
    cursor = projection["payload"]["successor_cursor"]
    state = single_cursor_state(cursor)
    stereo = closed_term_field(state, "stereo_state")
    occurrences = closed_term_field(stereo, "bond_occurrences")
    kept = [
        occurrence
        for occurrence in occurrences
        if int(closed_term_field(occurrence, "bond")) != bond
    ]
    if len(kept) == len(occurrences):
        raise AssertionError(f"missing successor bond occurrence for bond {bond}")
    occurrences[:] = kept
    old_state_digest = branch["payload"]["successor_state_digest"]
    refresh_cursor_digest(cursor, operation="test.cursor.digest")
    successor_state_digest = closed_term_digest(state, operation="test.directional.successor_bond_occurrence")
    propagate_text_projection_cursor_change(
        artifact,
        old_cursor_digest=branch["payload"]["successor_cursor_digest"],
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


def duplicate_directional_successor_bond_occurrence(artifact, *, bond: int) -> None:
    branch, manifest = directional_transition_branch_and_manifest(artifact, bond=bond)
    projection = text_projection_for_branch(artifact, branch)
    cursor = projection["payload"]["successor_cursor"]
    state = single_cursor_state(cursor)
    stereo = closed_term_field(state, "stereo_state")
    occurrences = closed_term_field(stereo, "bond_occurrences")
    matches = [
        occurrence
        for occurrence in occurrences
        if int(closed_term_field(occurrence, "bond")) == bond
    ]
    if len(matches) != 1:
        raise AssertionError(f"expected one successor bond occurrence for bond {bond}")
    occurrences.append(deepcopy(matches[0]))
    old_state_digest = branch["payload"]["successor_state_digest"]
    refresh_cursor_digest(cursor, operation="test.cursor.digest")
    successor_state_digest = closed_term_digest(state, operation="test.directional.duplicate_bond_occurrence")
    propagate_text_projection_cursor_change(
        artifact,
        old_cursor_digest=branch["payload"]["successor_cursor_digest"],
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


def mutate_directional_successor_snapshot_unrelated(artifact, *, bond: int) -> None:
    branch, manifest = directional_transition_branch_and_manifest(artifact, bond=bond)
    successor = closed_term_field(manifest["transition_term"], "successor_snapshot")
    closed_term_field(successor, "domains").append(
        [
            {
                "__dataclass__": "grimace._south_star1.residual_constraints.VarId",
                "fields": [["kind", "unrelated_directional_test"], ["key", [99]]],
            },
            [False, True],
        ]
    )
    digest = closed_term_digest(successor, operation="test.directional.successor_snapshot_unrelated")
    set_closed_term_field(manifest["transition_term"], "successor_snapshot_digest", digest)
    refresh_linked_raw_lifecycle_residual_digest(
        branch,
        manifest=manifest,
        field="successor_residual_snapshot_digest",
        digest=digest,
    )
    refresh_closed_term_digest_field(manifest, term_field="transition_term", digest_field="transition_digest", operation="test.transition.digest")
    reseal_support_artifact(artifact)
