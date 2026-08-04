"""Non-test support for rich support-artifact relationships."""

from grimace._south_star1.writer_envelope_terms import _identity_digest
from tests.south_star1.writer_artifact_test_support import artifact_object_by_id
from tests.south_star1.writer_artifact_test_support import refresh_kind_manifest_digest





def linked_tetra_lifecycle_manifest(
    *,
    branch,
    manifest,
    lifecycle_kind: str,
    certificate_kind: str,
):
    operation = (
        "WriterStereoLifecycleEvidence"
        if lifecycle_kind == "raw"
        else "WriterStereoBranchCertificate"
    )
    for item in branch["payload"]["obligation_manifests"]["stereo_lifecycle"]:
        if item["evidence_digest"] not in manifest["linked_lifecycle_digests"]:
            continue
        if item["operation"] != operation:
            continue
        if lifecycle_kind == "certificate" and item["certificate_kind"] != certificate_kind:
            continue
        return item
    raise AssertionError(f"missing linked tetra lifecycle manifest: {lifecycle_kind}")


def refresh_linked_raw_lifecycle_residual_digest(
    branch,
    *,
    manifest,
    field: str,
    digest: str,
) -> None:
    lifecycle = linked_tetra_lifecycle_manifest(
        branch=branch,
        manifest=manifest,
        lifecycle_kind="raw",
        certificate_kind="",
    )
    lifecycle[field] = digest


def text_projection_identity_digest(payload) -> str:
    return _identity_digest(
        {
            "source_cursor_digest": payload["source_cursor"]["digest"],
            "emitted_text": payload["emitted_text"],
            "successor_cursor_digest": payload["successor_cursor"]["digest"],
            "immediate_multiplicity": payload["immediate_multiplicity"],
            "support_count": payload["support_count"],
            "completion_count": payload["completion_count"],
            "branch_certificate_digests": payload["branch_certificate_digests"],
        },
    )


def propagate_text_projection_cursor_change(
    artifact,
    *,
    old_cursor_digest: str,
    new_cursor,
    old_state_digest: str | None = None,
    new_state_digest: str | None = None,
) -> None:
    for item in artifact["objects"]:
        if item["kind"] != "text_projection":
            continue
        payload = item["payload"]
        if payload["source_cursor"]["digest"] == old_cursor_digest:
            payload["source_cursor"] = new_cursor
            payload["digest"] = text_projection_identity_digest(payload)
            for branch_ref in payload["branch_support_refs"]:
                branch = artifact_object_by_id(artifact, branch_ref)
                branch["payload"]["source_cursor_digest"] = new_cursor["digest"]
                branch["payload"]["graph_ring_delta"]["manifest"][
                    "source_cursor_digest"
                ] = new_cursor["digest"]
                if (
                    old_state_digest is not None
                    and new_state_digest is not None
                    and branch["payload"]["source_state_digest"] == old_state_digest
                ):
                    branch["payload"]["source_state_digest"] = new_state_digest
                    branch["payload"]["graph_ring_delta"]["manifest"][
                        "source_state_digest"
                    ] = new_state_digest
                refresh_kind_manifest_digest(branch["payload"]["graph_ring_delta"], operation="test.graph_ring_delta.digest")
    for item in artifact["objects"]:
        if item["kind"] != "replay_path":
            continue
        if item["payload"]["final_cursor_digest"] == old_cursor_digest:
            item["payload"]["final_cursor_digest"] = new_cursor["digest"]
    for item in artifact["objects"]:
        if item["kind"] != "terminal_projection":
            continue
        payload = item["payload"]
        if payload["source_cursor"]["digest"] == old_cursor_digest:
            payload["source_cursor"] = new_cursor


