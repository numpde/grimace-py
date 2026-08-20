"""Non-test support for rich support-artifact relationships."""

from copy import deepcopy
from grimace._south_star1.writer_envelope_terms import _identity_digest
from tests.south_star1.writer_artifact_test_support import closed_term_digest
from tests.south_star1.writer_artifact_test_support import closed_term_field
from tests.south_star1.writer_support_artifact_queries import single_cursor_state
from tests.south_star1.writer_support_artifact_queries import text_projection_for_branch
from tests.south_star1.writer_support_artifact_transition_test_support import linked_tetra_lifecycle_manifest





def append_unrelated_raw_lifecycle(branch, *, manifest):
    linked = linked_tetra_lifecycle_manifest(
        branch=branch,
        manifest=manifest,
        lifecycle_kind="raw",
        certificate_kind="",
    )
    unrelated = deepcopy(linked)
    unrelated["evidence_digest"] = f"unrelated:{linked['evidence_digest']}"
    unrelated["linked_residual_work_digests"] = []
    unrelated["residual_work_digests"] = []
    unrelated["residual_work_operations"] = []
    branch["payload"]["obligation_manifests"]["stereo_lifecycle"].append(unrelated)
    return unrelated


def different_local_order_digest(artifact, *, branch, cursor_name: str, atom: int) -> str:
    projection = text_projection_for_branch(artifact, branch)
    cursor = projection["payload"][cursor_name]
    state = single_cursor_state(cursor)
    stereo = closed_term_field(state, "stereo_state")
    for record in closed_term_field(stereo, "local_orders"):
        if closed_term_field(record, "atom") != atom:
            return closed_term_digest(record, operation="test.tetra.local_order_alternate")
    raise AssertionError("missing alternate local-order record")


def refresh_local_order_event_identity_digest(event) -> None:
    identity = {
        "site": event["site"],
        "atom": event["atom"],
        "local_order": event["local_order"],
        "reference_order": event["reference_order"],
        "source_local_order_record_digest": event[
            "source_local_order_record_digest"
        ],
        "successor_local_order_record_digest": event[
            "successor_local_order_record_digest"
        ],
    }
    event["local_order_identity_digest"] = _identity_digest(identity)
