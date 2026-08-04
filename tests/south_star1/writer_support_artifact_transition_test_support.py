"""Non-test support for rich support-artifact relationships."""

from __future__ import annotations

"""Facts-bound writer support artifact verifier tests."""

from copy import deepcopy
from dataclasses import replace
from functools import lru_cache
from types import SimpleNamespace

from grimace._south_star1.errors import SouthStarError
from grimace._south_star1.errors import SouthStarErrorKind
from grimace._south_star1.facts import LigandKind
from grimace._south_star1.facts import LigandOccurrence
from grimace._south_star1.facts import SiteStatus
from grimace._south_star1.facts import StereoFacts
from grimace._south_star1.facts import TetraValue
from grimace._south_star1.ids import BondId
from grimace._south_star1.ids import OccurrenceId
from grimace._south_star1.ids import SiteId
from grimace._south_star1.ordinary_policy import OrdinaryPolicyOptions
from grimace._south_star1.ordinary_policy import ordinary_policy_for_facts
from grimace._south_star1.prepared_runtime import SouthStarRuntimeOptions
from grimace._south_star1.prepared_runtime import SouthStarWriterSurface
from grimace._south_star1.prepared_runtime import prepare_south_star_mol_from_facts
from grimace._south_star1.policy import SerializationLanguageMode
from grimace._south_star1.policy import DirectionMark
from grimace._south_star1.rdkit_adapter import RdkitOrdinaryExtractionOptions
from grimace._south_star1.rdkit_adapter import ordinary_molecule_facts_from_smiles
from grimace._south_star1.writer_envelope_terms import _digest_terms_bounded
from grimace._south_star1.writer_envelope_terms import _identity_digest
from grimace._south_star1.writer_envelope_work import WriterEnvelopeWorkBudget
from grimace._south_star1.writer_events import WriterRingEndpointEmitted
from grimace._south_star1.writer_events import WriterRingEndpointPaired
from grimace._south_star1.writer_frontier import _checked_writer_frontier_branch_supports
from grimace._south_star1.writer_frontier import initial_writer_frontier_cursor
from grimace._south_star1.writer_snapshot import WriterDecoderBoundary
from grimace._south_star1.writer_snapshot import capture_writer_frontier_snapshot
from grimace._south_star1.writer_snapshot_prefix_envelope import (
    writer_snapshot_prefix_read_envelope_for_emitted_texts,
)
import grimace._south_star1.writer_stereo as writer_stereo_module
import grimace._south_star1.writer_support_artifact_offline_verifier as offline_verifier_module
from grimace._south_star1.writer_support_artifact_checker import (
    verify_writer_support_artifact_consistency,
)
from grimace._south_star1.writer_support_artifact_fact_verifier import (
    OBJECT_KIND_OFFLINE_COVERAGE,
)
from grimace._south_star1.writer_support_artifact_fact_verifier import (
    verify_writer_support_artifact_for_facts,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    classify_residual_stereo_obligations_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    validate_writer_bracket_atom_text_against_facts,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_branch_projection_identities_offline,
)
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_count_dag_arithmetic,
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
from grimace._south_star1.writer_support_artifact_offline_verifier import (
    verify_writer_support_artifact_offline_replay,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    verify_writer_support_artifact_envelope,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_prefix_read,
)
from grimace._south_star1.writer_support_artifact_envelope import (
    writer_support_artifact_envelope_for_snapshot,
)
from tests.south_star1.helpers import cco_facts
from tests.south_star1.helpers import directional_facts
from tests.south_star1.helpers import shared_acyclic_directional_facts
from tests.south_star1.writer_artifact_resealing import reseal_support_artifact
from tests.south_star1.writer_artifact_test_support import artifact_object_by_id
from tests.south_star1.writer_artifact_test_support import closed_term_digest
from tests.south_star1.writer_artifact_test_support import closed_term_field
from tests.south_star1.writer_artifact_test_support import set_closed_term_field
from tests.south_star1.writer_artifact_test_support import refresh_cursor_digest
from tests.south_star1.writer_artifact_test_support import refresh_closed_term_digest_field
from tests.south_star1.writer_artifact_test_support import refresh_kind_manifest_digest
from tests.south_star1.writer_test_context import initial_writer_snapshot
from tests.south_star1.writer_test_context import prepare_writer_facts
from tests.south_star1.writer_test_context import writer_runtime_options
from tests.south_star1.helpers import tetrahedral_facts
from tests.south_star1.writer_test_fixtures import (
    directional_non_single_ring_carrier_facts,
)
from tests.south_star1.writer_test_fixtures import (
    directional_ring_carrier_facts,
)
from tests.south_star1.writer_test_fixtures import (
    shared_directional_ring_carrier_facts,
)
from tests.south_star1.helpers import two_atom_facts

RUN_SLOW_ENV = "SOUTH_STAR1_RUN_SLOW"



from tests.south_star1.writer_artifact_test_support import artifact_object_by_id
from tests.south_star1.writer_artifact_test_support import closed_term_field
from tests.south_star1.writer_artifact_test_support import closed_term_digest
from tests.south_star1.writer_artifact_test_support import refresh_kind_manifest_digest
from grimace._south_star1.writer_envelope_terms import _identity_digest

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


