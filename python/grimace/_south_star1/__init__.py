"""Private South Star 1 proof-kernel package.

This package is confined implementation scaffolding for the formal exact-support
model. It is not a public API and must not be re-exported from ``grimace``.

The core modules in this package are intentionally RDKit-free. RDKit belongs
only at the adapter/audit boundary.
"""

from __future__ import annotations

CORE_MODULES: tuple[str, ...] = (
    "annotation",
    "certificate_checker",
    "certificates",
    "completeness_checker",
    "constraints",
    "enumerate",
    "enumeration_trace",
    "errors",
    "facts",
    "fact_isomorphism",
    "finite_space_checker",
    "graph_index",
    "ids",
    "nonstereo_witness_search",
    "ordinary_ligand_equivalence",
    "ordinary_policy",
    "ordinary_semantics",
    "ordinary_stereo_closure",
    "ordinary_stereo_sites",
    "online_continuation",
    "online_decoder",
    "online_decoder_api",
    "online_decoder_state",
    "online_decisions",
    "online_render_sink",
    "online_search_vm",
    "online_stereo_witness",
    "exhaustive_online_traversal",
    "policy",
    "prepared_bench_matrix",
    "prepared_runtime",
    "proof_terms",
    "residual_constraints",
    "render",
    "ring_labels",
    "root_domains",
    "semantics",
    "semantic_relation_checker",
    "skeleton",
    "slots",
    "stereo_csp",
    "stereo_mapping",
    "stereo_templates",
    "support_artifact",
    "support_artifact_checker",
    "support_artifact_schema",
    "support_enumeration",
    "stereo_witness",
    "writer_branch_certificates",
    "writer_blocked_frontier_certificates",
    "writer_capability_certificates",
    "writer_events",
    "writer_closure_candidate_branch_certificates",
    "writer_closure_candidate_lifecycle",
    "writer_count_certificates",
    "writer_diagnostic_certificates",
    "writer_frontier",
    "writer_frontier_certificates",
    "writer_graph_obligations",
    "writer_online_decoder",
    "writer_projection_certificates",
    "writer_online_decoder_certificates",
    "writer_online_stats_certificates",
    "writer_residual_attachment_branch_certificates",
    "writer_residual_attachment_lifecycle",
    "writer_ring_lifecycle",
    "writer_runtime",
    "writer_snapshot",
    "writer_snapshot_certificates",
    "writer_state_delta_certificates",
    "writer_state",
    "writer_support_count_certificates",
    "writer_stereo",
    "writer_stereo_branch_certificates",
    "writer_stereo_non_neighbor",
    "writer_support",
    "writer_support_certificates",
    "writer_choice_count_certificates",
    "writer_terminal_certificates",
    "writer_transitions",
)

BOUNDARY_MODULES: tuple[str, ...] = (
    "audit_rdkit",
    "rdkit_adapter",
)

__all__ = ("BOUNDARY_MODULES", "CORE_MODULES")

from . import writer_stereo_non_neighbor as _writer_stereo_non_neighbor

_writer_stereo_non_neighbor.install()

del _writer_stereo_non_neighbor
