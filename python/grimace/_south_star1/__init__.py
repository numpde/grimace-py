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
    "online_traversal",
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
)

BOUNDARY_MODULES: tuple[str, ...] = (
    "audit_rdkit",
    "rdkit_adapter",
)

__all__ = ("BOUNDARY_MODULES", "CORE_MODULES")
