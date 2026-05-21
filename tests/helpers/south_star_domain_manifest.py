from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.aromatic_policy import (
    DEFAULT_SOUTH_STAR_AROMATIC_POLICY_CONTRACT,
    SOUTH_STAR_AROMATIC_TEXT_POLICY_CONTRACT,
)


SOUTH_STAR_FIRST_DOMAIN_POLICY = "south_star_first_domain_maximal_eligible_carrier"
SOUTH_STAR_EXPANDED_SUPPORT_POLICY = "south_star_expanded_domain_regression"
SOUTH_STAR_FIRST_DOMAIN_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_first_domain_directional_bond_stereo"
)
SOUTH_STAR_MARKERLESS_ACYCLIC_TREE_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_markerless_acyclic_tree"
)
SOUTH_STAR_NONSTEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_nonstereo_monocycle_ring_traversal"
)
SOUTH_STAR_NONSTEREO_POLYCYCLIC_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_nonstereo_polycyclic_closure_traversal"
)
SOUTH_STAR_RING_STEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_ring_stereo_monocycle_marker_obligations"
)
SOUTH_STAR_DISCONNECTED_COMPOSITION_WITNESS_AUTHORITY = (
    "temporary_witness_disconnected_composition_shared_records"
)
SOUTH_STAR_DISCONNECTED_COMPOSITION_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_disconnected_composition"
)
SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY = (
    "graph_native_regression_witness_with_semantic_parseback"
)
SOUTH_STAR_TETRAHEDRAL_ATOM_STEREO_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_tetrahedral_atom_stereo_obligations"
)
SOUTH_STAR_RING_TETRAHEDRAL_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_ring_tetrahedral_monocycle_obligations"
)
SOUTH_STAR_POLYCYCLIC_RING_STEREO_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_polycyclic_ring_stereo_marker_obligations"
)
SOUTH_STAR_SINGLE_ATOM_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_single_atom_atom_text"
)
SOUTH_STAR_TWO_ATOM_MARKERLESS_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_two_atom_markerless_atom_text"
)
SOUTH_STAR_DIRECTIONAL_COMPONENT_PRODUCT_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_directional_component_product"
)
SOUTH_STAR_DIRECTIONAL_TETRAHEDRAL_COMPOSITION_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_directional_tetrahedral_composition"
)
SOUTH_STAR_DISCONNECTED_MIXED_STEREO_COMPOSITION_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_disconnected_mixed_stereo_composition"
)
SOUTH_STAR_EXOCYCLIC_DIRECTIONAL_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_exocyclic_directional_monocycle_obligations"
)
SOUTH_STAR_RING_TETRAHEDRAL_EXOCYCLIC_DIRECTIONAL_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_ring_tetrahedral_exocyclic_directional_obligations"
)
SOUTH_STAR_AROMATIC_TEXT_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_aromatic_text_monocycle_obligations"
)

SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES: frozenset[str] = frozenset(
    {
        SOUTH_STAR_FIRST_DOMAIN_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_DISCONNECTED_COMPOSITION_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_DIRECTIONAL_COMPONENT_PRODUCT_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_DIRECTIONAL_TETRAHEDRAL_COMPOSITION_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_DISCONNECTED_MIXED_STEREO_COMPOSITION_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_EXOCYCLIC_DIRECTIONAL_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_RING_TETRAHEDRAL_EXOCYCLIC_DIRECTIONAL_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_AROMATIC_TEXT_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_MARKERLESS_ACYCLIC_TREE_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_NONSTEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_NONSTEREO_POLYCYCLIC_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_POLYCYCLIC_RING_STEREO_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_RING_STEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_RING_TETRAHEDRAL_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_SINGLE_ATOM_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_TETRAHEDRAL_ATOM_STEREO_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_TWO_ATOM_MARKERLESS_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
    }
)
SOUTH_STAR_TEMPORARY_WITNESS_AUTHORITIES: frozenset[str] = frozenset(
    {
        SOUTH_STAR_DISCONNECTED_COMPOSITION_WITNESS_AUTHORITY,
    }
)
SOUTH_STAR_REGRESSION_WITNESS_AUTHORITIES: frozenset[str] = frozenset(
    {
        SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY,
    }
)
SOUTH_STAR_TEMPORARY_WITNESS_FOLD_IN_PLANS: dict[str, str] = {
    SOUTH_STAR_DISCONNECTED_COMPOSITION_WITNESS_AUTHORITY: (
        "Fold into unified-reference fragment composition after fragment "
        "ordering and per-fragment traversal support are represented as shared "
        "facts and renderer inputs."
    ),
}

SOUTH_STAR_SHARED_PIPELINE_ELIGIBLE_EXPANDED_FEATURE_AREAS: frozenset[str] = frozenset(
    {
        "branched_saturated_monocycle",
        "aromatic_text_monocycle",
        "aromatic_text_branch",
        "charged_atom_text",
        "combined_atom_text_modifiers",
        "disconnected_markerless_fragments",
        "disconnected_stereo_fragments",
        "directional_tetrahedral_composition",
        "double_bond_text",
        "explicit_bracket_hydrogen",
        "exocyclic_directional_monocycle",
        "ring_tetrahedral_exocyclic_directional",
        "atom_map_text",
        "isotope_atom_text",
        "independent_directional_stereo_components",
        "markerless_acyclic_tree",
        "non_organic_bracket_atom_text",
        "nonstereo_polycyclic_skeleton",
        "polycyclic_ring_stereo",
        "radical_atom_text",
        "ring_stereo_monocycle",
        "ring_tetrahedral_monocycle",
        "simple_saturated_monocycle",
        "tetrahedral_atom_stereo",
        "triple_bond_text",
        "unsaturated_nonstereo_monocycle",
    }
)


@dataclass(frozen=True, slots=True)
class SouthStarDomainManifest:
    name: str
    schema_version: int
    fixture_policies: frozenset[str]
    support_authorities: frozenset[str]
    expanded_feature_areas: frozenset[str]
    annotation_policies: frozenset[str]
    aromatic_policy_contracts: frozenset[str]
    fragment_order_policies: frozenset[str]
    output_order_policies: frozenset[str]
    support_gate_blocker_categories: frozenset[str]


SOUTH_STAR_PRIVATE_DOMAIN = SouthStarDomainManifest(
    name="south_star_private_graph_native_seed",
    schema_version=1,
    fixture_policies=frozenset(
        {
            SOUTH_STAR_FIRST_DOMAIN_POLICY,
            SOUTH_STAR_EXPANDED_SUPPORT_POLICY,
        }
    ),
    support_authorities=frozenset(
        {
            *SOUTH_STAR_REGRESSION_WITNESS_AUTHORITIES,
            *SOUTH_STAR_TEMPORARY_WITNESS_AUTHORITIES,
            *SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES,
        }
    ),
    expanded_feature_areas=frozenset(
        {
            "aromatic_text_monocycle",
            "aromatic_text_branch",
            "simple_saturated_monocycle",
            "branched_saturated_monocycle",
            "disconnected_markerless_fragments",
            "disconnected_stereo_fragments",
            "directional_tetrahedral_composition",
            "double_bond_text",
            "exocyclic_directional_monocycle",
            "ring_tetrahedral_exocyclic_directional",
            "ring_stereo_monocycle",
            "nonstereo_polycyclic_skeleton",
            "polycyclic_ring_stereo",
            "tetrahedral_atom_stereo",
            "ring_tetrahedral_monocycle",
            "unsaturated_nonstereo_monocycle",
            "explicit_bracket_hydrogen",
            "atom_map_text",
            "combined_atom_text_modifiers",
            "isotope_atom_text",
            "independent_directional_stereo_components",
            "markerless_acyclic_tree",
            "non_organic_bracket_atom_text",
            "radical_atom_text",
            "charged_atom_text",
            "triple_bond_text",
        }
    ),
    annotation_policies=frozenset(
        {
            "maximal_eligible_carrier",
        }
    ),
    aromatic_policy_contracts=frozenset(
        {
            DEFAULT_SOUTH_STAR_AROMATIC_POLICY_CONTRACT.name,
            SOUTH_STAR_AROMATIC_TEXT_POLICY_CONTRACT.name,
        }
    ),
    fragment_order_policies=frozenset(
        {
            "all_fragment_orders",
        }
    ),
    output_order_policies=frozenset(
        {
            "first_occurrence_deduplication",
        }
    ),
    support_gate_blocker_categories=frozenset(
        {
            "aromatic_directional_surface",
            "aromatic_ring_surface",
            "atom_stereo",
            "dative_bond",
            "disconnected_molecule",
            "empty_molecule",
            "fused_or_polycyclic_ring",
            "metal_atom",
            "query_atom",
            "query_bond",
            "ring_molecule",
            "ring_stereo",
            "ring_tetrahedral_interaction",
            "unsupported_atom_text",
            "unsupported_bond_type",
            "unstated_component_equation",
        }
    ),
)
