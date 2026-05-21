from __future__ import annotations

from dataclasses import dataclass

from grimace._south_star.aromatic_policy import (
    DEFAULT_SOUTH_STAR_AROMATIC_POLICY_CONTRACT,
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
SOUTH_STAR_DISCONNECTED_COMPOSITION_WITNESS_AUTHORITY = (
    "temporary_witness_disconnected_composition_shared_records"
)
SOUTH_STAR_DISCONNECTED_COMPOSITION_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_disconnected_composition"
)
SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY = (
    "graph_native_regression_witness_with_semantic_parseback"
)
SOUTH_STAR_RING_STEREO_MONOCYCLE_WITNESS_AUTHORITY = (
    "temporary_witness_ring_stereo_monocycle_shared_records"
)
SOUTH_STAR_TETRAHEDRAL_ATOM_STEREO_WITNESS_AUTHORITY = (
    "temporary_witness_tetrahedral_atom_stereo"
)
SOUTH_STAR_RING_TETRAHEDRAL_MONOCYCLE_WITNESS_AUTHORITY = (
    "temporary_witness_ring_tetrahedral_monocycle_shared_records"
)
SOUTH_STAR_POLYCYCLIC_RING_STEREO_WITNESS_AUTHORITY = (
    "temporary_witness_polycyclic_ring_stereo_shared_records"
)
SOUTH_STAR_SINGLE_ATOM_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_single_atom_atom_text"
)
SOUTH_STAR_TWO_ATOM_MARKERLESS_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY = (
    "unified_reference_two_atom_markerless_atom_text"
)

SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES: frozenset[str] = frozenset(
    {
        SOUTH_STAR_FIRST_DOMAIN_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_DISCONNECTED_COMPOSITION_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_MARKERLESS_ACYCLIC_TREE_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_NONSTEREO_MONOCYCLE_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_SINGLE_ATOM_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
        SOUTH_STAR_TWO_ATOM_MARKERLESS_ATOM_TEXT_UNIFIED_REFERENCE_AUTHORITY,
    }
)
SOUTH_STAR_TEMPORARY_WITNESS_AUTHORITIES: frozenset[str] = frozenset(
    {
        SOUTH_STAR_DISCONNECTED_COMPOSITION_WITNESS_AUTHORITY,
        SOUTH_STAR_POLYCYCLIC_RING_STEREO_WITNESS_AUTHORITY,
        SOUTH_STAR_RING_STEREO_MONOCYCLE_WITNESS_AUTHORITY,
        SOUTH_STAR_RING_TETRAHEDRAL_MONOCYCLE_WITNESS_AUTHORITY,
        SOUTH_STAR_TETRAHEDRAL_ATOM_STEREO_WITNESS_AUTHORITY,
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
    SOUTH_STAR_POLYCYCLIC_RING_STEREO_WITNESS_AUTHORITY: (
        "Fold into unified-reference polycyclic stereo after closure-edge "
        "choices and marker equations solve through the shared traversal "
        "records."
    ),
    SOUTH_STAR_RING_STEREO_MONOCYCLE_WITNESS_AUTHORITY: (
        "Fold into unified-reference ring stereo after closure marker slots "
        "and parity equations are first-class shared constraint records."
    ),
    SOUTH_STAR_RING_TETRAHEDRAL_MONOCYCLE_WITNESS_AUTHORITY: (
        "Fold into unified-reference ring/tetrahedral support after emitted "
        "ligand order is represented as an atom-stereo obligation over shared "
        "ring traversal events."
    ),
    SOUTH_STAR_TETRAHEDRAL_ATOM_STEREO_WITNESS_AUTHORITY: (
        "Fold into unified-reference atom stereo after tetrahedral ligand-order "
        "constraints are solved as shared atom-stereo obligations."
    ),
}

SOUTH_STAR_SHARED_PIPELINE_ELIGIBLE_EXPANDED_FEATURE_AREAS: frozenset[str] = frozenset(
    {
        "branched_saturated_monocycle",
        "charged_atom_text",
        "disconnected_markerless_fragments",
        "disconnected_stereo_fragments",
        "explicit_bracket_hydrogen",
        "markerless_acyclic_tree",
        "nonstereo_polycyclic_skeleton",
        "polycyclic_ring_stereo",
        "radical_atom_text",
        "ring_stereo_monocycle",
        "ring_tetrahedral_monocycle",
        "simple_saturated_monocycle",
        "tetrahedral_atom_stereo",
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
    unsupported_feature_categories: frozenset[str]


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
            "simple_saturated_monocycle",
            "branched_saturated_monocycle",
            "disconnected_markerless_fragments",
            "disconnected_stereo_fragments",
            "ring_stereo_monocycle",
            "nonstereo_polycyclic_skeleton",
            "polycyclic_ring_stereo",
            "tetrahedral_atom_stereo",
            "ring_tetrahedral_monocycle",
            "unsaturated_nonstereo_monocycle",
            "explicit_bracket_hydrogen",
            "markerless_acyclic_tree",
            "radical_atom_text",
            "charged_atom_text",
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
    unsupported_feature_categories=frozenset(
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
