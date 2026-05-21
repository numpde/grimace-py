from __future__ import annotations

from dataclasses import dataclass


SOUTH_STAR_FIRST_DOMAIN_POLICY = "south_star_first_domain_maximal_eligible_carrier"
SOUTH_STAR_EXPANDED_SUPPORT_POLICY = "south_star_expanded_domain_regression"
SOUTH_STAR_FIRST_DOMAIN_WITNESS_AUTHORITY = (
    "temporary_witness_first_domain_shared_spine"
)
SOUTH_STAR_DISCONNECTED_COMPOSITION_WITNESS_AUTHORITY = (
    "temporary_witness_disconnected_composition_shared_records"
)
SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY = (
    "graph_native_regression_witness_with_semantic_parseback"
)
SOUTH_STAR_SATURATED_MONOCYCLE_WITNESS_AUTHORITY = (
    "temporary_witness_saturated_monocycle_shared_records"
)
SOUTH_STAR_NONSTEREO_MONOCYCLE_WITNESS_AUTHORITY = (
    "temporary_witness_nonstereo_monocycle_shared_records"
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

SOUTH_STAR_UNIFIED_REFERENCE_AUTHORITIES: frozenset[str] = frozenset()
SOUTH_STAR_TEMPORARY_WITNESS_AUTHORITIES: frozenset[str] = frozenset(
    {
        SOUTH_STAR_DISCONNECTED_COMPOSITION_WITNESS_AUTHORITY,
        SOUTH_STAR_FIRST_DOMAIN_WITNESS_AUTHORITY,
        SOUTH_STAR_NONSTEREO_MONOCYCLE_WITNESS_AUTHORITY,
        SOUTH_STAR_POLYCYCLIC_RING_STEREO_WITNESS_AUTHORITY,
        SOUTH_STAR_RING_STEREO_MONOCYCLE_WITNESS_AUTHORITY,
        SOUTH_STAR_RING_TETRAHEDRAL_MONOCYCLE_WITNESS_AUTHORITY,
        SOUTH_STAR_SATURATED_MONOCYCLE_WITNESS_AUTHORITY,
        SOUTH_STAR_TETRAHEDRAL_ATOM_STEREO_WITNESS_AUTHORITY,
    }
)
SOUTH_STAR_REGRESSION_WITNESS_AUTHORITIES: frozenset[str] = frozenset(
    {
        SOUTH_STAR_GRAPH_NATIVE_REGRESSION_AUTHORITY,
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
        }
    ),
    annotation_policies=frozenset(
        {
            "maximal_eligible_carrier",
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
            "unsupported_atom_charge",
            "unsupported_atom_isotope",
            "unsupported_atom_map",
            "unsupported_atom_text",
            "unsupported_bond_type",
            "unsupported_radical_atom",
            "unstated_component_equation",
        }
    ),
)
