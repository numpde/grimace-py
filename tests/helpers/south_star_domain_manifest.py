from __future__ import annotations

from dataclasses import dataclass


SOUTH_STAR_FIRST_DOMAIN_POLICY = "south_star_first_domain_maximal_eligible_carrier"
SOUTH_STAR_EXPANDED_SUPPORT_POLICY = "south_star_expanded_domain_regression"


@dataclass(frozen=True, slots=True)
class SouthStarDomainManifest:
    name: str
    schema_version: int
    fixture_policies: frozenset[str]
    support_authorities: frozenset[str]
    expanded_feature_areas: frozenset[str]
    annotation_policies: frozenset[str]
    fragment_order_policies: frozenset[str]
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
            "independent_first_domain_oracle",
            "graph_native_regression_with_semantic_parseback",
        }
    ),
    expanded_feature_areas=frozenset(
        {
            "simple_saturated_monocycle",
            "branched_saturated_monocycle",
            "disconnected_markerless_fragments",
            "disconnected_stereo_fragments",
            "tetrahedral_atom_stereo",
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
    unsupported_feature_categories=frozenset(
        {
            "aromatic_directional_surface",
            "atom_stereo",
            "dative_bond",
            "disconnected_molecule",
            "empty_molecule",
            "metal_atom",
            "query_atom",
            "query_bond",
            "ring_molecule",
            "ring_stereo",
            "unsupported_atom_text",
            "unsupported_bond_type",
            "unstated_component_equation",
        }
    ),
)
