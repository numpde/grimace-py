from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SouthStarAromaticPolicyContract:
    name: str
    molecule_fact_contract: str
    atom_text_policy: str
    bond_text_policy: str
    semantic_equivalence_relation: str
    directional_surface_policy: str
    supports_aromatic_facts: bool


DEFAULT_SOUTH_STAR_AROMATIC_POLICY_CONTRACT = SouthStarAromaticPolicyContract(
    name="non_aromatic_molecule_facts",
    molecule_fact_contract="non_aromatic_molecule_facts",
    atom_text_policy="non_aromatic_organic_and_bracket_atom_text",
    bond_text_policy="non_aromatic_single_double_bond_text",
    semantic_equivalence_relation="non_aromatic_parse_back_graph_stereo_identity",
    directional_surface_policy="unsupported_aromatic_directional_overlay",
    supports_aromatic_facts=False,
)
