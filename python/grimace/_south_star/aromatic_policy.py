from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SouthStarAromaticPolicyContract:
    name: str
    status: str
    molecule_fact_contract: str
    atom_text_policy: str
    bond_text_policy: str
    semantic_equivalence_relation: str
    directional_surface_policy: str
    support_gate_categories: tuple[str, ...]
    required_fixture_fields: tuple[str, ...]
    required_proof_obligations: tuple[str, ...]
    supports_aromatic_facts: bool


DEFAULT_SOUTH_STAR_AROMATIC_POLICY_CONTRACT = SouthStarAromaticPolicyContract(
    name="non_aromatic_molecule_facts",
    status="active",
    molecule_fact_contract="non_aromatic_molecule_facts",
    atom_text_policy="non_aromatic_organic_and_bracket_atom_text",
    bond_text_policy="non_aromatic_single_double_bond_text",
    semantic_equivalence_relation="non_aromatic_parse_back_graph_stereo_identity",
    directional_surface_policy="unsupported_aromatic_directional_overlay",
    support_gate_categories=("aromatic_ring_surface", "aromatic_directional_surface"),
    required_fixture_fields=(
        "source_smiles",
        "expected_support",
        "semantic_parseback",
    ),
    required_proof_obligations=(
        "non_aromatic_atom_text_obligations",
        "non_aromatic_bond_text_obligations",
        "non_aromatic_parseback_identity",
    ),
    supports_aromatic_facts=False,
)

SOUTH_STAR_NON_AROMATIC_KEKULE_FACTS_POLICY_CONTRACT = (
    SouthStarAromaticPolicyContract(
        name="non_aromatic_kekule_facts",
        status="candidate",
        molecule_fact_contract="caller_prepared_non_aromatic_kekule_facts",
        atom_text_policy="non_aromatic_organic_and_bracket_atom_text",
        bond_text_policy="explicit_kekule_single_double_bond_text",
        semantic_equivalence_relation="non_aromatic_parse_back_graph_stereo_identity",
        directional_surface_policy="unsupported_aromatic_directional_overlay",
        support_gate_categories=("aromatic_directional_surface",),
        required_fixture_fields=(
            "source_smiles",
            "preparation_contract",
            "expected_support",
            "semantic_parseback",
        ),
        required_proof_obligations=(
            "caller_prepared_kekule_fact_boundary",
            "explicit_single_double_bond_text_obligations",
            "non_aromatic_parseback_identity",
        ),
        supports_aromatic_facts=False,
    )
)

SOUTH_STAR_AROMATIC_TEXT_POLICY_CONTRACT = SouthStarAromaticPolicyContract(
    name="aromatic_text_policy",
    status="candidate",
    molecule_fact_contract="sanitized_aromatic_molecule_facts",
    atom_text_policy="lowercase_aromatic_atom_text",
    bond_text_policy="aromatic_bond_elision_or_explicit_aromatic_bond_text",
    semantic_equivalence_relation="aromatic_or_kekule_parse_back_semantic_identity",
    directional_surface_policy="undecided_aromatic_directional_overlay",
    support_gate_categories=("aromatic_directional_surface",),
    required_fixture_fields=(
        "source_smiles",
        "aromatic_fact_signature",
        "expected_support",
        "semantic_parseback",
        "equivalence_relation",
    ),
    required_proof_obligations=(
        "sanitized_aromatic_fact_boundary",
        "lowercase_aromatic_atom_text_obligations",
        "aromatic_bond_text_obligations",
        "aromatic_or_kekule_parseback_equivalence",
    ),
    supports_aromatic_facts=True,
)

SOUTH_STAR_AROMATIC_POLICY_FAMILY_CONTRACTS: tuple[
    SouthStarAromaticPolicyContract,
    ...
] = (
    DEFAULT_SOUTH_STAR_AROMATIC_POLICY_CONTRACT,
    SOUTH_STAR_NON_AROMATIC_KEKULE_FACTS_POLICY_CONTRACT,
    SOUTH_STAR_AROMATIC_TEXT_POLICY_CONTRACT,
)
