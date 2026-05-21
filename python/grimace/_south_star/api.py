from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.enum_s import SouthStarEnumSPrototypeResult
from grimace._south_star.enum_s import _mol_to_smiles_enum_s_graph_native_for_mol
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.policies import DEFAULT_SOUTH_STAR_POLICY_SET
from grimace._south_star.policies import SouthStarPolicySet


@dataclass(frozen=True, slots=True)
class SouthStarPrivateApiContract:
    provisional_name: str
    exported_from_public_package: bool
    accepted_input: str
    grammar_basis: str
    semantic_equivalence_checks: tuple[str, ...]
    parser_dependency: str
    annotation_policy: str
    fragment_order_policy: str
    output_order_policy: str
    diagnostic_boundaries: tuple[str, ...]
    unsupported_error_type: str
    rdkit_parity_surface: str


def south_star_private_api_contract(
    *,
    policy_set: SouthStarPolicySet = DEFAULT_SOUTH_STAR_POLICY_SET,
) -> SouthStarPrivateApiContract:
    """Return the inspectable contract for the private EnumS boundary."""
    return SouthStarPrivateApiContract(
        provisional_name="MolToSmilesEnumS",
        exported_from_public_package=False,
        accepted_input="rdkit.Chem.Mol",
        grammar_basis="south_star_declared_subset_grammar_v1",
        semantic_equivalence_checks=(
            "rdkit_parser_dependency",
            "rdkit_canonical_nonisomeric_parseback",
            "rdkit_canonical_isomeric_parseback",
        ),
        parser_dependency="RDKit parser parse-back evidence",
        annotation_policy=policy_set.annotation_policy.name,
        fragment_order_policy=policy_set.fragment_order_policy.name,
        output_order_policy=policy_set.output_order_policy.name,
        diagnostic_boundaries=(
            "result_policy_names",
            "result_generation_diagnostics",
            "support_gate_error_evidence",
            "semantic_parseback_test_evidence",
            "complexity_guardrail_test_evidence",
        ),
        unsupported_error_type="SouthStarUnsupportedFeatureError",
        rdkit_parity_surface="MolToSmilesEnum",
    )


def mol_to_smiles_enum_s_private(
    mol: Chem.Mol,
    *,
    policy_set: SouthStarPolicySet = DEFAULT_SOUTH_STAR_POLICY_SET,
) -> SouthStarEnumSPrototypeResult:
    """Private candidate boundary for a future `MolToSmilesEnumS` API."""
    if not isinstance(mol, Chem.Mol):
        raise TypeError("mol_to_smiles_enum_s_private requires an RDKit Mol")

    molecule_facts = SouthStarMoleculeFacts.from_mol(mol)
    molecule_facts.fail_if_unsupported()

    result = _mol_to_smiles_enum_s_graph_native_for_mol(
        mol,
        policy_set=policy_set,
        molecule_facts=molecule_facts,
    )
    if result.generation_diagnostics is None:
        raise AssertionError("private EnumS API requires generation diagnostics")
    return result
