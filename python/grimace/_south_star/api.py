from __future__ import annotations

from rdkit import Chem

from grimace._south_star.enum_s import SouthStarEnumSPrototypeResult
from grimace._south_star.enum_s import _mol_to_smiles_enum_s_graph_native_for_mol
from grimace._south_star.molecule_facts import SouthStarMoleculeFacts
from grimace._south_star.policies import DEFAULT_SOUTH_STAR_POLICY_SET
from grimace._south_star.policies import SouthStarPolicySet


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
