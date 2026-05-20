from __future__ import annotations

from rdkit import Chem

from grimace._south_star.enum_s import SouthStarEnumSPrototypeResult
from grimace._south_star.enum_s import _mol_to_smiles_enum_s_graph_native_for_mol
from grimace._south_star.policies import DEFAULT_SOUTH_STAR_POLICY_SET
from grimace._south_star.policies import SouthStarPolicySet
from grimace._south_star.support_gates import south_star_support_gate_report


def mol_to_smiles_enum_s_private(
    mol: Chem.Mol,
    *,
    policy_set: SouthStarPolicySet = DEFAULT_SOUTH_STAR_POLICY_SET,
) -> SouthStarEnumSPrototypeResult:
    """Private candidate boundary for a future `MolToSmilesEnumS` API."""
    if not isinstance(mol, Chem.Mol):
        raise TypeError("mol_to_smiles_enum_s_private requires an RDKit Mol")

    support_report = south_star_support_gate_report(mol)
    support_report.fail_if_unsupported()

    result = _mol_to_smiles_enum_s_graph_native_for_mol(
        mol,
        policy_set=policy_set,
    )
    if result.generation_diagnostics is None:
        raise AssertionError("private EnumS API requires generation diagnostics")
    return result
