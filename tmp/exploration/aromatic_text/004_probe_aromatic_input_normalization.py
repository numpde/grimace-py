"""Probe RDKit normalization of aromatic-looking As/Si inputs."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from grimace._south_star.support_gates import SouthStarUnsupportedFeatureError
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_grammar_conformance import south_star_grammar_conformance


@dataclass(frozen=True, slots=True)
class ProbeCase:
    label: str
    smiles: str


CASES: tuple[ProbeCase, ...] = (
    ProbeCase("aromatic-looking silicon input", "[si]1ccccc1"),
    ProbeCase("explicit Kekule silicon input", "[Si]1=CC=CC=C1"),
    ProbeCase("aromatic-looking arsenic input", "[as]1ccccc1"),
    ProbeCase("explicit Kekule arsenic input", "[As]1=CC=CC=C1"),
)


def main() -> None:
    print(
        "| label | source | writer | atom facts | bond facts | gate | grammar | "
        "EnumS support |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for case in CASES:
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None:
            print(f"| {case.label} | `{case.smiles}` | parse failed | | | | | |")
            continue
        writer = Chem.MolToSmiles(mol, canonical=False, doRandom=False)
        grammar = south_star_grammar_conformance(writer)
        print(
            f"| {case.label} | `{case.smiles}` | `{writer}` | "
            f"`{_atom_facts(mol)}` | `{_bond_facts(mol)}` | "
            f"`{_gate(mol)}` | `{'ok' if grammar.passed else grammar.rejection_code}` | "
            f"`{_enum_support_summary(case.smiles)}` |"
        )

    print()
    print(
        "Silicon normalized inputs have identical South Star support:",
        _same_outputs("[si]1ccccc1", "[Si]1=CC=CC=C1"),
    )


def _atom_facts(mol: Chem.Mol) -> str:
    return "; ".join(
        f"{atom.GetIdx()}:{atom.GetSymbol()}:arom{int(atom.GetIsAromatic())}"
        for atom in mol.GetAtoms()
    )


def _bond_facts(mol: Chem.Mol) -> str:
    return "; ".join(
        f"{bond.GetIdx()}:{bond.GetBondType().name}:arom{int(bond.GetIsAromatic())}"
        for bond in mol.GetBonds()
    )


def _gate(mol: Chem.Mol) -> str:
    return ",".join(sorted(south_star_support_gate_report(mol).categories)) or "ok"


def _enum_support_summary(smiles: str) -> str:
    try:
        result = mol_to_smiles_enum_s_graph_native(smiles)
    except SouthStarUnsupportedFeatureError as exc:
        return "unsupported:" + ",".join(sorted(exc.categories))
    return f"{len(result.outputs)} outputs"


def _same_outputs(left: str, right: str) -> bool:
    return (
        mol_to_smiles_enum_s_graph_native(left).outputs
        == mol_to_smiles_enum_s_graph_native(right).outputs
    )


if __name__ == "__main__":
    main()
