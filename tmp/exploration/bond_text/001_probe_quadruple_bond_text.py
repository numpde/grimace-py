"""Probe whether quadruple bond text is a narrow South Star bond-policy slice."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_grammar_conformance import south_star_grammar_conformance
from tests.helpers.south_star_semantic_identity import graph_signature


@dataclass(frozen=True, slots=True)
class QuadrupleBondProbeCase:
    case_id: str
    source_smiles: str
    note: str


CASES: tuple[QuadrupleBondProbeCase, ...] = (
    QuadrupleBondProbeCase(
        case_id="carbon_carbon_quadruple",
        source_smiles="C$C",
        note="smallest organic subset quadruple bond accepted by RDKit",
    ),
    QuadrupleBondProbeCase(
        case_id="bracket_carbon_quadruple",
        source_smiles="[C]$[C]",
        note="same molecule after explicit bracket input",
    ),
    QuadrupleBondProbeCase(
        case_id="metal_quadruple",
        source_smiles="[Mo]$[Mo]",
        note="accepted by RDKit but crosses South Star metal atom policy",
    ),
    QuadrupleBondProbeCase(
        case_id="opensmiles_example",
        source_smiles="[Ga+]$[As-]",
        note="OpenSMILES-style quadruple bond example with bracket atoms",
    ),
    QuadrupleBondProbeCase(
        case_id="invalid_carbon_valence_branch",
        source_smiles="CC$C",
        note="RDKit rejects sanitized valence, so it is not a renderer-policy witness",
    ),
)


def main() -> None:
    for case in CASES:
        mol = Chem.MolFromSmiles(case.source_smiles)
        print(f"\n## {case.case_id}")
        print(f"source: {case.source_smiles}")
        print(f"note: {case.note}")
        print(f"rdkit_parseable: {mol is not None}")
        if mol is None:
            continue

        random_outputs = tuple(
            dict.fromkeys(
                Chem.MolToSmiles(
                    mol,
                    canonical=False,
                    doRandom=True,
                    isomericSmiles=True,
                )
                for _ in range(32)
            )
        )
        print(f"canonical_smiles: {Chem.MolToSmiles(mol, canonical=True)}")
        print(f"random_outputs: {random_outputs}")
        print(f"graph_signature: {graph_signature(Chem.MolToSmiles(mol))}")
        print(
            "bond_facts: "
            + repr(
                tuple(
                    (
                        bond.GetIdx(),
                        str(bond.GetBondType()),
                        bond.GetIsAromatic(),
                        bond.GetBondDir().name,
                        bond.GetStereo().name,
                    )
                    for bond in mol.GetBonds()
                )
            )
        )
        report = south_star_support_gate_report(mol)
        print(f"south_star_supported: {report.supported}")
        print(f"south_star_categories: {tuple(sorted(report.categories))}")
        print(
            "grammar: "
            + repr(
                tuple(
                    (
                        output,
                        south_star_grammar_conformance(output).passed,
                        south_star_grammar_conformance(output).rejection_code,
                    )
                    for output in random_outputs
                )
            )
        )


if __name__ == "__main__":
    main()
