"""Inspect whether aromatic-bond directions are South Star semantic facts.

This probe distinguishes two surfaces:

- a manually assigned RDKit BondDir on an aromatic bond, which RDKit's SMILES
  writer drops and reparsing loses;
- an ordinary exocyclic directional alkene attached to an aromatic ring, whose
  directional carriers are non-aromatic and already belong to the normal
  directional-bond stereo model.
"""

from __future__ import annotations

from rdkit import Chem

from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_semantic_oracle import parse_smiles


def main() -> None:
    aromatic_overlay = parse_smiles("c1ccccc1")
    aromatic_overlay.GetBondWithIdx(0).SetBondDir(Chem.BondDir.ENDUPRIGHT)
    overlay_smiles = Chem.MolToSmiles(
        aromatic_overlay,
        canonical=False,
        isomericSmiles=True,
    )
    reparsed_overlay = parse_smiles(overlay_smiles)

    exocyclic_alkene = parse_smiles("c1ccccc1/C=C/Cl")

    print("manual aromatic direction writer output:", overlay_smiles)
    print(
        "manual aromatic direction survives reparse:",
        any(
            bond.GetBondDir() != Chem.BondDir.NONE
            for bond in reparsed_overlay.GetBonds()
        ),
    )
    print(
        "manual aromatic direction support categories:",
        sorted(south_star_support_gate_report(aromatic_overlay).categories),
    )
    print(
        "exocyclic aromatic alkene support categories:",
        sorted(south_star_support_gate_report(exocyclic_alkene).categories),
    )


if __name__ == "__main__":
    main()
