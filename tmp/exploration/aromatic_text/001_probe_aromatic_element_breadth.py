"""Probe RDKit and South Star treatment of bracket aromatic main-group atoms."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.atom_text import atom_text_obligation_for_supported_atom
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_grammar_conformance import south_star_grammar_conformance


@dataclass(frozen=True, slots=True)
class ProbeCase:
    label: str
    smiles: str


CASES: tuple[ProbeCase, ...] = (
    ProbeCase("selenophene", "[se]1cccc1"),
    ProbeCase("tellurophene", "[te]1cccc1"),
    ProbeCase("arsabenzene", "[as]1ccccc1"),
    ProbeCase("silabenzene", "[si]1ccccc1"),
    ProbeCase("bracket aromatic sulfur baseline", "s1cccc1"),
    ProbeCase("explicit bracket aromatic sulfur", "[s]1cccc1"),
    ProbeCase("capital selenium aromatic input", "[Se]1cccc1"),
    ProbeCase("selenium with explicit H", "[seH]1cccc1"),
    ProbeCase("mapped aromatic selenium", "[se:7]1cccc1"),
    ProbeCase("charged aromatic selenium", "[se+]1cccc1"),
)


def main() -> None:
    print(
        "| label | source | parses | writer | atoms | gate | grammar | atom text |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for case in CASES:
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None:
            print(
                f"| {case.label} | `{case.smiles}` | no | n/a | n/a | n/a | "
                "n/a | n/a |"
            )
            continue
        writer = Chem.MolToSmiles(mol, canonical=False, doRandom=False)
        atoms = ", ".join(
            f"{atom.GetIdx()}:{atom.GetSymbol()}:{int(atom.GetIsAromatic())}:"
            f"H{atom.GetNumExplicitHs()}:q{atom.GetFormalCharge()}:"
            f"map{atom.GetAtomMapNum()}"
            for atom in mol.GetAtoms()
        )
        gate = ",".join(sorted(south_star_support_gate_report(mol).categories)) or "ok"
        grammar = (
            "ok"
            if south_star_grammar_conformance(writer).passed
            else south_star_grammar_conformance(writer).rejection_code
        )
        atom_text = _atom_text_summary(mol)
        print(
            f"| {case.label} | `{case.smiles}` | yes | `{writer}` | "
            f"`{atoms}` | `{gate}` | `{grammar}` | `{atom_text}` |"
        )


def _atom_text_summary(mol: Chem.Mol) -> str:
    summaries = []
    for atom in mol.GetAtoms():
        if not atom.GetIsAromatic():
            continue
        try:
            obligation = atom_text_obligation_for_supported_atom(atom)
        except NotImplementedError as exc:
            summaries.append(f"{atom.GetIdx()}:{type(exc).__name__}")
            continue
        summaries.append(
            f"{atom.GetIdx()}:{obligation.emitted_text}:"
            f"{obligation.token_family}"
        )
    return ",".join(summaries)


if __name__ == "__main__":
    main()
