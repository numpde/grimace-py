"""Probe bracket-only main-group atom-text candidates as a policy family."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.atom_text import (
    atom_text_obligation_for_supported_atom,
    south_star_atom_text_fields,
    unsupported_atom_text_reasons,
)
from grimace._south_star.support_gates import METAL_ATOMIC_NUMBERS
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_grammar_conformance import south_star_grammar_conformance


@dataclass(frozen=True, slots=True)
class ProbeCase:
    label: str
    smiles: str


CASES: tuple[ProbeCase, ...] = (
    ProbeCase("supported silicon hydride", "[SiH3]C"),
    ProbeCase("supported selenium hydride", "[SeH]"),
    ProbeCase("supported tellurium hydride", "[TeH]"),
    ProbeCase("arsine", "[AsH3]"),
    ProbeCase("arsine methyl", "[AsH2]C"),
    ProbeCase("germane", "[GeH4]"),
    ProbeCase("germyl methyl", "[GeH3]C"),
    ProbeCase("stibine", "[SbH3]"),
    ProbeCase("arsenic Kekule ring", "[As]1=CC=CC=C1"),
    ProbeCase("germanium Kekule ring", "[Ge]1=CC=CC=C1"),
    ProbeCase("tin hydride metal-boundary comparator", "[SnH4]"),
    ProbeCase("sodium cation metal comparator", "[Na+]"),
    ProbeCase("magnesium cation metal comparator", "[Mg+2]"),
)


def main() -> None:
    print(
        "| label | source | parses | writer | atom facts | metal atoms | gate | "
        "unsupported atom text | grammar | atom-text obligation |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for case in CASES:
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None:
            print(
                f"| {case.label} | `{case.smiles}` | no | n/a | n/a | n/a | "
                "n/a | n/a | n/a | n/a |"
            )
            continue
        writer = Chem.MolToSmiles(mol, canonical=False, doRandom=False)
        grammar = south_star_grammar_conformance(writer)
        print(
            f"| {case.label} | `{case.smiles}` | yes | `{writer}` | "
            f"`{_atom_facts(mol)}` | `{_metal_atoms(mol)}` | "
            f"`{_gate(mol)}` | `{_unsupported_atom_text(mol)}` | "
            f"`{'ok' if grammar.passed else grammar.rejection_code}` | "
            f"`{_atom_text_obligations(mol)}` |"
        )


def _atom_facts(mol: Chem.Mol) -> str:
    return "; ".join(
        f"{atom.GetIdx()}:{atom.GetSymbol()}:Z{atom.GetAtomicNum()}:"
        f"H{atom.GetNumExplicitHs()}:q{atom.GetFormalCharge()}:"
        f"rad{atom.GetNumRadicalElectrons()}:arom{int(atom.GetIsAromatic())}"
        for atom in mol.GetAtoms()
    )


def _metal_atoms(mol: Chem.Mol) -> str:
    indices = tuple(
        atom.GetIdx()
        for atom in mol.GetAtoms()
        if atom.GetAtomicNum() in METAL_ATOMIC_NUMBERS
    )
    return ",".join(str(idx) for idx in indices) or "none"


def _gate(mol: Chem.Mol) -> str:
    return ",".join(sorted(south_star_support_gate_report(mol).categories)) or "ok"


def _unsupported_atom_text(mol: Chem.Mol) -> str:
    summaries: list[str] = []
    for atom in mol.GetAtoms():
        reasons = unsupported_atom_text_reasons(south_star_atom_text_fields(atom))
        if reasons:
            summaries.extend(f"{atom.GetIdx()}:{reason.category}" for reason in reasons)
    return "; ".join(summaries) or "none"


def _atom_text_obligations(mol: Chem.Mol) -> str:
    summaries: list[str] = []
    for atom in mol.GetAtoms():
        try:
            obligation = atom_text_obligation_for_supported_atom(atom)
        except NotImplementedError as exc:
            summaries.append(f"{atom.GetIdx()}:{type(exc).__name__}")
            continue
        summaries.append(
            f"{atom.GetIdx()}:{obligation.emitted_text}:"
            f"{'/'.join(obligation.bracket_obligations)}"
        )
    return "; ".join(summaries)


if __name__ == "__main__":
    main()
