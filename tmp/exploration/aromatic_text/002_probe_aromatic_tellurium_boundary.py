"""Probe whether tellurium is a clean bracket-only aromatic text slice."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.atom_text import (
    SOUTH_STAR_BRACKET_ONLY_AROMATIC_ATOM_TEXT_TOKENS,
    SOUTH_STAR_BRACKET_ONLY_ATOM_TEXT_TOKENS,
    atom_text_obligation_for_supported_atom,
)
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.south_star_grammar_conformance import south_star_grammar_conformance


@dataclass(frozen=True, slots=True)
class ProbeCase:
    label: str
    smiles: str


CASES: tuple[ProbeCase, ...] = (
    ProbeCase("tellurophene lowercase", "[te]1cccc1"),
    ProbeCase("tellurophene capital", "[Te]1cccc1"),
    ProbeCase("mapped tellurophene", "[te:7]1cccc1"),
    ProbeCase("tellurium explicit H", "[teH]1cccc1"),
    ProbeCase("charged tellurium", "[te+]1cccc1"),
    ProbeCase("acyclic tellurium baseline", "[TeH]"),
    ProbeCase("selenophene implemented comparator", "[se]1cccc1"),
)


def main() -> None:
    print(
        "| label | source | parses | writer | atoms | current South Star gate | "
        "grammar | atom-text obligation | missing policy token |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for case in CASES:
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None:
            print(
                f"| {case.label} | `{case.smiles}` | no | n/a | n/a | n/a | "
                "n/a | n/a | n/a |"
            )
            continue
        writer = Chem.MolToSmiles(mol, canonical=False, doRandom=False)
        atom_facts = "; ".join(_atom_fact(atom) for atom in mol.GetAtoms())
        gate = ",".join(sorted(south_star_support_gate_report(mol).categories)) or "ok"
        grammar = south_star_grammar_conformance(writer)
        obligations = "; ".join(_atom_text_obligation(atom) for atom in mol.GetAtoms())
        missing_policy_token = _missing_policy_token(mol)
        print(
            f"| {case.label} | `{case.smiles}` | yes | `{writer}` | "
            f"`{atom_facts}` | `{gate}` | "
            f"`{'ok' if grammar.passed else grammar.rejection_code}` | "
            f"`{obligations}` | `{missing_policy_token}` |"
        )


def _atom_fact(atom: Chem.Atom) -> str:
    return (
        f"{atom.GetIdx()}:{atom.GetSymbol()}:Z{atom.GetAtomicNum()}:"
        f"arom{int(atom.GetIsAromatic())}:H{atom.GetNumExplicitHs()}:"
        f"q{atom.GetFormalCharge()}:map{atom.GetAtomMapNum()}"
    )


def _atom_text_obligation(atom: Chem.Atom) -> str:
    try:
        obligation = atom_text_obligation_for_supported_atom(atom)
    except NotImplementedError as exc:
        return f"{atom.GetIdx()}:{type(exc).__name__}"
    return f"{atom.GetIdx()}:{obligation.emitted_text}:{obligation.token_family}"


def _missing_policy_token(mol: Chem.Mol) -> str:
    missing: list[str] = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol() == "Te" and "Te" not in SOUTH_STAR_BRACKET_ONLY_ATOM_TEXT_TOKENS:
            missing.append("Te bracket-only atom")
        if (
            atom.GetSymbol() == "Te"
            and atom.GetIsAromatic()
            and "te" not in SOUTH_STAR_BRACKET_ONLY_AROMATIC_ATOM_TEXT_TOKENS
        ):
            missing.append("te bracket-only aromatic atom")
    return ",".join(sorted(set(missing))) or "none"


if __name__ == "__main__":
    main()
