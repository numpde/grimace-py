"""Probe atom-map composition with bracket-only aromatic selenium text."""

from __future__ import annotations

from dataclasses import dataclass

from rdkit import Chem

from grimace._south_star.atom_text import atom_text_obligation_for_supported_atom
from grimace._south_star.support_gates import south_star_support_gate_report
from tests.helpers.mols import parse_smiles
from tests.helpers.south_star_exact_support import load_south_star_expanded_support_cases
from tests.helpers.south_star_grammar_conformance import south_star_grammar_conformance


@dataclass(frozen=True, slots=True)
class ProbeCase:
    label: str
    smiles: str


CASES: tuple[ProbeCase, ...] = (
    ProbeCase("mapped aromatic selenium", "[se:7]1cccc1"),
    ProbeCase("mapped capital selenium input", "[Se:7]1cccc1"),
    ProbeCase("unmodified selenium comparator", "[se]1cccc1"),
)


def main() -> None:
    print("| label | source | writer | gate | grammar | atom-text obligations |")
    print("| --- | --- | --- | --- | --- | --- |")
    for case in CASES:
        mol = parse_smiles(case.smiles)
        writer = Chem.MolToSmiles(mol, canonical=False, doRandom=False)
        grammar = south_star_grammar_conformance(writer)
        print(
            f"| {case.label} | `{case.smiles}` | `{writer}` | "
            f"`{_gate(mol)}` | `{'ok' if grammar.passed else grammar.rejection_code}` | "
            f"`{_atom_text_obligations(mol)}` |"
        )

    base = _selenophene_fixture_support()
    mapped = tuple(output.replace("[se]", "[se:7]") for output in base)
    parse_failures = tuple(output for output in mapped if Chem.MolFromSmiles(output) is None)
    grammar_failures = tuple(
        output
        for output in mapped
        if not south_star_grammar_conformance(output).passed
    )
    map_failures = tuple(output for output in mapped if not _has_one_mapped_aromatic_se(output))

    print()
    print("Derived support by substituting `[se]` with `[se:7]` in the pinned")
    print("selenophene support:")
    print(f"- base outputs: {len(base)}")
    print(f"- derived mapped outputs: {len(mapped)}")
    print(f"- parse failures: {len(parse_failures)}")
    print(f"- grammar failures: {len(grammar_failures)}")
    print(f"- selenium atom-map failures: {len(map_failures)}")
    if parse_failures or grammar_failures or map_failures:
        print(f"- first parse failure: {parse_failures[:1]}")
        print(f"- first grammar failure: {grammar_failures[:1]}")
        print(f"- first map failure: {map_failures[:1]}")


def _gate(mol: Chem.Mol) -> str:
    return ",".join(sorted(south_star_support_gate_report(mol).categories)) or "ok"


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


def _selenophene_fixture_support() -> tuple[str, ...]:
    for case in load_south_star_expanded_support_cases():
        if case.case_id == "aromatic_selenium_text_selenophene":
            return case.expected_support
    raise AssertionError("missing aromatic_selenium_text_selenophene fixture")


def _has_one_mapped_aromatic_se(smiles: str) -> bool:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mapped_selenium_atoms = [
        atom
        for atom in mol.GetAtoms()
        if atom.GetSymbol() == "Se"
        and atom.GetIsAromatic()
        and atom.GetAtomMapNum() == 7
    ]
    return len(mapped_selenium_atoms) == 1


if __name__ == "__main__":
    main()
