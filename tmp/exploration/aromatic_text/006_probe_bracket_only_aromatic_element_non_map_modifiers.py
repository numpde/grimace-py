"""Probe non-map modifiers on bracket-only aromatic element text."""

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
    ProbeCase("isotope tellurium", "[15te]1cccc1"),
    ProbeCase("isotope selenium", "[15se]1cccc1"),
    ProbeCase("isotope mapped tellurium", "[15te:7]1cccc1"),
    ProbeCase("isotope mapped selenium", "[15se:7]1cccc1"),
    ProbeCase("explicit-H tellurium", "[teH]1cccc1"),
    ProbeCase("explicit-H selenium", "[seH]1cccc1"),
    ProbeCase("charged tellurium", "[te+]1cccc1"),
    ProbeCase("charged selenium", "[se+]1cccc1"),
    ProbeCase("chiral tellurium input", "[te@]1cccc1"),
    ProbeCase("chiral selenium input", "[se@]1cccc1"),
)


def main() -> None:
    print(
        "| label | source | parses | writer | atom facts | gate | grammar | "
        "atom-text obligations |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- |")
    for case in CASES:
        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None:
            print(
                f"| {case.label} | `{case.smiles}` | no | n/a | n/a | "
                "n/a | n/a | n/a |"
            )
            continue
        writer = Chem.MolToSmiles(mol, canonical=False, doRandom=False)
        grammar = south_star_grammar_conformance(writer)
        print(
            f"| {case.label} | `{case.smiles}` | yes | `{writer}` | "
            f"`{_atom_facts(mol)}` | `{_gate(mol)}` | "
            f"`{'ok' if grammar.passed else grammar.rejection_code}` | "
            f"`{_atom_text_obligations(mol)}` |"
        )

    _print_derived_support("tellurium isotope", "[te]", "[15te]")
    _print_derived_support("selenium isotope", "[se]", "[15se]")
    _print_derived_support("mapped tellurium isotope", "[te:7]", "[15te:7]")
    _print_derived_support("mapped selenium isotope", "[se:7]", "[15se:7]")


def _atom_facts(mol: Chem.Mol) -> str:
    return "; ".join(
        f"{atom.GetIdx()}:{atom.GetSymbol()}:iso{atom.GetIsotope()}:"
        f"H{atom.GetNumExplicitHs()}:q{atom.GetFormalCharge()}:"
        f"rad{atom.GetNumRadicalElectrons()}:map{atom.GetAtomMapNum()}:"
        f"chiral{atom.GetChiralTag()}:arom{int(atom.GetIsAromatic())}"
        for atom in mol.GetAtoms()
    )


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


def _print_derived_support(label: str, old: str, new: str) -> None:
    base = _fixture_support_containing(old)
    derived = tuple(output.replace(old, new) for output in base)
    parse_failures = tuple(output for output in derived if Chem.MolFromSmiles(output) is None)
    grammar_failures = tuple(
        output
        for output in derived
        if not south_star_grammar_conformance(output).passed
    )
    isotope_failures = tuple(output for output in derived if not _has_isotope(output))

    print()
    print(f"Derived {label} support by substituting `{old}` with `{new}`:")
    print(f"- base outputs: {len(base)}")
    print(f"- derived outputs: {len(derived)}")
    print(f"- parse failures: {len(parse_failures)}")
    print(f"- grammar failures: {len(grammar_failures)}")
    print(f"- isotope failures: {len(isotope_failures)}")


def _fixture_support_containing(token: str) -> tuple[str, ...]:
    for case in load_south_star_expanded_support_cases():
        if any(token in output for output in case.expected_support):
            return case.expected_support
    raise AssertionError(f"missing fixture support containing {token!r}")


def _has_isotope(smiles: str) -> bool:
    mol = parse_smiles(smiles)
    return any(atom.GetIsotope() != 0 for atom in mol.GetAtoms())


if __name__ == "__main__":
    main()
