from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from rdkit import Chem, rdBase

from grimace import _core, _runtime


SUPPORTED_STEREO_FLAGS = _runtime.MolToSmilesFlags(
    isomeric_smiles=True,
    kekule_smiles=False,
    rooted_at_atom=-1,
    canonical=False,
    all_bonds_explicit=False,
    all_hs_explicit=False,
    do_random=True,
    ignore_atom_map_numbers=False,
)


@dataclass(frozen=True, slots=True)
class Candidate:
    case_id: str
    smiles: str
    rationale: str


CANDIDATES = (
    Candidate(
        case_id="isolated_single_candidate_alkene",
        smiles="C/C=C/C",
        rationale="minimal isolated stereo double bond; both sides have one carrier candidate",
    ),
    Candidate(
        case_id="coupled_single_candidate_diene",
        smiles="C/C=C/C=C/C",
        rationale="minimal coupled diene; begin sides have one carrier candidate",
    ),
    Candidate(
        case_id="coupled_two_candidate_branched_diene",
        smiles="CC/C(C)=C/C=C/C",
        rationale="small coupled diene where one begin side has two carrier candidates",
    ),
)


def branch_counts(smiles: str) -> tuple[int, Counter[str], dict[str, tuple[str, dict[str, object]]]]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"candidate did not parse: {smiles!r}")
    prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
    rows = _core._stereo_constraint_output_facts(prepared)
    counts: Counter[str] = Counter()
    examples: dict[str, tuple[str, dict[str, object]]] = {}
    for row in rows:
        for component in row["component_token_phase"]:
            inputs = component["token_flip_inference_inputs"]
            branch = str(inputs["inference_branch"])
            counts[branch] += 1
            examples.setdefault(branch, (str(row["smiles"]), dict(inputs)))
    return len(rows), counts, examples


def main() -> None:
    print(f"RDKit version: {rdBase.rdkitVersion}")
    for candidate in CANDIDATES:
        support_count, counts, examples = branch_counts(candidate.smiles)
        print()
        print(candidate.case_id)
        print(f"  smiles: {candidate.smiles}")
        print(f"  rationale: {candidate.rationale}")
        print(f"  support: {support_count}")
        print(f"  branch_counts: {dict(sorted(counts.items()))}")
        for branch, (example_smiles, inputs) in sorted(examples.items()):
            print(f"  example {branch}: {example_smiles}")
            print(
                "    required_input_facts: "
                f"{tuple(inputs['required_input_facts'])}"
            )
            print(
                "    phase/begin: "
                f"{inputs['effective_phase']}@{inputs['effective_begin_atom_idx']} "
                f"from {inputs['phase_source']}/{inputs['begin_atom_source']}"
            )
            print(
                "    begin side: "
                f"{inputs['begin_side_idx']} candidates="
                f"{inputs['begin_side_candidate_count']} selected="
                f"{inputs['selected_begin_neighbor_idx']} token="
                f"{inputs['selected_begin_token']}"
            )
            print(
                "    first emitted known: "
                f"{inputs['first_emitted_candidate_known']} adjustment="
                f"{inputs['rdkit_token_flip_adjustment']} inferred="
                f"{inputs['inferred_token_flip']}"
            )


if __name__ == "__main__":
    main()
