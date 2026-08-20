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
    Candidate(
        case_id="coupled_adjacent_two_candidate_token_adjustment",
        smiles="C/C=C/C(C)=C/C",
        rationale="small coupled diene where the RDKit token adjustment uses an adjacent two-candidate side",
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


def adjustment_reason_counts(
    smiles: str,
) -> tuple[Counter[str], dict[str, tuple[str, dict[str, object]]]]:
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
            facts = {fact["fact"]: fact for fact in inputs["input_observation_facts"]}
            adjustment = facts["rdkit_token_flip_adjustment"]
            if adjustment["value"]:
                counts["value_true"] += 1
                examples.setdefault("value_true", (str(row["smiles"]), dict(inputs)))
            if adjustment["root_begin_side_adjustment"]:
                counts["root_begin_side_orientation"] += 1
                examples.setdefault(
                    "root_begin_side_orientation", (str(row["smiles"]), dict(inputs))
                )
            if adjustment["adjacent_two_candidate_adjustment"]:
                counts["adjacent_two_candidate_first_emitted"] += 1
                examples.setdefault(
                    "adjacent_two_candidate_first_emitted",
                    (str(row["smiles"]), dict(inputs)),
                )
    return counts, examples


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
        adjustment_counts, adjustment_examples = adjustment_reason_counts(candidate.smiles)
        print(f"  adjustment_reason_counts: {dict(sorted(adjustment_counts.items()))}")
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
        for reason, (example_smiles, inputs) in sorted(adjustment_examples.items()):
            adjustment = {
                fact["fact"]: fact for fact in inputs["input_observation_facts"]
            }["rdkit_token_flip_adjustment"]
            print(f"  adjustment {reason}: {example_smiles}")
            print(
                "    begin/root/orientation: "
                f"{inputs['effective_begin_atom_idx']} "
                f"root={adjustment['begin_atom_is_root']} "
                f"orientation={adjustment['begin_side_orientation']}"
            )
            print(
                "    adjacent: "
                f"side={adjustment['adjacent_two_candidate_side_idx']} "
                f"selected_root={adjustment['selected_neighbor_is_root']} "
                f"first={adjustment['adjacent_first_emitted_candidate_idx']} "
                f"first_not_begin="
                f"{adjustment['adjacent_first_emitted_is_not_begin']}"
            )


if __name__ == "__main__":
    main()
