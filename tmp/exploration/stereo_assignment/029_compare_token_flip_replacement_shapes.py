from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass

from rdkit import rdBase

from grimace import _core, _runtime
from tests.helpers.mols import parse_smiles
from tests.helpers.stereo_constraint_model import (
    load_pinned_stereo_constraint_model_cases,
)


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
class ComponentObservation:
    case_id: str
    branch: str
    inferred_token_flip: str
    forced_model_token_flip: str
    required_input_facts: tuple[str, ...]
    phase_source: str
    begin_atom_source: str
    selected_begin_token: str | None
    begin_side_candidate_count: int
    first_emitted_candidate_known: bool
    rdkit_token_flip_adjustment: bool
    before_count: int
    after_count: int
    model_explains_inferred: bool


def component_observations() -> tuple[ComponentObservation, ...]:
    out: list[ComponentObservation] = []
    for case in load_pinned_stereo_constraint_model_cases(rdBase.rdkitVersion):
        mol = parse_smiles(case.smiles)
        prepared = _runtime.prepare_smiles_graph(mol, flags=SUPPORTED_STEREO_FLAGS)
        for row in _core._stereo_constraint_output_facts(prepared):
            for component in row["component_token_phase"]:
                inputs = component["token_flip_inference_inputs"]
                inferred = component["inferred_token_flip"]
                forced = component["forced_model_token_flip"]
                if inferred is None:
                    continue
                out.append(
                    ComponentObservation(
                        case_id=case.case_id,
                        branch=str(inputs["inference_branch"]),
                        inferred_token_flip=str(inferred),
                        forced_model_token_flip=str(forced),
                        required_input_facts=tuple(inputs["required_input_facts"]),
                        phase_source=str(inputs["phase_source"]),
                        begin_atom_source=str(inputs["begin_atom_source"]),
                        selected_begin_token=inputs["selected_begin_token"],
                        begin_side_candidate_count=int(
                            inputs["begin_side_candidate_count"]
                        ),
                        first_emitted_candidate_known=bool(
                            inputs["first_emitted_candidate_known"]
                        ),
                        rdkit_token_flip_adjustment=bool(
                            inputs["rdkit_token_flip_adjustment"]
                        ),
                        before_count=int(
                            component["token_phase_assignment_count_before_token"]
                        ),
                        after_count=int(
                            component["token_phase_assignment_count_after_token"]
                        ),
                        model_explains_inferred=bool(
                            component["token_phase_dimension_explains_inferred_flip"]
                        ),
                    )
                )
    return tuple(out)


def print_counter(title: str, counter: Counter[object]) -> None:
    print(f"\n{title}")
    for key, count in counter.most_common():
        print(f"  {key!r}: {count}")


def summarize_adapter_shape(observations: tuple[ComponentObservation, ...]) -> None:
    """Alternative A: final token-flip fact adapter."""

    print("\nAlternative A: token-flip fact adapter")
    print(f"  emitted final token-flip facts: {len(observations)}")
    print(
        "  model forced flip matches adapter fact: "
        f"{sum(obs.inferred_token_flip == obs.forced_model_token_flip for obs in observations)}"
        f"/{len(observations)}"
    )
    print(
        "  model row reduction is exactly 2x: "
        f"{sum(obs.before_count == 2 * obs.after_count for obs in observations)}"
        f"/{len(observations)}"
    )


def summarize_observation_fact_shape(
    observations: tuple[ComponentObservation, ...],
) -> None:
    """Alternative B: typed observations that imply token-phase filtering."""

    fact_shapes = Counter(obs.required_input_facts for obs in observations)
    source_shapes = Counter(
        (obs.phase_source, obs.begin_atom_source) for obs in observations
    )
    token_shapes = Counter(
        (
            obs.branch,
            obs.selected_begin_token,
            obs.begin_side_candidate_count,
            obs.first_emitted_candidate_known,
            obs.rdkit_token_flip_adjustment,
        )
        for obs in observations
    )

    print("\nAlternative B: typed token-observation facts")
    print(
        "  observation-filtered model explains inferred flip: "
        f"{sum(obs.model_explains_inferred for obs in observations)}"
        f"/{len(observations)}"
    )
    print_counter("  required fact shapes", fact_shapes)
    print_counter("  phase/begin-atom source shapes", source_shapes)
    print_counter("  branch/token/adjustment shapes", token_shapes)


def summarize_expanded_row_shape(
    observations: tuple[ComponentObservation, ...],
) -> None:
    """Alternative C: estimate extra axes a richer startup table would need."""

    by_case: dict[str, list[ComponentObservation]] = defaultdict(list)
    for observation in observations:
        by_case[observation.case_id].append(observation)

    print("\nAlternative C: expanded token-phase rows")
    for case_id, case_observations in sorted(by_case.items()):
        unique_branch_axes = {
            (
                obs.branch,
                obs.phase_source,
                obs.begin_atom_source,
                obs.begin_side_candidate_count,
                obs.first_emitted_candidate_known,
            )
            for obs in case_observations
        }
        print(
            f"  {case_id}: {len(case_observations)} output-component rows, "
            f"{len(unique_branch_axes)} observed row-axis shapes"
        )

    max_before = max((obs.before_count for obs in observations), default=0)
    max_after = max((obs.after_count for obs in observations), default=0)
    print(f"  max token-phase rows before observation: {max_before}")
    print(f"  max token-phase rows after observation: {max_after}")
    print(
        "  implication: current fixtures do not require expanding startup rows "
        "to explain token flips; observation facts already force the existing "
        "token-phase dimension."
    )


def main() -> None:
    observations = component_observations()
    print(f"RDKit version: {rdBase.rdkitVersion}")
    print(f"component observations with inferred token flips: {len(observations)}")
    print_counter("branches", Counter(obs.branch for obs in observations))
    print_counter("cases", Counter(obs.case_id for obs in observations))
    summarize_adapter_shape(observations)
    summarize_observation_fact_shape(observations)
    summarize_expanded_row_shape(observations)


if __name__ == "__main__":
    main()
