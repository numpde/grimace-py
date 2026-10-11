"""Measure bounded South Star 2 preparation, token-choice, and prefix-frontier costs for small C/O graphs."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import math
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any

from rdkit import Chem

from grimace._south_star2.preparation import PreparedMolecule, prepare_molecule


def _cases() -> list[tuple[str, str]]:
    cases = [(f"carbon_chain_{size}", "C" * size) for size in (3, 6, 9, 12)]
    cases.extend(
        (f"four_arm_symmetric_branch_arm_{length}", "C" + "".join(f"({ 'C' * length })" for _ in range(3)) + "C" * length)
        for length in (1, 2, 3)
    )
    cases.extend((f"disconnected_ethane_components_{count}", ".".join("CC" for _ in range(count))) for count in (2, 3, 4, 5))
    return cases


def _median_spread(values: list[int]) -> dict[str, float | int]:
    return {
        "median_ns": statistics.median(values),
        "min_ns": min(values),
        "max_ns": max(values),
        "samples": len(values),
    }


def _semantic_check(smiles: str, output: str) -> None:
    source = Chem.MolFromSmiles(smiles)
    parsed = Chem.MolFromSmiles(output)
    if source is None or parsed is None:
        raise RuntimeError(f"RDKit failed to parse source/output: {smiles!r} -> {output!r}")
    if (source.GetNumAtoms(), source.GetNumBonds()) != (parsed.GetNumAtoms(), parsed.GetNumBonds()):
        raise RuntimeError(f"output atom/bond counts differ: {smiles!r} -> {output!r}")
    if Chem.MolToSmiles(source, canonical=True) != Chem.MolToSmiles(parsed, canonical=True):
        raise RuntimeError(f"output graph differs semantically: {smiles!r} -> {output!r}")


def _walk(
    initial: Any,
    *,
    max_frontier: int,
    max_tokens: int,
    max_seconds: float,
    selection_repeats: int,
) -> dict[str, Any]:
    prefix = initial
    chosen_text: list[str] = []
    steps: list[dict[str, int]] = []
    started = time.perf_counter()
    while not prefix.is_accepted:
        before = prefix.frontier_size
        if before > max_frontier:
            return {"status": "frontier_cap", "frontier_size": before, "steps": steps}
        if len(chosen_text) >= max_tokens:
            return {"status": "token_cap", "frontier_size": before, "steps": steps}
        if time.perf_counter() - started > max_seconds:
            return {"status": "time_cap", "frontier_size": before, "steps": steps}

        choice_started = time.perf_counter_ns()
        choices = prefix.choices()
        choices_ns = time.perf_counter_ns() - choice_started
        if not choices:
            raise RuntimeError(f"nonaccepted prefix has no choices after {''.join(chosen_text)!r}")
        successor_frontiers = [choice[1].frontier_size for choice in choices]
        largest_successor = max(successor_frontiers)
        total_successor_states = sum(successor_frontiers)
        if largest_successor > max_frontier or total_successor_states > max_frontier:
            steps.append(
                {
                    "frontier_before": before,
                    "choice_text_count": len(choices),
                    "largest_generated_successor_frontier": largest_successor,
                    "total_states_across_generated_choice_frontiers": total_successor_states,
                    "choices_ns": choices_ns,
                    "cap_rejected_successor_batch": 1,
                }
            )
            return {
                "status": "frontier_cap",
                "frontier_size": before,
                "steps": steps,
            }

        # The bridge has no text-only select operation: choices() already
        # builds every successor. This measures consuming the first returned
        # token/successor pair, not a native reselection or a semantic merge.
        select_started = time.perf_counter_ns()
        selected = choices[0]
        select_ns = time.perf_counter_ns() - select_started
        lookup_started = time.perf_counter_ns()
        for _ in range(selection_repeats):
            selected = choices[0]
        repeated_lookup_ns = time.perf_counter_ns() - lookup_started
        token, successor = selected
        after = successor.frontier_size
        steps.append(
            {
                "frontier_before": before,
                "choice_text_count": len(choices),
                "chosen_successor_frontier": after,
                "largest_generated_successor_frontier": largest_successor,
                "total_states_across_generated_choice_frontiers": total_successor_states,
                "choices_ns": choices_ns,
                "first_successor_selection_ns": select_ns,
                "repeated_first_successor_lookup_ns_per_op": repeated_lookup_ns // selection_repeats,
            }
        )
        chosen_text.append(token)
        prefix = successor

    output = "".join(chosen_text)
    return {"status": "accepted", "frontier_size": prefix.frontier_size, "output": output, "steps": steps}


def _measure_case(
    name: str,
    smiles: str,
    *,
    repeats: int,
    max_frontier: int,
    max_tokens: int,
    max_seconds: float,
    selection_repeats: int,
) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid benchmark SMILES: {smiles!r}")
    prepared_facts = prepare_molecule(mol)
    core = importlib.import_module("grimace._core")
    raw_atoms = list(prepared_facts.atoms)
    raw_bonds = list(prepared_facts.bonds)

    results: dict[str, Any] = {
        "case": name,
        "source_smiles": smiles,
        "atom_count": len(raw_atoms),
        "bond_count": len(raw_bonds),
        "ordering_policy": "native AnyOrder; the adapter has no InputOrder option",
        "limits": {
            "max_frontier_states_before_choices": max_frontier,
            "max_tokens": max_tokens,
            "max_seconds_per_walk": max_seconds,
        },
        "regimes": {},
    }

    for regime in ("raw_any_order", "molecular_preparation"):
        timings: dict[str, list[int]] = {key: [] for key in ("rdkit_fact_extraction_ns", "adapter_prepare_ns", "native_input_materialization_ns", "native_writer_construction_ns", "initial_prefix_ns")}
        walks = []
        direct_locality = []
        for _ in range(repeats):
            if regime == "molecular_preparation":
                stage = time.perf_counter_ns()
                facts: PreparedMolecule = prepare_molecule(mol)
                timings["adapter_prepare_ns"].append(time.perf_counter_ns() - stage)
                atom_facts = facts.atoms
                bond_facts = facts.bonds
            else:
                # Extract only the raw graph facts, bypassing adapter validation
                # and PreparedMolecule construction while retaining the same
                # RDKit graph as the molecular-preparation lane.
                stage = time.perf_counter_ns()
                atom_facts = tuple("C" if atom.GetAtomicNum() == 6 else "O" for atom in mol.GetAtoms())
                bond_facts = tuple((bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), "") for bond in mol.GetBonds())
                timings["rdkit_fact_extraction_ns"].append(time.perf_counter_ns() - stage)

            stage = time.perf_counter_ns()
            atoms = list(atom_facts)
            bonds = list(bond_facts)
            timings["native_input_materialization_ns"].append(time.perf_counter_ns() - stage)
            stage = time.perf_counter_ns()
            writer = core._SouthStar2PreparedWriter(atoms, bonds)
            timings["native_writer_construction_ns"].append(time.perf_counter_ns() - stage)
            stage = time.perf_counter_ns()
            initial = writer.initial_prefix()
            timings["initial_prefix_ns"].append(time.perf_counter_ns() - stage)
            walk = _walk(
                initial,
                max_frontier=max_frontier,
                max_tokens=max_tokens,
                max_seconds=max_seconds,
                selection_repeats=selection_repeats,
            )
            if walk["status"] == "accepted":
                _semantic_check(smiles, walk["output"])
            walks.append(walk)
            if name.startswith("carbon_chain_") or name.startswith("disconnected_ethane_components_"):
                state_started = time.perf_counter_ns()
                direct = writer.initial_state()
                direct_initial_state_ns = time.perf_counter_ns() - state_started
                roots_started = time.perf_counter_ns()
                root_choices = direct.choices()
                root_choices_ns = time.perf_counter_ns() - roots_started
                if not root_choices or root_choices[0][0] != "C":
                    raise RuntimeError(f"expected the first atom root of {smiles!r} to emit C")
                active = root_choices[0][1]
                local_started = time.perf_counter_ns()
                local_choices = active.choices()
                local_choices_ns = time.perf_counter_ns() - local_started
                if not local_choices:
                    raise RuntimeError(f"expected an active first-ethane transition for {smiles!r}")
                direct_locality.append(
                    {
                        "initial_state_ns": direct_initial_state_ns,
                        "root_choice_count": len(root_choices),
                        "root_choices_ns": root_choices_ns,
                        "fixed_first_atom_neighbor_choice_count": len(local_choices),
                        "fixed_first_atom_neighbor_choices_ns": local_choices_ns,
                    }
                )

        measured_steps = [step for walk in walks for step in walk["steps"]]
        per_position = []
        for position in range(max((len(walk["steps"]) for walk in walks), default=0)):
            at_position = [walk["steps"][position] for walk in walks if len(walk["steps"]) > position]
            row: dict[str, Any] = {"token_position": position}
            for field in (
                "frontier_before",
                "choice_text_count",
                "chosen_successor_frontier",
                "largest_generated_successor_frontier",
                "total_states_across_generated_choice_frontiers",
                "choices_ns",
                "repeated_first_successor_lookup_ns_per_op",
            ):
                values = [step[field] for step in at_position if field in step]
                if values:
                    row[field] = _median_spread(values)
            per_position.append(row)
        first_selection_samples = [
            step["first_successor_selection_ns"]
            for step in measured_steps
            if "first_successor_selection_ns" in step
        ]
        repeated_selection_samples = [
            step["repeated_first_successor_lookup_ns_per_op"]
            for step in measured_steps
            if "repeated_first_successor_lookup_ns_per_op" in step
        ]
        results["regimes"][regime] = {
            "setup_timing": {key: _median_spread(values) for key, values in timings.items() if values},
            "per_choice_sample_summary": {
                "choices_ns": _median_spread([step["choices_ns"] for step in measured_steps]),
                "first_successor_selection_ns": _median_spread(first_selection_samples)
                if first_selection_samples
                else None,
                "first_successor_selection_sample_count": len(first_selection_samples),
                "repeated_first_successor_lookup_ns_per_op": _median_spread(repeated_selection_samples)
                if repeated_selection_samples
                else None,
                "repeated_first_successor_lookup_sample_count": len(repeated_selection_samples),
                "step_samples": len(measured_steps),
            },
            "per_token_position": per_position,
            "direct_locality_probe": {
                "meaning": "one concrete Walker after choosing the first atom root, then choices() for its first atom-to-neighbor transition; later chain atoms test connected remote extension and later disconnected components are inactive spectators",
                "timings": {
                    key: _median_spread([sample[key] for sample in direct_locality])
                    for key in (
                        "initial_state_ns",
                        "root_choices_ns",
                        "fixed_first_atom_neighbor_choices_ns",
                    )
                }
                if direct_locality
                else None,
                "root_choice_count": _median_spread([sample["root_choice_count"] for sample in direct_locality]) if direct_locality else None,
                "fixed_first_atom_neighbor_choice_count": _median_spread([sample["fixed_first_atom_neighbor_choice_count"] for sample in direct_locality]) if direct_locality else None,
                "samples": direct_locality,
            },
            "walks": walks,
        }
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-frontier", type=int, default=2500)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-seconds", type=float, default=3.0)
    parser.add_argument("--selection-repeats", type=int, default=1000)
    args = parser.parse_args()
    if args.repeats < 1 or args.selection_repeats < 1:
        parser.error("repeats and selection-repeats must be positive")
    if not math.isfinite(args.max_seconds):
        parser.error("--max-seconds must be finite")

    core = importlib.import_module("grimace._core")
    core_path = Path(core.__file__).resolve()
    document = {
        "schema": "south_star2_prefix_frontier_measurement_v1",
        "python": sys.version,
        "rdkit_version": Chem.rdBase.rdkitVersion,
        "core_extension_path": str(core_path),
        "core_extension_size_bytes": core_path.stat().st_size,
        "core_extension_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
        "measurement_scope": "raw neutral C/O facts over private native bridge versus same facts through real RDKit molecular preparation",
        "native_policy": "AnyOrder; no InputOrder bridge policy is measured",
        "frontier_meaning": "retained physical native states; not support multiplicity and no equivalent-state merging is asserted",
        "selection_meaning": "choices() builds grouped successors; selected pair retrieval is Python-side consumption and does not re-run native transition logic",
        "cap_meaning": "the retained chosen path and post-call generated successor batch stop at max-frontier; one choices() call constructs successors before the cap can be checked and cannot be interrupted; time is checked between native calls",
        "parameters": vars(args) | {"output": str(args.output)},
        "cases": [
            _measure_case(
                name,
                smiles,
                repeats=args.repeats,
                max_frontier=args.max_frontier,
                max_tokens=args.max_tokens,
                max_seconds=args.max_seconds,
                selection_repeats=args.selection_repeats,
            )
            for name, smiles in _cases()
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "core_extension_path": document["core_extension_path"], "cases": len(document["cases"])}, indent=2))


if __name__ == "__main__":
    main()
