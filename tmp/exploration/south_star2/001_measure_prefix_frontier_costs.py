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
import tracemalloc
from pathlib import Path
from typing import Any

from rdkit import Chem

from grimace._south_star2.preparation import PreparedMolecule, VirtualHydrogen, prepare_molecule


def _cases() -> list[tuple[str, str]]:
    cases = [(f"carbon_chain_{size}", "C" * size) for size in (3, 6, 9, 12)]
    cases.extend(
        (f"four_arm_symmetric_branch_arm_{length}", "C" + "".join(f"({ 'C' * length })" for _ in range(3)) + "C" * length)
        for length in (1, 2, 3)
    )
    cases.extend((f"disconnected_ethane_components_{count}", ".".join("CC" for _ in range(count))) for count in (2, 3, 4, 5))
    cases.extend(
        (
            ("stereo_tetra_ccw_tree", "C[C@H](O)CC"),
            ("stereo_e_butene_tree", "C/C=C/C"),
            ("stereo_z_butene_tree", "C/C=C\\C"),
            ("stereo_mixed_alkene_tetra_tree", "C/C=C/[C@H](O)C"),
        )
    )
    return cases


def _fixed_component_locality_cases() -> list[tuple[str, str]]:
    return [
        (f"fixed_ccc_ethane_spectators_{count}", "CCC" + ".CC" * count)
        for count in range(5)
    ]


def _median_spread(values: list[int]) -> dict[str, float | int]:
    return {
        "median_ns": statistics.median(values),
        "min_ns": min(values),
        "max_ns": max(values),
        "samples": len(values),
    }


def _count_spread(values: list[int]) -> dict[str, float | int]:
    return {
        "median_count": statistics.median(values),
        "min_count": min(values),
        "max_count": max(values),
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
        raise RuntimeError(f"output graph or configured stereo differs semantically: {smiles!r} -> {output!r}")
    for mapping in parsed.GetSubstructMatches(source, useChirality=True):
        if any(
            source.GetAtomWithIdx(index).GetTotalNumHs()
            != parsed.GetAtomWithIdx(mapping[index]).GetTotalNumHs()
            for index in range(source.GetNumAtoms())
        ):
            continue
        if any(
            (target := parsed.GetBondBetweenAtoms(
                mapping[bond.GetBeginAtomIdx()], mapping[bond.GetEndAtomIdx()]
            )) is None or target.GetBondType() != bond.GetBondType()
            for bond in source.GetBonds()
        ):
            continue
        break
    else:
        raise RuntimeError(f"no graph/stereo mapping preserves per-atom hydrogen counts: {smiles!r} -> {output!r}")


def _directional_wire(facts: PreparedMolecule) -> tuple[tuple[int, int, int, int, int, bool], ...]:
    return tuple(
        (
            relation.bond,
            relation.left_endpoint,
            relation.left_carrier,
            relation.right_endpoint,
            relation.right_carrier,
            relation.side_phase_xor,
        )
        for relation in facts.directional
    )


def _native_atom_facts(facts: PreparedMolecule) -> list[object]:
    atoms: list[object] = []
    for atom in facts.atoms:
        if type(atom) is str:
            atoms.append(atom)
        else:
            reference_order = tuple(
                "virtual_hydrogen" if ligand is VirtualHydrogen.INSTANCE else ligand
                for ligand in atom.reference_order
            )
            atoms.append(("tetrahedral", reference_order, atom.text_by_parity))
    return atoms


def _walk(
    initial: Any,
    *,
    target_output: str,
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

        matching = [choice for choice in choices if target_output.startswith("".join(chosen_text) + choice[0])]
        if len(matching) != 1:
            raise RuntimeError(
                f"RDKit witness {target_output!r} has {len(matching)} matching choices after {''.join(chosen_text)!r}"
            )
        # The bridge has no text-only select operation: choices() already
        # builds every successor. This consumes the choice matching one actual
        # pinned-regime RDKit output witness.
        select_started = time.perf_counter_ns()
        selected = matching[0]
        select_ns = time.perf_counter_ns() - select_started
        lookup_started = time.perf_counter_ns()
        for _ in range(selection_repeats):
            selected = matching[0]
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
    if output != target_output:
        raise RuntimeError(f"native accepted path {output!r} differs from RDKit witness {target_output!r}")
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
    writer_regime: str,
) -> dict[str, Any]:
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"invalid benchmark SMILES: {smiles!r}")
    prepared_facts = prepare_molecule(mol)
    core = importlib.import_module("grimace._core")
    target_output = Chem.MolToRandomSmilesVect(mol, 1, 1, True, False, False, False)[0]
    has_stereo_facts = bool(prepared_facts.directional) or any(
        type(atom) is not str for atom in prepared_facts.atoms
    )
    raw_atoms = _native_atom_facts(prepared_facts)
    raw_bonds = list(prepared_facts.bonds)

    results: dict[str, Any] = {
        "case": name,
        "source_smiles": smiles,
        "atom_count": len(raw_atoms),
        "bond_count": len(raw_bonds),
        "writer_regime": writer_regime,
        "rdkit_witness_seed": 1,
        "rdkit_witness_output": target_output,
        "limits": {
            "max_frontier_states_before_choices": max_frontier,
            "max_tokens": max_tokens,
            "max_seconds_per_walk": max_seconds,
        },
        "regimes": {},
    }

    regimes = ("molecular_preparation",) if has_stereo_facts else ("raw_any_order", "molecular_preparation")
    for regime in regimes:
        timings: dict[str, list[int]] = {key: [] for key in ("rdkit_fact_extraction_ns", "adapter_prepare_ns", "native_input_materialization_ns", "native_writer_construction_ns", "initial_prefix_ns")}
        walks = []
        direct_locality = []
        for _ in range(repeats):
            if regime == "molecular_preparation":
                stage = time.perf_counter_ns()
                facts: PreparedMolecule = prepare_molecule(mol)
                timings["adapter_prepare_ns"].append(time.perf_counter_ns() - stage)
                atom_facts = _native_atom_facts(facts)
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
            directional = _directional_wire(facts) if regime == "molecular_preparation" else ()
            timings["native_input_materialization_ns"].append(time.perf_counter_ns() - stage)
            stage = time.perf_counter_ns()
            writer = core._SouthStar2PreparedWriter(atoms, bonds, writer_regime, directional)
            timings["native_writer_construction_ns"].append(time.perf_counter_ns() - stage)
            stage = time.perf_counter_ns()
            initial = writer.initial_prefix()
            timings["initial_prefix_ns"].append(time.perf_counter_ns() - stage)
            walk = _walk(
                initial,
                target_output=target_output,
                max_frontier=max_frontier,
                max_tokens=max_tokens,
                max_seconds=max_seconds,
                selection_repeats=selection_repeats,
            )
            if walk["status"] == "accepted":
                _semantic_check(smiles, walk["output"])
            walks.append(walk)
            if name.startswith("disconnected_ethane_components_"):
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
                    row[field] = (
                        _median_spread(values)
                        if field.endswith("_ns") or field.endswith("_ns_per_op")
                        else _count_spread(values)
                    )
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
            "component_boundary_probe": {
                "meaning": "one concrete Walker after choosing the first atom root of the first ethane component, then choices() for the atom-to-neighbor transition that completes that component; this cost may include normalization and root selection for the next component",
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
                "root_choice_count": _count_spread([sample["root_choice_count"] for sample in direct_locality]) if direct_locality else None,
                "fixed_first_atom_neighbor_choice_count": _count_spread([sample["fixed_first_atom_neighbor_choice_count"] for sample in direct_locality]) if direct_locality else None,
                "samples": direct_locality,
            },
            "walks": walks,
        }
    return results


def _measure_fixed_component_locality(
    *,
    repeats: int,
    writer_regime: str,
    case_name: str | None,
) -> list[dict[str, Any]]:
    cases = [
        (name, smiles)
        for name, smiles in _fixed_component_locality_cases()
        if case_name is None or name == case_name
    ]
    samples: dict[str, list[dict[str, int]]] = {name: [] for name, _smiles in cases}
    # Interleave spectator sizes within each repeat to limit order effects.
    for _ in range(repeats):
        for name, smiles in cases:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"invalid locality probe SMILES: {smiles!r}")
            started = time.perf_counter_ns()
            facts: PreparedMolecule = prepare_molecule(mol)
            prepare_ns = time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            atoms = _native_atom_facts(facts)
            bonds = list(facts.bonds)
            directional = _directional_wire(facts)
            materialize_ns = time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            writer = importlib.import_module("grimace._core")._SouthStar2PreparedWriter(
                atoms, bonds, writer_regime, directional
            )
            construct_ns = time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            initial = writer.initial_state()
            initial_state_ns = time.perf_counter_ns() - started
            started = time.perf_counter_ns()
            root_choices = initial.choices()
            root_choices_ns = time.perf_counter_ns() - started
            if not root_choices or root_choices[0][0] != "C":
                raise RuntimeError(f"expected input-order root zero to emit C for {smiles!r}")
            started = time.perf_counter_ns()
            root_atom_choices = root_choices[0][1].choices()
            root_atom_choices_ns = time.perf_counter_ns() - started
            if not root_atom_choices or root_atom_choices[0][0] != "C":
                raise RuntimeError(f"expected root-zero neighbor to emit C for {smiles!r}")
            active_component_state = root_atom_choices[0][1]
            started = time.perf_counter_ns()
            component_completion_choices = active_component_state.choices()
            component_completion_choices_ns = time.perf_counter_ns() - started
            if not component_completion_choices or any(
                token != "C" for token, _successor in component_completion_choices
            ):
                raise RuntimeError(f"expected atom 1-to-2 choices inside CCC for {smiles!r}")
            completion_successor_labels = [
                [token for token, _successor in successor.choices()]
                for _token, successor in component_completion_choices
            ]
            spectator_count = int(name.rsplit("_", 1)[1])
            successors_match_component_boundary = (
                all(not labels for labels in completion_successor_labels)
                if not spectator_count
                else all(labels and all(token == "." for token in labels) for labels in completion_successor_labels)
            )
            if not successors_match_component_boundary:
                raise RuntimeError(
                    f"atom 1-to-2 should complete CCC before fragment handling for {smiles!r}: "
                    f"{completion_successor_labels!r}"
                )
            samples[name].append(
                {
                    "prepare_ns": prepare_ns,
                    "native_input_materialization_ns": materialize_ns,
                    "native_writer_construction_ns": construct_ns,
                    "initial_state_ns": initial_state_ns,
                    "root_choice_ns": root_choices_ns,
                    "root_choice_count": len(root_choices),
                    "root_zero_neighbor_choice_ns": root_atom_choices_ns,
                    "root_zero_neighbor_choice_count": len(root_atom_choices),
                    "component_completion_choice_ns": component_completion_choices_ns,
                    "component_completion_choice_count": len(component_completion_choices),
                    "component_completion_generated_successor_count": len(component_completion_choices),
                }
            )

    results = []
    for name, smiles in cases:
        records = samples[name]
        fields = tuple(records[0])
        results.append(
            {
                "case": name,
                "source_smiles": smiles,
                "first_component": "CCC",
                "ethane_spectator_count": int(name.rsplit("_", 1)[1]),
                "meaning": "root_zero_neighbor_choice_ns times 0-to-1 and its returned successor remains inside CCC; the following component_completion_choice_ns times 1-to-2 and completes CCC (then EOS for CCC or fragment-root choices for CCC.CC plus spectators).",
                "timings": {
                    field: _median_spread([sample[field] for sample in records]) if field.endswith("_ns") else _count_spread([sample[field] for sample in records])
                    for field in fields
                    if field.endswith("_ns")
                },
                "counts": {
                    field: _count_spread([sample[field] for sample in records])
                    for field in fields
                    if field.endswith("_count")
                },
                "samples": records,
            }
        )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--writer-regime",
        choices=(
            "generic_any_order",
            "generic_input_order",
            "rdkit_random_noncanonical",
        ),
        required=True,
    )
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--max-frontier", type=int, default=2500)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--max-seconds", type=float, default=3.0)
    parser.add_argument("--selection-repeats", type=int, default=1000)
    parser.add_argument("--memory-pass", action="store_true")
    parser.add_argument("--case", help="measure only the named main case (useful for one-process-per-case RSS runs)")
    parser.add_argument(
        "--locality-case",
        choices=[name for name, _smiles in _fixed_component_locality_cases()],
        help="measure only the named fixed-component locality case; with no --case, omits all main cases",
    )
    args = parser.parse_args()
    if args.repeats < 1 or args.selection_repeats < 1:
        parser.error("repeats and selection-repeats must be positive")
    if not math.isfinite(args.max_seconds):
        parser.error("--max-seconds must be finite")
    if args.writer_regime != "rdkit_random_noncanonical":
        parser.error("this probe measures only the pinned RDKit writer regime")
    if args.memory_pass and args.case is None:
        parser.error("--memory-pass requires --case to isolate one main case")
    if args.memory_pass and args.locality_case is not None:
        parser.error("--memory-pass measures one main case and cannot include the locality lane")
    if Chem.rdBase.rdkitVersion != "2026.03.1":
        parser.error(f"expected pinned RDKit 2026.03.1, found {Chem.rdBase.rdkitVersion}")

    core = importlib.import_module("grimace._core")
    core_path = Path(core.__file__).resolve()
    selected_cases = [(name, smiles) for name, smiles in _cases() if args.case is None or name == args.case]
    if args.locality_case is not None and args.case is None:
        selected_cases = []
    if args.case is not None and not selected_cases:
        parser.error(f"unknown case {args.case!r}")
    measured_cases = []
    for name, smiles in selected_cases:
        if args.memory_pass:
            tracemalloc.start()
        case_result = _measure_case(
            name,
            smiles,
            repeats=args.repeats,
            max_frontier=args.max_frontier,
            max_tokens=args.max_tokens,
            max_seconds=args.max_seconds,
            selection_repeats=args.selection_repeats,
            writer_regime=args.writer_regime,
        )
        if args.memory_pass:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            for regime_result in case_result["regimes"].values():
                regime_result["python_heap_memory"] = {
                    "method": "tracemalloc peak for this one case in a separate memory-pass run; Python allocator only, excludes native Rust allocations",
                    "current_bytes_after_case": current,
                    "peak_bytes": peak,
                }
        measured_cases.append(case_result)
    document = {
        "schema": "south_star2_prefix_frontier_measurement_v2",
        "python": sys.version,
        "rdkit_version": Chem.rdBase.rdkitVersion,
        "core_extension_path": str(core_path),
        "core_extension_size_bytes": core_path.stat().st_size,
        "core_extension_sha256": hashlib.sha256(core_path.read_bytes()).hexdigest(),
        "measurement_scope": "nonstereo C/O cases compare raw facts with real preparation; tetrahedral, E/Z, and mixed cases measure only preparation; fixed CCC lane separates the verified 0-to-1 interior choice from 1-to-2 component completion while varying disconnected ethane spectators",
        "native_writer_regime": args.writer_regime,
        "writer_parameters": {
            "rootedAtAtom": -1,
            "canonical": False,
            "doRandom": True,
            "doIsomericSmiles": True,
            "doKekule": False,
            "allBondsExplicit": False,
            "allHsExplicit": False,
        },
        "memory_pass": args.memory_pass,
        "process_memory_method": "run each --case in a fresh process under /usr/bin/time -v; whole-process peak RSS includes interpreter/import baseline, Python objects, and native allocations and cannot isolate Rust memory",
        "source_membership_note": "each path follows one seed-1 MolToRandomSmilesVect output from RDKit under the pinned writer regime; this is an actual source output witness, not support completeness",
        "frontier_meaning": "retained physical native states; not support multiplicity and no equivalent-state merging is asserted",
        "selection_meaning": "choices() builds grouped successors; selected pair retrieval is Python-side consumption and does not re-run native transition logic",
        "cap_meaning": "the retained chosen path and post-call generated successor batch stop at max-frontier; one choices() call constructs successors before the cap can be checked and cannot be interrupted; time is checked between native calls",
        "parameters": vars(args) | {"output": str(args.output)},
        "cases": measured_cases,
        "fixed_component_interior_locality": _measure_fixed_component_locality(
            repeats=args.repeats,
            writer_regime=args.writer_regime,
            case_name=args.locality_case if args.locality_case is not None else (None if args.case is None else ""),
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(args.output), "core_extension_path": document["core_extension_path"], "cases": len(document["cases"])}, indent=2))


if __name__ == "__main__":
    main()
