#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import asdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rdkit import Chem

from grimace._south_star.enum_s import mol_to_smiles_enum_s_graph_native
from tests.helpers.south_star_exact_support import (
    load_south_star_exact_first_domain_cases,
    load_south_star_expanded_support_cases,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_ROOT / "notes" / "perf_reports" / "south_star_enum_s_v1.json"


@dataclass(frozen=True, slots=True)
class BenchmarkCase:
    case_id: str
    fixture_family: str
    domain_label: str
    source_smiles: str
    expected_output_count: int


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record South Star semantic EnumS benchmark timings."
    )
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.repeats <= 0:
        raise ValueError("--repeats must be positive")
    if args.warmups < 0:
        raise ValueError("--warmups must be non-negative")

    cases = _benchmark_cases()
    for _ in range(args.warmups):
        for case in cases:
            _run_case(case)

    rows = [_benchmark_case(case, repeats=args.repeats) for case in cases]
    report = {
        "schema_version": 1,
        "kind": "south_star_semantic_enum_s_benchmark",
        "command": _command_line(args),
        "metadata": _metadata(),
        "repeats": args.repeats,
        "warmups": args.warmups,
        "policy_set": {
            "annotation_policy": "maximal_eligible_carrier",
            "fragment_order_policy": "all_fragment_orders",
            "output_order_policy": "first_occurrence_deduplication",
        },
        "scope_note": (
            "Measures the private South Star semantic enumerator "
            "mol_to_smiles_enum_s_graph_native on pinned semantic fixtures. "
            "This is not an RDKit writer-parity benchmark."
        ),
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")


def _benchmark_cases() -> tuple[BenchmarkCase, ...]:
    return tuple(
        BenchmarkCase(
            case_id=case.case_id,
            fixture_family="exact_first_domain",
            domain_label="first_domain_directional_bond_stereo",
            source_smiles=case.source_smiles,
            expected_output_count=len(case.expected_support),
        )
        for case in load_south_star_exact_first_domain_cases()
    ) + tuple(
        BenchmarkCase(
            case_id=case.case_id,
            fixture_family="expanded_support",
            domain_label=case.feature_area,
            source_smiles=case.source_smiles,
            expected_output_count=len(case.expected_support),
        )
        for case in load_south_star_expanded_support_cases()
    )


def _benchmark_case(case: BenchmarkCase, *, repeats: int) -> dict[str, Any]:
    timings = []
    output_count = 0
    for _ in range(repeats):
        start = time.perf_counter()
        output_count = _run_case(case)
        timings.append(time.perf_counter() - start)
    if output_count != case.expected_output_count:
        raise AssertionError(
            f"{case.case_id} produced {output_count} outputs, expected "
            f"{case.expected_output_count}"
        )
    return {
        **asdict(case),
        "output_count": output_count,
        "time_mean_seconds": statistics.fmean(timings),
        "time_stdev_seconds": statistics.stdev(timings) if len(timings) > 1 else 0.0,
        "time_min_seconds": min(timings),
        "time_max_seconds": max(timings),
    }


def _run_case(case: BenchmarkCase) -> int:
    result = mol_to_smiles_enum_s_graph_native(
        case.source_smiles,
        case_id=case.case_id,
    )
    return len(result.outputs)


def _metadata() -> dict[str, str]:
    return {
        "git_commit": _git_commit(),
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "machine": platform.machine(),
        "rdkit_version": Chem.rdBase.rdkitVersion,
    }


def _git_commit() -> str:
    return subprocess.check_output(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT,
        text=True,
    ).strip()


def _command_line(args: argparse.Namespace) -> str:
    return (
        "PYTHONPATH=python:. python3 scripts/benchmark_south_star_enum_s.py "
        f"--repeats {args.repeats} --warmups {args.warmups} "
        f"--output {args.output}"
    )


if __name__ == "__main__":
    main()
