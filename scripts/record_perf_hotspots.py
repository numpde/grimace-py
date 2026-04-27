#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import textwrap

from tests.perf._history import (
    REPORT_DIR,
    append_history_record,
    current_run_metadata,
    parse_perf_report_top_symbols,
    sanitize_label,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Record an internal perf-hotspot history entry for Grimace benchmarks.",
    )
    parser.add_argument("--label", required=True, help="Short internal label for this profile run")
    parser.add_argument("--smiles", required=True, help="SMILES to benchmark")
    parser.add_argument(
        "--mode",
        choices=("enum", "decoder", "determinized-decoder"),
        default="enum",
        help="Which Grimace API surface to profile",
    )
    parser.add_argument(
        "--rooted-at-atom",
        type=int,
        default=-1,
        help="Root index to use for the selected API (default: -1)",
    )
    parser.add_argument(
        "--isomeric-smiles",
        action="store_true",
        help="Profile isomeric enumeration instead of non-isomeric",
    )
    parser.add_argument(
        "--all-roots",
        action="store_true",
        help="For decoder modes, exhaust all roots explicitly instead of using rootedAtAtom directly",
    )
    parser.add_argument(
        "--loops",
        type=int,
        default=2,
        help="How many benchmark runs to perform inside the profiled process",
    )
    parser.add_argument(
        "--warmup-loops",
        type=int,
        default=2,
        help="How many warmup runs to perform before profiling",
    )
    parser.add_argument(
        "--percent-limit",
        type=float,
        default=1.5,
        help="perf report --percent-limit threshold",
    )
    parser.add_argument(
        "--top",
        type=int,
        default=20,
        help="How many top hotspot symbols to keep in the JSONL record",
    )
    parser.add_argument(
        "--keep-report",
        action="store_true",
        help="Persist the full perf report text under notes/perf_reports/",
    )
    return parser.parse_args()


def build_payload_script(args: argparse.Namespace) -> str:
    if args.mode == "enum":
        run_body = textwrap.dedent(
            f"""
            set(
                grimace.MolToSmilesEnum(
                    mol,
                    rootedAtAtom={args.rooted_at_atom},
                    isomericSmiles={args.isomeric_smiles},
                    canonical=False,
                    doRandom=True,
                )
            )
            """
        ).strip()
    else:
        decoder_cls = (
            "grimace.MolToSmilesDecoder"
            if args.mode == "decoder"
            else "grimace.MolToSmilesDeterminizedDecoder"
        )
        if args.all_roots:
            run_body = textwrap.dedent(
                f"""
                outputs = set()
                for root_idx in range(mol.GetNumAtoms()):
                    stack = [{decoder_cls}(
                        mol,
                        rootedAtAtom=root_idx,
                        isomericSmiles={args.isomeric_smiles},
                        canonical=False,
                        doRandom=True,
                    )]
                    while stack:
                        state = stack.pop()
                        if state.is_terminal:
                            outputs.add(state.prefix)
                            continue
                        stack.extend(choice.next_state for choice in reversed(state.next_choices))
                """
            ).strip()
        else:
            run_body = textwrap.dedent(
                f"""
                outputs = set()
                stack = [{decoder_cls}(
                    mol,
                    rootedAtAtom={args.rooted_at_atom},
                    isomericSmiles={args.isomeric_smiles},
                    canonical=False,
                    doRandom=True,
                )]
                while stack:
                    state = stack.pop()
                    if state.is_terminal:
                        outputs.add(state.prefix)
                        continue
                    stack.extend(choice.next_state for choice in reversed(state.next_choices))
                """
            ).strip()

    payload_lines = [
        "from rdkit import Chem",
        "import grimace",
        "",
        f"mol = Chem.MolFromSmiles({args.smiles!r})",
        f"for _ in range({args.warmup_loops}):",
        textwrap.indent(run_body, "    "),
        f"for _ in range({args.loops}):",
        textwrap.indent(run_body, "    "),
        "",
    ]
    return "\n".join(payload_lines)


def main() -> int:
    args = parse_args()
    label = sanitize_label(args.label)

    with tempfile.TemporaryDirectory(prefix="grimace-perf-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        payload_path = temp_dir_path / "payload.py"
        data_path = temp_dir_path / "profile.data"
        payload_path.write_text(build_payload_script(args), encoding="utf-8")

        env = os.environ.copy()

        subprocess.run(
            [
                "perf",
                "record",
                "-g",
                "-o",
                str(data_path),
                "--",
                sys.executable,
                str(payload_path),
            ],
            cwd=REPO_ROOT,
            env=env,
            check=True,
        )

        report = subprocess.check_output(
            [
                "perf",
                "report",
                "--stdio",
                "-i",
                str(data_path),
                "--percent-limit",
                str(args.percent_limit),
            ],
            cwd=REPO_ROOT,
            env=env,
            text=True,
        )

    hotspots = parse_perf_report_top_symbols(report, limit=args.top)
    report_path: str | None = None
    if args.keep_report:
        REPORT_DIR.mkdir(parents=True, exist_ok=True)
        metadata = current_run_metadata(change_label=args.label)
        report_name = (
            f"{metadata['recorded_at_utc'].replace(':', '').replace('+00:00', 'Z')}"
            f"_{metadata['git_commit']}_{label}.perf.txt"
        )
        report_file = REPORT_DIR / report_name
        report_file.write_text(report, encoding="utf-8")
        report_path = str(report_file.relative_to(REPO_ROOT))
    else:
        metadata = current_run_metadata(change_label=args.label)

    append_history_record(
        {
            "kind": "perf_hotspots",
            **metadata,
            "benchmark": {
                "enum": "MolToSmilesEnum",
                "decoder": "MolToSmilesDecoder",
                "determinized-decoder": "MolToSmilesDeterminizedDecoder",
            }[args.mode],
            "label": args.label,
            "smiles": args.smiles,
            "mode": args.mode,
            "rooted_at_atom": args.rooted_at_atom,
            "all_roots": args.all_roots,
            "isomeric_smiles": args.isomeric_smiles,
            "warmup_loops": args.warmup_loops,
            "loops": args.loops,
            "percent_limit": args.percent_limit,
            "report_path": report_path,
            "top_hotspots": hotspots,
        }
    )

    print(f"Appended hotspot history entry for {args.label}")
    if report_path is not None:
        print(f"Saved perf report: {report_path}")
    for hotspot in hotspots:
        print(
            f"{hotspot['inclusive_pct']:5.2f}% {hotspot['self_pct']:5.2f}% "
            f"{hotspot['dso']} :: {hotspot['symbol']}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
