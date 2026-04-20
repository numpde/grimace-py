"""Scan the bundled RDKit-backed dataset for writer-support mismatches.

The controller walks the local molecule fixture, filters it to the requested
public-surface mode, and evaluates one molecule per subprocess. The worker
subprocess computes either:

- the single deterministic RDKit serialization for the chosen writer flags
- a many-draw RDKit sample set with a simple saturation heuristic
- Grimace's exact support for the same public mode

The deterministic scan succeeds for a molecule when the deterministic RDKit
string is a member of Grimace's exact support. The sampled scan classifies one
case as:

- `clean` when the sampled RDKit outputs stay within Grimace support and either
  match it exactly or the deterministic member check passes with no sampled
  discrepancy
- `rdkit_only` when RDKit emits a sampled string that Grimace cannot produce
- `grimace_only` only after a plateaued single-seed miss survives a
  higher-budget confirmation pass across extra seeds
- `uncertain` when the sample remains a strict subset of Grimace support but the
  plateau heuristic never triggered

Running each molecule in a subprocess keeps timeouts and crashes local to the
current case, which is useful when mining large or pathological inputs.

The controller can also append one JSON record per event to a `.jsonl` file.
That gives long scans a durable audit trail and an automatic resume cursor.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

from rdkit import Chem, rdBase

import grimace
from grimace._reference.dataset import iter_default_molecule_cases, molecule_is_connected

_CONFIRM_GRIMACE_ONLY_SEED_COUNT = 4
_CONFIRM_GRIMACE_ONLY_DRAWS_PER_ROUND = 100
_CONFIRM_GRIMACE_ONLY_STAGNATION_ROUNDS = 8
_CONFIRM_GRIMACE_ONLY_MAX_DRAWS = 20_000


@dataclass(frozen=True, slots=True)
class ScanConfig:
    root_mode: str
    isomeric_smiles: bool
    kekule_smiles: bool
    all_bonds_explicit: bool
    all_hs_explicit: bool
    ignore_atom_map_numbers: bool
    rdkit_mode: str
    draws_per_round: int
    stagnation_rounds: int
    max_draws: int
    seed: int
    connected_mode: str
    max_atoms: int | None
    limit: int
    start_after: str | None
    timeout: float


@dataclass(frozen=True, slots=True)
class ResumeState:
    checked: int
    start_after: str | None


def _parse_bool(value: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Unsupported boolean value: {value!r}")


def _rooted_at_atom(root_mode: str, mol: Chem.Mol) -> int | None:
    """Resolve the CLI root selector against one parsed molecule."""
    if root_mode == "none":
        return None
    if root_mode == "zero":
        return 0
    if root_mode == "last":
        return mol.GetNumAtoms() - 1
    try:
        root_idx = int(root_mode)
    except ValueError as exc:
        raise ValueError(f"Unsupported root mode: {root_mode!r}") from exc
    return root_idx


def _include_case(connected_mode: str, mol: Chem.Mol) -> bool:
    is_connected = molecule_is_connected(mol)
    if connected_mode == "connected":
        return is_connected
    if connected_mode == "disconnected":
        return not is_connected
    if connected_mode == "all":
        return True
    raise ValueError(f"Unsupported connected mode: {connected_mode!r}")


def _grimace_support(
    mol: Chem.Mol,
    *,
    rooted_at_atom: int | None,
    isomeric_smiles: bool,
    kekule_smiles: bool,
    all_bonds_explicit: bool,
    all_hs_explicit: bool,
    ignore_atom_map_numbers: bool,
) -> set[str]:
    kwargs: dict[str, Any] = {
        "isomericSmiles": isomeric_smiles,
        "kekuleSmiles": kekule_smiles,
        "canonical": False,
        "allBondsExplicit": all_bonds_explicit,
        "allHsExplicit": all_hs_explicit,
        "doRandom": True,
        "ignoreAtomMapNumbers": ignore_atom_map_numbers,
    }
    if rooted_at_atom is not None:
        kwargs["rootedAtAtom"] = rooted_at_atom
    return set(grimace.MolToSmilesEnum(mol, **kwargs))


def _rdkit_expected(
    mol: Chem.Mol,
    *,
    rooted_at_atom: int | None,
    isomeric_smiles: bool,
    kekule_smiles: bool,
    all_bonds_explicit: bool,
    all_hs_explicit: bool,
    ignore_atom_map_numbers: bool,
) -> str:
    kwargs: dict[str, Any] = {
        "isomericSmiles": isomeric_smiles,
        "kekuleSmiles": kekule_smiles,
        "canonical": False,
        "allBondsExplicit": all_bonds_explicit,
        "allHsExplicit": all_hs_explicit,
        "doRandom": False,
        "ignoreAtomMapNumbers": ignore_atom_map_numbers,
    }
    if rooted_at_atom is not None:
        kwargs["rootedAtAtom"] = rooted_at_atom
    return Chem.MolToSmiles(Chem.Mol(mol), **kwargs)


def _sample_rdkit_support(
    mol: Chem.Mol,
    *,
    rooted_at_atom: int | None,
    isomeric_smiles: bool,
    kekule_smiles: bool,
    all_bonds_explicit: bool,
    all_hs_explicit: bool,
    ignore_atom_map_numbers: bool,
    draws_per_round: int,
    stagnation_rounds: int,
    max_draws: int,
    seed: int,
) -> tuple[set[str], int, bool]:
    rdBase.SeedRandomNumberGenerator(seed)
    kwargs: dict[str, Any] = {
        "isomericSmiles": isomeric_smiles,
        "kekuleSmiles": kekule_smiles,
        "canonical": False,
        "allBondsExplicit": all_bonds_explicit,
        "allHsExplicit": all_hs_explicit,
        "doRandom": True,
        "ignoreAtomMapNumbers": ignore_atom_map_numbers,
    }
    if rooted_at_atom is not None:
        kwargs["rootedAtAtom"] = rooted_at_atom

    sampled: set[str] = set()
    draw_count = 0
    stalled_rounds = 0

    while draw_count < max_draws and stalled_rounds < stagnation_rounds:
        before_round = len(sampled)
        round_budget = min(draws_per_round, max_draws - draw_count)
        for _ in range(round_budget):
            sampled.add(Chem.MolToSmiles(Chem.Mol(mol), **kwargs))
            draw_count += 1
        if len(sampled) == before_round:
            stalled_rounds += 1
        else:
            stalled_rounds = 0

    return sampled, draw_count, stalled_rounds >= stagnation_rounds


def _preview_strings(values: set[str], *, limit: int = 8) -> list[str]:
    return sorted(values)[:limit]


def _resume_mode_signature(config: ScanConfig) -> dict[str, Any]:
    """Return the compatibility-relevant mode subset for JSONL resume."""
    signature = asdict(config)
    for key in ("limit", "start_after", "timeout"):
        signature.pop(key, None)
    return signature


def _append_jsonl_record(path: str | None, record: dict[str, Any]) -> None:
    if path is None:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True))
        handle.write("\n")


def _iter_jsonl_records(path: str | Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, raw_line in enumerate(handle, start=1):
            line = raw_line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Malformed JSONL record at {path}:{line_number}"
                ) from exc
            if not isinstance(record, dict):
                raise ValueError(
                    f"JSONL record at {path}:{line_number} is not an object"
                )
            records.append(record)
    return records


def _load_resume_state(path: str, config: ScanConfig) -> ResumeState:
    """Infer the effective resume cursor from one prior JSONL scan log."""
    records = _iter_jsonl_records(path)
    expected_signature = _resume_mode_signature(config)
    recorded_signature: dict[str, Any] | None = None
    checked = 0
    last_cid: str | None = None

    for record in records:
        record_type = record.get("record_type")
        if record_type == "mode":
            mode = record.get("mode")
            if not isinstance(mode, dict):
                raise ValueError("JSONL mode record is missing the mode object")
            signature = {
                key: mode[key]
                for key in expected_signature
                if key in mode
            }
            if set(signature) != set(expected_signature):
                raise ValueError("JSONL mode record is missing required mode fields")
            if recorded_signature is None:
                recorded_signature = signature
            elif signature != recorded_signature:
                raise ValueError("JSONL file mixes incompatible scan modes")
            continue

        if record_type in {"case", "timeout", "error"}:
            cid = record.get("cid")
            if not isinstance(cid, str):
                raise ValueError("JSONL case record is missing the CID")
            checked += 1
            last_cid = cid

    if recorded_signature is not None and recorded_signature != expected_signature:
        raise ValueError("JSONL resume mode does not match the requested scan mode")
    return ResumeState(checked=checked, start_after=last_cid)


def _classify_support_comparison(
    *,
    sampled_support: set[str],
    grimace_support: set[str],
    plateau_reached: bool,
) -> dict[str, Any]:
    rdkit_only = sampled_support - grimace_support
    grimace_only = grimace_support - sampled_support
    if rdkit_only:
        status = "rdkit_only"
    elif not grimace_only:
        status = "clean"
    elif plateau_reached:
        status = "grimace_only"
    else:
        status = "uncertain"
    return {
        "status": status,
        "rdkit_only_count": len(rdkit_only),
        "grimace_only_count": len(grimace_only),
        "rdkit_only_preview": _preview_strings(rdkit_only),
        "grimace_only_preview": _preview_strings(grimace_only),
    }


def _confirm_grimace_only_support(
    mol: Chem.Mol,
    *,
    rooted_at_atom: int | None,
    isomeric_smiles: bool,
    kekule_smiles: bool,
    all_bonds_explicit: bool,
    all_hs_explicit: bool,
    ignore_atom_map_numbers: bool,
    seed: int,
) -> tuple[set[str], int, bool, int]:
    """Re-sample a suspected grimace_only case with more budget and more seeds."""
    combined_support: set[str] = set()
    total_draw_count = 0
    all_plateau_reached = True

    for seed_offset in range(1, _CONFIRM_GRIMACE_ONLY_SEED_COUNT + 1):
        sampled_support, draw_count, plateau_reached = _sample_rdkit_support(
            mol,
            rooted_at_atom=rooted_at_atom,
            isomeric_smiles=isomeric_smiles,
            kekule_smiles=kekule_smiles,
            all_bonds_explicit=all_bonds_explicit,
            all_hs_explicit=all_hs_explicit,
            ignore_atom_map_numbers=ignore_atom_map_numbers,
            draws_per_round=_CONFIRM_GRIMACE_ONLY_DRAWS_PER_ROUND,
            stagnation_rounds=_CONFIRM_GRIMACE_ONLY_STAGNATION_ROUNDS,
            max_draws=_CONFIRM_GRIMACE_ONLY_MAX_DRAWS,
            seed=seed + seed_offset,
        )
        combined_support.update(sampled_support)
        total_draw_count += draw_count
        all_plateau_reached = all_plateau_reached and plateau_reached

    return (
        combined_support,
        total_draw_count,
        all_plateau_reached,
        _CONFIRM_GRIMACE_ONLY_SEED_COUNT,
    )


def _classify_support_with_confirmation(
    *,
    sampled_support: set[str],
    grimace_support: set[str],
    plateau_reached: bool,
    confirmation_support: set[str] | None = None,
    confirmation_plateau_reached: bool | None = None,
) -> dict[str, Any]:
    """Classify one sampled comparison, optionally confirming grimace_only."""
    initial = _classify_support_comparison(
        sampled_support=sampled_support,
        grimace_support=grimace_support,
        plateau_reached=plateau_reached,
    )
    if initial["status"] != "grimace_only" or confirmation_support is None:
        return initial

    confirmed = _classify_support_comparison(
        sampled_support=sampled_support | confirmation_support,
        grimace_support=grimace_support,
        plateau_reached=plateau_reached and bool(confirmation_plateau_reached),
    )
    confirmed["initial_status"] = initial["status"]
    confirmed["grimace_only_confirmed"] = confirmed["status"] == "grimace_only"
    return confirmed


def _worker_main(args: argparse.Namespace) -> int:
    """Evaluate one concrete molecule/mode pair and emit a compact JSON result."""
    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {args.smiles!r}")
    rooted_at_atom = None if args.rooted_at_atom < 0 else args.rooted_at_atom
    support = _grimace_support(
        mol,
        rooted_at_atom=rooted_at_atom,
        isomeric_smiles=args.isomeric,
        kekule_smiles=args.kekule,
        all_bonds_explicit=args.all_bonds_explicit,
        all_hs_explicit=args.all_hs_explicit,
        ignore_atom_map_numbers=args.ignore_atom_map_numbers,
    )
    expected = _rdkit_expected(
        mol,
        rooted_at_atom=rooted_at_atom,
        isomeric_smiles=args.isomeric,
        kekule_smiles=args.kekule,
        all_bonds_explicit=args.all_bonds_explicit,
        all_hs_explicit=args.all_hs_explicit,
        ignore_atom_map_numbers=args.ignore_atom_map_numbers,
    )
    payload: dict[str, Any] = {
        "expected": expected,
        "support_size": len(support),
    }
    if args.rdkit_mode == "deterministic":
        payload.update(
            {
                "status": "clean" if expected in support else "rdkit_only",
                "contains": expected in support,
            }
        )
    elif args.rdkit_mode == "sampled":
        sampled_support, draw_count, plateau_reached = _sample_rdkit_support(
            mol,
            rooted_at_atom=rooted_at_atom,
            isomeric_smiles=args.isomeric,
            kekule_smiles=args.kekule,
            all_bonds_explicit=args.all_bonds_explicit,
            all_hs_explicit=args.all_hs_explicit,
            ignore_atom_map_numbers=args.ignore_atom_map_numbers,
            draws_per_round=args.draws_per_round,
            stagnation_rounds=args.stagnation_rounds,
            max_draws=args.max_draws,
            seed=args.seed,
        )
        classification = _classify_support_comparison(
            sampled_support=sampled_support,
            grimace_support=support,
            plateau_reached=plateau_reached,
        )
        if classification["status"] == "grimace_only":
            (
                confirmation_support,
                confirmation_draw_count,
                confirmation_plateau_reached,
                confirmation_seed_count,
            ) = _confirm_grimace_only_support(
                mol,
                rooted_at_atom=rooted_at_atom,
                isomeric_smiles=args.isomeric,
                kekule_smiles=args.kekule,
                all_bonds_explicit=args.all_bonds_explicit,
                all_hs_explicit=args.all_hs_explicit,
                ignore_atom_map_numbers=args.ignore_atom_map_numbers,
                seed=args.seed,
            )
            payload.update(
                {
                    "confirmation_draw_count": confirmation_draw_count,
                    "confirmation_plateau_reached": confirmation_plateau_reached,
                    "confirmation_sampled_size": len(confirmation_support),
                    "confirmation_seed_count": confirmation_seed_count,
                }
            )
            classification = _classify_support_with_confirmation(
                sampled_support=sampled_support,
                grimace_support=support,
                plateau_reached=plateau_reached,
                confirmation_support=confirmation_support,
                confirmation_plateau_reached=confirmation_plateau_reached,
            )
        payload.update(classification)
        if expected not in support:
            payload["status"] = "rdkit_only"
            payload["rdkit_only_count"] = payload.get("rdkit_only_count", 0) + 1
            payload["rdkit_only_preview"] = _preview_strings(
                set(payload.get("rdkit_only_preview", ())) | {expected}
            )
        payload.update(
            {
                "contains": expected in support,
                "sampled_size": len(sampled_support),
                "draw_count": draw_count,
                "plateau_reached": plateau_reached,
            }
        )
    else:
        raise ValueError(f"Unsupported RDKit comparison mode: {args.rdkit_mode!r}")
    print(
        json.dumps(payload, sort_keys=True)
    )
    return 0


def _controller_main(args: argparse.Namespace) -> int:
    """Iterate the dataset and stop at the first concrete regression."""
    config = ScanConfig(
        root_mode=args.root,
        isomeric_smiles=args.isomeric,
        kekule_smiles=args.kekule,
        all_bonds_explicit=args.all_bonds_explicit,
        all_hs_explicit=args.all_hs_explicit,
        ignore_atom_map_numbers=args.ignore_atom_map_numbers,
        rdkit_mode=args.rdkit_mode,
        draws_per_round=args.draws_per_round,
        stagnation_rounds=args.stagnation_rounds,
        max_draws=args.max_draws,
        seed=args.seed,
        connected_mode=args.connected,
        max_atoms=args.max_atoms,
        limit=args.limit,
        start_after=args.start_after,
        timeout=args.timeout,
    )
    if args.resume_jsonl and args.jsonl_output is None:
        raise ValueError("--resume-jsonl requires --jsonl-output")
    if args.resume_jsonl and config.start_after is not None:
        raise ValueError("--resume-jsonl cannot be combined with --start-after")

    resume_state = ResumeState(checked=0, start_after=config.start_after)
    if args.resume_jsonl and args.jsonl_output is not None and Path(args.jsonl_output).exists():
        resume_state = _load_resume_state(args.jsonl_output, config)

    print(json.dumps({"mode": asdict(config)}, sort_keys=True), flush=True)
    if args.resume_jsonl:
        print(
            json.dumps(
                {
                    "resume": {
                        "checked": resume_state.checked,
                        "start_after": resume_state.start_after,
                        "path": args.jsonl_output,
                    }
                },
                sort_keys=True,
            ),
            flush=True,
        )
    _append_jsonl_record(
        args.jsonl_output,
        {
            "record_type": "mode",
            "mode": asdict(config),
            "resume_checked": resume_state.checked,
            "resume_start_after": resume_state.start_after,
        },
    )

    seen_start = resume_state.start_after is None
    checked = resume_state.checked

    if checked >= config.limit:
        _append_jsonl_record(
            args.jsonl_output,
            {
                "record_type": "stop",
                "reason": "limit",
                "checked": checked,
            },
        )
        return 0

    for idx, case in enumerate(iter_default_molecule_cases(), start=1):
        # `start_after` is a resume cursor for long scans, not a filter value.
        if not seen_start:
            if case.cid == resume_state.start_after:
                seen_start = True
            continue

        mol = Chem.MolFromSmiles(case.smiles)
        if mol is None or mol.GetNumAtoms() == 0:
            continue
        if config.max_atoms is not None and mol.GetNumAtoms() > config.max_atoms:
            continue
        if not _include_case(config.connected_mode, mol):
            continue

        rooted_at_atom = _rooted_at_atom(config.root_mode, mol)
        checked += 1

        # Run each case in a fresh interpreter so timeouts and crashes do not
        # poison the whole scan.
        worker_cmd = [
            sys.executable,
            __file__,
            "--worker",
            "--smiles",
            case.smiles,
            "--isomeric",
            "true" if config.isomeric_smiles else "false",
            "--kekule",
            "true" if config.kekule_smiles else "false",
            "--all-bonds-explicit",
            "true" if config.all_bonds_explicit else "false",
            "--all-hs-explicit",
            "true" if config.all_hs_explicit else "false",
            "--ignore-atom-map-numbers",
            "true" if config.ignore_atom_map_numbers else "false",
            "--rdkit-mode",
            config.rdkit_mode,
            "--draws-per-round",
            str(config.draws_per_round),
            "--stagnation-rounds",
            str(config.stagnation_rounds),
            "--max-draws",
            str(config.max_draws),
            "--seed",
            str(config.seed),
            "--rooted-at-atom",
            str(-1 if rooted_at_atom is None else rooted_at_atom),
        ]
        try:
            proc = subprocess.run(
                worker_cmd,
                cwd=os.getcwd(),
                env=os.environ.copy(),
                capture_output=True,
                text=True,
                timeout=config.timeout,
                check=False,
            )
        except subprocess.TimeoutExpired:
            _append_jsonl_record(
                args.jsonl_output,
                {
                    "record_type": "timeout",
                    "checked": checked,
                    "idx": idx,
                    "cid": case.cid,
                    "smiles": case.smiles,
                    "atoms": mol.GetNumAtoms(),
                    "root": rooted_at_atom,
                },
            )
            print(
                f"TIMEOUT checked={checked} idx={idx} cid={case.cid} atoms={mol.GetNumAtoms()}",
                flush=True,
            )
            continue

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()
            _append_jsonl_record(
                args.jsonl_output,
                {
                    "record_type": "error",
                    "checked": checked,
                    "idx": idx,
                    "cid": case.cid,
                    "smiles": case.smiles,
                    "atoms": mol.GetNumAtoms(),
                    "root": rooted_at_atom,
                    "detail": detail,
                },
            )
            print(
                f"ERROR checked={checked} idx={idx} cid={case.cid} atoms={mol.GetNumAtoms()}",
                flush=True,
            )
            print(case.smiles, flush=True)
            if detail:
                print(detail, flush=True)
            return 1

        # The worker only prints one JSON payload on success.
        payload = json.loads(proc.stdout.strip())
        _append_jsonl_record(
            args.jsonl_output,
            {
                "record_type": "case",
                "checked": checked,
                "idx": idx,
                "cid": case.cid,
                "smiles": case.smiles,
                "atoms": mol.GetNumAtoms(),
                "root": rooted_at_atom,
                **payload,
            },
        )
        summary = (
            f"{payload['status'].upper()} checked={checked} idx={idx} cid={case.cid} "
            f"atoms={mol.GetNumAtoms()} root={rooted_at_atom} support={payload['support_size']}"
        )
        if config.rdkit_mode == "sampled":
            summary += (
                f" sampled={payload['sampled_size']} draws={payload['draw_count']} "
                f"plateau={payload['plateau_reached']} contains={payload['contains']}"
            )
        else:
            summary += f" contains={payload['contains']}"
        print(
            summary,
            flush=True,
        )
        if payload["status"] in {"rdkit_only", "grimace_only"}:
            print(case.smiles, flush=True)
            print(f"expected {payload['expected']}", flush=True)
            for key in ("rdkit_only_preview", "grimace_only_preview"):
                preview = payload.get(key)
                if preview:
                    print(f"{key} {preview}", flush=True)
            _append_jsonl_record(
                args.jsonl_output,
                {
                    "record_type": "stop",
                    "reason": payload["status"],
                    "checked": checked,
                    "cid": case.cid,
                },
            )
            return 1

        if checked >= config.limit:
            _append_jsonl_record(
                args.jsonl_output,
                {
                    "record_type": "stop",
                    "reason": "limit",
                    "checked": checked,
                    "cid": case.cid,
                },
            )
            break

    else:
        _append_jsonl_record(
            args.jsonl_output,
            {
                "record_type": "stop",
                "reason": "complete",
                "checked": checked,
            },
        )

    return 0


def _build_parser() -> argparse.ArgumentParser:
    """Build one parser shared by the controller and hidden worker mode."""
    parser = argparse.ArgumentParser(
        description="Mine the local RDKit fixture for Grimace writer regressions.",
    )
    parser.add_argument(
        "--worker",
        action="store_true",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--smiles",
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--rooted-at-atom",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--root",
        default="none",
        help="Root policy: none, zero, last, or an explicit atom index",
    )
    parser.add_argument(
        "--isomeric",
        type=_parse_bool,
        default=True,
        help="Whether to compare the isomeric writer surface (true/false)",
    )
    parser.add_argument(
        "--kekule",
        type=_parse_bool,
        default=False,
        help="Whether to compare kekulized writer output (true/false)",
    )
    parser.add_argument(
        "--all-bonds-explicit",
        type=_parse_bool,
        default=False,
        help="Whether to require explicit bond tokens in the writer output (true/false)",
    )
    parser.add_argument(
        "--all-hs-explicit",
        type=_parse_bool,
        default=False,
        help="Whether to require explicit hydrogens in the writer output (true/false)",
    )
    parser.add_argument(
        "--ignore-atom-map-numbers",
        type=_parse_bool,
        default=False,
        help="Whether to ignore atom map numbers in the writer output (true/false)",
    )
    parser.add_argument(
        "--rdkit-mode",
        choices=("deterministic", "sampled"),
        default="deterministic",
        help="Compare deterministic RDKit output or a plateau-sampled RDKit support subset",
    )
    parser.add_argument(
        "--draws-per-round",
        type=int,
        default=40,
        help="RDKit random draws per sampling round when --rdkit-mode=sampled",
    )
    parser.add_argument(
        "--stagnation-rounds",
        type=int,
        default=5,
        help="Consecutive no-new-output rounds required to call the RDKit sample saturated",
    )
    parser.add_argument(
        "--max-draws",
        type=int,
        default=400,
        help="Maximum RDKit random draws per molecule when --rdkit-mode=sampled",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=12345,
        help="Seed passed to RDKit's RNG when --rdkit-mode=sampled",
    )
    parser.add_argument(
        "--connected",
        choices=("connected", "disconnected", "all"),
        default="connected",
        help="Restrict the scan to connected, disconnected, or all parsed molecules",
    )
    parser.add_argument(
        "--max-atoms",
        type=int,
        default=None,
        help="Skip molecules larger than this atom count",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=100,
        help="Maximum number of checked molecules after filtering",
    )
    parser.add_argument(
        "--start-after",
        default=None,
        help="Skip all cases up to and including this CID",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=12.0,
        help="Per-molecule subprocess timeout in seconds",
    )
    parser.add_argument(
        "--jsonl-output",
        default=None,
        help="Append JSONL progress records to this path",
    )
    parser.add_argument(
        "--resume-jsonl",
        action="store_true",
        help="Resume from the last CID recorded in --jsonl-output",
    )
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.worker:
        return _worker_main(args)
    return _controller_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
