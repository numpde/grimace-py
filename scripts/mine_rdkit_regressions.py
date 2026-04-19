"""Scan the bundled RDKit-backed dataset for writer-support mismatches.

The controller walks the local molecule fixture, filters it to the requested
public-surface mode, and evaluates one molecule per subprocess. The worker
subprocess computes:

- the single deterministic RDKit serialization for the chosen writer flags
- Grimace's exact support for the same public mode

The scan succeeds for a molecule when the deterministic RDKit string is a
member of Grimace's exact support. Running each molecule in a subprocess keeps
timeouts and crashes local to the current case, which is useful when mining
large or pathological inputs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass
from typing import Any

from rdkit import Chem

import grimace
from grimace._reference.dataset import iter_default_molecule_cases, molecule_is_connected


@dataclass(frozen=True, slots=True)
class ScanConfig:
    root_mode: str
    isomeric_smiles: bool
    kekule_smiles: bool
    all_bonds_explicit: bool
    all_hs_explicit: bool
    ignore_atom_map_numbers: bool
    connected_mode: str
    max_atoms: int | None
    limit: int
    start_after: str | None
    timeout: float


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


def _worker_main(args: argparse.Namespace) -> int:
    """Evaluate one concrete molecule/mode pair and emit a compact JSON result."""
    mol = Chem.MolFromSmiles(args.smiles)
    if mol is None:
        raise ValueError(f"RDKit failed to parse SMILES: {args.smiles!r}")
    rooted_at_atom = None if args.rooted_at_atom < 0 else args.rooted_at_atom
    expected = _rdkit_expected(
        mol,
        rooted_at_atom=rooted_at_atom,
        isomeric_smiles=args.isomeric,
        kekule_smiles=args.kekule,
        all_bonds_explicit=args.all_bonds_explicit,
        all_hs_explicit=args.all_hs_explicit,
        ignore_atom_map_numbers=args.ignore_atom_map_numbers,
    )
    support = _grimace_support(
        mol,
        rooted_at_atom=rooted_at_atom,
        isomeric_smiles=args.isomeric,
        kekule_smiles=args.kekule,
        all_bonds_explicit=args.all_bonds_explicit,
        all_hs_explicit=args.all_hs_explicit,
        ignore_atom_map_numbers=args.ignore_atom_map_numbers,
    )
    print(
        json.dumps(
            {
                "expected": expected,
                "contains": expected in support,
                "support_size": len(support),
            },
            sort_keys=True,
        )
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
        connected_mode=args.connected,
        max_atoms=args.max_atoms,
        limit=args.limit,
        start_after=args.start_after,
        timeout=args.timeout,
    )
    print(json.dumps({"mode": asdict(config)}, sort_keys=True), flush=True)

    seen_start = config.start_after is None
    checked = 0

    for idx, case in enumerate(iter_default_molecule_cases(), start=1):
        # `start_after` is a resume cursor for long scans, not a filter value.
        if not seen_start:
            if case.cid == config.start_after:
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
            print(
                f"TIMEOUT checked={checked} idx={idx} cid={case.cid} atoms={mol.GetNumAtoms()}",
                flush=True,
            )
            continue

        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout).strip()
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
        print(
            f"MATCH checked={checked} idx={idx} cid={case.cid} atoms={mol.GetNumAtoms()} "
            f"root={rooted_at_atom} support={payload['support_size']} contains={payload['contains']}",
            flush=True,
        )
        if not payload["contains"]:
            print(case.smiles, flush=True)
            print(f"expected {payload['expected']}", flush=True)
            return 1

        if checked >= config.limit:
            break

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
    return parser


def main() -> int:
    parser = _build_parser()
    args = parser.parse_args()
    if args.worker:
        return _worker_main(args)
    return _controller_main(args)


if __name__ == "__main__":
    raise SystemExit(main())
