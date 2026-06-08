"""Generate RDKit random-writer support-count fixture shards.

The generated fixture records count evidence, not full support strings. The
adaptive criterion is intentionally conservative and explicit in the output:
each seed must independently satisfy the same saturation rule.
"""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

from rdkit_writer_support_count_surfaces import surface_name


CRITERION_VERSION = 1


def writer_flags_from_args(args: argparse.Namespace) -> dict[str, bool]:
    return {
        "isomericSmiles": bool(args.isomeric_smiles),
        "canonical": False,
        "doRandom": True,
        "kekuleSmiles": bool(args.kekule_smiles),
        "allBondsExplicit": bool(args.all_bonds_explicit),
        "allHsExplicit": bool(args.all_hs_explicit),
        "ignoreAtomMapNumbers": bool(args.ignore_atom_map_numbers),
    }


def estimated_missing_variants(singleton_count: int, doubleton_count: int) -> float:
    if singleton_count == 0:
        return 0.0
    if doubleton_count == 0:
        return float("inf")
    return (singleton_count * singleton_count) / (2 * doubleton_count)


def adaptive_patience(support_count: int) -> int:
    return max(10_000, 20 * support_count)


def run_is_saturated(
    *,
    draw_count: int,
    support_count: int,
    consecutive_draws_without_new_variant: int,
    singleton_count: int,
    doubleton_count: int,
    min_draws: int,
    unseen_mass_threshold: float,
    allowed_missing_variants: float,
) -> bool:
    if draw_count < min_draws:
        return False
    if consecutive_draws_without_new_variant < adaptive_patience(support_count):
        return False
    estimated_unseen_mass = singleton_count / draw_count
    if estimated_unseen_mass > unseen_mass_threshold:
        return False
    return (
        estimated_missing_variants(singleton_count, doubleton_count)
        <= allowed_missing_variants
    )


def _load_input_cases(path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{path} must contain readable JSON") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    raw_cases = payload.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise ValueError(f"{path} must define a nonempty cases list")
    cases = []
    seen_ids: set[str] = set()
    for raw_case in raw_cases:
        if not isinstance(raw_case, dict):
            raise ValueError(f"{path} contains a non-object case")
        for field in ("id", "source", "smiles", "rooted_at_atom"):
            if field not in raw_case:
                raise ValueError(f"{path} case is missing {field!r}: {raw_case!r}")
        if not isinstance(raw_case["id"], str) or not raw_case["id"]:
            raise ValueError(f"{path} case has invalid id: {raw_case!r}")
        if not isinstance(raw_case["source"], str) or not raw_case["source"]:
            raise ValueError(f"{path} case has invalid source: {raw_case!r}")
        if not isinstance(raw_case["smiles"], str) or not raw_case["smiles"]:
            raise ValueError(f"{path} case has invalid smiles: {raw_case!r}")
        if type(raw_case["rooted_at_atom"]) is not int:
            raise ValueError(f"{path} case has invalid rooted_at_atom: {raw_case!r}")
        if raw_case["id"] in seen_ids:
            raise ValueError(f"{path} contains duplicate case id: {raw_case['id']!r}")
        seen_ids.add(raw_case["id"])
        cases.append(raw_case)
    return cases


def _run_case_seed(
    *,
    mol: object,
    root_idx: int,
    flags: dict[str, bool],
    seed: int,
    min_draws: int,
    max_draws: int,
    unseen_mass_threshold: float,
    allowed_missing_variants: float,
) -> dict[str, object]:
    from rdkit import Chem, rdBase

    rdBase.SeedRandomNumberGenerator(seed)
    counts: Counter[str] = Counter()
    last_new_at = 0
    final_draw_count = max_draws

    for draw_count in range(1, max_draws + 1):
        output = Chem.MolToSmiles(
            Chem.Mol(mol),
            rootedAtAtom=root_idx,
            **flags,
        )
        before = len(counts)
        counts[output] += 1
        if len(counts) != before:
            last_new_at = draw_count

        consecutive = draw_count - last_new_at
        if draw_count < min_draws or consecutive < adaptive_patience(len(counts)):
            continue

        frequencies = Counter(counts.values())
        singleton_count = frequencies.get(1, 0)
        doubleton_count = frequencies.get(2, 0)
        if run_is_saturated(
            draw_count=draw_count,
            support_count=len(counts),
            consecutive_draws_without_new_variant=consecutive,
            singleton_count=singleton_count,
            doubleton_count=doubleton_count,
            min_draws=min_draws,
            unseen_mass_threshold=unseen_mass_threshold,
            allowed_missing_variants=allowed_missing_variants,
        ):
            final_draw_count = draw_count
            break

    frequencies = Counter(counts.values())
    singleton_count = frequencies.get(1, 0)
    doubleton_count = frequencies.get(2, 0)
    return {
        "seed": seed,
        "draw_count": final_draw_count,
        "support_count": len(counts),
        "consecutive_draws_without_new_variant": final_draw_count - last_new_at,
        "singleton_count": singleton_count,
        "doubleton_count": doubleton_count,
        "estimated_unseen_mass": singleton_count / final_draw_count,
        "estimated_missing_variants": estimated_missing_variants(
            singleton_count,
            doubleton_count,
        ),
    }


def generate_fixture(args: argparse.Namespace) -> dict[str, object]:
    from rdkit import Chem, rdBase

    flags = writer_flags_from_args(args)
    if args.output.stem != surface_name(flags):
        raise ValueError(
            f"output filename must be {surface_name(flags)!r}.json for the "
            "selected flags"
        )

    cases = []
    for raw_case in _load_input_cases(args.input):
        mol = Chem.MolFromSmiles(raw_case["smiles"])
        if mol is None:
            raise ValueError(f"RDKit failed to parse {raw_case['id']!r}")
        runs = [
            _run_case_seed(
                mol=mol,
                root_idx=raw_case["rooted_at_atom"],
                flags=flags,
                seed=seed,
                min_draws=args.min_draws,
                max_draws=args.max_draws,
                unseen_mass_threshold=args.unseen_mass_threshold,
                allowed_missing_variants=args.allowed_missing_variants,
            )
            for seed in args.seed
        ]
        support_counts = {run["support_count"] for run in runs}
        if len(support_counts) != 1:
            raise ValueError(
                f"{raw_case['id']!r} did not reach multi-seed count agreement: "
                f"{sorted(support_counts)!r}"
            )
        for run in runs:
            if not run_is_saturated(
                draw_count=int(run["draw_count"]),
                support_count=int(run["support_count"]),
                consecutive_draws_without_new_variant=int(
                    run["consecutive_draws_without_new_variant"]
                ),
                singleton_count=int(run["singleton_count"]),
                doubleton_count=int(run["doubleton_count"]),
                min_draws=args.min_draws,
                unseen_mass_threshold=args.unseen_mass_threshold,
                allowed_missing_variants=args.allowed_missing_variants,
            ):
                raise ValueError(
                    f"{raw_case['id']!r} seed {run['seed']!r} did not satisfy "
                    "adaptive saturation"
                )
        support_count = int(runs[0]["support_count"])
        cases.append(
            {
                "id": raw_case["id"],
                "source": raw_case["source"],
                "smiles": raw_case["smiles"],
                "rooted_at_atom": raw_case["rooted_at_atom"],
                "support_count": support_count,
                "evidence": {
                    "method": "rdkit_random_adaptive_saturation",
                    "criterion_version": CRITERION_VERSION,
                    "min_draws": args.min_draws,
                    "unseen_mass_threshold": args.unseen_mass_threshold,
                    "allowed_missing_variants": args.allowed_missing_variants,
                    "runs": runs,
                },
            }
        )

    return {
        "rdkit_version": rdBase.rdkitVersion,
        "flags": flags,
        "cases": cases,
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate RDKit writer support-count fixture shards.",
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--seed", type=int, action="append", required=True)
    parser.add_argument("--min-draws", type=int, default=20_000)
    parser.add_argument("--max-draws", type=int, default=200_000)
    parser.add_argument("--unseen-mass-threshold", type=float, default=1e-4)
    parser.add_argument("--allowed-missing-variants", type=float, default=1.0)
    parser.add_argument("--isomeric-smiles", action="store_true")
    parser.add_argument("--kekule-smiles", action="store_true")
    parser.add_argument("--all-bonds-explicit", action="store_true")
    parser.add_argument("--all-hs-explicit", action="store_true")
    parser.add_argument("--ignore-atom-map-numbers", action="store_true")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    if len(args.seed) < 2:
        raise SystemExit("at least two --seed values are required")
    if len(args.seed) != len(set(args.seed)):
        raise SystemExit("--seed values must be unique")
    if args.min_draws <= 0 or args.max_draws < args.min_draws:
        raise SystemExit("--max-draws must be at least --min-draws > 0")
    if not (0 < args.unseen_mass_threshold <= 1):
        raise SystemExit("--unseen-mass-threshold must be in (0, 1]")
    if args.allowed_missing_variants < 0:
        raise SystemExit("--allowed-missing-variants must be nonnegative")
    if args.output.exists() and not args.force:
        raise SystemExit(f"{args.output} already exists; pass --force to overwrite it")

    payload = generate_fixture(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
