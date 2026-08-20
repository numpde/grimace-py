from __future__ import annotations

import argparse

from grimace._reference import (
    DEFAULT_CORE_SELECTION_LIMIT,
    DEFAULT_RDKIT_RANDOM_POLICY_PATH,
    ReferencePolicy,
    write_core_exact_sets_artifact,
    write_full_metrics_artifact,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--policy",
        default=str(DEFAULT_RDKIT_RANDOM_POLICY_PATH),
        help="Path to the policy JSON file",
    )
    parser.add_argument(
        "--core-limit",
        type=int,
        default=DEFAULT_CORE_SELECTION_LIMIT,
        help="Number of molecules to keep in the exact-set core snapshot",
    )
    parser.add_argument(
        "--metrics-limit",
        type=int,
        default=None,
        help="Optional limit for the broad metrics snapshot; defaults to the full source",
    )
    parser.add_argument(
        "--metrics-max-smiles-length",
        type=int,
        default=None,
        help="Optional maximum SMILES length for the broad metrics snapshot",
    )
    parser.add_argument(
        "--skip-core",
        action="store_true",
        help="Do not write the core exact-set artifact",
    )
    parser.add_argument(
        "--skip-metrics",
        action="store_true",
        help="Do not write the full metrics artifact",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    policy = ReferencePolicy.from_path(args.policy)

    if not args.skip_core:
        core_path = write_core_exact_sets_artifact(policy, limit=args.core_limit)
        print(core_path)

    if not args.skip_metrics:
        metrics_path = write_full_metrics_artifact(
            policy,
            limit=args.metrics_limit,
            max_smiles_length=args.metrics_max_smiles_length,
        )
        print(metrics_path)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
