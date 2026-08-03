"""Explicit South Star qualification runner backed by the plan registry."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import unittest

from tests.south_star1.qualification_plan import (
    CONTINUATION_AUTHORITY_DIAGNOSTIC_LAYERS,
    CONTINUATION_AUTHORITY_PRODUCT_LAYERS,
    SLOW_QUALIFICATION_LAYERS,
    SLOW_QUALIFICATION_SHARDS,
    bind_slow_qualification_shard,
    reset_slow_qualification_shard,
    slow_cases_for_shard,
    validate_qualification_plan,
)

SLOW_PRODUCT_LAYERS = CONTINUATION_AUTHORITY_PRODUCT_LAYERS
SLOW_DIAGNOSTIC_LAYERS = CONTINUATION_AUTHORITY_DIAGNOSTIC_LAYERS


def validate_selection(shard: str | None, layer: str | None) -> None:
    validate_qualification_plan()
    if not shard or shard not in SLOW_QUALIFICATION_SHARDS:
        raise ValueError(f"unknown slow qualification shard: {shard!r}")
    if not layer or layer not in SLOW_QUALIFICATION_LAYERS:
        raise ValueError(f"unknown slow qualification layer: {layer!r}")


def load_selected_layer(
    loader: unittest.TestLoader, shard: str, layer: str
) -> tuple[unittest.TestSuite, object]:
    validate_selection(shard, layer)
    token = bind_slow_qualification_shard(shard)
    suite = unittest.TestSuite()
    for test_id in SLOW_QUALIFICATION_LAYERS[layer].test_ids:
        suite.addTests(loader.loadTestsFromName(test_id))
    return suite, token


def _run_child_layers(shard: str, layers: tuple[str, ...]) -> int:
    for layer in layers:
        command = [sys.executable, "-m", __name__]
        env = os.environ.copy()
        env.update(
            SOUTH_STAR1_RUN_SLOW="1",
            SOUTH_STAR1_SLOW_SHARD=shard,
            SOUTH_STAR1_SLOW_LAYER=layer,
        )
        print(f"running qualification layer: {shard}/{layer}", flush=True)
        completed = subprocess.run(command, env=env)
        if completed.returncode:
            return completed.returncode
    return 0


def _describe_plan() -> int:
    validate_qualification_plan()
    print("South Star qualification plan:")
    for definition in SLOW_QUALIFICATION_LAYERS.values():
        print(f"  {definition.name} [{definition.kind}] {definition.role}")
        for test_id in definition.test_ids:
            print(f"    {test_id}")
    for shard in SLOW_QUALIFICATION_SHARDS.values():
        print(f"  shard={shard.name} cases={','.join(shard.case_names)}")
        print(f"    product={','.join(shard.product_layers)}")
    return 0


def _run_one(shard: str, layer: str) -> int:
    try:
        validate_selection(shard, layer)
        cases = slow_cases_for_shard(shard)
    except ValueError as error:
        print(error, file=sys.stderr)
        return 2
    print("south-star1 slow qualification:")
    print(f"  shard={shard}")
    print(f"  cases={','.join(case.name for case in cases)}")
    print(f"  layer={layer}")
    started = time.monotonic()
    suite, token = load_selected_layer(unittest.defaultTestLoader, shard, layer)
    try:
        result = unittest.TextTestRunner(verbosity=2).run(suite)
    finally:
        reset_slow_qualification_shard(token)
    print(f"elapsed_seconds={time.monotonic() - started:.3f}")
    return 0 if result.wasSuccessful() else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-shard", metavar="SHARD")
    parser.add_argument("--run-all-product", action="store_true")
    parser.add_argument("--describe-plan", action="store_true")
    arguments = parser.parse_args(argv)
    if arguments.describe_plan:
        return _describe_plan()
    if arguments.run_shard and arguments.run_all_product:
        parser.error("--run-shard and --run-all-product are mutually exclusive")
    if arguments.run_shard:
        validate_selection(arguments.run_shard, SLOW_PRODUCT_LAYERS[0])
        return _run_child_layers(
            arguments.run_shard,
            SLOW_QUALIFICATION_SHARDS[arguments.run_shard].product_layers,
        )
    if arguments.run_all_product:
        shard = os.environ.get("SOUTH_STAR1_SLOW_SHARD")
        if shard:
            validate_selection(shard, SLOW_PRODUCT_LAYERS[0])
            return _run_child_layers(shard, SLOW_QUALIFICATION_SHARDS[shard].product_layers)
        for shard_definition in SLOW_QUALIFICATION_SHARDS.values():
            status = _run_child_layers(
                shard_definition.name,
                shard_definition.product_layers,
            )
            if status:
                return status
        return 0
    return _run_one(
        os.environ.get("SOUTH_STAR1_SLOW_SHARD"),
        os.environ.get("SOUTH_STAR1_SLOW_LAYER"),
    )


if __name__ == "__main__":
    raise SystemExit(main())
