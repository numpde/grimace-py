#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from tests.helpers.south_star_enum_s_benchmark_cases import (
    SOUTH_STAR_ENUM_S_BENCHMARK_POLICY_SET,
    SOUTH_STAR_ENUM_S_CASE_MANIFEST_SCOPE_NOTE,
    south_star_enum_s_benchmark_cases,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = (
    REPO_ROOT / "notes" / "perf_reports" / "south_star_enum_s_case_manifest_v1.json"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the deterministic South Star EnumS benchmark case manifest."
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = {
        "schema_version": 1,
        "kind": "south_star_semantic_enum_s_case_manifest",
        "policy_set": SOUTH_STAR_ENUM_S_BENCHMARK_POLICY_SET,
        "scope_note": SOUTH_STAR_ENUM_S_CASE_MANIFEST_SCOPE_NOTE,
        "rows": [asdict(case) for case in south_star_enum_s_benchmark_cases()],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
