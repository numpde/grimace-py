"""Run rich support-artifact contract domains in separate child processes."""

from __future__ import annotations

import argparse
import subprocess
import sys
import time

from tests.south_star1.writer_support_artifact_test_plan import domain_by_name
from tests.south_star1.writer_support_artifact_test_plan import bounded_domains
from tests.south_star1.writer_support_artifact_test_plan import WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS
from tests.south_star1.writer_support_artifact_test_plan import test_ids_for_domain
from tests.south_star1.writer_support_artifact_test_plan import (
    validate_writer_support_artifact_test_plan,
)


def _run_domain(domain_name: str) -> int:
    domain = domain_by_name(domain_name)
    started = time.monotonic()
    test_count = len(test_ids_for_domain(domain))
    print(f"domain_started={domain.name}", flush=True)
    print(f"domain_module_count={len(domain.modules)}", flush=True)
    print(f"domain_test_count={test_count}", flush=True)
    command = [sys.executable, "-m", "unittest", *domain.modules, "-q"]
    result = subprocess.run(command, check=False)
    elapsed = time.monotonic() - started
    if result.returncode == 0:
        print(f"domain_passed={domain.name}", flush=True)
    else:
        print(f"domain_failed={domain.name}", flush=True)
        print(f"domain_return_code={result.returncode}", flush=True)
    print(f"domain_elapsed_seconds={elapsed:.6f}", flush=True)
    return result.returncode


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain")
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--describe-plan", action="store_true")
    args = parser.parse_args()
    validate_writer_support_artifact_test_plan()
    if args.describe_plan:
        for domain in WRITER_SUPPORT_ARTIFACT_TEST_DOMAINS:
            print(f"{domain.name}: {', '.join(domain.modules)} [{domain.kind}] {domain.role}")
        return 0
    if bool(args.domain) == args.all:
        parser.error("choose exactly one of --domain or --all")
    if args.all:
        for domain in bounded_domains():
            result = _run_domain(domain.name)
            if result:
                return result
        return 0
    return _run_domain(args.domain)


if __name__ == "__main__":
    raise SystemExit(main())
