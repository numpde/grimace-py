# Tests

The test suite is organized by intent first, then by feature.

## Layout

- `tests/contract/`: API, serialization, dataset, policy, and export invariants.
- `tests/reference/`: internal `_reference` behavior checks, split into `prepared/`, `nonstereo/`, and `stereo/`.
- `tests/rdkit_serialization/`: RDKit-derived writer conformance tests, organized by behavior and mapped onto Grimace's public support/decoder surface, including pinned serializer regression cases.
- `tests/parity/`: Rust kernel versus Python reference parity checks on curated and representative slices.
- `tests/integration/`: import, end-to-end smoke coverage, and kernel dataset contract checks.
- `tests/south_star1/`: South Star 1 proof-kernel and writer-shaped runtime checks.
- `tests/perf/`: opt-in timing checks that are excluded by default.
- `tests/helpers/`: shared case selectors, policy loaders, molecule parsers, and assertion helpers.
- `tests/rdkit_serialization/known_stereo_gaps.py`: opt-in diagnostic checks
  for known coupled directional-stereo gaps, excluded from default discovery.

## Commands

- Default suite: `python3 scripts/run_bounded_tests.py --wall-seconds 300 --rss-mib 4096 --cpu-seconds 300 --address-space-mib 8192 -- env PYTHONPATH=python:. python3 -m unittest discover -s tests -t .`
- South Star 1 source-tree semantics: `python3 scripts/run_bounded_tests.py --wall-seconds 120 --rss-mib 2048 --cpu-seconds 120 --address-space-mib 4096 -- env PYTHONPATH=python:. python3 -m unittest tests.run_south_star_semantics -q`
- Exact public invariants: `python3 scripts/run_bounded_tests.py --wall-seconds 120 --rss-mib 2048 --cpu-seconds 120 --address-space-mib 4096 -- env PYTHONPATH=python:. python3 -m unittest tests.run_exact_public_invariants -q`
- Installed-artifact correctness subset: `python3 scripts/run_bounded_tests.py --wall-seconds 300 --rss-mib 4096 --cpu-seconds 300 --address-space-mib 8192 -- python3 -m unittest tests.run_installed_package_correctness -q`
- Pinned RDKit parity subset: `python3 scripts/run_bounded_tests.py --wall-seconds 300 --rss-mib 4096 --cpu-seconds 300 --address-space-mib 8192 -- env PYTHONPATH=python:. python3 -m unittest tests.run_pinned_rdkit_parity -q`
- Known stereo-gap diagnostics: `python3 scripts/run_bounded_tests.py --wall-seconds 300 --rss-mib 4096 --cpu-seconds 300 --address-space-mib 8192 -- env PYTHONPATH=python:. python3 -m unittest tests.run_known_stereo_gaps -q`
- Stereo constraint diagnostics: `python3 scripts/run_bounded_tests.py --wall-seconds 300 --rss-mib 4096 --cpu-seconds 300 --address-space-mib 8192 -- env PYTHONPATH=python:. python3 -m unittest tests.run_stereo_constraint_diagnostics -q`
- Perf suite: `python3 scripts/run_bounded_tests.py --wall-seconds 300 --rss-mib 4096 --cpu-seconds 300 --address-space-mib 8192 -- env RUN_PERF_TESTS=1 PYTHONPATH=python:. python3 -m unittest discover -s tests/perf -t .`
- Runner supervisor self-tests: `make test-bounded-runner` (bootstrapped independently with GNU `timeout` and shell CPU/address-space limits).

The runner's wall budget includes lock wait, startup, and command execution;
cleanup has a separate finite allowance. If it reports `cleanup_unverified`,
stop launching repository tests or builds and inspect the command's owned
processes on the host before retrying. An empty sandbox process view does not
prove cleanup.

The runner serializes these commands across checkouts, applies inherited CPU
and per-process address-space limits, and enforces a wall-clock limit. Its
process-tree RSS is a sampled sum from Linux `/proc`; shared pages may count
more than once, short peaks between samples may be missed, and the value is not
a hard kernel memory cap. The runner fails closed when its own `/proc` view is
unavailable or an owned descendant cannot be accounted for, including a
cross-namespace descendant. It verifies that no live process-tree members remain
before reporting completion. A runner killed with
SIGKILL cannot clean up its process tree.

CI intentionally runs named layers instead of reusing full source-tree discovery
for every pull request. The source-tree job runs the South Star 1 semantics
runner; exact public invariants and pinned RDKit parity run as separate jobs;
installed wheel and sdist jobs run the installed-artifact correctness subset.

## Rules

- Performance assertions do not belong in correctness suites.
- Keep principled semantic checks separate from RDKit writer-parity checks.
  Parsed-object equivalence is useful evidence, but it must not silently
  replace exact string equality in RDKit-parity tests.
- RDKit-specific traversal, rooting, fragment-order, and directional-bond
  placement behavior should be named as RDKit writer behavior, not generic
  SMILES semantics.
- Kernel parity tests belong under `tests/parity/`, not `tests/reference/`.
- Dataset-backed kernel contract checks belong under `tests/integration/` once they stop being cross-language parity checks.
- RDKit-derived writer expectations belong under `tests/rdkit_serialization/`, not scattered through smoke tests.
- Bulky serializer expected-output sets belong in version-keyed JSON fixtures
  under `tests/fixtures/rdkit_serializer_regressions/`, with a source reference
  for each case.
- Deterministic RDKit writer-output membership cases belong in version-keyed
  fixtures under `tests/fixtures/rdkit_writer_membership/`.
- Deterministic RDKit rooted random-writer cases belong in version-keyed
  fixtures under `tests/fixtures/rdkit_rooted_random/`.
- Isolated RDKit behaviors that are unusual but still may need to be mirrored
  belong in version-keyed fixtures under `tests/fixtures/rdkit_known_quirks/`.
- RDKit disconnected sampling input suites belong under
  `tests/fixtures/rdkit_disconnected_sampling/`.
- Large pinned RDKit fixture corpora may use version-directory JSON files under
  the fixture root; keep file names ordered by source area or serializer feature.
- Pinned RDKit JSON fixtures should reuse `tests/helpers/pinned_rdkit_fixtures.py`
  for version, id, source, and canonical expected-set validation.
- Exact public invariant checks should be runnable through `tests.run_exact_public_invariants`.
- Exact RDKit-parity tests should be version-keyed and runnable through `tests.run_pinned_rdkit_parity`.
- `tests.run_pinned_rdkit_parity` must fail, not silently skip, when the
  installed RDKit version has no checked-in pinned fixtures.
- Known gap tests should be runnable through a named diagnostic runner and
  excluded from default discovery until they become passing conformance tests.
- Large stereo-constraint model witnesses should be marked `test_tier:
  "diagnostic"` in their version-keyed fixture and run through
  `tests.run_stereo_constraint_diagnostics`, not default discovery.
- Shared case selectors and policy overrides belong in `tests/helpers/`, not duplicated across files.
- Prefer strengthening Rust-native tests before expanding parity breadth.
