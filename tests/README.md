# Tests

The test suite is organized by intent first, then by feature.

## Layout

- `tests/contract/`: API, serialization, dataset, policy, and export invariants.
- `tests/reference/`: internal `_reference` behavior checks, split into `prepared/`, `nonstereo/`, and `stereo/`.
- `tests/rdkit_serialization/`: RDKit-derived writer conformance tests, organized by behavior and mapped onto Grimace's public support/decoder surface, including pinned serializer regression cases.
- `tests/parity/`: Rust kernel versus Python reference parity checks on curated and representative slices.
- `tests/integration/`: import, end-to-end smoke coverage, and kernel dataset contract checks.
- `tests/perf/`: opt-in timing checks that are excluded by default.
- `tests/helpers/`: shared case selectors, policy loaders, molecule parsers, and assertion helpers.

## Commands

- Default suite: `PYTHONPATH=python:. python3 -m unittest discover -s tests -t .`
- Exact public invariants: `PYTHONPATH=python:. python3 -m unittest tests.run_exact_public_invariants -q`
- Installed-artifact correctness subset: `python3 -m unittest tests.run_installed_package_correctness -q`
- Pinned RDKit parity subset: `PYTHONPATH=python:. python3 -m unittest tests.run_pinned_rdkit_parity -q`
- Perf suite: `RUN_PERF_TESTS=1 PYTHONPATH=python:. python3 -m unittest discover -s tests/perf -t .`

CI runs the exact public invariants and pinned RDKit parity layers as separate
source-tree jobs, and reuses them inside the installed-artifact correctness
subset.

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
- Large pinned RDKit fixture corpora may use `VERSION/*.json` shards under the
  fixture root; keep shard names ordered by source area or serializer feature.
- Pinned RDKit JSON fixtures should reuse `tests/helpers/pinned_rdkit_fixtures.py`
  for version, id, source, and canonical expected-set validation.
- Exact public invariant checks should be runnable through `tests.run_exact_public_invariants`.
- Exact RDKit-parity tests should be version-keyed and runnable through `tests.run_pinned_rdkit_parity`.
- `tests.run_pinned_rdkit_parity` must fail, not silently skip, when the
  installed RDKit version has no checked-in pinned fixtures.
- Shared case selectors and policy overrides belong in `tests/helpers/`, not duplicated across files.
- Prefer strengthening Rust-native tests before expanding parity breadth.
