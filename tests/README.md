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
- `tests/rdkit_serialization/known_stereo_gaps.py`: opt-in diagnostic checks
  for known coupled directional-stereo gaps, excluded from default discovery.

## Commands

- Container checks: `make checks`
- Container installed-artifact correctness: `make test`
- Container package test: `make test-package`
- Container enum timing artifacts: `make timings-enum`
- Default suite: `PYTHONPATH=python:. python3 -m unittest discover -s tests -t .`
- Exact public invariants: `PYTHONPATH=python:. python3 -m unittest tests.run_exact_public_invariants -q`
- Installed-artifact correctness subset: `python3 -m unittest tests.run_installed_package_correctness -q`
- Pinned RDKit parity subset: `PYTHONPATH=python:. python3 -m unittest tests.run_pinned_rdkit_parity -q`
- Known stereo-gap diagnostics, expected to fail until the pinned gaps are fixed:
  `PYTHONPATH=python:. python3 -m unittest tests.run_known_stereo_gaps -q`
- Perf suite wrapper: `RUN_PERF_TESTS=1 PYTHONPATH=python:. python3 -m unittest discover -s tests/perf -t .`

CI runs the exact public invariants and pinned RDKit parity layers as separate
source-tree jobs, and reuses them inside the installed-artifact correctness
subset. The installed-artifact subset also runs count-only RDKit writer support
evidence because those fixtures validate release artifacts against checked
support cardinalities.

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
- Deterministic RDKit writer outputs that were not observed in bounded
  random-writer sampling belong in diagnostic fixtures under
  `tests/fixtures/rdkit_deterministic_unobserved/`, not in passing membership
  fixtures.
- RDKit random-writer support-count evidence belongs in version-keyed shards
  under `tests/fixtures/rdkit_writer_support_counts/`; keep its adaptive
  saturation evidence explicit.
- Deterministic RDKit rooted random-writer cases belong in version-keyed
  fixtures under `tests/fixtures/rdkit_rooted_random/`.
- Isolated RDKit behaviors that are unusual but still may need to be mirrored
  belong in version-keyed fixtures under `tests/fixtures/rdkit_known_quirks/`.
- Pinned RDKit writer outputs that Grimace does not yet produce belong in
  version-keyed diagnostic fixtures under
  `tests/fixtures/rdkit_known_stereo_gaps/`.
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
- `tests.run_installed_package_correctness` must also fail, not silently skip,
  when its count-only RDKit writer support fixtures are missing for the
  installed RDKit version. Direct discovery may skip those version-keyed count
  tests for exploratory local RDKit installs.
- Known gap tests should be runnable through a named diagnostic runner and
  excluded from default discovery until they become passing conformance tests.
- Shared case selectors and policy overrides belong in `tests/helpers/`, not duplicated across files.
- Prefer strengthening Rust-native tests before expanding parity breadth.
- Prepared graph input equivalence and serialized `PreparedMol` equivalence are
  different matrices. Use `prepared_graph_input_variants()` only for RDKit mol,
  reference prepared graph, and core prepared graph coverage; keep raw/zstd
  byte round-trips in PreparedMol-specific tests.
