# Tests

The test suite is organized by intent first, then by feature.

## Layout

- `tests/contract/`: API, serialization, dataset, policy, and export invariants.
- `tests/reference/`: internal `_reference` behavior checks, split into `prepared/`, `nonstereo/`, and `stereo/`.
- `tests/parity/`: Rust kernel versus Python reference parity checks on curated and representative slices.
- `tests/integration/`: import, end-to-end smoke coverage, and kernel dataset contract checks.
- `tests/perf/`: opt-in timing checks that are excluded by default.
- `tests/helpers/`: shared case selectors, policy loaders, molecule parsers, and assertion helpers.

## Commands

- Default suite: `PYTHONPATH=python:. python3 -m unittest discover -s tests -t .`
- Perf suite: `RUN_PERF_TESTS=1 PYTHONPATH=python:. python3 -m unittest discover -s tests/perf -t .`

## Rules

- Performance assertions do not belong in correctness suites.
- Kernel parity tests belong under `tests/parity/`, not `tests/reference/`.
- Dataset-backed kernel contract checks belong under `tests/integration/` once they stop being cross-language parity checks.
- Shared case selectors and policy overrides belong in `tests/helpers/`, not duplicated across files.
- Prefer strengthening Rust-native tests before expanding parity breadth.
