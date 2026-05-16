# AGENTS.md

Scope: this repository (`/home/coder/repos/grimace-py`) and all subdirectories
unless a deeper `AGENTS.md` overrides.

## Project Intent

Grimace is a Rust-first Python package for exact rooted SMILES support and
online next-token decoding for RDKit's `canonical=False, doRandom=True` writer
regime.

The project has two correctness layers that must stay separate:

- principled SMILES/chemistry semantics: emitted strings should parse back to
  the intended graph and stereo assignment
- RDKit writer parity: emitted strings should match RDKit's actual writer
  support for the supported regime

The public runtime is currently an RDKit writer-parity API. Do not silently
substitute parsed-object equivalence for exact RDKit string support unless the
test or note explicitly states it is a semantic-equivalence check.

## Core Rules For Coding Agents

- Prefer fail-fast behavior over silent fallback.
- Do not swallow exceptions unless explicitly requested.
- Avoid defensive code paths that hide data, fixture, or parity issues.
- Use assertions or explicit errors when assumptions are required.
- Keep behavior transparent and inspectable.
- Prefer simple, readable code over abstraction-heavy code.
- Do not add RDKit-matching special cases without naming the RDKit behavior and
  backing it with a pinned fixture, source reference, or focused exploration.

## Rust/Python Boundary

- The compiled Rust extension is the public runtime path.
- Python owns RDKit interop, input preparation, packaging, and reference/helper
  utilities.
- Rust owns core enumeration, support, and decoder behavior.
- Keep RDKit-specific writer policy distinct from generic SMILES or chemistry
  semantics. Name RDKit traversal, rooting, fragment-order, and directional-bond
  placement behavior as RDKit writer behavior.
- On stereo work, prefer explicit facts, constraints, and row/state models over
  scattered procedural repairs.

## Coding Style

- Keep code close to the invariant it enforces.
- Add comments only when they explain a real "why" that is not already visible
  in the code.
- Avoid incidental line churn: do not rewrap, rename, or reformat neighboring
  code while making a focused fix.
- Use specific names when they distinguish real concepts, not as decorative
  verbosity.
- Keep dummy/API-compatibility values deterministic and document why they exist
  at the boundary.
- Keep defensive code at true boundaries only: external inputs, sparse raw data,
  mode-dependent RDKit/runtime outputs, and explicit compatibility seams.
- Prefer one clear consistency check at the boundary over repeated downstream
  validation of the same assumption.

## Fixtures And SSoT

- RDKit-derived expectations belong in version-keyed JSON fixtures under
  `tests/fixtures/`, not inline in tests.
- Use `tests/helpers/pinned_rdkit_fixtures.py` and nearby fixture loaders for
  version, id, source, and expected-set validation.
- Every pinned RDKit case should carry enough source metadata to explain where
  the expectation came from.
- Large fixture corpora may use `VERSION/*.json` shards; keep shard names
  ordered by source area or serializer feature.
- `known_quirks` means "RDKit does this, for some reason, and Grimace may need
  to mirror it"; it does not mean "out of scope."
- Avoid duplicating expected SMILES sets across tests. Put them in fixtures and
  share loaders/assertion helpers.

## Testing

Prefer focused tests while iterating, then broaden when the touched surface is
larger.

Common commands:

- Default Python suite:
  `PYTHONPATH=python:. python3 -m unittest discover -s tests -t .`
- Exact public invariants:
  `PYTHONPATH=python:. python3 -m unittest tests.run_exact_public_invariants -q`
- Pinned RDKit parity:
  `PYTHONPATH=python:. python3 -m unittest tests.run_pinned_rdkit_parity -q`
- Installed-artifact correctness:
  `python3 -m unittest tests.run_installed_package_correctness -q`
- Rust unit tests:
  `cargo test --lib`
- Performance timings, opt-in only:
  `RUN_PERF_TESTS=1 PYTHONPATH=python:. python3 -m unittest discover -s tests/perf -t .`

Testing rules:

- Performance assertions do not belong in correctness suites.
- Pinned RDKit parity tests must be keyed to the exact RDKit version. The
  current pinned writer convention is RDKit `2026.03.1`.
- `tests.run_pinned_rdkit_parity` should fail, not silently redefine
  expectations, when the installed RDKit version has no checked-in fixture set.
- Sampled RDKit subset checks are useful sanity checks, not proof of exact
  support equality.
- Prefer strengthening Rust-native tests before expanding broad parity coverage.
- Keep principled semantic tests separate from RDKit writer-parity tests.

## Exploration And Notes

- Keep long-lived planning and architecture notes under repo-root `notes/`.
- Name new notes with the next numeric prefix plus a concrete topic, e.g.
  `014_shared_marker_obligation_alternatives.md`.
- Keep reusable probes under `tmp/exploration/<topic>/` with chronological names
  such as `031_inspect_shared_bridge_token_basis.py`.
- Give kept exploration scripts a concise top-level docstring explaining the
  question they answer.
- Do not turn exploratory scripts into runtime dependencies.
- For substantial refactors, write down the concept split, ownership
  boundaries, serious alternatives, potential regrets, and minimal-regret plan
  before broad implementation.

## Change Management

- Make small, focused commits.
- Split commits by conceptual unit; do not bundle code, tests, notes, and
  cleanup unless they are one inseparable change.
- Do not modify unrelated files.
- Do not revert user changes unless explicitly requested.
- Stage one path or concept at a time and check `git status -sb` before each
  commit.
- Use concrete commit messages that state the change.
- Do not commit generated run outputs, plot/perf output directories, or
  unrelated untracked artifacts unless explicitly requested.

## Release And Packaging Notes

- The PyPI distribution is `grimace-py`; the import package is `grimace`.
- Plain `pip install grimace` installs an unrelated package and must not appear
  as an install instruction.
- Release notes should summarize the change log for the release, not duplicate
  the README.
- Keep version changes consistent between `pyproject.toml` and `Cargo.toml`.
