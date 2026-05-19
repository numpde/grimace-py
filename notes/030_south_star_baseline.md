# South Star Baseline

Branch: `south-star`

Date: 2026-05-20

Task: `South Star 01: Baseline branch and add semantic runner`

## Baseline

Before adding the South Star semantic runner, the branch was green on:

- `cargo test --lib`: 84 passed.
- `PYTHONPATH=python:. python3 -m unittest tests.run_exact_public_invariants -q`:
  43 passed.

## Runner

The first South Star runner is intentionally minimal:

- `tests.run_south_star_semantics`
- `tests/south_star/`

It creates a separate place for semantic-investigation tests before any runtime
excision or public API changes. The runner is scoped to `tests.south_star.*` and
does not name pinned RDKit exact-string parity modules.

The first harness tests are not semantic witnesses yet. They only protect the
starting surface so later South Star work has a separate entry point.
