# Rust-First Layout

## Goal

The repository should treat the Rust crate as the source of truth for runtime
behavior, while Python remains the packaging, bindings, and reference-oracle
layer.

## Ownership

- Rust owns:
  - prepared-graph runtime shape
  - exact-support algorithms
  - next-token walkers
  - runtime invariants and schema evolution
- Python owns:
  - PyO3 binding surface
  - RDKit interop
  - policy loading
  - dataset and artifact tooling
  - oracle/reference checks

## Current Transition State

The runtime is still partly duplicated between Rust and the legacy
pure-Python implementation under `python/smiles_next_token/reference/`.

The intended direction is:

1. Rust stays authoritative for runtime behavior.
2. Python top-level imports prefer `_core` bindings.
3. `smiles_next_token.reference` remains available for compatibility and tests.
4. `smiles_next_token.rdkit_reference` is the explicit home for oracle and
   fixture-generation workflows.

## Implications

- Performance work should happen in Rust first.
- Bug fixes for runtime walkers should land in Rust first.
- Python parity tests should validate Rust, not define behavior independently.
- RDKit-backed checks remain valuable, but as oracle coverage rather than as
  the main implementation path.

## Preferred Surface

- Runtime callers should prefer `smiles_next_token`.
- Reference-oracle and fixture-generation workflows should prefer
  `smiles_next_token.rdkit_reference`.
- `smiles_next_token.reference` remains available for compatibility, tests, and
  debugging, but it is not the preferred runtime API.

## Test Authority

The intended order of authority is:

1. Rust-native tests for prepared-graph validation and walker/enumerator
   behavior.
2. Python contract and integration tests for the public API and transport
   boundary.
3. Python parity tests as a cross-language regression net on curated and
   representative slices.
4. RDKit-backed reference tests as oracle coverage.

When coverage overlaps, prefer strengthening items earlier in this list instead
of expanding later ones.

## Change Rules

- New runtime behavior should land in Rust first.
- If Python fallback behavior is required, keep it explicitly limited to the
  top-level runtime wrapper or to `rdkit_reference`.
- Avoid adding new dual implementations unless there is a concrete oracle or
  debugging need.
