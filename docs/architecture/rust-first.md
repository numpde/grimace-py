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
