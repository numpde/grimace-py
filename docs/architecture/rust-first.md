# Rust-First Layout

## Goal

Build a Rust-first, tests-first library where Rust is the source of truth for
runtime behavior and Python provides a thin public façade plus internal
RDKit-based bridge and oracle code.

## Ownership

- Rust owns:
  - prepared-graph runtime shape
  - exact-support algorithms
  - next-token walkers
  - runtime invariants and schema evolution
- Python owns:
  - the thin public wrapper over `_core`
  - RDKit interop and transport construction
  - policy loading
  - dataset and artifact tooling
  - oracle/reference checks

## Implications

- Performance work should happen in Rust first.
- Bug fixes for runtime walkers should land in Rust first.
- Python parity tests should validate Rust, not define behavior independently.
- RDKit-backed checks remain valuable, but as oracle coverage rather than as
  the main implementation path.

## Public And Internal Boundaries

- `smiles_next_token` is the only supported public Python API.
- `smiles_next_token._core` is required and remains hidden implementation detail.
- `smiles_next_token._runtime` is internal bridge code.
- `smiles_next_token._reference` is internal oracle/reference code.

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
- Avoid adding dual implementations unless there is a concrete oracle or test
  need.
- Public API changes should happen in `smiles_next_token`, not in internal modules.
