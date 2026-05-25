---
title: Rust-first layout
---

## Goal

This page is for contributors changing internals. The design goal is a
Rust-first, tests-first library where Rust is the source of truth for runtime
behavior and Python provides the public façade plus RDKit bridge code.

## Ownership

- Rust owns:
  - prepared molecule storage after RDKit preparation
  - prepared-graph runtime shape
  - prepared molecule byte encoding
  - exact-support algorithms
  - next-token walkers
  - runtime invariants and schema evolution
- Python owns:
  - the thin public wrapper over `_core`
  - RDKit interop at the `PrepareMol` boundary
  - dataset-backed test tooling
  - oracle/reference checks

## Implications

- Performance work should happen in Rust first.
- Bug fixes for runtime walkers should land in Rust first.
- Python parity tests should validate Rust, not define behavior independently.
- RDKit-backed checks remain valuable, but as oracle coverage rather than as
  the main implementation path.

## Public and internal boundaries

- `grimace` is the only supported public Python API.
- `grimace._core` is required and remains hidden implementation detail.
- `grimace._runtime` is internal bridge code.
- `grimace._reference` is internal oracle/reference code.
- `PrepareMol` may use RDKit; runtime consumption of `PreparedMol` should not.

## Test authority

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

## Change rules

- New runtime behavior should land in Rust first.
- Avoid adding dual implementations unless there is a concrete oracle or test
  need.
- Public API changes should happen in `grimace`, not in internal modules.
