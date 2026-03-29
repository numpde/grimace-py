# Correctness

## Source of truth

Rust is the source of truth for runtime behavior.

Python builds the RDKit bridge and exposes the public API, but runtime
enumeration and next-token decoding are Rust-backed.

## What is tested

The test suite is layered:

1. Rust-native tests for core runtime behavior
2. Python integration tests for the public API
3. Python parity tests for cross-language regression checks
4. RDKit-backed reference checks

## What the reference code is for

The internal `_reference` package is used for tests, fixtures, and oracle
checks. It is not the supported public runtime API.

## What this means for users

If a runtime bug is fixed, the fix should land in Rust first.
