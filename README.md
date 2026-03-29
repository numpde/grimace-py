# grimace-py

Rust-first SMILES and next-token engine with Python bindings.

## Direction

The repository is organized around a Rust core in [`rust/src/`](/home/ra/repos/grimace-py/rust/src), surfaced to Python through [`python/smiles_next_token/`](/home/ra/repos/grimace-py/python/smiles_next_token).

The intended roles are:

- Rust: source of truth for runtime data structures, validation, and exact-support algorithms.
- Python: thin public API over the Rust core, plus internal RDKit bridge code.
- Internal reference code: oracle checks, fixtures, policy handling, and artifact production.

## Package Surface

The only supported public Python surface is `smiles_next_token`.

Current top-level exports:

- `MolToSmilesSupport`

The compiled extension `smiles_next_token._core` is required for the public runtime package.
`MolToSmilesSupport(...)` mirrors RDKit `MolToSmiles` flag names, but currently
supports only rooted random support generation on singly-connected molecules.

## Docs

- [`docs/README.md`](/home/ra/repos/grimace-py/docs/README.md)
- [`docs/architecture/rust-first.md`](/home/ra/repos/grimace-py/docs/architecture/rust-first.md)
- [`docs/api/python.md`](/home/ra/repos/grimace-py/docs/api/python.md)
