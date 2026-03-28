# grimace-py

Rust-first SMILES and next-token engine with Python bindings and RDKit-backed reference tooling.

## Direction

The repository is organized around a Rust core in [`rust/src/`](/home/ra/repos/grimace-py/rust/src), surfaced to Python through PyO3 bindings in [`python/smiles_next_token/`](/home/ra/repos/grimace-py/python/smiles_next_token).

The intended roles are:

- Rust: source of truth for runtime data structures, walkers, and exact-support algorithms.
- Python bindings: thin runtime API over the Rust core.
- RDKit reference tooling: oracle checks, fixture generation, policy handling, and artifact production.

The legacy pure-Python implementation remains under [`python/smiles_next_token/reference/`](/home/ra/repos/grimace-py/python/smiles_next_token/reference) as reference-oracle code, not the preferred runtime path.

## Package Surface

- `smiles_next_token`: top-level Python API, preferring `_core` bindings when present.
- `smiles_next_token.rdkit_reference`: RDKit-backed oracle, dataset, and artifact helpers.
- `smiles_next_token.reference`: compatibility surface for the existing pure-Python reference implementation.

## Docs

- [`docs/README.md`](/home/ra/repos/grimace-py/docs/README.md)
- [`docs/architecture/rust-first.md`](/home/ra/repos/grimace-py/docs/architecture/rust-first.md)
- [`docs/api/python.md`](/home/ra/repos/grimace-py/docs/api/python.md)
