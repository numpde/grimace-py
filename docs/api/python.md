# Python API

## Runtime Surface

The preferred runtime entrypoint is `smiles_next_token`.

Current top-level exports:

- `PreparedSmilesGraph`
- `RootedConnectedNonStereoWalker`
- `RootedConnectedStereoWalker`
- `enumerate_rooted_connected_nonstereo_smiles_support`
- `enumerate_rooted_connected_stereo_smiles_support`
- `make_nonstereo_walker`
- `make_stereo_walker`
- `make_prepared_graph`
- `prepare_smiles_graph`
- `CONNECTED_NONSTEREO_SURFACE`
- `CONNECTED_STEREO_SURFACE`
- `HAVE_CORE_BINDINGS`

When the compiled extension is available:

- `prepare_smiles_graph` returns a `_core.PreparedSmilesGraph`
- the `enumerate_*` helpers dispatch through the Rust implementation
- the walker factories return `_core` walker objects

When the compiled extension is absent:

- non-stereo runtime helpers fall back to the pure-Python reference implementation
- stereo enumeration falls back to the reference enumerator
- stereo walker construction raises, because there is no pure-Python stereo walker

## Reference Surface

Reference-only helpers live under:

- `smiles_next_token.rdkit_reference`
- `smiles_next_token.reference`

Use these for:

- RDKit random sampling
- dataset iteration
- policy loading
- artifact generation
- oracle and parity workflows

They are not the preferred runtime API.
