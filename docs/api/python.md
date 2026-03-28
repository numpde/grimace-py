# Python API

## Runtime Surface

The preferred runtime entrypoint is `smiles_next_token`.

Current top-level exports:

- `PreparedSmilesGraph`
- `RootedConnectedNonStereoWalker`
- `RootedConnectedStereoWalker`
- `prepare_smiles_graph`
- `CONNECTED_NONSTEREO_SURFACE`
- `CONNECTED_STEREO_SURFACE`
- `HAVE_CORE_BINDINGS`

When the compiled extension is available, the walker and prepared-graph types
come from `smiles_next_token._core`.

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
