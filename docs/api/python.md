# Python API

## Public Surface

The only supported public Python API is `smiles_next_token`.

Current top-level exports:

- `ReferencePolicy`
- `enumerate_rooted_connected_nonstereo_smiles_support`
- `enumerate_rooted_connected_stereo_smiles_support`
- `enumerate_rooted_nonstereo_smiles_support`
- `enumerate_rooted_smiles_support`

The compiled extension `smiles_next_token._core` is required. There is no
public runtime fallback.

## Internal Modules

The package also contains internal support code:

- `smiles_next_token._runtime`
  Internal RDKit-to-core bridge helpers.
- `smiles_next_token._reference`
  Internal pure-Python oracle/reference implementation used by tests, fixtures,
  and artifact workflows.

These are not part of the supported public API.
