# Python API

## Public Surface

The only supported public Python API is `smiles_next_token`.

Current top-level exports:

- `MolToSmilesSupport`

The compiled extension `smiles_next_token._core` is required. There is no
public runtime fallback.

## MolToSmilesSupport

`MolToSmilesSupport(mol, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This is the supported exact-support runtime entrypoint. Its keyword surface
mirrors RDKit `MolToSmiles`, but the current engine only implements rooted
random support generation.

Supported combination:

- `rootedAtAtom >= 0`
- `canonical=False`
- `doRandom=True`

Supported writer flags:

- `isomericSmiles`
- `kekuleSmiles`
- `allBondsExplicit`
- `allHsExplicit`
- `ignoreAtomMapNumbers`

Unsupported combinations fail fast with `NotImplementedError`. Molecules with
multiple disconnected fragments also raise `NotImplementedError`.

## Internal Modules

The package also contains internal support code:

- `smiles_next_token._runtime`
  Internal RDKit-to-core bridge helpers.
- `smiles_next_token._reference`
  Internal pure-Python oracle/reference implementation used by tests, fixtures,
  and artifact workflows.

These are not part of the supported public API.
