# Python API

## Public Surface

The only supported public Python API is `smiles_next_token`.

Current top-level exports:

- `ReferencePolicy`
- `MolToSmilesSupport`

The compiled extension `smiles_next_token._core` is required. There is no
public runtime fallback.

## MolToSmilesSupport

`MolToSmilesSupport(mol, *, rootedAtAtom, isomericSmiles=True, connectedOnly=True, policy=None)`

This is the supported exact-support runtime entrypoint.

- `rootedAtAtom` selects the root atom, following RDKit keyword style.
- `isomericSmiles=True` dispatches to the stereochemical surface.
- `isomericSmiles=False` dispatches to the nonstereo surface.
- `connectedOnly` is currently required to stay `True`; `False` raises `NotImplementedError`.
- `policy` accepts a `ReferencePolicy` and controls the RDKit bridge/oracle settings used to prepare the transport graph.

## Internal Modules

The package also contains internal support code:

- `smiles_next_token._runtime`
  Internal RDKit-to-core bridge helpers.
- `smiles_next_token._reference`
  Internal pure-Python oracle/reference implementation used by tests, fixtures,
  and artifact workflows.

These are not part of the supported public API.
