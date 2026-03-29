# Python API

## Public surface

The only supported public Python API is `smiles_next_token`.

Current top-level exports:

- `MolToSmilesDecoder`
- `MolToSmilesEnum`

The compiled extension `smiles_next_token._core` is required. There is no
public runtime fallback.

## MolToSmilesEnum

`MolToSmilesEnum(mol, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This yields the complete exact support as whole SMILES strings.

```python
outputs = list(
    smiles_next_token.MolToSmilesEnum(
        mol,
        rootedAtAtom=0,
        canonical=False,
        doRandom=True,
    )
)
```

The keyword names mirror RDKit `MolToSmiles`, but the current engine only
implements rooted random support generation.

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

## MolToSmilesDecoder

`MolToSmilesDecoder(mol, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This is the incremental next-token API. It accepts the same flags and the same
current limits as `MolToSmilesEnum(...)`.

```python
decoder = smiles_next_token.MolToSmilesDecoder(
    mol,
    rootedAtAtom=0,
    canonical=False,
    doRandom=True,
)
tokens = decoder.nextTokens()
```

Available methods:

- `nextTokens() -> tuple[str, ...]`
- `advance(token: str)`
- `prefix() -> str`
- `isTerminal() -> bool`
- `copy() -> MolToSmilesDecoder`

## Internal Modules

The package also contains internal support code:

- `smiles_next_token._runtime`
  Internal RDKit-to-core bridge helpers.
- `smiles_next_token._reference`
  Internal pure-Python oracle/reference implementation used by tests and
  fixtures.

These are not part of the supported public API.
