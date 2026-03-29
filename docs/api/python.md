# Python API

## Public surface

The only supported public Python API is `grimace`.

Current top-level exports:

- `MolToSmilesDecoder`
- `MolToSmilesEnum`

The compiled extension `grimace._core` is required. There is no
public runtime fallback.

## MolToSmilesEnum

`MolToSmilesEnum(mol, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This yields the complete exact support as whole SMILES strings.

```python
outputs = list(
    grimace.MolToSmilesEnum(
        mol,
        rootedAtAtom=0,
        canonical=False,
        doRandom=True,
    )
)
```

The keyword names mirror RDKit `MolToSmiles`, but the current engine only
implements rooted random support generation.

This is the important semantic point:

- in RDKit, `canonical=False, doRandom=True` returns one sampled SMILES string
- here, `MolToSmilesEnum(...)` yields the full exact support of that same writer mode

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

It exposes next-token support for the same rooted random writer mode. It does
not expose RDKit's internal writer directly.

```python
decoder = grimace.MolToSmilesDecoder(
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

## Decoder model

`MolToSmilesDecoder(...)` exposes the same runtime as a stateful decoder:

- `nextTokens()` returns the allowed next SMILES fragments
- `advance(token)` moves the decoder forward
- `prefix()` returns the current prefix

The tokens are literal SMILES fragments, not token ids. A token may be one
character or many characters, for example `"C"`, `"[C@H]"`, or `"%12"`.

The decoder does not expose enough information to reconstruct the full support
by itself, because one token can lead to more than one successor state.

## Correctness

Rust is the source of truth for runtime behavior.

Python builds the RDKit bridge and exposes the public API, but runtime
enumeration and next-token decoding are Rust-backed.

The test suite is layered:

1. Rust-native tests for core runtime behavior
2. Python integration tests for the public API
3. Python parity tests for cross-language regression checks
4. RDKit-backed reference checks

## Internal Modules

The package also contains internal support code:

- `grimace._runtime`
  Internal RDKit-to-core bridge helpers.
- `grimace._reference`
  Internal pure-Python oracle/reference implementation used by tests and
  fixtures.

These are not part of the supported public API.
