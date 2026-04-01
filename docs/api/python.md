# Python API

## Public surface

The only supported public Python API is `grimace`.

Current top-level exports:

- `MolToSmilesDecoder`
- `MolToSmilesEnum`
- `MolToSmilesTokenInventory`

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
multiple disconnected fragments are supported here and in
`MolToSmilesDecoder(...)`, but not yet in `MolToSmilesTokenInventory(...)`.

## MolToSmilesDecoder

`MolToSmilesDecoder(mol, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This is the incremental next-token API. It accepts the same flags and the same
current limits as `MolToSmilesEnum(...)`.

It exposes online next choices for the same rooted random writer mode. The
decoder is online: it shows the legal next choices for the current emitted
prefix, and each choice already carries the next decoder state.

```python
decoder = grimace.MolToSmilesDecoder(
    mol,
    rootedAtAtom=0,
    isomericSmiles=False,
    canonical=False,
    doRandom=True,
)
while decoder.prefix != "CC(=O)Oc1c":
    decoder = decoder.next_choices[0].next_state

decoder.prefix       # 'CC(=O)Oc1c'
[choice.text for choice in decoder.next_choices]  # ['(', 'c']
```

Available interface:

- `next_choices: tuple[MolToSmilesChoice, ...]`
- `prefix: str`
- `is_terminal: bool`
- `copy() -> MolToSmilesDecoder`

Each `MolToSmilesChoice` has:

- `text: str`
- `next_state: MolToSmilesDecoder`

## Decoder model

`MolToSmilesDecoder(...)` exposes the same runtime as a stateful decoder:

- `next_choices` returns the allowed next logical choices
- `choice.text` is the emitted SMILES fragment for that choice
- `choice.next_state` is the decoder state after taking that choice
- `prefix` returns the current prefix

The tokens are literal SMILES fragments, not token ids. A token may be one
character or many characters, for example `"C"`, `"[C@H]"`, or `"%12"`.

Important semantic point:

- the decoder is online
- it reports the legal next choices for the current emitted prefix
- choosing a choice advances it to the next decoder state
- it does not precompute one fixed full output before you start stepping

So:

- `MolToSmilesEnum(...)` gives exact full support
- `MolToSmilesDecoder(...)` lets you step through that support one token at a time

## MolToSmilesTokenInventory

`MolToSmilesTokenInventory(mol, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=None, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This returns a sorted tuple of possible decoder tokens for one molecule under
the same public writer flags.

This first implementation is fast and local-graph based. It is intended as a
token-inventory helper, not as a proof that every returned token is reachable.

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
