# Python API

## Public surface

The only supported public Python import name is `grimace`.

The public runtime is intentionally narrow: exact support and decoding for
RDKit's `canonical=False, doRandom=True` writer regime under the current
stable writer convention.

The public signatures keep RDKit-like defaults for surface compatibility, but
those defaults are not currently supported. Pass `canonical=False` and
`doRandom=True` explicitly.

Current top-level exports:

- `MolToSmilesChoice`
- `MolToSmilesDecoder`
- `MolToSmilesDeterminizedDecoder`
- `MolToSmilesEnum`
- `MolToSmilesTokenInventory`

The compiled extension `grimace._core` is required. There is no
public runtime fallback. Run the public API from the same Python environment
where the extension was built or installed.

Environment requirements:

- package metadata declares Python `>=3.11`
- `rdkit>=2026.3`
- for source builds, a Rust toolchain with `rustc >= 1.83`

Install-path caveat:

- this project is not currently published on PyPI
- plain `pip install grimace` installs an unrelated older package, not this
  project
- use a release wheel, sdist, or a local source build instead
- example wheel install:
  `python -m pip install`
  `https://github.com/numpde/grimace-py/releases/download/v0.1.6/grimace-0.1.6-cp312-cp312-manylinux_2_28_x86_64.whl`

Current continuously exercised matrix:

- Linux source-tree tests on CPython `3.12`
- Linux wheel build and smoke tests on CPython `3.12` and `3.13`
- source distribution build plus `twine check` metadata validation

Other Python versions and non-Linux platforms are expected source-build paths,
not part of the current release asset or CI matrix.
Python `3.11` is in that source-build category today: declared, but not part of
the current CI matrix.
The published sdist is not currently installed and smoke-tested in CI as an
artifact.

## MolToSmilesEnum

`MolToSmilesEnum(mol, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This yields the complete exact support as whole SMILES strings.

Although the signature mirrors RDKit defaults, the current runtime does not
support those defaults. A naive `grimace.MolToSmilesEnum(mol)` call raises
`NotImplementedError`; pass `canonical=False` and `doRandom=True` explicitly.

When `rootedAtAtom < 0`, the result is the exact union across all valid roots
for the requested writer flags. `rootedAtAtom=-1` is the preferred public
spelling for that all-roots mode. `rootedAtAtom=None` is not supported; omit
the argument or use `-1` instead.

Set semantics are the contract here. `MolToSmilesEnum(...)` yields the exact
support, but callers should not rely on the yielded iteration order as a
stable public ordering guarantee.

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

The serialization target is the current stable RDKit writer convention,
currently `RDKit 2026.03.1`. Older slash/backslash serialization conventions
are out of scope. The dependency floor is `rdkit>=2026.3`, but exact output
parity is only validated against that current stable writer convention; newer
RDKit releases may still require fixture or expectation updates.

This is the important semantic point:

- in RDKit, `canonical=False, doRandom=True` returns one sampled SMILES string
- here, `MolToSmilesEnum(...)` yields the full exact support of that same writer mode

Supported combination:

- omit `rootedAtAtom` or pass `rootedAtAtom=-1` for all-roots behavior
- pass `rootedAtAtom >= 0` for one explicit root
- other negative integer `rootedAtAtom` values are also accepted for RDKit
  compatibility and behave like `-1`, but `-1` is the preferred public
  spelling
- `rootedAtAtom=None` is not supported
- `canonical=False`
- `doRandom=True`

Supported writer flags:

- `isomericSmiles`
- `kekuleSmiles`
- `allBondsExplicit`
- `allHsExplicit`
- `ignoreAtomMapNumbers`

Unsupported flag combinations fail fast with `NotImplementedError`. Other
invalid public inputs can still raise more specific exceptions such as
`IndexError` or `ValueError`. Molecules with multiple disconnected fragments
are supported here, in
`MolToSmilesDecoder(...)`, in `MolToSmilesDeterminizedDecoder(...)`, and in
`MolToSmilesTokenInventory(...)`.

For disconnected molecules, a nonnegative `rootedAtAtom` does not reorder
fragments. It selects the rooted fragment and the local root atom within that
fixed fragment order, but non-rooted fragments can still vary internally.

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
while decoder.prefix != "CC":
    decoder = decoder.next_choices[0].next_state

decoder.prefix       # 'CC'
[choice.text for choice in decoder.next_choices]  # ['(', '(']
```

Available interface:

- `next_choices: tuple[MolToSmilesChoice, ...]`
- `prefix: str`
- `is_terminal: bool`
- `copy() -> MolToSmilesDecoder`

Each `MolToSmilesChoice` has:

- `text: str`
- `next_state`: the same decoder class as the parent choice came from

Two different choices may therefore share the same `text` while leading to
different successor states.

## MolToSmilesDeterminizedDecoder

`MolToSmilesDeterminizedDecoder(mol, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This is the merged alternative to `MolToSmilesDecoder(...)`. It accepts the
same flags and current limits, but it returns at most one next choice per token
text by merging same-text continuations into one successor state.

```python
decoder = grimace.MolToSmilesDeterminizedDecoder(
    mol,
    rootedAtAtom=9,
    isomericSmiles=False,
    canonical=False,
    doRandom=True,
)
decoder = decoder.next_choices[0].next_state  # 'c'
decoder = decoder.next_choices[0].next_state  # 'c1'
decoder = decoder.next_choices[0].next_state  # 'c1('

[choice.text for choice in decoder.next_choices]  # ['C', 'c']
```

Available interface:

- `next_choices: tuple[MolToSmilesChoice, ...]`
- `prefix: str`
- `is_terminal: bool`
- `copy() -> MolToSmilesDeterminizedDecoder`

## MolToSmilesChoice

`MolToSmilesChoice` is the public helper object returned from
`decoder.next_choices`.

Available interface:

- `text: str`
- `next_state`: `MolToSmilesDecoder` or `MolToSmilesDeterminizedDecoder`,
  matching the parent decoder class

## Decoder model

`MolToSmilesDecoder(...)` exposes the same runtime as a stateful decoder:

- `next_choices` returns the allowed next logical choices
- `choice.text` is the emitted SMILES fragment for that choice
- `choice.next_state` is the decoder state after taking that choice
- `prefix` returns the current prefix

The tokens are literal SMILES fragments, not token ids. A token may be one
character or many characters, for example `"C"`, `"[C@H]"`, or `"%12"`.

`prefix` is the literal concatenation of the tokens emitted so far on that
decoder path.

Important semantic point:

- the decoder is online
- it reports the legal next choices for the current emitted prefix
- choosing a choice advances it to the next decoder state
- it does not precompute one fixed full output before you start stepping

Terminology:

- branch-preserving means `next_choices` may contain duplicate `choice.text`
  values when they represent different underlying branches
- determinized frontier means same-text continuations would be merged into one
  combined successor state for that token

Public semantic choice:

- `MolToSmilesDecoder(...)` is branch-preserving
- duplicate same-text choices are therefore meaningful and may appear in
  `next_choices`
- this preserves branch identity instead of hiding distinct continuations
  behind one merged token choice
- two decoder states may therefore share the same `prefix` while exposing
  different `next_choices`

- `MolToSmilesDeterminizedDecoder(...)` is determinized
- same-text continuations are merged into one combined successor state
- this gives at most one public choice per token text
- the merged successor can still represent many underlying branches

So:

- `MolToSmilesEnum(...)` gives exact full support
- `MolToSmilesDecoder(...)` lets you step through that support one token at a time
- `MolToSmilesDeterminizedDecoder(...)` lets you step through the same support
  with same-text branches merged

## MolToSmilesTokenInventory

`MolToSmilesTokenInventory(mol, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This returns the exact sorted tuple of reachable decoder tokens for one
molecule under the same public writer flags.

When `rootedAtAtom < 0`, it unions the exact reachable token inventories
across all roots. When `rootedAtAtom >= 0`, it reports the inventory for that
rooted public runtime. Omitting `rootedAtAtom` means the same thing as passing
`-1`, and `-1` is the preferred public spelling for that all-roots mode. For
disconnected molecules it includes the `"."` separator token when fragment
transitions are reachable under the requested root mode. `rootedAtAtom=None`
is not supported; omit the argument or use `-1` instead.

This is an exact runtime inventory, not a probabilistic distribution and not a
general-purpose tokenizer vocabulary.

## Correctness

Rust is the source of truth for runtime behavior.

Python builds the RDKit bridge and exposes the public API, but runtime
enumeration and next-token decoding are Rust-backed.

The test suite is layered:

1. Rust-native tests for core runtime behavior
2. Python integration tests for the public API
3. Python parity tests for cross-language regression checks
4. RDKit-backed reference checks

## Internal modules

The package also contains internal support code:

- `grimace._runtime`
  Internal RDKit-to-core bridge helpers.
- `grimace._reference`
  Internal pure-Python oracle/reference implementation used by tests and
  fixtures.

These are not part of the supported public API.
