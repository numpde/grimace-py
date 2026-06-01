---
title: API
---

## Public surface

The only supported public Python import name is `grimace`.

This page is a reference. For the supported flag combinations and root
semantics, start with [Runtime](../runtime.md). For terminology, see
[Concepts](../concepts.md).

Current top-level exports:

- `MolToSmilesChoice`
- `MolToSmilesDecoder`
- `MolToSmilesDeterminizedDecoder`
- `MolToSmilesEnum`
- `MolToSmilesDeviation`
- `MolToSmilesTokenInventory`
- `MolToSmilesTokenInventorySuperset`
- `PreparedMol`
- `PrepareMol`
- `SmilesDeviation`

The compiled extension `grimace._core` is required. There is no public runtime
fallback.

## PreparedMol

`PrepareMol(mol, *, isomericSmiles=True, kekuleSmiles=False, allBondsExplicit=False, allHsExplicit=False, ignoreAtomMapNumbers=False)`

`PreparedMol.to_bytes(*, compression=None, dictionary_level=3, level=3)`

`PreparedMol.from_bytes(data)`

Prepares an RDKit molecule once under a fixed writer surface and returns an
opaque `PreparedMol`. See [Prepared molecules](../guides/prepared-mol.md) for
the workflow.

```python
prepared = grimace.PrepareMol(mol, isomericSmiles=False)
payload = prepared.to_bytes()
restored = grimace.PreparedMol.from_bytes(payload)
```

`PreparedMol` is accepted anywhere the public runtime accepts a molecule.

The writer-surface flags passed to `PrepareMol` are baked into the prepared
object. Runtime calls with conflicting writer flags raise `ValueError`.
`rootedAtAtom`, `canonical`, and `doRandom` remain runtime options.

`PreparedMol.to_bytes()` returns a versioned binary payload owned by the Rust
core. `PreparedMol.from_bytes(...)` accepts that payload and reconstructs an
opaque object ready for the runtime.

`PreparedMol.to_bytes(compression="zstd")` writes a zstd frame using the
default shipped dictionary and compression level. `PreparedMol.from_bytes(...)`
detects compressed payloads from the zstd frame and selects the matching
shipped dictionary from the frame dictionary id.

## MolToSmilesEnum

`MolToSmilesEnum(mol, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This yields the complete exact support of Grimace's supported writer language as
whole SMILES strings.

Although the signature mirrors RDKit defaults, the current runtime does not
support those defaults. Use the supported options from [Runtime](../runtime.md).

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

This is the important semantic point:

- in RDKit, `canonical=False, doRandom=True` returns one sampled SMILES string
- here, `MolToSmilesEnum(...)` yields the full exact support of Grimace's supported
  language for that writer mode

## MolToSmilesDecoder

`MolToSmilesDecoder(mol, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This is the incremental next-token API for the same support language as
`MolToSmilesEnum(...)`. It shows the legal next choices for the current emitted
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

State interface:

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

This is the determinized alternative to `MolToSmilesDecoder(...)`. It returns at
most one next choice per token text by merging same-text continuations into one
successor state.

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

State interface:

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

The decoder APIs expose the support language as stateful next-token choices.
For the conceptual model and the difference between branch-preserving and
determinized choices, see [Concepts](../concepts.md).

Both decoder classes expose `prefix`, `next_choices`, `is_terminal`, and
`copy()`.

## MolToSmilesDeviation

`MolToSmilesDeviation(mol, candidate, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This diagnoses the first place where a candidate leaves the molecule's
supported SMILES language under the requested public writer flags. See
[Deviation diagnostics](../guides/deviation.md) for examples.

It returns `None` when the candidate is accepted. Otherwise it returns a
`SmilesDeviation` with:

- `reason`: `"unexpected_text"`, `"unexpected_token"`, or `"incomplete"`
- `char_index`: character offset in the concatenated candidate text
- `token_index`: token offset for sequence candidates, or `None` for string candidates
- `offset_in_token`: offset within the external token for sequence candidates,
  or `None` for string candidates
- `accepted_text`: accepted candidate prefix
- `rejected_text`: remaining candidate text at the deviation
- `legal_next_tokens`: sorted legal next Grimace token texts

`candidate` may be a string or a sequence of external token strings. String
candidates are matched as text. Sequence candidates are atomic: each item must
match one legal Grimace decoder token text.

String input and external token sequence input have different boundary
semantics. The guide shows both cases.

## MolToSmilesTokenInventory

`MolToSmilesTokenInventory(mol, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This returns the exact sorted tuple of reachable decoder tokens for one
molecule under the requested public writer flags.

When `rootedAtAtom < 0`, it unions the exact reachable token inventories
across all roots. When `rootedAtAtom >= 0`, it reports the inventory for that
rooted public runtime. Omitting `rootedAtAtom` means the same thing as passing
`-1`, and `-1` is the preferred public spelling for that all-roots mode. For
disconnected molecules it includes the `"."` separator token when fragment
transitions are reachable under the requested root mode. `rootedAtAtom=None`
is not supported; omit the argument or use `-1` instead.

Use this when you need exact per-molecule coverage for Grimace decoder tokens.

## MolToSmilesTokenInventorySuperset

`MolToSmilesTokenInventorySuperset(mol, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This returns a sorted conservative token inventory for one molecule under the
requested public writer flags.

The main use is fast vocabulary-building and coverage checks over molecular
datasets. See [Token inventories](../guides/token-inventory.md).

For the same molecule and flags, the exact inventory is contained in the
superset inventory.

When `rootedAtAtom < 0`, it unions conservative token inventories across all
roots.
For disconnected molecules it includes the `"."` separator token. `PreparedMol`
inputs are accepted when their writer flags match the requested public options.
