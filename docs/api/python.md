---
title: API
---

## Contents

- [Public surface](#public-surface)
- [PrepareMol and PreparedMol](#preparemol-and-preparedmol)
- [MolToSmilesEnum](#moltosmilesenum)
- [Decoder model](#decoder-model)
- [MolToSmilesDecoder](#moltosmilesdecoder)
- [MolToSmilesDeterminizedDecoder](#moltosmilesdeterminizeddecoder)
- [MolToSmilesChoice](#moltosmileschoice)
- [MolToSmilesSample](#moltosmilessample)
- [MolToSmilesDeviation](#moltosmilesdeviation)
- [MolToSmilesTokenInventory](#moltosmilestokeninventory)
- [MolToSmilesTokenInventorySuperset](#moltosmilestokeninventorysuperset)

## Public surface

Import the package as `grimace` and use the names documented on this page.
Underscore-prefixed modules are internal. The compiled extension
`grimace._core` is required; there is no Python fallback runtime.

For supported flag combinations and root semantics, start with
[Runtime](../runtime.html). For terminology, see [Concepts](../concepts.html).

Examples below use:

```python
from rdkit import Chem
import grimace

mol = Chem.MolFromSmiles("CCO")
```

## PrepareMol and PreparedMol

`PrepareMol(mol, *, isomericSmiles=True, kekuleSmiles=False, allBondsExplicit=False, allHsExplicit=False, ignoreAtomMapNumbers=False)`

`PreparedMol.to_bytes(*, compression=None, dictionary_level=3, level=3)`

`PreparedMol.from_bytes(data)`

Prepares an RDKit molecule once under a fixed writer surface and returns an
opaque `PreparedMol`. See [Prepared molecules](../guides/prepared-mol.html) for
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
shipped dictionary from the frame dictionary id. The default is
`dictionary_level=3, level=3`.

## MolToSmilesEnum

`MolToSmilesEnum(mol, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This yields the complete exact support of Grimace's supported writer language as
whole SMILES strings.

Although the signature mirrors RDKit defaults, the current runtime does not
support those defaults. Use the supported options from [Runtime](../runtime.html).

When `rootedAtAtom < 0`, the result is the exact union across all valid roots
for the requested writer flags. `rootedAtAtom=-1` is the preferred public
spelling for that all-roots mode. `rootedAtAtom=None` is not supported; omit
the argument or use `-1` instead.

Set semantics are the contract here. `MolToSmilesEnum(...)` yields the exact
support, but callers should not rely on the yielded iteration order as a
stable public ordering guarantee.

```python
outputs = sorted(
    grimace.MolToSmilesEnum(
        mol,
        rootedAtAtom=-1,
        isomericSmiles=False,
        canonical=False,
        doRandom=True,
    )
)
outputs  # ['C(C)O', 'C(O)C', 'CCO', 'OCC']
```

This is the important semantic point:

- in RDKit, `canonical=False, doRandom=True` returns one sampled SMILES string
- here, `MolToSmilesEnum(...)` yields the full exact support of Grimace's supported
  language for that writer mode

## Decoder model

The decoder APIs expose the same support language as stateful next-token
choices.

Both decoder classes expose:

- `next_choices: tuple[MolToSmilesChoice, ...]`
- `choices() -> tuple[MolToSmilesChoice, ...]`
- `prefix: str`
- `is_terminal: bool`
- `copy()`

`choices()` returns the same cached tuple as `next_choices`.
`MolToSmilesChoice.next_state` is cached after first access.

Use `MolToSmilesDecoder(...)` when each branch-preserving writer choice matters.
Use `MolToSmilesDeterminizedDecoder(...)` when you want at most one visible
choice per token text. See [Concepts](../concepts.html) for the branch-preserving
and determinized model.

## MolToSmilesDecoder

`MolToSmilesDecoder(mol, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This is the incremental next-token API for the same support language as
`MolToSmilesEnum(...)`. It shows the legal next choices for the current emitted
prefix, and each choice exposes the next decoder state.

```python
decoder = grimace.MolToSmilesDecoder(
    mol,
    rootedAtAtom=-1,
    isomericSmiles=False,
    canonical=False,
    doRandom=True,
)
[(choice.text, choice.branch_count) for choice in decoder.next_choices]
# [('C', 1), ('C', 1), ('O', 1)]
```

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
    rootedAtAtom=-1,
    isomericSmiles=False,
    canonical=False,
    doRandom=True,
)
[(choice.text, choice.branch_count) for choice in decoder.next_choices]
# [('C', 2), ('O', 1)]
```

## MolToSmilesChoice

`MolToSmilesChoice` is the public helper object returned from
`decoder.next_choices`.

Choice attributes:

- `text: str`
- `branch_count: int`
- `next_state`: `MolToSmilesDecoder` or `MolToSmilesDeterminizedDecoder`,
  matching the parent decoder class

`branch_count` is local to the current prefix. For `MolToSmilesDecoder`, it is
`1`. For `MolToSmilesDeterminizedDecoder`, it is the number of
branch-preserving choices hidden behind that exposed token. It is not final
support size, probability mass, or RDKit random-writer frequency.

## MolToSmilesSample

`MolToSmilesSample(mol, *, seed, decoder_view="determinized", sampling_mode="uniform_token", isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This draws one complete Grimace-supported token path and records the visible
token choices at each step. It is useful when you need a legal sampled SMILES
string together with the next-token context seen along that path.

`seed` is required and must be an unsigned 64-bit integer. The seed is a
Grimace sampler seed; it does not reproduce RDKit random-writer ordering.
No uniformity over finished SMILES strings is implied.

Accepted `decoder_view`/`sampling_mode` pairs are:

- `"determinized"` / `"uniform_token"`: sample uniformly from unique visible
  next-token choices
- `"determinized"` / `"branch_multiplicity"`: sample visible next-token choices
  weighted by their hidden branch count
- `"branch_preserving"` / `"branch_preserving"`: sample uniformly from
  branch-preserving choices, then report the selected visible token bucket

The result is a `SmilesSample`:

- `tokens: tuple[str, ...]`
- `smiles: str`
- `decoder_view: str`
- `sampling_mode: str`
- `steps: tuple[SmilesSampleStep, ...]`

Each `SmilesSampleStep` has:

- `choice_tokens: tuple[str, ...]`
- `choice_branch_counts: tuple[int, ...]`
- `selected_index: int`
- `selected_token: str`

`choice_tokens` are unique visible token texts for the current prefix.
`choice_branch_counts` are the hidden branch counts behind those visible
tokens. `selected_index` points into `choice_tokens`, and
`selected_token == choice_tokens[selected_index]`.

```python
sample = grimace.MolToSmilesSample(
    mol,
    seed=17,
    rootedAtAtom=-1,
    isomericSmiles=False,
    canonical=False,
    doRandom=True,
)

sample.smiles  # 'OCC'
sample.tokens  # ('O', 'C', 'C')
sample.steps[0].choice_tokens  # ('C', 'O')
sample.steps[0].choice_branch_counts  # (2, 1)
sample.steps[0].selected_token  # 'O'
```

## MolToSmilesDeviation

`MolToSmilesDeviation(mol, candidate, *, isomericSmiles=True, kekuleSmiles=False, rootedAtAtom=-1, canonical=True, allBondsExplicit=False, allHsExplicit=False, doRandom=False, ignoreAtomMapNumbers=False)`

This diagnoses the first place where a candidate leaves the molecule's
supported SMILES language under the requested public writer flags. See
[Deviation diagnostics](../guides/deviation.html) for examples.

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
datasets. See [Token inventories](../guides/token-inventory.html).

For the same molecule and flags, the exact inventory is contained in the
superset inventory.

When `rootedAtAtom < 0`, it unions conservative token inventories across all
roots.
For disconnected molecules it includes the `"."` separator token. `PreparedMol`
inputs are accepted when their writer flags match the requested public options.
