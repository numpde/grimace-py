---
title: Runtime
---

This page is the public contract for flags, roots, and install targets. Grimace
mirrors RDKit option names, but its runtime operates on the random-writer
support language, not RDKit's default canonical writer call.

For runtime operations that mirror RDKit's `MolToSmiles(...)` options, start
with:

```python
FLAGS = dict(canonical=False, doRandom=True)
```

and either omit `rootedAtAtom` or pass `rootedAtAtom=-1` for all roots.

## Required runtime mode

| Option | Supported value | Meaning |
|---|---|---|
| `canonical` | `False` | Use the non-canonical writer surface. |
| `doRandom` | `True` | Use RDKit's random-writer mode as the language to enumerate, decode, or sample. |
| `rootedAtAtom` | omit or `-1` | Use all valid roots. |
| `rootedAtAtom` | nonnegative atom index | Use one explicit root. |

Other negative integer `rootedAtAtom` values are accepted for RDKit
compatibility and behave like `-1`, but `-1` is the preferred public spelling.
`rootedAtAtom=None` is not supported.

Although public signatures mirror RDKit defaults, those defaults are not the
supported Grimace runtime mode. Unsupported flag combinations fail fast with
`NotImplementedError`. Other invalid public inputs can still raise more
specific exceptions such as `IndexError` or `ValueError`.

## Writer flags

Writer flags change how valid strings are rendered. They are part of the writer
surface and are supported in any combination:

- `isomericSmiles`
- `kekuleSmiles`
- `allBondsExplicit`
- `allHsExplicit`
- `ignoreAtomMapNumbers`

If you use `PrepareMol(...)`, these flags are baked into the prepared molecule.
Later runtime calls must use matching writer flags. `canonical`, `doRandom`,
and `rootedAtAtom` remain runtime options.

For the difference between writer parity and chemical equivalence, see
[Correctness contracts](correctness-contracts.html).

For current runtime scope and known gaps, see
[Limitations](current-limitations.html).

## All-roots behavior

`rootedAtAtom=-1` means all valid roots, but each public API exposes that
all-roots state in the shape natural for the operation:

| API | All-roots behavior |
|---|---|
| `MolToSmilesEnum(...)` | Yields the exact union of complete strings across all root atoms. |
| `MolToSmilesDecoder(...)` | Starts from one branch-preserving all-roots decoder state. |
| `MolToSmilesDeterminizedDecoder(...)` | Starts from one all-roots decoder state with same-text choices merged. |
| `MolToSmilesSample(...)` | Draws one seeded walk from the requested all-roots decoder view. |
| `MolToSmilesDeviation(...)` | Checks the candidate against the all-roots supported language. |
| `MolToSmilesTokenInventory(...)` | Returns the exact union of reachable decoder tokens across all roots. |
| `MolToSmilesTokenInventorySuperset(...)` | Returns a conservative token inventory unioned across all roots. |

Use a nonnegative `rootedAtAtom` only when you need one explicit traversal
start.

## Disconnected molecules

For disconnected molecules, fragment order is preserved. A nonnegative
`rootedAtAtom` selects the rooted fragment and its local root atom within that
fixed fragment order, but non-rooted fragments can still vary internally.

## Install matrix

Package metadata declares Python `>=3.11` and `rdkit>=2026.3`.

The currently exercised release matrix publishes Linux `x86_64` wheels for
CPython `3.12` and `3.13`, plus a source distribution. Other Python versions
and non-Linux platforms are expected source-build paths today.
