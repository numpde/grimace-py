---
title: Runtime requirements
---

This page is the source of truth for supported public runtime options.

The public signatures mirror RDKit flag names and defaults, but the current
runtime supports a strict subset. A call that looks like RDKit's default
`MolToSmiles(...)` call is not the supported Grimace path.

Pass these options explicitly:

- `canonical=False`
- `doRandom=True`
- omit `rootedAtAtom` or pass `rootedAtAtom=-1` for all-roots behavior
- pass `rootedAtAtom >= 0` for one explicit root

Other negative integer `rootedAtAtom` values are accepted for RDKit
compatibility and behave like `-1`, but `-1` is the preferred public spelling.
`rootedAtAtom=None` is not supported.

Unsupported flag combinations fail fast with `NotImplementedError`. Other
invalid public inputs can still raise more specific exceptions such as
`IndexError` or `ValueError`.

Use this copy-paste base while experimenting:

```python
FLAGS = dict(canonical=False, doRandom=True)
```

## Writer flags

The supported writer flags are:

- `isomericSmiles`
- `kekuleSmiles`
- `allBondsExplicit`
- `allHsExplicit`
- `ignoreAtomMapNumbers`

The current writer target is RDKit writer parity for the supported regime,
currently validated against `RDKit 2026.03.1`.

For the difference between writer parity and chemical equivalence, see
[Correctness contracts](correctness-contracts.md).

## Rooting

- `rootedAtAtom=<idx>` uses one explicit starting atom for connected molecules.
- `rootedAtAtom=-1` for `MolToSmilesEnum(...)` returns the exact support
  unioned across all root atoms.
- `rootedAtAtom=-1` for the decoder classes starts from one merged all-roots
  decoder state.
- `rootedAtAtom=-1` for `MolToSmilesTokenInventory(...)` and
  `MolToSmilesTokenInventorySuperset(...)` returns the token inventory unioned
  across all root atoms.

For disconnected molecules, fragment order is preserved. A nonnegative
`rootedAtAtom` selects the rooted fragment and its local root atom within that
fixed fragment order, but non-rooted fragments can still vary internally.

## Install matrix

Package metadata declares Python `>=3.11` and `rdkit>=2026.3`.

The currently exercised release matrix publishes Linux `x86_64` wheels for
CPython `3.12` and `3.13`, plus a source distribution. Other Python versions
and non-Linux platforms are expected source-build paths today.
