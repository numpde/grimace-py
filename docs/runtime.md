---
title: Runtime requirements
---

The public signatures mirror RDKit flag names and defaults, but the current
runtime intentionally supports only a strict subset.

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

## Writer flags

The supported writer flags are:

- `isomericSmiles`
- `kekuleSmiles`
- `allBondsExplicit`
- `allHsExplicit`
- `ignoreAtomMapNumbers`

The current writer target is RDKit writer parity for the supported regime,
currently validated against `RDKit 2026.03.1`.

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
