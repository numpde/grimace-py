---
title: Current limitations
---

Grimace's current public runtime is exact for its supported writer language,
not for every string RDKit can serialize.

## Runtime options

The supported runtime mode is:

- `canonical=False`
- `doRandom=True`
- `rootedAtAtom=-1` for all roots, or `rootedAtAtom >= 0` for one explicit root

Supported writer-surface flags are listed in
[Runtime requirements](runtime.md).

## RDKit serializer parity

Full RDKit serializer parity is not complete. The current reviewed RDKit
`2026.03.1` serializer ledger has:

- `54 covered`
- `6 known-gap`
- `0 needs-fixture`
- `0 unreviewed`

Known gaps are concentrated in coupled directional double-bond and ring-closure
stereo cases. They are pinned as executable known-gap fixtures and summarized
in [RDKit serializer coverage](rdkit-serializer-coverage.md).

## Install surface

Package metadata declares Python `>=3.11` and `rdkit>=2026.3`.

The exercised release assets are Linux `x86_64` wheels for CPython `3.12` and
`3.13`, plus a source distribution. Other Python versions and non-Linux
platforms are expected source-build paths today.
