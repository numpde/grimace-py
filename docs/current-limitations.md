---
title: Limitations
---

Grimace's current public runtime is exact inside its supported writer language.
That is narrower than “every valid SMILES string for the molecule” and narrower
than full RDKit serializer parity.

## Runtime options

The supported runtime mode is:

- `canonical=False`
- `doRandom=True`
- `rootedAtAtom=-1` for all roots, or `rootedAtAtom >= 0` for one explicit root

Supported writer-surface flags are listed in [Runtime](runtime.html). The
writer-parity contract is explained in
[Correctness contracts](correctness-contracts.html).

## RDKit serializer parity

Full RDKit serializer parity is not complete. The current reviewed RDKit
`2026.03.1` serializer ledger is checked in and summarized by:

```bash
python scripts/report_rdkit_serializer_coverage.py
```

Known gaps are concentrated in coupled directional double-bond and ring-closure
stereo cases. They are pinned as executable known-gap fixtures and summarized
in [RDKit serializer coverage](rdkit-serializer-coverage.html).

## Install surface

Package metadata declares Python `>=3.11` and `rdkit>=2026.3`.

The exercised release assets are Linux `x86_64` wheels for CPython `3.12` and
`3.13`, plus a source distribution. Other Python versions and non-Linux
platforms are expected source-build paths today.
