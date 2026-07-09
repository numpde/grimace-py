---
title: Known gaps
---

Known gaps are pinned RDKit writer-parity cases that Grimace does not yet
fully mirror. They are executable parity debt: the fixtures record exact RDKit
outputs for a pinned RDKit version, and the diagnostic runner keeps those
outputs visible until the implementation matches them.

Keep this separate from chemical validity. A known-gap string can be a valid
SMILES string for the intended molecule; the gap is that the current Grimace
writer language does not yet contain the same string that RDKit emits for the
same reviewed writer surface.

## Current cluster

The current known gaps are coupled directional-stereo cases. They involve
slash/backslash bond-direction tokens whose placement depends on more than the
local double bond: ring closures, traversal order, and neighboring directional
carriers can all interact.

That is narrower than "stereo is unsupported." Many stereo cases are already
covered by passing exact-support or writer-membership fixtures. The known-gap
cluster records the remaining RDKit-compatible directional-token placement
work.

## Pinned cases

The checked-in fixture is:

- [`tests/fixtures/rdkit_known_stereo_gaps/2026.03.1.json`](https://github.com/numpde/grimace-py/blob/main/tests/fixtures/rdkit_known_stereo_gaps/2026.03.1.json)

It currently contains 16 RDKit `2026.03.1` cases:

| Group | Cases | Why it is hard |
|---|---:|---|
| Directional ring-closure canonical output | 1 | Ring-closure spelling and slash/backslash orientation are coupled. |
| Manual multi-double-bond stereo outputs | 4 | Neighboring directional carriers interact across more than one double bond. |
| Manual stereo-atom mutation surfaces | 2 | RDKit can expose writer surfaces that depend on explicit stereo-atom assignments. |
| CHEMBL3623347 canonical output | 1 | Decoder-path membership is pinned, but full all-roots support is too large for the normal suite. |
| CHEMBL409450 random-vector outputs | 8 | Deterministic RDKit random-writer outputs exercise the same coupled directional-token family. |

The RDKit-source-tree traceability view is
[RDKit serializer coverage](rdkit-serializer-coverage.html). It maps these
fixtures back to RDKit's own serializer tests and source blocks, including the
RDKit #4582 / #3967 cases and manual bond-stereo wrapper tests.

For a concrete failing parity case, see
[Parity examples](parity-examples.html).

## How diagnostics run

Known gaps are intentionally outside default passing test discovery. Run them
explicitly when working on directional-stereo parity:

```bash
PYTHONPATH=python:. python3 -m unittest tests.run_known_stereo_gaps -q
```

The runner is expected to fail until the pinned gaps are fixed. That is why the
fixtures are not mixed into the passing writer-membership corpus.

## Closing a gap

A known gap is closed only when the matching RDKit output belongs to Grimace's
supported writer language under the same pinned RDKit version and writer
surface.

The usual promotion path is:

1. Implement the missing RDKit-compatible directional-token behavior.
2. Move the case into the strongest passing fixture family that applies:
   exact support when the support is small enough, otherwise writer membership
   or bounded decoder-path membership.
3. Update the serializer ledger from `known-gap` to `covered`.
4. Keep the fixture version-keyed to the RDKit version that supplied the
   expected output.

For the broader correctness vocabulary, see
[Correctness contracts](correctness-contracts.html). For the full fixture
inventory, see [Testing fixtures](testing-fixtures.html).
