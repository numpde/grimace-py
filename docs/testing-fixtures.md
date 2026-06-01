---
title: Testing fixtures
---

Fixtures are checked-in data that Grimace's correctness tests load. They make a
claim executable when the claim depends on concrete data: exact support sets,
token inventories, RDKit writer outputs, known gaps, or RDKit source snapshots.

A fixture case answers two questions:

- What behavior does the test assert? That is the fixture family.
- Why is this molecule or serializer case in the suite? That is the source
  class.

Source classes include RDKit's own source tree, Grimace local probes,
dataset-derived molecules, random-writer observations, and known-gap
diagnostics. Cases with source class `upstream-rdkit` are also indexed in
[RDKit serializer coverage](rdkit-serializer-coverage.html), which points back
to the reviewed RDKit files and source blocks.

## Current coverage

Snapshot for RDKit `2026.03.1`, generated from:

```bash
python scripts/report_correctness_coverage.py
```

| Fixture family | Cases | What the tests check |
|---|---:|---|
| `rdkit_exact_small_support` | 76 | Exact support and token-inventory equality for small saturable cases. |
| `rdkit_serializer_regressions` | 130 | Exact support and inventory regressions for serializer edge cases. |
| `rdkit_writer_membership` | 56 | Deterministic RDKit writer outputs must be in Grimace support. |
| `rdkit_writer_support_counts` | 14 | Count-only RDKit random-writer support evidence. |
| `rdkit_rooted_random` | 1 | Version-pinned rooted random-writer output. |
| `rdkit_known_stereo_gaps` | 16 | Executable parity debt, outside the passing parity lane. |
| `rdkit_known_quirks` | 1 | Isolated RDKit behavior observation. |

By source class:

| Source class | Cases | Meaning |
|---|---:|---|
| `upstream-rdkit` | 171 | Case came from RDKit's own tests or source blocks. |
| `random-writer-observation` | 31 | Case was observed from RDKit's random writer. |
| `local-probe` | 30 | Case was designed in Grimace to probe a specific behavior. |
| `dataset-derived` | 45 | Case came from a molecule dataset and was promoted after review. |
| `known-rdkit-gap` | 16 | Case records a known current parity gap. |
| `rdkit-quirk` | 1 | Case records isolated RDKit behavior worth pinning. |

The first table says what the tests assert. The second says why cases were
included. A source class can feed more than one fixture family.

## How to read a case

Read a fixture case in this order:

1. Fixture family: tells you the assertion strength, such as exact support
   equality or writer-output membership.
2. RDKit version: pins RDKit-derived claims to one `rdBase.rdkitVersion`.
3. Case ID: gives a stable test identifier.
4. `source`: records provenance, such as RDKit source-tree tests, local probes,
   dataset mining, random-writer observations, or known gaps.
5. Expected fields: define the executable claim.

Documentation explains intent; loaders and tests enforce it.

## Fixture paths

| Path | Contents |
|---|---|
| `tests/fixtures/rdkit_exact_small_support/` | Passing exact support and inventory parity. |
| `tests/fixtures/rdkit_serializer_regressions/` | Passing serializer edge-case support and inventory parity. |
| `tests/fixtures/rdkit_writer_membership/` | Passing deterministic RDKit writer-membership parity. |
| `tests/fixtures/rdkit_writer_support_counts/` | Count-only RDKit random-writer support evidence. |
| `tests/fixtures/rdkit_rooted_random/` | Passing rooted random-writer observations. |
| `tests/fixtures/rdkit_known_stereo_gaps/` | Failing opt-in diagnostics for known stereo gaps. |
| `tests/fixtures/rdkit_known_quirks/` | Isolated RDKit behavior observations. |
| `tests/fixtures/rdkit_disconnected_sampling/` | Compatibility sampling inputs, not exact-version parity claims. |
| `tests/fixtures/rdkit_stereo_regressions/` | Reusable stereo members and rejected members. |
| `tests/fixtures/rdkit_upstream_serializer_sources/` | Local RDKit source snapshots used for serializer audit. |
| `tests/fixtures/rdkit_upstream_serializer_coverage/` | Reviewed map from RDKit source-tree serializer blocks to Grimace evidence. |

## Promotion rules

Mining output is candidate data only. Promote a mined case only when it has:

- a stable classification
- a clear `source`
- an executable assertion
- evidence that is not a near-duplicate of an existing fixture

Prefer exact support and inventory equality for small cases. Use deterministic
writer membership when exact support is too large. Keep `rdkit_only` cases as
known gaps while they fail. Do not promote `uncertain` mined cases.

Use writer support-count fixtures only when storing every support string is not
worth the noise. These fixtures retain the count plus the RDKit sampling
evidence: per-seed draw counts, no-new-variant streaks, singleton/doubleton
counts, and unseen-mass estimates. They are saturation-backed evidence, not a
mathematical exhaustive proof.

## Maintenance checklist

When adding or changing fixtures:

1. Put RDKit-derived claims under an exact RDKit version.
2. Give every case a stable ID and clear `source`.
3. Keep expected string lists sorted and unique when the loader requires it.
4. Add typed loader validation for new fields.
5. Add contract tests for new schema rules.
6. Add runtime or parity tests for the actual behavior claim.
7. Link RDKit source-tree serializer claims through the coverage ledger when a
   fixture exists to cover the RDKit source block.
