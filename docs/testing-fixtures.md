---
title: Testing fixtures
---

Checked-in fixtures are the executable RDKit evidence behind Grimace's
correctness tests. They live in JSON when the data itself is part of the claim:
support sets, inventories, writer outputs, known gaps, and source snapshots.

## Current coverage

Snapshot for RDKit `2026.03.1`, generated from:

```bash
python scripts/report_correctness_coverage.py
```

| Fixture family | Cases | Claim |
|---|---:|---|
| `rdkit_exact_small_support` | 76 | Exact support and token-inventory equality for small saturable cases. |
| `rdkit_serializer_regressions` | 130 | Exact support and inventory regressions for serializer edge cases. |
| `rdkit_writer_membership` | 56 | Deterministic RDKit writer outputs must be in Grimace support. |
| `rdkit_rooted_random` | 1 | Version-pinned rooted random-writer output. |
| `rdkit_known_stereo_gaps` | 16 | Executable parity debt, outside the passing parity lane. |
| `rdkit_known_quirks` | 1 | Isolated RDKit behavior observation. |

By provenance:

| Source class | Cases |
|---|---:|
| `upstream-rdkit` | 171 |
| `random-writer-observation` | 31 |
| `local-probe` | 28 |
| `dataset-derived` | 33 |
| `known-rdkit-gap` | 16 |
| `rdkit-quirk` | 1 |

The fixture-family counts describe assertion type. The provenance counts
describe where the evidence came from. Serializer-ledger coverage is summarized
in [RDKit serializer coverage](rdkit-serializer-coverage.md).

## How to read a case

Read a fixture case in this order:

1. Fixture family: tells you the assertion strength, such as exact support
   equality or writer-output membership.
2. RDKit version: pins RDKit-derived claims to one `rdBase.rdkitVersion`.
3. Case ID: gives a stable test identifier.
4. `source`: records provenance, such as upstream RDKit tests, local probes,
   dataset mining, random-writer observations, or known gaps.
5. Expected fields: define the executable claim.

Documentation explains intent; loaders and tests enforce it.

## Fixture families

| Path | Role |
|---|---|
| `tests/fixtures/rdkit_exact_small_support/` | Passing exact support and inventory parity. |
| `tests/fixtures/rdkit_serializer_regressions/` | Passing serializer edge-case support and inventory parity. |
| `tests/fixtures/rdkit_writer_membership/` | Passing deterministic RDKit writer-membership parity. |
| `tests/fixtures/rdkit_rooted_random/` | Passing rooted random-writer observations. |
| `tests/fixtures/rdkit_known_stereo_gaps/` | Failing opt-in diagnostics for known stereo gaps. |
| `tests/fixtures/rdkit_known_quirks/` | Isolated RDKit behavior observations. |
| `tests/fixtures/rdkit_disconnected_sampling/` | Compatibility sampling inputs, not exact-version parity claims. |
| `tests/fixtures/rdkit_stereo_regressions/` | Reusable stereo members and rejected members. |
| `tests/fixtures/rdkit_upstream_serializer_sources/` | Local RDKit source snapshots used for serializer audit. |
| `tests/fixtures/rdkit_upstream_serializer_coverage/` | Reviewed map from upstream serializer blocks to Grimace evidence. |

## Promotion rules

Mining output is candidate data only. Promote a mined case only when it has:

- a stable classification
- a clear `source`
- an executable assertion
- evidence that is not a near-duplicate of an existing fixture

Prefer exact support and inventory equality for small cases. Use deterministic
writer membership when exact support is too large. Keep `rdkit_only` cases as
known gaps while they fail. Do not promote `uncertain` mined cases.

## Maintenance checklist

When adding or changing fixtures:

1. Put RDKit-derived claims under an exact RDKit version.
2. Give every case a stable ID and clear `source`.
3. Keep expected string lists sorted and unique when the loader requires it.
4. Add typed loader validation for new fields.
5. Add contract tests for new schema rules.
6. Add runtime or parity tests for the actual behavior claim.
7. Link upstream serializer claims through the coverage ledger when a fixture
   exists to cover an upstream RDKit source block.
