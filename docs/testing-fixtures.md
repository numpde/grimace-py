---
title: Testing fixture guide
---

This guide explains the checked-in test fixtures: what claim each family makes,
where the data comes from, and which tests enforce it. Most users should start
with [Correctness contracts](correctness-contracts.md).

Grimace keeps RDKit-derived test data in JSON fixtures instead of inline Python
constants when the data is part of the correctness evidence.

Fixture claims follow the separation in
[Correctness contracts](correctness-contracts.md): exact RDKit writer-parity
fixtures are string-level RDKit-version claims, while semantic equivalence and
known RDKit quirks should be classified separately.

This guide is not the machine source of truth. Fixture validity is enforced by
typed loader helpers under `tests/helpers/`, contract tests under
`tests/contract/`, and the relevant runtime/parity tests.

## How to read a fixture

Start with the fixture family, then the RDKit version, then the case `source`.

- Family: tells you the kind of claim, such as exact support equality or writer
  output membership.
- Version: pins RDKit-derived claims to one `rdBase.rdkitVersion`.
- Case ID: gives a stable test identifier.
- Source: explains where the evidence came from: upstream RDKit tests, local
  exact-support probes, dataset mining, random-writer observations, or known
  quirks/gaps.
- Expected fields: define the executable claim. Examples include exact support,
  token inventory, deterministic writer output, sampled RDKit confirmation, or
  known rejected members.

If a fixture claim should be continuously enforced, it belongs in code, not
only in documentation.

## RDKit-pinned parity fixtures

These fixtures are keyed by exact `rdBase.rdkitVersion`.  They are correctness
evidence only for that RDKit build.

- `tests/fixtures/rdkit_exact_small_support/`: exact support and
  token-inventory equality for small saturable cases.
- `tests/fixtures/rdkit_serializer_regressions/`: exact Grimace support and
  inventory regressions for serializer edge cases, including optional RDKit
  sampling confirmation.
- `tests/fixtures/rdkit_writer_membership/`: deterministic RDKit writer
  outputs that must be members of Grimace support.  These are not full
  support-equality claims.
- `tests/fixtures/rdkit_rooted_random/`: deterministic rooted outputs from
  RDKit rooted writer tests.
- `tests/fixtures/rdkit_known_quirks/`: neutral, version-pinned observations
  of RDKit behavior that need to stay isolated for later discussion or
  implementation work.  "Known quirk" does not mean Grimace is allowed to
  diverge; it means RDKit does this in the pinned version and the behavior is
  unusual enough to keep separate from ordinary parity fixtures.

Large pinned corpora may use `VERSION/*.json` shards under their fixture root.
Shard names should keep review order stable by source area or serializer
feature.

The pinned parity runner requires checked-in fixtures for the installed RDKit
version before running those claims.

## RDKit compatibility fixtures

These fixtures are not exact-version parity corpora.  They support broader
behavioral checks against the installed RDKit build.

- `tests/fixtures/rdkit_disconnected_sampling/`: disconnected molecule inputs
  for RDKit sampling compatibility checks.
- `tests/fixtures/rdkit_stereo_regressions/`: reusable stereo regression
  members and rejected members shared across reference and public-surface
  tests.

## RDKit upstream source fixtures

- `tests/fixtures/rdkit_upstream_serializer_sources/`: local copies of the
  RDKit serializer source and test files used to audit serializer coverage.
  These files are fixture data, not runtime code.  Each versioned copy includes
  `manifest.json` with upstream commit metadata and SHA-256 digests, plus the
  RDKit BSD-3-Clause license text.
- `tests/fixtures/rdkit_upstream_serializer_coverage/`: parser-generated
  inventory of upstream RDKit serializer test blocks.  The extractor owns
  upstream file, line range, parser kind, matched terms, and snippet hash.
  Reviewed fields map each upstream block to a coverage status and, when
  covered, `grimace_links` pointing at concrete fixture files and case IDs.

Use [RDKit serializer coverage guide](rdkit-serializer-coverage.md) when you
need to understand how upstream RDKit source blocks map to fixture cases.

## Reference dataset fixtures

- `tests/fixtures/reference/`: reference-policy fixtures and generated
  artifacts for the Python reference layer.
- `tests/fixtures/top_100000_CIDs.tsv.gz`: source molecule list used by
  reference dataset loaders.

## Maintenance rule

When adding or changing fixtures:

1. Put RDKit-derived claims under an exact RDKit version.
2. Give every case a stable ID and a clear `source`.
3. Keep expected string lists sorted and unique when the loader requires it.
4. Add typed loader validation for new fields.
5. Add contract tests for new schema rules.
6. Add runtime/parity tests for the actual behavior claim.
7. Link upstream serializer claims through the coverage ledger when the case
   exists to cover an upstream RDKit source block.

Do not use documentation as the machine source of truth. If a fixture family
needs enforcement, put that enforcement in one of these places:

- a typed loader in `tests/helpers/`
- loader contract tests in `tests/contract/`
- the pinned RDKit parity runner when the fixture is exact-version parity
  evidence
- CI, when the claim should be continuously exercised

Documentation should link to the fixture families and explain their intent,
but tests should enforce schemas, version keys, duplicate IDs, sorted expected
outputs, and availability for pinned RDKit parity runs.
