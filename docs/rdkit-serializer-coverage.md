---
title: RDKit serializer coverage guide
---

This guide explains the RDKit serializer audit trail: which upstream RDKit
serializer tests were reviewed, which ones matter for Grimace's public surface,
and where the corresponding Grimace evidence lives.

The coverage ledger lives at:

- `tests/fixtures/rdkit_upstream_serializer_coverage/2026.03.1.json`

The local RDKit source files audited by that ledger live at:

- `tests/fixtures/rdkit_upstream_serializer_sources/2026.03.1/`

The ledger is keyed to RDKit `2026.03.1`.  Its claims should not be read as
evidence for a different RDKit serializer version unless a new versioned ledger
is generated and reviewed.

## How to read it

There are three layers:

1. RDKit source snapshots: local copies of the RDKit serializer source and test
   files, with upstream commit metadata and SHA-256 digests.
2. Coverage ledger: parser-generated entries for serializer-related upstream
   blocks, plus human-reviewed status and notes.
3. Grimace fixtures: executable JSON cases linked from the ledger when an
   upstream claim is covered or intentionally tracked as a known gap.

The ledger is not itself the parity test.  It is the map from an upstream RDKit
claim to the fixture cases that enforce the claim.

Each in-scope ledger entry should answer four questions:

- What exact RDKit source block is being discussed?
- Does that source block map to Grimace's current public writer surface?
- If yes, which fixture case IDs enforce it?
- If no, why is it out of scope or still a known gap?

Use `grimace_links` for executable evidence.  Avoid prose-only coverage claims
when a claim can be represented by a fixture case.

## Where fixture cases come from

RDKit-parity cases are not all copied from one place.  The main sources are:

- Upstream RDKit serializer tests: cases derived from RDKit `SmilesWrite.cpp`,
  `catch_tests.cpp`, `rough_test.py`, Java wrapper SMILES tests, and related
  serializer checks copied into the source snapshot.
- Local exact-support probes: small molecules chosen so Grimace can assert
  complete support and token-inventory equality, often guided by specific RDKit
  writer branches.
- Dataset-derived regressions: cases mined from the bundled molecule fixture
  with `scripts/mine_rdkit_regressions.py`.
- RDKit random-writer observations: version-pinned random/rooted outputs that
  Grimace must include in support.
- Known quirks and known gaps: version-pinned RDKit observations kept separate
  from ordinary passing parity claims.

Every pinned fixture case should carry a `source` string that states which of
those paths produced the evidence.  The source string is review context; the
loader and parity tests are the enforcement.

## Current status

Generate the current reviewed counts from the checked-in ledger:

```bash
python scripts/report_rdkit_serializer_coverage.py
```

For the broader fixture-family and source-class summary, run:

```bash
python scripts/report_correctness_coverage.py
```

The contract test
`tests.contract.test_rdkit_upstream_serializer_coverage` enforces the ledger
schema, checks source snippet hashes and line spans, validates fixture links,
and now fails if any entry remains `unreviewed` or `needs-fixture`.

## Status meanings

`covered` means the upstream serializer claim has corresponding Grimace
correctness evidence.  That evidence may be exact support equality, token
inventory equality, deterministic RDKit writer output membership, or a bounded
decoder-path membership check when full support materialization is too large.
The concrete claim is stated in the entry's `notes` and `grimace_links`.

`known-gap` means the upstream serializer claim is relevant and has executable
pinned fixture coverage, but at least one parity assertion intentionally fails
against the current implementation.  These are not ignored observations.  They
are red tests for work that remains, primarily coupled directional double-bond
and ring-closure stereo parity.

`out-of-scope` means the upstream test does not map to the current Grimace
surface.  Common examples are CXSMILES extension serialization, wrapper API
smoke tests, canonical-ranking behavior outside the supported
`canonical=False, doRandom=True` regime, and internal RDKit helper APIs that
Grimace does not expose.

`needs-fixture` is a temporary triage state only.  It means the entry is
relevant but has not yet been mapped to a fixture or deliberately classified.
The current ledger has no `needs-fixture` entries; adding one should be treated
as unfinished work.

`unreviewed` is also temporary.  It is used only when regenerated extractor
output introduces a new upstream block whose reviewed fields have not been
assigned yet.

## Known gaps

The current `known-gap` entries are concentrated in RDKit's #4582 and manual
bond-stereo regressions.  They pin RDKit outputs that Grimace should eventually
accept but currently does not:

- GitHub #4582 bulk random double-bond/ring-closure outputs for CHEMBL409450.
- GitHub #4582 continued / #3967 part 2 directional ring-closure output.
- Manual multi-double-bond stereo outputs from `testBondSetStereoDifficultCase`.
- Manual stereo-atom mutation outputs from `testBondSetStereoAtoms`.

These point at the same implementation family: RDKit-equivalent traversal-order
state for coupled directional stereo tokens.

## Maintenance workflow

When updating RDKit serializer coverage:

1. Refresh or add the local RDKit source snapshot and manifest.
2. Regenerate parser-owned coverage fields:

```bash
python scripts/extract_rdkit_serializer_cases.py --write
```

3. Review every new `unreviewed` entry and assign one of the stable statuses
   above.
4. Add or link executable fixtures for every in-scope claim.
5. Run the report and contract tests before treating the audit as complete.

To inspect the serializer ledger:

```bash
python scripts/report_rdkit_serializer_coverage.py
```

To summarize pinned RDKit correctness evidence across fixture families, source
classes, and serializer-ledger statuses:

```bash
python scripts/report_correctness_coverage.py
```

To fail explicitly on unfinished triage:

```bash
python scripts/report_rdkit_serializer_coverage.py --fail-untriaged
```

The contract tests enforce the same policy during normal test runs.
