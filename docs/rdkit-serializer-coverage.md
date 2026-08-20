# RDKit Serializer Coverage

This project keeps a version-pinned ledger of upstream RDKit serializer tests
that were reviewed for relevance to Grimace's current public surface.

The ledger lives at:

- `tests/fixtures/rdkit_upstream_serializer_coverage/2026.03.1.json`

The local RDKit source files audited by that ledger live at:

- `tests/fixtures/rdkit_upstream_serializer_sources/2026.03.1/`

The ledger is keyed to RDKit `2026.03.1`.  Its claims should not be read as
evidence for a different RDKit serializer version unless a new versioned ledger
is generated and reviewed.

## Current Status

Current reviewed counts:

- `54 covered`
- `6 known-gap`
- `209 out-of-scope`
- `0 needs-fixture`
- `0 unreviewed`

The contract test
`tests.contract.test_rdkit_upstream_serializer_coverage` enforces the ledger
schema, checks source snippet hashes and line spans, validates fixture links,
and now fails if any entry remains `unreviewed` or `needs-fixture`.

## Status Meanings

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

## Known Gaps

The current `known-gap` entries are concentrated in RDKit's #4582 and manual
bond-stereo regressions.  They pin RDKit outputs that Grimace should eventually
accept but currently does not:

- GitHub #4582 bulk random double-bond/ring-closure outputs for CHEMBL409450.
- GitHub #4582 continued / #3967 part 2 directional ring-closure output.
- Manual multi-double-bond stereo outputs from `testBondSetStereoDifficultCase`.
- Manual stereo-atom mutation outputs from `testBondSetStereoAtoms`.

These point at the same implementation family: RDKit-equivalent traversal-order
state for coupled directional stereo tokens.

## Maintenance Workflow

When the local RDKit serializer source fixture changes, regenerate the parser
owned fields:

```bash
python scripts/extract_rdkit_serializer_cases.py --write
```

Then review every new `unreviewed` entry and assign one of the stable statuses
above.  Use fixture links rather than prose-only claims whenever the entry is
in scope.

To inspect the ledger:

```bash
python scripts/report_rdkit_serializer_coverage.py
```

To fail explicitly on unfinished triage:

```bash
python scripts/report_rdkit_serializer_coverage.py --fail-untriaged
```

The contract tests enforce the same policy during normal test runs.
