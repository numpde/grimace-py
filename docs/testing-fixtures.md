# Testing Fixtures

Grimace keeps RDKit-derived test data in JSON fixtures instead of inline
Python constants when the data is part of the correctness evidence.

Fixture claims follow the separation in
[Correctness contracts](correctness-contracts.md): exact RDKit writer-parity
fixtures are string-level RDKit-version claims, while semantic equivalence and
known RDKit quirks should be classified separately.

The source of truth for fixture validity is not this page.  Fixture validity is
enforced by typed loader helpers under `tests/helpers/` and their contract
tests under `tests/contract/`.  This page only explains why the fixture
families exist and what claim each family supports.

## RDKit-Pinned Parity Fixtures

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

## RDKit Compatibility Fixtures

These fixtures are not exact-version parity corpora.  They support broader
behavioral checks against the installed RDKit build.

- `tests/fixtures/rdkit_disconnected_sampling/`: disconnected molecule inputs
  for RDKit sampling compatibility checks.
- `tests/fixtures/rdkit_stereo_regressions/`: reusable stereo regression
  members and rejected members shared across reference and public-surface
  tests.

## RDKit Upstream Source Fixtures

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
  See [rdkit-serializer-coverage.md](rdkit-serializer-coverage.md) for the
  current reviewed counts and status policy.

## Reference Dataset Fixtures

- `tests/fixtures/reference/`: reference-policy fixtures and generated
  artifacts for the Python reference layer.
- `tests/fixtures/top_100000_CIDs.tsv.gz`: source molecule list used by
  reference dataset loaders.

## Maintenance Rule

Do not use a documentation index as the machine source of truth.  If a fixture
family needs enforcement, put that enforcement in one of these places:

- a typed loader in `tests/helpers/`
- loader contract tests in `tests/contract/`
- the pinned RDKit parity runner when the fixture is exact-version parity
  evidence
- CI, when the claim should be continuously exercised

Documentation should link to the fixture families and explain their intent,
but tests should enforce schemas, version keys, duplicate IDs, sorted expected
outputs, and availability for pinned RDKit parity runs.
