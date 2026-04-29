# Test Fixtures

This directory stores test data that should not live as inline Python
constants.  RDKit-derived fixtures are separated by what they prove.

## RDKit-Pinned Parity Fixtures

These fixtures are keyed by exact `rdBase.rdkitVersion`.  They are correctness
evidence only for that RDKit build.

- `rdkit_exact_small_support/`: exact support and token-inventory equality for
  small saturable cases.
- `rdkit_serializer_regressions/`: exact Grimace support and inventory
  regressions for serializer edge cases, including optional RDKit sampling
  confirmation.
- `rdkit_writer_membership/`: deterministic RDKit writer outputs that must be
  members of Grimace support.  These are not full support-equality claims.
- `rdkit_rooted_random/`: deterministic rooted outputs from RDKit rooted writer
  tests.

Large pinned corpora may use `VERSION/*.json` shards under their fixture root.
Shard names should keep review order stable by source area or serializer
feature.

## RDKit Compatibility Fixtures

These fixtures are not exact-version parity corpora.  They support broader
behavioral checks against the installed RDKit build.

- `rdkit_disconnected_sampling/`: disconnected molecule inputs for RDKit
  sampling compatibility checks.
- `rdkit_stereo_regressions/`: reusable stereo regression members and rejected
  members shared across reference and public-surface tests.

## Reference Dataset Fixtures

- `reference/`: reference-policy fixtures and generated artifacts for the
  Python reference layer.
- `top_100000_CIDs.tsv.gz`: source molecule list used by reference dataset
  loaders.
