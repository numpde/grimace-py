# Reference Artifacts

Reference outputs are organized by policy kind, human-readable policy name, and a
content-derived policy digest.

Layout:

- `rdkit_random/branches/<branch_family>/policies/<policy_name>.json`
- `rdkit_random/branches/<branch_family>/snapshots/<policy_name>/<policy_digest>/...`

If any tracked policy field changes, the digest changes and outputs are written to
a new snapshot directory.

Policies may also constrain the input surface via `input_source.filters`, for
example `connected_only` or `stereochemistry=forbid`.
