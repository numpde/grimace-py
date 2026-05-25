# Retired Reference Artifacts

The old `rdkit_random` reference-artifact tree has been retired.

It stored policy JSON files plus generated RDKit random-sampling snapshots:

- exact sampled sets for small selected slices
- compressed metrics with sampled `distinct_count`
- duplicated copies under both `tests/fixtures/reference/` and
  `grimace._reference._data/reference/`

Those artifacts were not part of the current pinned RDKit `2026.03.1`
correctness evidence. They also encoded an older RDKit policy, duplicated
package data, and made sampled counts look more authoritative than they were.

Useful lessons retained:

- Count evidence is useful, but it should be its own explicit lane.
- A sampled `distinct_count` is not an exact support count unless the case is
  confidence-labeled and the sampling evidence is recorded.
- Count evidence should be version-pinned to the RDKit build used to produce
  it.
- Passing evidence should distinguish exact support, deterministic writer
  membership, saturated random sampling, lower bounds, and known mismatches.
- Development/mining artifacts should not be bundled as package data unless
  runtime code needs them.

The `top_100000_CIDs.tsv.gz` molecule fixture remains under `tests/fixtures/`
for active mining and reference tests. The retired part is the package-data
copy and the generated policy/snapshot store built from it.

The replacement direction is a small, version-keyed support-count fixture lane,
not a generic snapshot/policy artifact store.
