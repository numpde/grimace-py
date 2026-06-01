# RDKit Writer Support Counts

This note defines a count-only RDKit writer fixture lane.

## Purpose

`rdkit_writer_support_counts` records the cardinality of RDKit's random-writer
support for selected molecules under a pinned RDKit version and one exact
serializer flag surface.

It is not a replacement for `rdkit_exact_small_support`. When full support
strings are cheap to store and compare, keep the stronger full-support fixture.
This lane is for larger supports where retaining every string is noisy but a
checked support count is still useful.

## Fixture Layout

```text
tests/fixtures/rdkit_writer_support_counts/
  2026.03.1/
    nonisomeric__random.json
    isomeric__random.json
    nonisomeric__random_kekule.json
    nonisomeric__random_all_bonds_explicit.json
    nonisomeric__random_all_hs_explicit.json
```

Each shard owns one complete flag surface:

```json
{
  "rdkit_version": "2026.03.1",
  "flags": {
    "isomericSmiles": false,
    "canonical": false,
    "doRandom": true,
    "kekuleSmiles": false,
    "allBondsExplicit": false,
    "allHsExplicit": false,
    "ignoreAtomMapNumbers": false
  },
  "cases": []
}
```

No flag defaults are inferred. The shard filename must match the flag surface.
Case-level flag overrides are not allowed.

## Evidence

The first supported evidence method is
`rdkit_random_adaptive_saturation`. It is saturation-backed count evidence, not
a mathematical exhaustive proof.

Each case records:

- `support_count`: the count claim
- `evidence.method`
- `evidence.criterion_version`
- `evidence.min_draws`
- `evidence.unseen_mass_threshold`
- `evidence.allowed_missing_variants`
- at least two independent seed runs

Each seed run records:

- `seed`
- `draw_count`
- `support_count`
- `consecutive_draws_without_new_variant`
- `singleton_count`
- `doubleton_count`
- `estimated_unseen_mass`
- `estimated_missing_variants`

Every seed run must independently satisfy the adaptive criterion and agree on
the same `support_count`.

## Adaptive Criterion

For a run with `n` draws, `k` observed variants, singleton count `f1`, and
doubleton count `f2`:

- unseen mass estimate: `f1 / n`
- missing variant estimate:
  - `0` when `f1 == 0`
  - `f1 * f1 / (2 * f2)` when `f2 > 0`
  - unstable otherwise
- no-new-variant patience: `max(10000, 20 * k)`

A run is accepted only when:

- `n >= min_draws`
- `consecutive_draws_without_new_variant >= max(10000, 20 * k)`
- `estimated_unseen_mass <= unseen_mass_threshold`
- `estimated_missing_variants <= allowed_missing_variants`

These fields are recorded so the strength of the count evidence is visible
without re-running RDKit.

## Test Contract

The fixture loader validates schema, flags, filename/flag consistency, evidence
method, criterion version, per-seed saturation, seed uniqueness, and multi-seed
count agreement.

The runtime test checks:

```text
len(grimace.MolToSmilesEnum(...)) == support_count
```

This fixture family has a dedicated runtime test. Keep it separate from
full-support fixtures because it asserts a count, not explicit member strings.

## Dataset Mining Workflow

Mining is two-stage.

First, scan the checked-in molecule fixture for promising candidates:

```bash
python scripts/mine_rdkit_writer_support_count_candidates.py \
  --surface nonisomeric__random \
  --molecule-filter any \
  --limit 1000 \
  --min-support-count 100 \
  --max-support-count 2000 \
  --max-candidates 20 \
  --output notes/support_count_mining/nonisomeric_random_candidates_next.json
```

The miner uses Grimace exact enumeration as a pre-screen. Its output is a
ranked report plus a `generator_input` block. It does not claim anything about
RDKit random-writer saturation.

Second, copy the selected `generator_input` cases into a candidate input file
and run the saturation generator into a review shard:

```bash
python scripts/generate_rdkit_writer_support_counts.py \
  --input notes/support_count_mining/nonisomeric_random_input.json \
  --output notes/support_count_mining/nonisomeric__random_review.json \
  --seed 12345 \
  --seed 54321
```

Only promote cases whose independent RDKit seed runs satisfy the adaptive
criterion and agree on the count. The fixture remains curated; mining reports
are evidence discovery, not evidence acceptance.

Both scripts refuse to overwrite an existing output path unless `--force` is
passed. Use `--force` only when intentionally replacing a local review artifact.

The checked-in `top_100000` molecule fixture has no atom-map labels. Do not add
a dataset-mined `ignore_atom_maps` support-count shard from that source: it
would exercise a flag value on molecules where the flag has no semantic effect.
Mapped-atom behavior belongs in the existing exact-support and writer-flag
fixtures unless a deliberately constructed large mapped case justifies a
count-only shard.
