# Remaining Stereo Gap Triage After Omitted-Marker Promotion

Date: 2026-05-19
Branch: `stereo-constraint-model`
Baseline commit: `db98278`

## Summary

The omitted deferred-marker-before-atom runtime promotion closed exactly one
pinned RDKit writer gap:

- `github4582_chembl409450_random_vector_seed1_index0`
- previous state: `support_missing`
- current state: `support_present`

The known stereo-gap fixture now has:

- `support_present`: 7
- `support_missing`: 8
- `decoder_path_only`: 1

This is a useful result, but not a general solution for the remaining red
directional-stereo writer gaps.  The promotion is target-independent and now
part of the normal runtime boundary, but it only covers the shape where a
deferred directional marker is immediately followed by atom entry and an
omitted-marker event still passes the support boundary.

## Remaining Support-Missing Cases

All remaining missing cases are still classified as `missing_rdkit_writer_policy`.

| Case | Same-skeleton support | Parse-equivalent current support | Parse-mismatch current support |
| --- | ---: | ---: | ---: |
| `github3967_part2_directional_ring_closure_canonical` | 2 | 0 | 2 |
| `github4582_chembl409450_random_vector_seed1_index2` | 2 | 0 | 2 |
| `github4582_chembl409450_random_vector_seed1_index3` | 2 | 1 | 1 |
| `github4582_chembl409450_random_vector_seed1_index5` | 2 | 0 | 2 |
| `github4582_chembl409450_random_vector_seed1_index6` | 2 | 1 | 1 |
| `github4582_chembl409450_random_vector_seed1_index7` | 2 | 0 | 2 |
| `github4582_chembl409450_random_vector_seed1_index8` | 2 | 0 | 2 |
| `github4582_chembl409450_random_vector_seed1_index9` | 4 | 1 | 3 |

## Boundary Grouping

There are two useful buckets now.

First, five cases have no same-skeleton parse-equivalent Grimace output:

- `github3967_part2_directional_ring_closure_canonical`
- `github4582_chembl409450_random_vector_seed1_index2`
- `github4582_chembl409450_random_vector_seed1_index5`
- `github4582_chembl409450_random_vector_seed1_index7`
- `github4582_chembl409450_random_vector_seed1_index8`

These are stronger evidence that Grimace is still missing a necessary
marker-placement basis shape, not merely an RDKit string preference among
already-semantic-equivalent outputs.

Second, three cases already have at least one same-skeleton parse-equivalent
Grimace output, but not the RDKit writer string:

- `github4582_chembl409450_random_vector_seed1_index3`
- `github4582_chembl409450_random_vector_seed1_index6`
- `github4582_chembl409450_random_vector_seed1_index9`

These are better candidates for isolating RDKit writer ordering/idiosyncrasy
after the semantic support basis is already present.

## Next Slices

1. Re-run target-guided diagnostics for the five no-parse-equivalent cases and
   group their first failure rows by `target_alignment_gap`,
   `deferred_marker_basis_rows`, and marker-event facts.
2. Separately inspect the three parse-equivalent-present cases for writer string
   preference: compare RDKit target marker slots with Grimace same-skeleton
   parse-equivalent marker slots.
3. Keep behavior changes narrow.  The next runtime promotion should name one
   target-independent fact, prove it on one witness, and then reclassify only
   the affected fixture cases.
