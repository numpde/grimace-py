# Remaining Stereo Gap Triage After Marker-Boundary Promotions

Date: 2026-05-19
Branch: `stereo-constraint-model`
Baseline commits: `db98278`, graph-marker quotient slice

## Summary

Two target-independent runtime promotions have closed pinned RDKit writer gaps:

- `github4582_chembl409450_random_vector_seed1_index0`
  - mechanism: omitted deferred-marker-before-atom path
  - previous state: `support_missing`
  - current state: `support_present`
- `github3967_part2_directional_ring_closure_canonical`
  - mechanism: graph-marker-equation-backed writer slot quotient
  - previous state: `support_missing`
  - current state: `support_present`

The known stereo-gap fixture now has:

- `support_present`: 8
- `support_missing`: 7
- `decoder_path_only`: 1

This is a useful result, but not a general solution for the remaining red
directional-stereo writer gaps.  Both promotions are target-independent and now
part of the normal runtime boundary, but each is intentionally narrow:

- omitted-marker promotion covers the shape where a deferred directional marker
  is immediately followed by atom entry and an omitted-marker event still passes
  the support boundary
- graph-marker quotient promotion covers the shape where a candidate marker's
  local token phase conflicts with an already-forced selected-carrier phase,
  while the complete component graph-marker equation accepts the emitted marker
  slots and covers every side in that component

## Remaining Support-Missing Cases

All remaining missing cases are still classified as `missing_rdkit_writer_policy`.

| Case | Same-skeleton support | Parse-equivalent current support | Parse-mismatch current support |
| --- | ---: | ---: | ---: |
| `github4582_chembl409450_random_vector_seed1_index2` | 2 | 0 | 2 |
| `github4582_chembl409450_random_vector_seed1_index3` | 2 | 1 | 1 |
| `github4582_chembl409450_random_vector_seed1_index5` | 2 | 0 | 2 |
| `github4582_chembl409450_random_vector_seed1_index6` | 2 | 1 | 1 |
| `github4582_chembl409450_random_vector_seed1_index7` | 2 | 0 | 2 |
| `github4582_chembl409450_random_vector_seed1_index8` | 2 | 0 | 2 |
| `github4582_chembl409450_random_vector_seed1_index9` | 4 | 1 | 3 |

## Boundary Grouping

There are two useful remaining buckets now.

First, four cases have no same-skeleton parse-equivalent Grimace output:

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

1. Re-run target-guided diagnostics for the four no-parse-equivalent cases and
   group their first failure rows by `target_alignment_gap`,
   `deferred_marker_basis_rows`, and marker-event facts.
2. Separately inspect the three parse-equivalent-present cases for writer string
   preference: compare RDKit target marker slots with Grimace same-skeleton
   parse-equivalent marker slots.
3. Keep behavior changes narrow.  The next runtime promotion should name one
  target-independent fact, prove it on one witness, and then reclassify only
  the affected fixture cases.

## Historical Target-Guided Scan Of No-Parse-Equivalent Cases

The original five no-parse-equivalent cases were scanned with
`_stereo_target_guided_marker_basis_diagnostics(..., root_idx=-1,
max_steps=5000)`.

### github3967 Terminal Marker Basis

This was the cleanest next target and is now closed by the graph-marker
equation quotient:

- case: `github3967_part2_directional_ring_closure_canonical`
- root: `0`
- longest relevant prefix: `C1=CC/C=C2\C3=C`
- target remainder: `\CC=CC=CC3C2C=C1`
- previous next token support: `/`
- RDKit target token at that boundary: `\`
- `target_alignment_gap`: `no_successor_prefix_matches_target`
- deferred marker rows: `/` attempt leaves one row but graph-marker equations
  reject; `\` attempt has graph-marker equations accepting but zero rows after
  marker events

The implemented runtime path keeps selected-carrier token-phase facts as the
normal constraint, but admits a competing deferred marker only when the full
component graph-marker equation covers every side and accepts the emitted
marker slots.  Target-guided replay now reaches the full RDKit target.

### CHEMBL No-Parse-Equivalent Cases

The four CHEMBL no-parse-equivalent cases share an early root-`3` failure:

- cases: seed-1 indices `2`, `5`, `7`, and `8`
- root: `3`
- prefix: `N1`
- target remainder starts with atom text `C...`
- current next token support: `/`, `\`
- `target_alignment_gap`: `target_atom_before_directional_marker_successor`
- deferred marker rows: two candidate rows; stored `/` and stored `\` each
  preserve twelve rows after marker events in one attempt, but graph-marker
  equations reject; flipped attempts go to zero rows

The same cases also have longer-prefix failures on root `11`, but those are
not yet the best next target:

- indices `2` and `5`: prefix `N1C(/C(c2c1cc`, target wants `c`, support wants
  `(`
- index `7`: prefix `N1C(/C(=C2/`, target wants `C`, support wants `N`
- index `8`: prefix `N1C(=O)/C(c2c1cc(Br)cc2)=C1/`, target wants `C`, support
  wants `N`

Those longer-prefix failures look more like traversal/ring-order choices after
some writer policy has already diverged.  The CHEMBL root-`3` mismatch is now
the smaller remaining no-parse-equivalent bucket.

## github3967 Root-Cause Slice

The focused deferred-marker diagnostic at prefix `C1=CC/C=C2\C3=C` originally
had two candidate rows:

- candidate `/`: current support accepts it; selected-carrier basis forces the
  component token phase to `stored`; one row survives marker events; graph
  marker equations reject because only side ids `[0, 1]` are covered
- candidate `\`: current support rejected it; the candidate implies `flipped`;
  the base selected-carrier fact has already forced `stored`; token-phase
  assignment count drops to zero before marker-event filtering; graph marker
  equations accept and cover side ids `[0, 1, 2, 3]`

So the immediate failure is not that terminal marker events alone erase a valid
row.  The earlier selected-carrier token-phase commitment is too strong for
this writer surface: it commits `stored` locally, while the complete
marker-slot equation says the RDKit target is satisfiable with `flipped`.

The target-independent fact is a graph-marker-equation-backed quotient:

- keep selected-carrier token-phase facts as the normal online constraint
- at a support boundary with a complete graph-marker equation, allow a
  competing token phase if the full equation covers all component side ids and
  accepts the candidate marker slots
- expose the quotient as an explicit writer-policy fact, with row ids and graph
  marker equation evidence, so it remains separable from principled semantic
  constraints

This is narrower and more inspectable than weakening selected-carrier facts
globally.

## Parse-Equivalent-Present Writer Gaps

The three remaining support-missing cases with existing same-skeleton
parse-equivalent Grimace outputs are writer-policy gaps, not missing semantic
support-basis gaps.

### Extra Redundant Marker

Two cases have the RDKit target slots as a subset of Grimace's parse-equivalent
same-skeleton slots:

- `github4582_chembl409450_random_vector_seed1_index3`
  - RDKit target slots: `(16, \)`, `(18, /)`, `(23, \)`
  - Grimace parse-equivalent slots: `(12, /)`, `(16, \)`, `(18, /)`,
    `(23, \)`
- `github4582_chembl409450_random_vector_seed1_index6`
  - RDKit target slots: `(8, /)`, `(19, \)`, `(41, \)`
  - Grimace parse-equivalent slots: `(6, /)`, `(8, /)`, `(19, \)`,
    `(41, \)`

These look like redundant-marker omission: Grimace emits one earlier marker
that is semantically compatible but RDKit's writer omits it in the sampled
surface.  The smallest useful policy here is not a new semantic constraint; it
is a writer representative rule that can prove the earlier marker is redundant
given later marker equations.

### Global Phase Representative

One case has a parse-equivalent same-skeleton Grimace output with all four
marker slots inverted relative to the RDKit target:

- `github4582_chembl409450_random_vector_seed1_index9`
  - RDKit target slots: `(13, /)`, `(18, \)`, `(23, \)`, `(27, /)`
  - Grimace parse-equivalent slots: `(13, \)`, `(18, /)`, `(23, /)`,
    `(27, \)`

That is likely a writer representative choice over a globally inverted
directional-marker phase.  It should be handled separately from redundant
marker omission and from the already-promoted github3967 graph-marker-equation
quotient.

The practical implication is that the parse-equivalent-present cases are later
polish for exact RDKit string support.  They should not drive the next
semantic-basis work; github3967 remains the cleaner next runtime target.
