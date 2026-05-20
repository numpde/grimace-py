# South Star Seed Complexity Checkpoint

Branch: `south-star`

Date: 2026-05-20

Task: `South Star 24: Measure EnumS complexity and performance`

## Scope

This is a checkpoint for the current graph-native `EnumS` seed, not a
performance claim and not a correctness proof.

Current seed limitations:

- one deterministic connected-acyclic atom-index traversal;
- component marker assignment products only;
- no root, branch-order, or traversal support expansion yet;
- no ring-closure or disconnected support.

The useful question at this stage is whether the implementation still exposes
component-factorized complexity counters and whether output counts match the
component assignment product on the current semantic fixture corpus.

## Measured Command

```sh
PYTHONPATH=python:. python3 - <<'PY'
from time import perf_counter
from tests.helpers.south_star_enum_s import (
    mol_to_smiles_enum_s_graph_native_for_case,
)
from tests.helpers.south_star_semantics import load_south_star_semantic_cases

for case in load_south_star_semantic_cases():
    t0 = perf_counter()
    result = mol_to_smiles_enum_s_graph_native_for_case(case)
    dt = (perf_counter() - t0) * 1000
    ...
PY
```

## Snapshot

| case | components | local assignments | product estimate | outputs | time ms |
|---|---:|---:|---:|---:|---:|
| isolated_alkene_z | 1 | 2 | 2 | 2 | 5.864 |
| linear_diene_same_phase | 1 | 2 | 2 | 2 | 0.406 |
| branched_substituted_alkene | 1 | 2 | 2 | 2 | 0.308 |
| hetero_imine_carrier | 1 | 2 | 2 | 2 | 0.310 |
| independent_two_alkenes | 2 | 2 x 2 | 4 | 4 | 0.549 |
| linear_diene_opposite_phase | 1 | 2 | 2 | 2 | 0.354 |
| same_side_alternate_carriers | 1 | 2 | 2 | 2 | 0.268 |

The first row includes Python/RDKit/helper warmup effects and should not be
used as a timing baseline.

## Interpretation

The current seed matches the intended component-factorized shape on the
fixture corpus:

- single-component cases expose two local assignments: source marker pattern
  and global flip;
- coupled shared-carrier diene cases stay one component with two assignments,
  not one assignment axis per stereo feature;
- independent components multiply at the final support level;
- output counts match the product estimate for the current deterministic
  traversal.

This is the desired complexity direction for the South Star model. The main
remaining risk is traversal expansion: adding roots, branch orders, and other
walk choices must not collapse the implementation back into one global row
filter over all stereo facts.

## Follow-Up

`South Star 18C: Expand graph-native traversal support` should preserve these
counters while widening traversal coverage. `South Star 24` should be repeated
after that work; only then will timing numbers be meaningful enough to compare
against other enumeration paths.
