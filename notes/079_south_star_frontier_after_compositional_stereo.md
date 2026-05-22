# South Star Frontier After Compositional Stereo

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 221: Refresh frontier after compositional stereo admission`

## Purpose

Refresh the South Star frontier after admitting the first compositional stereo
fixtures. The goal is to state what the current one-truth spine now proves,
what it still does not prove, and what the next straight-line implementation
slice should be.

This note follows notes 076-078. It does not redefine the South Star target:
fixture strings remain checked artifacts. The authority is the unified
reference path: molecule facts, typed obligations, component partition,
traversal/rendering skeletons, assignments, rendering, first-occurrence
deduplication, and semantic parse-back.

## Current Admission

The compositional stereo sequence now has three proof-backed fixtures:

| Case | Source | Authority | Outputs |
| --- | --- | --- | ---: |
| `compositional_stereo_two_tetra_separated` | `F[C@H](Cl)C[C@H](Br)I` | `unified_reference_compositional_stereo_product` | 48 |
| `compositional_stereo_two_tetra_adjacent` | `F[C@H](Cl)[C@H](Br)I` | `unified_reference_compositional_stereo_coupled_component` | 40 |
| `compositional_stereo_two_tetra_disconnected` | `F[C@H](Cl)Br.F[C@H](Cl)I` | `unified_reference_compositional_stereo_product` | 288 |

All three promote through package-readiness checks as unified-reference-backed
cases. The current readiness matrix has 77 unified-reference-backed cases.

## What This Proves

The branch now proves that the South Star spine can handle more than a single
local stereo obligation in three important shapes:

- independent connected tetrahedral centers;
- adjacent connected tetrahedral centers with a coupled component label;
- disconnected fragment products, including all fragment orders.

The disconnected fixture is important because it confirms that product support
is not limited to one connected traversal skeleton. It exercises fragment-level
proof outputs, fragment-order composition, first-occurrence deduplication, and
runtime equality as a cross-check.

## What This Does Not Prove

The current compositional fixtures are still deliberately narrow:

- they do not prove mixed ring/tetrahedral multi-center composition;
- they do not prove mixed directional/tetrahedral/ring composition beyond
  already-pinned existing baseline cases;
- they do not admit query or unspecified bonds;
- they do not admit dative or coordination bonds;
- they do not decide aromatic directional overlay policy.

These are not defects in the admitted fixtures. They are boundaries that should
remain explicit until each has a proof model.

## Remaining Fixed-Molecule Compositional Frontier

The next fixed-molecule frontier should stay with ordinary molecules and
ordinary stereo semantics before opening query or coordination semantics.

The current inventory still has two useful mixed ring/tetra witnesses:

| Case | Source | Current status | Outputs |
| --- | --- | --- | ---: |
| `ring_tetra_plus_branch_tetra` | `F[C@H]1CCCC([C@H](Cl)Br)C1` | supported; needs proof classification | 576 |
| `polycyclic_ring_tetra_plus_branch_tetra` | `F[C@H]1CC2CCC1C2[C@H](Cl)Br` | supported; needs proof classification | 3160 |

The smallest one is the right next investigation target. It is not just
"another tetrahedral fixture": it asks whether the ring/tetrahedral obligation
and the branch tetrahedral obligation can be expressed by the same component
partition vocabulary without a new local patch.

## Recommendation

Proceed in this order:

1. Inventory mixed ring/tetra compositional witnesses with the same
   proof-stage vocabulary used for the first compositional fixtures.
2. Pin the smallest mixed ring/tetra fixture only if the proof helper can name
   the component partition and coupling relation clearly.
3. If the partition is unclear, do not pin a large support file. Open narrower
   proof-model tasks instead.
4. After the mixed ring/tetra slice, reassess broader South Star frontiers:
   query bonds, dative bonds, aromatic directional overlays, and larger mixed
   stereo products.

## Notion Backlog Created

- `South Star 222: Inventory mixed ring/tetra compositional witnesses`
- `South Star 223: Add first mixed ring/tetra compositional fixture`
- `South Star 224: Reassess post-compositional semantic frontiers`
