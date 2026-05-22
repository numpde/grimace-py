# South Star Polycyclic Directional Witness Inventory

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 229: Inventory polycyclic ring/tetra directional witnesses`

## Purpose

Inventory fixed-molecule witnesses that combine polycyclic ring/tetrahedral
obligations with directional obligations. This is an inventory slice only: no
support gate changes and no fixture pinning.

## Probe

Reusable script:

`tmp/exploration/frontier/004_inventory_polycyclic_directional_witnesses.py`

The script reports support-gate status, raw directional features, raw
directional components, tetrahedral facts, ring/tetrahedral obligations,
runtime support behavior, and the proof-model facts that appear to be needed.

For inventory purposes, the script inspects raw directional features even when
the support gate rejects the molecule. That is deliberate: support-gated
witnesses still need fact inventory before any proof-model decision.

## Results

| Case | Source | Supported | Categories | Frags | Tetra | Ring/Tetra | Directional Features | Directional Components | Runtime |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `monocycle_ring_tetra_directional_supported_baseline` | `F[C@H]1CCCC(/C=C/Cl)C1` | true | - | 1 | 1 | 1 | 1 | 1 | 576 outputs |
| `polycyclic_ring_tetra_branch_tetra_supported_baseline` | `F[C@H]1CC2CCC1C2[C@H](Cl)Br` | true | - | 1 | 2 | 2 | 0 | 0 | 3160 outputs |
| `polycyclic_ring_tetra_directional_branch` | `F[C@H]1CC2CCC1C2/C=C/Cl` | false | `fused_or_polycyclic_ring`, `ring_molecule`, `ring_tetrahedral_interaction` | 1 | 1 | 1 | 1 | 1 | `SouthStarUnsupportedFeatureError` |
| `disconnected_polycyclic_ring_tetra_directional` | `F[C@H]1CC2CCC1C2.F/C=C/Cl` | false | `fused_or_polycyclic_ring`, `ring_tetrahedral_interaction` | 2 | 1 | 1 | 1 | 1 | `SouthStarUnsupportedFeatureError` |

## Interpretation

The connected witness is the best first proof-model target:

`F[C@H]1CC2CCC1C2/C=C/Cl`

It has one fragment, one ring/tetrahedral obligation, and one directional
component. It is rejected because the current support gate does not yet admit
the combined polycyclic ring/tetra plus directional surface.

The disconnected witness is important but should come second:

`F[C@H]1CC2CCC1C2.F/C=C/Cl`

It adds fragment-product composition on top of the same polycyclic
ring/tetrahedral and directional ingredients. That makes it a better stress
case after the connected proof model is clear.

The supported polycyclic branch-tetra witness has 3160 outputs, but it has no
directional component. It remains useful scale evidence, not the next proof
shape.

## Proof-Model Needs

A coherent proof model for the connected frontier needs to expose:

- polycyclic traversal facts;
- polycyclic ring/tetrahedral obligations;
- directional component facts;
- component partition between the ring/tetrahedral obligation and the
  directional obligation;
- rendering skeleton facts that combine ring-closure, tetrahedral atom text,
  and directional marker slots;
- semantic parse-back evidence.

Do not expand the support gate before those facts have an explicit ownership
boundary.

## Recommended Next Step

Proceed to `South Star 230`: define the mixed polycyclic
ring/tetra/directional proof model. Start with the connected witness. Treat the
disconnected witness as a second-wave fragment-product stress case.
