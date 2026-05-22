# South Star Ring/Tetrahedral Frontier Witnesses

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 206: Inventory ring/tetrahedral frontier witnesses`

## Purpose

Inventory small natural witnesses for the next chosen South Star family:
ring/tetrahedral interaction expansion. This is a planning/probe slice only;
it does not change runtime support.

## Probe

Reusable inventory script:

`tmp/exploration/frontier/002_inventory_ring_tetrahedral_witnesses.py`

The script prints support-gate categories, ring topology facts, and extracted
ring/tetrahedral obligations for supported baselines and unsupported frontier
witnesses.

## Baselines

Current supported baselines:

| Case | Source | Atoms | Rings | Fused/poly | Obligation |
| --- | --- | ---: | ---: | --- | --- |
| `supported_monocycle_center` | `F[C@H]1CCCC(C)C1` | 8 | 1 | false | center in ring, two ring ligands, one acyclic ligand, one implicit H |
| `supported_adjacent_monocycle` | `F[C@H](Cl)C1CCCCC1` | 9 | 1 | false | center outside ring, one ring ligand, two acyclic ligands, one implicit H |
| `supported_monocycle_directional_branch` | `F[C@H]1CCCC(/C=C/Cl)C1` | 10 | 1 | false | center in ring plus exocyclic directional branch |

These prove that South Star already has a useful monocycle ring/tetrahedral
obligation vocabulary. The next frontier is not "any ring tetrahedral center";
it is ring/tetrahedral coupling across fused, bridged, or otherwise polycyclic
ring systems.

## Unsupported Witnesses

Current useful unsupported witnesses:

| Case | Source | Atoms | Rings | Categories | Obligation |
| --- | --- | ---: | ---: | --- | --- |
| `unsupported_small_polycyclic` | `F[C@H]1CC2CC1C2` | 7 | 3 | `fused_or_polycyclic_ring`, `ring_molecule`, `ring_tetrahedral_interaction` | center in ring; ring ligands `(2, 5)`; acyclic ligand `(0)`; implicit H |
| `unsupported_small_bridged` | `F[C@H]1CC2CCC1C2` | 8 | 2 | `fused_or_polycyclic_ring`, `ring_molecule`, `ring_tetrahedral_interaction` | center in ring; ring ligands `(2, 6)`; acyclic ligand `(0)`; implicit H |
| `unsupported_bridge_variant` | `F[C@H]1CCC2CC1C2` | 8 | 3 | `fused_or_polycyclic_ring`, `ring_molecule`, `ring_tetrahedral_interaction` | center in ring; ring ligands `(2, 6)`; acyclic ligand `(0)`; implicit H |
| `unsupported_known_fused_witness` | `C1CC2CCCC2[C@H]1F` | 9 | 2 | `fused_or_polycyclic_ring`, `ring_molecule`, `ring_tetrahedral_interaction` | center in ring; ring ligands `(6, 0)`; acyclic ligand `(8)`; implicit H |

The smallest witness is `F[C@H]1CC2CC1C2`, but it is a compact three-ring
polycyclic system. The cleaner first fixture candidate is probably
`F[C@H]1CC2CCC1C2`: it is still small, has two perceived rings, and exposes the
same center-in-ring/two-ring-ligand obligation shape without making the
smallest-case topology the design driver.

## Non-Witnesses

Two compact polycyclic-looking inputs are supported because they do not produce
a ring/tetrahedral interaction obligation:

- `F[C@H]1C2CC1C2`
- `F[C@H]1CC2CC2C1`

They are useful negative controls for the next proof-shape work: the boundary
is not just "polycyclic plus chiral tag." It is whether a supported tetrahedral
fact has ring ligands that require traversal/ring-closure evidence.

## Coupled Facts

Every current unsupported witness has the same essential coupling:

- tetrahedral center is in a ring;
- source token is orientation-sensitive;
- two ligands are ring ligands;
- one ligand is acyclic;
- one implicit hydrogen participates in the ligand order;
- support gate also reports fused/polycyclic ring topology.

The proof shape must therefore combine:

- polycyclic traversal basis;
- ring-closure event placement and labels;
- tetrahedral source ligand order;
- emitted ligand order from traversal events;
- semantic parse-back identity.

## Recommendation For South Star 207

Start with the cleaner 8-atom bridged witness `F[C@H]1CC2CCC1C2`, keep the
7-atom witness `F[C@H]1CC2CC1C2` as a stress/minimality check, and keep
`C1CC2CCCC2[C@H]1F` as the continuity check against the older notes.

The next slice should define a unified-reference proof authority for this
family before pinning expected support strings.
