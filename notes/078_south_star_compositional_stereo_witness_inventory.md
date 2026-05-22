# South Star Compositional Stereo Witness Inventory

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 215: Inventory compositional stereo witnesses`

## Purpose

Inventory small witnesses for fixed-molecule compositional stereo scaling after
the proof contract in note 077.

This is an exploration slice. It does not promote fixtures and does not change
runtime support.

## Probe

Reusable script:

`tmp/exploration/frontier/003_inventory_compositional_stereo_witnesses.py`

The script records support-gate status, fragment count, tetrahedral facts,
ring/tetrahedral obligations, directional components, and current runtime
output count where supported.

One caveat: `directional components` currently means the existing directional
component extractor. Tetrahedral components are counted separately as facts and
obligations; a future compositional proof model should expose a unified
component partition across both families.

## Inventory

| Case | Source | Classification | Supported | Categories | Frags | Tetra | Ring/Tetra Obligations | Directional Components | Outputs |
| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| `acyclic_two_tetra_separated` | `F[C@H](Cl)C[C@H](Br)I` | independent product candidate | yes | - | 1 | 2 | 0 | 0 | 48 |
| `acyclic_two_tetra_adjacent` | `F[C@H](Cl)[C@H](Br)I` | coupled component candidate | yes | - | 1 | 2 | 0 | 0 | 40 |
| `disconnected_two_tetra` | `F[C@H](Cl)Br.F[C@H](Cl)I` | independent product candidate | yes | - | 2 | 2 | 0 | 0 | 288 |
| `directional_tetrahedral_acyclic_existing` | `F/C=C/[C@H](Cl)Br` | existing mixed-component baseline | yes | - | 1 | 1 | 0 | 1 | 40 |
| `independent_directional_diene_existing` | `F/C=C/C/C=C/Cl` | existing independent-directional baseline | yes | - | 1 | 0 | 0 | 2 | 48 |
| `monocycle_ring_tetra_directional_existing` | `F[C@H]1CCCC(/C=C/Cl)C1` | existing ring/directional/tetra baseline | yes | - | 1 | 1 | 1 | 1 | 576 |
| `ring_tetra_plus_branch_tetra` | `F[C@H]1CCCC([C@H](Cl)Br)C1` | coupled or shared-ring-ligand candidate | yes | - | 1 | 2 | 2 | 0 | 576 |
| `polycyclic_ring_tetra_plus_branch_tetra` | `F[C@H]1CC2CCC1C2[C@H](Cl)Br` | coupled or shared-ring-ligand candidate | yes | - | 1 | 2 | 2 | 0 | 3160 |
| `disconnected_polycyclic_ring_tetra_directional` | `F[C@H]1CC2CCC1C2.F/C=C/Cl` | unsupported disconnected-product boundary | no | `fused_or_polycyclic_ring`, `ring_tetrahedral_interaction` | 2 | 1 | 1 | 0 | - |
| `polycyclic_ring_tetra_directional_branch` | `F[C@H]1CC2CCC1C2/C=C/Cl` | unsupported polycyclic mixed boundary | no | `fused_or_polycyclic_ring`, `ring_molecule`, `ring_tetrahedral_interaction` | 1 | 1 | 1 | 0 | - |

## Interpretation

The cleanest first compositional fixtures are not the largest mixed ring cases.
They are the small multi-tetrahedral cases:

- `F[C@H](Cl)C[C@H](Br)I` has two separated tetrahedral centers and 48 outputs.
- `F[C@H](Cl)[C@H](Br)I` has two adjacent centers and 40 outputs.
- `F[C@H](Cl)Br.F[C@H](Cl)I` has two disconnected tetrahedral fragments and
  288 outputs.

These are ordinary fixed-molecule cases, already supported by the current
runtime, and small enough to use as the first proof-contract fixtures.

The ring/tetrahedral plus branch-tetra cases are useful but less clean as first
fixtures:

- both tetrahedral centers produce ring/tetrahedral obligations, so the branch
  center is not an independent acyclic add-on;
- the polycyclic variant already has 3160 outputs;
- they are better second-wave coupled-component witnesses after the component
  partition vocabulary is explicit.

The polycyclic ring/tetrahedral plus directional cases are still support-gated.
They should not be forced into the first compositional fixture slice. They are
better as follow-up support-gate/domain expansion after the independent and
adjacent multi-tetrahedral cases are proven cleanly.

## Recommended First Fixture Candidates

Use this order:

1. `acyclic_two_tetra_separated`: first independent-product candidate.
2. `acyclic_two_tetra_adjacent`: first coupled-component candidate, if the proof
   model can name the coupling cleanly.
3. `disconnected_two_tetra`: fragment-product stress after the connected cases.

Do not start with the 3160-output polycyclic mixed case. It is useful, but it
is not the clearest proof-contract exercise.

## Follow-Up Needed

Before pinning fixtures, add or extract a proof helper that can report:

- tetrahedral obligation ids for each center;
- component partition over tetrahedral obligations;
- independent-product versus coupled-component classification;
- assignment counts before rendering;
- rendered output equality with current runtime;
- semantic parse-back evidence.

## Proof-Stage Refresh After Initial Admission

Task: `South Star 222: Inventory mixed ring/tetra compositional witnesses`

After admitting the separated, adjacent, and disconnected two-tetra fixtures,
the same inventory script now reports proof-stage data from
`compositional_stereo_proof_report(...)`: proof classification, component
partition, coupling reasons, and proof/runtime equality.

The mixed ring/tetra candidates classify as follows:

| Case | Source | Proof class | Components | Coupling reasons | Proof/runtime | Outputs |
| --- | --- | --- | --- | --- | --- | ---: |
| `monocycle_ring_tetra_directional_existing` | `F[C@H]1CCCC(/C=C/Cl)C1` | `independent_product` | `tetrahedral:1`; `directional:component:0` | - | `576/576 match` | 576 |
| `ring_tetra_plus_branch_tetra` | `F[C@H]1CCCC([C@H](Cl)Br)C1` | `coupled_component` | `tetrahedral:1+tetrahedral:6` | `shared_ring_tetrahedral_system` | `576/576 match` | 576 |
| `polycyclic_ring_tetra_plus_branch_tetra` | `F[C@H]1CC2CCC1C2[C@H](Cl)Br` | `coupled_component` | `tetrahedral:1+tetrahedral:8` | `shared_ring_tetrahedral_system` | `3160/3160 match` | 3160 |

The smallest mixed ring/tetra next fixture candidate is therefore
`F[C@H]1CCCC([C@H](Cl)Br)C1`: it has the same output count as the already
admitted monocycle ring/tetra/directional baseline, but it exercises two
tetrahedral obligations coupled by the ring/tetrahedral system rather than a
product of one ring/tetrahedral and one directional component.

Do not pin the 3160-output polycyclic branch case first. It has the same proof
shape as the smaller monocyclic witness and should follow only if the smaller
fixture is coherent.

The refreshed inventory also shows that some older directional labels are
coarser than the new proof classification. In particular,
`F/C=C/C/C=C/Cl` remains an existing fixture baseline, but the compositional
proof helper currently classifies the two directional obligations as coupled by
`adjacent_directional_obligation`. That is outside this mixed ring/tetra slice;
it should be reconciled before using directional component labels as a general
product/coupling authority.
