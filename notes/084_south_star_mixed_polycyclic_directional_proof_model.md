# South Star Mixed Polycyclic Directional Proof Model

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 230: Define mixed polycyclic directional proof model`

## Purpose

Define the proof model for connected polycyclic ring/tetrahedral plus
directional composition before changing support gates or pinning fixtures.

Primary witness:

`F[C@H]1CC2CCC1C2/C=C/Cl`

## Domain

The first domain should be deliberately narrow:

- one connected molecule;
- one fused or polycyclic ring system;
- one tetrahedral center with a ring/tetrahedral interaction obligation;
- one directional component outside the tetrahedral center;
- no shared directional carrier with another directional component;
- no query, dative, aromatic directional overlay, or RDKit writer-parity
  behavior.

The disconnected witness
`F[C@H]1CC2CCC1C2.F/C=C/Cl` is a second-wave fragment-product stress case, not
the first proof-model target.

## Current Witness Facts

For `F[C@H]1CC2CCC1C2/C=C/Cl`:

- tetrahedral center: atom `1`;
- ring/tetrahedral obligation count: `1`;
- directional feature: bond `8`, central bond `(8, 9)`;
- directional carrier edges: `(7, 8)` and `(9, 10)`;
- directional component count: `1`;
- directional coupling causes: none;
- support gate today rejects with `fused_or_polycyclic_ring`, `ring_molecule`,
  and `ring_tetrahedral_interaction`.

The directional component does not share the tetrahedral center atom. It should
therefore be modeled as an independent directional obligation composed with a
polycyclic ring/tetrahedral obligation over a shared traversal/rendering
skeleton.

## Ownership Boundary

Use the existing ownership split:

- polycyclic graph and ring-closure facts belong to the polycyclic traversal
  and ring/tetrahedral proof spine;
- tetrahedral token orientation belongs to the tetrahedral proof input and
  ring/tetrahedral obligation;
- directional marker slots and component-local marker equations belong to the
  directional component support state;
- composition belongs to a mixed proof helper that combines those facts and
  checks product/renderer consistency.

The mixed helper must not reimplement polycyclic traversal, directional
component extraction, or tetrahedral orientation logic.

## Component Partition

The connected witness should be classified as an independent product of:

1. a polycyclic ring/tetrahedral obligation component;
2. a directional marker component.

The proof must show:

- the directional component has no extractor-level coupling causes;
- the directional component does not share the tetrahedral center atom;
- directional marker equations remain component-local;
- changing the directional marker assignment does not change the valid
  tetrahedral token assignment set;
- rendered output equality is checked after first-occurrence deduplication.

If any of these fail during implementation, the witness should move to a
coupled-component proof model instead of being forced into product language.

## Required Proof Stages

The helper for this domain should report:

1. molecule facts: graph topology, fragments, ring-system facts, atom/bond
   facts, tetrahedral facts, and directional components;
2. ring/tetrahedral obligations: center, source ligand order, ring ligands,
   acyclic ligands, implicit hydrogen, ring-closure event fields;
3. directional obligations: source feature id, central bond, carrier edges,
   source markers, component id, coupling causes;
4. traversal skeletons: connected graph traversal plans, atom events,
   ring-closure events, directional marker slots, and renderer inputs;
5. assignment model: tetrahedral proof input per relevant atom event and
   directional component marker assignments;
6. product evidence: component-local directional equations and stable
   tetrahedral orientation evidence across directional assignments;
7. rendering: strings rendered from traversal events and assignments;
8. semantic evidence: grammar conformance, RDKit parseability, graph signature
   equality, stereo semantic signature equality;
9. deduplication: first-occurrence deduplication, with expected fixture strings
   used only as checked artifacts.

## Proposed Authority

If the model proves coherent, use:

`unified_reference_polycyclic_ring_tetrahedral_directional_composition`

Do not register the authority until the first fixture using it is added.

## Implementation Gate

Before support-gate expansion, add a proof helper that can run against the
currently rejected witness by explicitly opting into this domain. The helper
should fail fast if any required fact is missing.

Only after that helper renders proof outputs and validates semantic parse-back
should `South Star 231` expand support and pin a fixture.
