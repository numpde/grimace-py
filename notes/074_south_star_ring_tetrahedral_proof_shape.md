# South Star Ring/Tetrahedral Proof Shape

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 207: Define ring/tetrahedral proof shape`

## Purpose

Define the unified-reference proof shape for the next South Star family:
polycyclic ring systems with tetrahedral centers whose ligand order depends on
ring traversal and ring-closure events.

This note names the intended authority before any expected support strings are
pinned.

## Authority

Use a new authority name:

`unified_reference_polycyclic_ring_tetrahedral_obligations`

This authority should mean:

- support is generated from molecule facts, traversal/ring-closure events, and
  tetrahedral ligand-order constraints;
- expected SMILES strings are not hand-authored as the proof source;
- every rendered output parses back to the source graph and stereo semantics;
- runtime output equality is a cross-check after support is implemented, not
  the source of the fixture expectation.

## Domain Boundary

The first domain should be intentionally narrow:

- one connected molecule;
- ordinary fixed-molecule graph, not query or dative semantics;
- fused, bridged, or otherwise polycyclic ring topology;
- at least one supported tetrahedral center with a ring ligand;
- no unsupported atom text, bond text, fragment composition, or aromatic
  directional overlay;
- no attempt to cover all ring stereochemistry in the first slice.

Initial witnesses from notes/073:

- first candidate: `F[C@H]1CC2CCC1C2`;
- minimality stress: `F[C@H]1CC2CC1C2`;
- continuity check: `C1CC2CCCC2[C@H]1F`;
- negative controls: `F[C@H]1C2CC1C2` and `F[C@H]1CC2CC2C1`, which do not
  produce ring/tetrahedral obligations.

## Required Proof Stages

### 1. Molecule Facts

The proof starts from typed facts:

- ring-system topology: ring count, shared atoms, shared bonds, and whether the
  system is fused/polycyclic;
- tetrahedral center facts: center atom, source token, source ligand order,
  explicit neighbors, implicit hydrogen count;
- ring/tetrahedral obligations: center-in-ring flag, ring ligand atom indices,
  acyclic ligand atom indices, and required event fields.

These facts already exist in pieces:

- `SouthStarMoleculeFacts`;
- `SouthStarTetrahedralCenterFact`;
- `SouthStarRingTetrahedralInteractionObligation`.

### 2. Traversal And Ring-Closure Events

The proof needs a traversal/event enumerator that can run before the public
support gate is opened for the new family.

This is the key engineering boundary. Do not pin expected support by manually
writing strings, and do not make the proof depend on an already-supported
runtime path that cannot enumerate the new witnesses yet.

The no-regret shape is to extract or expose a shared traversal/ring-closure
event spine:

- graph traversal plan;
- atom event sequence;
- ring-closure event pair;
- ring-closure label assignment;
- bond text needed by the ring closure;
- renderer inputs attached to events.

Runtime and proof code may both consume this spine, but the proof must remain
the fixture authority while the runtime is being extended.

### 3. Tetrahedral Observations

For each traversal, derive the emitted ligand order for every involved
tetrahedral center from structured events:

- parent ligand;
- emitted child ligands;
- ring-closure ligand atom indices;
- implicit hydrogen position;
- ring-closure labels needed to locate closure ligands.

This should reuse the existing tetrahedral observation language instead of
introducing fixture-local calculations.

### 4. Constraint Assignment

For each tetrahedral observation, compute the preserving token from:

- source token;
- source ligand order;
- emitted ligand order.

This is the same mathematical relation already used by the current tetrahedral
proofs. The new family should add ring-closure traversal evidence, not a new
chirality rule.

### 5. Rendering And Deduplication

Render outputs from the traversal events and tetrahedral token assignments.
Then deduplicate by first occurrence, consistent with the current South Star
output-order policy.

### 6. Semantic Parse-Back

Every output must pass:

- South Star grammar conformance;
- RDKit parseability;
- graph signature equality with the source;
- semantic stereo signature equality with the source.

Exact equality with RDKit writer support is not part of this authority.

## What Not To Do

Do not:

- use checked-in expected strings as the only proof;
- add runtime special cases for the selected witnesses;
- collapse query/dative semantics into this family;
- treat the smallest 7-atom witness as the design target if the 8-atom witness
  exposes the same math more clearly;
- open public support gates before the proof authority exists.

## Immediate Follow-Up

The next Backlog item should not pin fixtures yet. It should first implement or
extract the shared traversal/ring-closure proof spine needed by this authority.

After that, fixture pinning can add the first minimal cases under the new
authority and require package-readiness coverage.
