# South Star Tetrahedral Constraint Family

Branch: `south-star`

Date: 2026-05-20

Task: `South Star 39: Defer tetrahedral stereo as separate constraint family`

## Purpose

Tetrahedral atom stereo should be a distinct South Star constraint family. It
should not be folded into directional double-bond carrier equations, and it
should not reuse slash/backslash marker-slot machinery by analogy.

The existing double-bond path answers: which directional bond marker slots
preserve alkene/imine stereo? Tetrahedral stereo answers a different question:
which neighbor-order observation at a chiral atom, plus which `@`/`@@` token,
preserves the intended tetrahedral orientation?

## Current Boundary

The current private South Star support gate reports `atom_stereo` and fails
fast for molecules such as `C[C@H](F)Cl`. That boundary should remain until a
dedicated atom-stereo component model exists.

The first implementation should be additive:

- keep directional double-bond components unchanged;
- introduce tetrahedral components beside them;
- let traversal events expose atom-local neighbor order facts;
- solve atom-stereo token choices through an atom-stereo solver path.

## Required Concept Split

Directional double-bond components:
: Use carrier edges, directional markers, marker slots, and parity equations
  over slash/backslash assignments.

Tetrahedral atom components:
: Use chiral center atoms, ligand order, atom token position, and `@`/`@@`
  choices. They do not have carrier edges.

Traversal facts:
: The traversal determines the order in which a chiral atom's neighbors appear
  in the emitted SMILES syntax, including parent edge, main child, branch
  children, ring-closure neighbors, and implicit hydrogen position.

Annotation policy:
: A policy decides whether a tetrahedral center is annotated at all and whether
  both `@` and `@@` spellings are emitted when both are semantically valid under
  different traversal orders. This is separate from double-bond marker policy.

## Fact Model

A tetrahedral component needs at least:

- `center_atom_idx`;
- intended source chirality;
- ligand identities, including an implicit hydrogen if present;
- local source ligand order used to define intended orientation;
- supported atom-token rendering form for the center;
- whether the center can be represented without brackets in the current scope.

A traversal observation needs:

- `center_atom_idx`;
- emitted atom token site id;
- ordered ligand references as seen by the SMILES parser;
- parent ligand, if the center is reached from a parent atom;
- child/branch ligand positions;
- ring-closure ligand positions, once rings are supported;
- implicit hydrogen position, if present.

The solver input should be a center-local equation:

```text
orientation(emitted_ligand_order, atom_stereo_token) == intended_orientation
```

The solver output should be one atom-token assignment for that traversal site:

```text
site_id -> "@" | "@@"
```

It should not produce slash/backslash marker assignments.

## Traversal/Event Implications

Atom events need to carry enough structure for the atom-stereo solver:

- atom index;
- token site id;
- rendered atom body without `@`/`@@`;
- neighbor-order observation after branch/ring placement is known;
- optional atom-stereo slot.

Rendering should then be:

1. build traversal events;
2. build atom-stereo equations from atom-stereo slots;
3. solve `@`/`@@` assignments;
4. render the atom token with the solved assignment.

This mirrors the double-bond slot discipline without sharing its carrier-edge
equation model.

## Solver Shape

The first solver can be small and exact:

- for each tetrahedral center, compute orientation parity for `@` and `@@`
  under the traversal's ligand order;
- keep the token choices that preserve the intended orientation;
- require exactly one chosen token for a fully specified center in a fully
  specified traversal;
- fail before rendering if no token preserves the center.

Independent tetrahedral centers should multiply as separate component domains,
just like independent double-bond components. Coupled cases should only appear
if the syntax policy creates a real shared observation; they should not be
invented by a global row model.

## First Fixtures

Start with atom-stereo cases that do not require rings:

1. `C[C@H](F)Cl`: one explicit tetrahedral center with implicit hydrogen.
2. `N[C@H](C)O`: heteroatom substituents and branch-order variation.
3. `C[C@](F)(Cl)Br`: quaternary center with no implicit hydrogen.
4. `C[C@H](F)[C@H](Cl)Br`: two independent tetrahedral centers.

Then add ring-containing centers only after the ring traversal event model is
stable.

For each fixture, assert:

- emitted strings parse back to the intended graph and atom stereo assignment;
- negative `@`/`@@` flips are rejected by the semantic oracle;
- support-size expectations are expressed through center-local assignments and
  traversal observations, not RDKit writer parity.

## Guardrails

- Keep `atom_stereo` fail-fast until atom-stereo slots and equations exist.
- Do not encode tetrahedral behavior as a special case in the renderer.
- Do not use RDKit exact writer strings as the semantic authority.
- Do not merge atom-stereo facts into double-bond carrier components.
- Do not add a completed-string parser filter to repair wrong `@`/`@@` choices.

## Minimal Implementation Path

1. Add atom-stereo component extraction that only reports facts and still keeps
   the public gate fail-fast.
2. Add atom event slot structure for synthetic tests, without changing emitted
   support.
3. Add a tiny orientation-parity helper with table-driven tests independent of
   traversal.
4. Connect traversal neighbor-order observations to atom-stereo equations for a
   single acyclic center.
5. Lift the `atom_stereo` gate for the first pinned fixture only after emitted
   strings pass semantic conformance by construction.
