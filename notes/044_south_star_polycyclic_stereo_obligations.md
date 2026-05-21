# South Star Polycyclic Stereo Obligations

Task: `South Star 103: Plan polycyclic stereo obligations`

## Boundary

South Star currently supports non-aromatic nonstereo polycyclic skeleton
traversal. It does not yet support stereo obligations whose carrier choices
interact with a polycyclic ring system. This is a deliberate boundary: the next
implementation should extend the shared fact/event/equation model, not add
ring-specific string repairs.

The existing support gate should therefore keep these cases distinct:

- a nonstereo polycyclic skeleton is supported;
- directional source text that RDKit parses without stereo remains skeleton
  support if all molecule facts are nonstereo;
- a real stereo component attached to or inside a polycyclic system remains
  fail-fast until closure choices and marker obligations are modeled together;
- aromatic and ring/tetrahedral surfaces remain separate blockers.

## Obligation Shape

Polycyclic stereo needs obligations over a traversal, not over a completed
SMILES string:

1. choose a connected spanning tree;
2. classify every non-tree ring edge as a closure edge;
3. allocate closure labels for that closure-edge set;
4. emit traversal events with atom events, bond events, ring-open events, and
   ring-close events;
5. attach marker slots to event-local carrier positions, including closure
   carrier positions when a carrier edge is not part of the spanning tree;
6. build marker-slot parity equations from semantic stereo components and the
   event-local carrier contexts;
7. solve the equations for that traversal and render only solved assignments.

The important constraint is locality: a marker obligation is attached to the
event that emits the carrier, while the semantic component remains a molecule
fact. Multiple closure choices may expose different valid carrier events for
the same semantic component, so the solver must run per traversal skeleton.

## Complexity Shape

The expected product is not "all strings, then filter." It is:

```text
sum over spanning-tree / closure-edge choices:
  traversal_orderings(choice)
  * solved_marker_assignments(choice, traversal)
```

Diagnostics should continue to expose the layers separately:

- `spanning_tree_count`;
- `closure_edge_count`;
- `closure_label_count`;
- `traversal_skeleton_count`;
- `marker_slot_count`;
- `local_assignment_count`;
- `solved_assignment_count`;
- `raw_output_count`;
- `output_count`.

If a future slice cannot account for those fields without post-render repair,
it is not aligned with the South Star model.

## First Guardrail Witnesses

- `C1CC2CCC1C2`: supported nonstereo bridged skeleton.
- `C1/C=C\C2CCCC2C1`: supported because RDKit parses no stereo component from
  the directional source text.
- `F/C=C\C1CC2CCC1C2`: unsupported because a real external alkene stereo
  component is attached to a polycyclic system whose stereo/traversal
  composition is not modeled yet.
- `F[C@H](Cl)C1CCCCC1`: unsupported by `ring_tetrahedral_interaction`, not by
  the polycyclic stereo plan.
- `c1ccccc1`: unsupported by `aromatic_ring_surface`, not by the polycyclic
  stereo plan.

These are planning witnesses. They should not become a fixture-by-fixture
definition of the final surface.

## Minimal Implementation Sequence

1. Expose per-traversal closure-edge-set identity in diagnostics/tests.
2. Extend marker-slot equation tests to include closure carrier positions in
   polycyclic traversals.
3. Add one external-stereo/polycycle witness as fail-fast until equations are
   stated for every traversal choice.
4. Promote the witness only after the same semantic component equations solve
   through both ordinary bond events and ring-closure events.

