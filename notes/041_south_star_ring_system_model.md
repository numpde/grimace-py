# South Star Ring-System Model

Task: `South Star 73: Generalize ring-system traversal model`

## Current Boundary

South Star currently supports simple monocycles only. Fused, bridged,
spiro-like, and otherwise polycyclic ring systems remain fail-fast unsupported
under `fused_or_polycyclic_ring`.

The important change is conceptual: ring-system information now has a named
fact boundary. Future traversal work should consume ring-system facts instead
of rediscovering ring shape inside the renderer or patching completed strings.

## Minimal Fact Model

The ring-system fact model should describe graph structure, not writer policy:

- atom rings and bond rings from molecule facts;
- flat ring atom and bond membership;
- per-atom and per-bond ring membership counts;
- shared ring atoms and shared ring bonds;
- ring count;
- graph cyclomatic number, i.e. the number of non-tree closure edges that a
  connected traversal will eventually need to choose;
- whether the graph is a simple monocycle;
- whether the graph is spiro-like;
- whether the graph is fused/polycyclic and therefore outside current support.

That is not yet enough to enumerate polycycles. It is enough to keep the next
modeling work honest about what must be chosen before rendering.

## Alternatives Considered

1. Keep only flat `ring_count` and ring membership.

   This is too weak. It tells the gate that the molecule has multiple rings but
   does not give future traversal code a stable place to ask which atom/bond
   cycles are involved.

2. Import RDKit traversal or ring-label behavior.

   This is not South Star semantics. RDKit writer behavior can remain a
   comparison target, but ring-system support needs its own choices for
   spanning trees, non-tree closure edges, label allocation, and closure-event
   ordering.

3. Define a graph-native ring-system fact boundary first.

   This is the current path. It keeps support narrow, but future code has a
   named input shape for closure-edge selection and traversal-event generation.

## Future Enumeration Choices

Polycyclic support requires explicit choices before any renderer changes:

- choose a spanning tree for the connected component;
- classify every non-tree ring edge as a closure edge;
- allocate closure labels independently from RDKit writer order;
- decide closure-event order at both endpoints;
- render closure-side bond text from bond facts;
- attach any directional marker slots to event-local closure positions.

Those choices belong in traversal and marker-slot layers, not in a post-render
repair phase.

## Guardrail Witness

The guardrail witnesses are deliberately unsupported:

- `C1CC2CCCC2C1` exposes a fused/polycyclic shape with shared ring atoms and a
  shared ring bond;
- `C1CCC2(CC1)CCCC2` exposes a spiro-like shape with a shared ring atom and no
  shared ring bond;
- `C1CC2CCC1C2` exposes a bridged shape with multiple shared ring atoms and
  shared ring bonds.

All three should expose ring-system facts with two rings, cyclomatic number
two, and no partial support until the graph-native polycycle traversal model
exists.
