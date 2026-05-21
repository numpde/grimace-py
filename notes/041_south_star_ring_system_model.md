# South Star Ring-System Model

Task: `South Star 73: Generalize ring-system traversal model`

## Current Boundary

South Star now supports simple monocycles, the first non-aromatic nonstereo
polycyclic skeleton slice, and a narrow unmodified fused aromatic ring-system
slice. Broader polycyclic stereo, ring/tetrahedral interactions, modified
aromatic atom text, aromatic directional overlays, and unsupported atom/bond
text still remain fail-fast unsupported under their named categories.

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
- whether the graph is fused/polycyclic and therefore needs a polycyclic
  traversal path rather than the simple-monocycle path.

That fact boundary is now enough for nonstereo skeleton traversal and narrow
fused aromatic closure traversal. It is not enough for broader polycyclic
stereo, aromatic directional overlays, or modified aromatic atom text.

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

3. Define a graph-native ring-system fact boundary first, then route support
   through spanning-tree and closure-edge choices.

   This is the current path. It keeps support narrow, but supported polycyclic
   skeletons and narrow fused aromatic systems now use named inputs for
   closure-edge selection and traversal-event generation.

## Future Enumeration Choices

Polycyclic skeleton support requires explicit choices before rendering:

- choose a spanning tree for the connected component;
- classify every non-tree ring edge as a closure edge;
- allocate closure labels independently from RDKit writer order;
- decide closure-event order at both endpoints;
- render closure-side bond text from bond facts;
- attach any directional marker slots to event-local closure positions.

Those choices belong in traversal and marker-slot layers, not in a post-render
repair phase.

## Regression Witnesses

The first polycyclic witnesses are no longer unsupported merely because they
are polycyclic. They are regression witnesses for the skeleton traversal path:

- `C1CC2CCCC2C1` exposes a fused/polycyclic shape with shared ring atoms and a
  shared ring bond;
- `C1CCC2(CC1)CCCC2` exposes a spiro-like shape with a shared ring atom and no
  shared ring bond;
- `C1CC2CCC1C2` exposes a bridged shape with multiple shared ring atoms and
  shared ring bonds.

All three should expose ring-system facts with two rings and cyclomatic number
two. `C1CC2CCC1C2` is pinned as the first graph-native regression fixture.
The remaining boundary is not "polycyclic" in general; it is polycyclic
surfaces whose semantics require additional models, especially stereo,
aromaticity, and ring/tetrahedral interactions.

`South Star 182` adds the first fused aromatic witnesses under the same
closure-traversal spine:

- `c1ccc2ccccc2c1` for an all-carbon fused aromatic system;
- `c1ccc2ncccc2c1` for a fused aromatic system with unmodified aromatic
  nitrogen;
- `c1ccc2occc2c1` for a fused aromatic system with unmodified aromatic oxygen.

These are not RDKit writer-parity cases. They are South Star semantic fixtures
for sanitized aromatic molecule facts, lowercase aromatic atom text, elided
aromatic bond text, graph-native closure choices, and parse-back semantic
acceptance.
