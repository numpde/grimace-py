# South Star Next Frontier Choice

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 205: Choose next semantic frontier family`

## Purpose

Choose the next South Star semantic family after the atom-text expansion and
frontier refresh. The choice should optimize for the South Star goal: a
principled, fixed-molecule, mathematically anchored SMILES support enumerator,
not local convenience.

## Candidate Families

### Query Or Unspecified Bonds

Small witness: `C~C`

Pros:

- It is one of the only two current adversarial unsupported witnesses.
- The failure mode is easy to pin and diagnose.

Cons:

- Query semantics are not ordinary fixed-molecule semantics.
- Supporting this cleanly may require a query-SMILES product model rather than
  the current graph/stereo support model.
- It risks broadening the South Star surface in a direction that does not help
  ordinary molecule support.

Recommendation: do not choose this as the next expansion. Keep it as a clear
fail-fast boundary unless a separate query-SMILES product is defined.

### Dative Or Coordination Semantics

Small witness: `N->[O]`

Pros:

- It is the other current adversarial unsupported witness.
- It is a real molecule-surface issue rather than merely a fixture gap.

Cons:

- It is coupled to coordination chemistry, metal handling, and RDKit-specific
  writer quirks.
- The semantic contract is not just "render another bond token"; parse-back,
  valence, charge, and coordination interpretation need a separate model.
- It is easy to accidentally bolt on a writer-shaped special case.

Recommendation: do not choose this next. It should become its own deliberate
coordination-family track, not an incidental bond-text slice.

### Aromatic Directional Overlays

Small witness: aromatic bond manually tagged directional.

Pros:

- It is a named current fail-fast surface.
- It directly relates to marker placement and directional annotation policy.

Cons:

- The natural input space is less clear than ordinary aliphatic directional
  double-bond stereo.
- It may be a policy question before it is an implementation question: decide
  whether directional markers on aromatic surfaces are meaningful South Star
  annotations or invalid input facts.
- The existing aromatic support work is mostly markerless text and traversal
  breadth, not directional constraint semantics.

Recommendation: keep as a later policy/probe track. Do not use it as the next
main expansion unless a real, natural molecule witness is pinned first.

### Ring/Tetrahedral Interaction Expansion

Representative blocker: ring-local tetrahedral ligand-order dependence outside
the current monocycle proof family.

Pros:

- It stays inside fixed-molecule graph/stereo semantics.
- It is ordinary chemistry surface, not query or coordination semantics.
- It exercises the core South Star mathematical model: traversal order,
  ligand order, ring closure placement, stereo observations, and semantic
  parse-back identity.
- It is the highest-leverage route toward broader `MolToSmilesEnumS`
  readiness because ordinary users will hit ring/stereo interactions before
  they need query or coordination support.

Cons:

- It is harder than atom/bond text expansion.
- The fixture authority must be designed carefully to avoid case-by-case
  expected-string accumulation.
- It may require extending the unified-reference spine rather than just adding
  renderer policy.

Recommendation: choose this as the next main South Star frontier.

## Decision

Choose ring/tetrahedral interaction expansion as the next semantic family.

This does not mean implementing arbitrary ring stereochemistry immediately.
The next slice should be a probe and fixture-authority slice:

1. identify the smallest natural unsupported witnesses;
2. classify which facts couple traversal, ring closure, and tetrahedral
   ligand order;
3. define what the unified reference must generate;
4. pin a minimal SSoT fixture set only after the authority is named.

Query bonds and dative bonds remain explicit fail-fast boundaries. They should
not be silently promoted just because they are the only small adversarial
unsupported witnesses after atom-text expansion.

## Follow-Up Tasks

Update the Backlog so the generic next-probe row points at this chosen family.
The next tasks should be:

1. Build a ring/tetrahedral frontier witness inventory.
2. Define the unified-reference proof shape for those witnesses.
3. Add the first minimal SSoT fixtures only after the proof shape is explicit.
