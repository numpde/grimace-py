# Shared Marker Obligation Alternatives

Branch: `stereo-constraint-model`

## Context

The committed shared-deferred-marker change lets one emitted directional
marker constrain more than one stereo component. That is necessary, but not
sufficient.

The remaining small witnesses show a sharper problem. A visible marker edge is
not necessarily the selected stereo-atom carrier edge for every component that
touches it. In the manual difficult diene cases, RDKit's selected stereo-atom
edges are:

- component 0: `(1, 2)` and `(3, 4)`;
- component 1: `(4, 5)` and `(7, 9)`.

But the visible markers are on `(1, 2)`, `(3, 4)`, and `(7, 8)`. The bridge
edge `(3, 4)` is a selected carrier for component 0, but only a visible
candidate edge for component 1. The terminal visible marker `(7, 8)` is also
not the selected carrier edge for component 1; RDKit selected `(7, 9)`.

So the current local assumption is wrong:

> visible marker on a candidate edge means that edge is the selected carrier.

For two-candidate sides, RDKit can place a visible marker on the other
candidate edge and still use it to encode the selected stereo relation.

## Alternatives

### A. Visible Edge Selects Carrier

Shape:

- When a directional marker is emitted on a side candidate edge, select that
  candidate neighbor immediately.
- Treat the marker as the chosen token for that side.

Advantages:

- Simple and close to the old walker state.
- Works for one-candidate sides.

Problems:

- False for the manual difficult witnesses.
- Conflates marker placement with selected stereo-atom choice.
- Cannot explain RDKit outputs where `(7, 8)` is visibly marked while `(7, 9)`
  is the selected stereo atom.

Verdict:

Reject as the general rule. It can remain only as the degenerate
one-candidate-side behavior.

### B. Visible Edge Is Immediate Token-Flip Fact

Shape:

- Do not necessarily treat the edge as selected.
- Convert the visible marker immediately into a component token-flip fact,
  usually by comparing it to a physical edge basis or local model basis.

Advantages:

- Smaller than a new state dimension.
- Matches some currently passing cases.

Problems:

- Still collapses too early.
- Physical-basis interpretation fixes the `trans_*` bridge cases but misses
  the `cis_*` bridge cases.
- Model-basis interpretation can admit part of the `cis_*` path, but regresses
  previously passing `trans_*` behavior because later marker placement still
  depends on selected carrier choice.
- It cannot represent "marker is on the complement candidate edge" without
  already knowing the side's selected neighbor.

Verdict:

Reject as the target model. It is another adapter, not the principled
enumerator.

### C. Visible Edge Is Marker Obligation

Shape:

- Emitting a directional marker creates a typed marker-placement fact:
  component, side, emitted edge, marker token, physical edge orientation,
  local side basis, traversal role, and output slot.
- The fact does not by itself select the carrier neighbor.
- It filters assignment rows that include both carrier choice and token phase.
- For a two-candidate side, the same marker fact can be compatible with either
  the selected candidate or its complement, depending on the row.

Advantages:

- Separates the variables RDKit keeps separate:
  carrier selection, visible marker placement, and token interpretation.
- Handles shared bridge edges naturally: one physical marker can generate
  obligations for multiple adjacent components, each interpreted in that
  component's local coordinate system.
- Preserves online enumeration. The marker fact is created at emission time;
  it is not a post-hoc string cleanup.
- Provides the right seam for RDKit-specific traversal/writer rules: semantic
  carrier constraints stay separate from marker-placement policy.

Problems:

- Requires a new assignment dimension or richer token-phase rows.
- Completion filtering must understand pending marker obligations.
- More code than another local branch patch.

Verdict:

Directionally right, but not quite the best formulation. "Obligation" is a
useful implementation word when a marker has to be discharged later, but the
more general model is an online marker-placement fact that filters assignment
rows immediately.

### D. RDKit Stack Emulation

Shape:

- Mirror RDKit's writer stack and bond-direction choices more directly.
- Keep local procedural state close to `SmilesWrite.cpp` behavior.

Advantages:

- May converge quickly on RDKit exactness.
- Easier to compare with upstream code in narrow cases.

Problems:

- Bolted-on and hard to justify as a principled enumerator.
- Keeps RDKit quirks mixed into traversal.
- Does not help with semantic-vs-RDKit layer separation.

Verdict:

Useful as an oracle while investigating, not as the architecture.

### E. Marker-Placement Rows Filtered By Online Events

Shape:

- Extend each component row from carrier choice plus token phase to carrier
  choice plus token phase plus marker-placement choices.
- A directional-marker emission is a positive online event:
  `(side, edge, marker, traversal role, slot)`.
- Passing an eligible edge without emitting a marker is a negative online
  event: `(side, edge, no-marker)`.
- Runtime state stores remaining row ids. Token support is the set of marker
  or non-marker emissions that keep at least one row alive.

Advantages:

- This is the cleanest single-source-of-truth variant.
- It separates the three variables that RDKit keeps separate:
  selected carrier edge, visible marker edge, and token-phase interpretation.
- It handles complement-candidate marker placement directly. A marker on
  `(7, 8)` can be compatible with selected stereo atom `(7, 9)` because both
  are columns in the same row.
- It preserves online enumeration. Every token emitted either filters rows now
  or is rejected now; there is no post-hoc string cleanup.
- It gives a clean home to RDKit-specific behavior: marker-placement rows can
  be layer-specific, while semantic carrier rows remain separate.

Problems:

- It adds a row dimension.
- It requires negative facts for eligible marker slots that are passed without
  a marker.
- It will require replacing current local helpers such as
  `should_defer_unknown_two_candidate_side_commit` and
  `forced_shared_candidate_neighbor`, not just wrapping them.

Feasibility:

`tmp/exploration/stereo_assignment/034_estimate_marker_row_expansion.py`
estimates the current fixture scale. The largest observed marker-placement
row estimate is 256 rows for the larger pinned polyene components. The manual
difficult diene component is 32 rows. That is small enough to pursue explicit
rows before considering a custom propagator.

Verdict:

Best target. It subsumes marker obligations as an implementation detail but
uses a cleaner assignment-state formulation.

### F. Native Constraint Propagator For Marker Events

Shape:

- Keep variables for carrier choices, token phases, and marker placements.
- Apply positive and negative marker events through a small propagator instead
  of explicit row filtering.

Advantages:

- Avoids row growth if future components get much larger.
- Closest to the Z3 exploration model.

Problems:

- More machinery and harder to audit.
- Premature while explicit row counts remain small.
- Harder to expose deterministic diagnostics like "remaining row ids".

Verdict:

Keep as a fallback if explicit rows become too large. Do not start here.

## Recommended Direction

Proceed with Alternative E.

The next clean slice should not try to fix the remaining witnesses by changing
`model_token_flip_for_chosen_token` or by choosing a different immediate basis.
Instead:

1. Add diagnostic-only marker-placement row summaries for each model
   component.
2. Include, for each row, carrier choices, token phases, and marker-placement
   choices for each eligible side/candidate edge.
3. Populate positive marker events for emitted directional marker edges,
   including shared edges.
4. Populate negative marker events when traversal passes an eligible marker
   edge without a marker.
5. Verify in diagnostics that all four manual difficult witnesses have
   nonempty row state under their RDKit marker-event sequence.
6. Only after that, route token support through the row-filtering query.

The final runtime state should eventually have:

- carrier assignment state;
- token-phase assignment state;
- marker-placement assignment state;
- no independent "visible marker selected this carrier" shortcut.

## Investigation Artifacts

The current scratch scripts are:

- `tmp/exploration/stereo_assignment/031_inspect_shared_bridge_token_basis.py`
- `tmp/exploration/stereo_assignment/032_compare_shared_marker_alternatives.py`
- `tmp/exploration/stereo_assignment/033_classify_rdkit_marker_placement_samples.py`
- `tmp/exploration/stereo_assignment/034_estimate_marker_row_expansion.py`

The second script compares:

- visible edge selects carrier;
- visible edge is immediate token-flip fact;
- visible edge is marker obligation.

The third script samples RDKit outputs and shows that complement-candidate
marker placement is not just one canonical-output accident. The fourth script
checks that explicit marker-placement rows are not currently large enough to
justify a more complex native propagator.
