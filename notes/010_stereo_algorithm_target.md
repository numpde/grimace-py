# Stereo Algorithm Target

Branch: `stereo-constraint-model`

## Purpose

This note fixes the implementation goal in algorithmic terms. The target is
not a formal proof artifact and not a runtime Z3 dependency. The target is a
principled enumeration algorithm: precompute the finite stereo choices implied
by the molecule, then let the online walker narrow those choices as traversal
facts become known.

The current code has useful diagnostics and fixtures, but the runtime still
contains local procedural heuristics. Those should be replaced by one shared
algorithmic source of truth.

## Algorithmic Goal

Startup should compile each molecule into small independent stereo components.
Each component contains a finite set of valid assignments over carrier choices
and orientation/phase choices. During enumeration or decoding, walker state
stores only the remaining compatible assignments, plus any marker-placement
obligations whose placement has not yet become forced.

The online loop should look like this:

1. Build stereo components and allowed assignments before traversal.
2. Start each component with its full allowed-assignment set.
3. When traversal observes a carrier edge, ring action, branch action, or atom
   context, emit a typed fact.
4. Restrict each touched component to assignments compatible with the fact.
5. If no assignment remains, prune the branch.
6. If a directional marker is forced at the current output boundary, emit it.
7. If several marker choices remain possible, branch online.
8. If a marker is not placeable yet, keep a pending obligation tied to the
   component, side, carrier edge, and traversal event.
9. Accept terminal states only if all components have at least one compatible
   assignment and all marker obligations are discharged.

This is the intended replacement for scattered local repairs. The walker
should ask the model what remains possible; it should not patch selections or
flip tokens after constructing a successor.

## Role of Z3

Z3 is an exploration tool, not the production engine. It is useful for finding
the constraint language and checking small witnesses, especially when RDKit
and OpenSMILES intuition disagree.

Production should use a deterministic Rust analog:

- enumerate small component assignment domains directly;
- store assignments as compact tuples or bitsets;
- precompute transition/filter tables where useful;
- answer completion and forced-emission queries without invoking a solver.

If a component becomes too large for explicit tuples, the production analog can
move to a native propagator. That is still different from depending on Z3.

## What Makes Current Runtime Heuristic-Like

The issue is not that individual helpers are necessarily wrong. The issue is
that truth is distributed across traversal-time conditionals instead of one
component assignment state.

Current replacement targets, in algorithmic order:

- `resolved_selected_neighbors_from_fields`: forces ambiguous shared-edge
  groups after the fact. Target shape: shared-edge coupling is encoded in the
  allowed assignment table; resolution is reading the remaining assignment set.
- `forced_shared_candidate_neighbor`: nudges unresolved sides toward a shared
  carrier candidate. Target shape: observing one side filters the component;
  if the other side is then forced, the model reports that directly.
- `normalize_component_token_flips`: checks token-flip consistency after
  successor construction. Target shape: token orientation/phase is part of the
  assignment state, so inconsistent branches are never constructed.
- `defer_coupled_component_phase_if_begin_side_is_unresolved` and
  `commit_coupled_component_phase_from_deferred_part`: timing-dependent phase
  commitments. Target shape: phase remains an assignment dimension until facts
  force it.
- `emitted_edge_part_generic` and `emitted_isolated_edge_part`: mix edge
  emission, carrier selection, token deferral, and RDKit writer quirks. Target
  shape: edge emission produces facts; model queries decide selected carrier
  and marker obligations.

## Current Progress In This Frame

Useful scaffolding already exists:

- `StereoConstraintModel` separates semantic, RDKit-local, and RDKit-traversal
  layers.
- Pinned fixtures capture RDKit-versioned assignment counts and witness
  behavior.
- `_stereo_constraint_output_facts` exposes marker provenance as graph and
  traversal facts.
- Rust diagnostics now build traversal facts and project one ring-marker
  placement rule.

But this is still diagnostic. Runtime support is not yet driven by the finite
assignment state, and the traversal layer does not yet define the full marker
placement algorithm.

## Non-Goals

These are explicitly not the target:

- a formal specification that is not executable by the walker;
- a post-hoc string projection pass over completed SMILES;
- a growing list of RDKit-specific local `if` branches in traversal code;
- a runtime dependency on an external solver.

## Implementation Direction

The next implementation phases should be algorithm-first:

### Phase A: Assignment State Boundary

Add a runtime-side component assignment state that stores the remaining allowed
assignment ids per component. Initially keep behavior unchanged and update this
state in shadow mode.

Concrete work:

- Give every allowed component assignment a stable id.
- Add functions to filter assignment ids by `StereoConstraintFact`.
- Add shadow state to walker diagnostics or debug state.
- Assert current selected-neighbor facts always leave at least one semantic
  assignment.

### Phase B: Replace Shared-Carrier Repair

Move shared-carrier coupling out of `resolved_selected_neighbors_from_fields`
and `forced_shared_candidate_neighbor` into assignment filtering.

Concrete work:

- Add tests around the reduced porphyrin traversal-coupling fixture.
- Show that observing one shared carrier side forces or excludes the coupled
  side through assignment-state narrowing.
- Keep old helpers only as adapters until outputs match.

### Phase C: Token Phase as Assignment Dimension

Move component token flip/phase consistency into the model state.

Concrete work:

- Add assignment fields for token orientation/phase.
- Replace `normalize_component_token_flips` checks with model filtering.
- Ensure branch construction never creates a state whose phase has no
  compatible assignment.

### Phase D: Marker Obligations

Replace eager marker placement with pending obligations driven by facts and
remaining assignments.

Concrete work:

- Define obligation fields: component, side, carrier edge, marker token,
  source traversal event, eligible future slots.
- Emit obligations when a selected carrier edge becomes known but placement is
  not forced.
- Discharge obligations at online output boundaries.
- Keep the Rust projection diagnostic until runtime behavior matches it.

### Phase E: Remove Heuristics

Delete or shrink each suspicious helper only after the assignment-state path
explains the same behavior and tests pin the replacement.

## Success Criteria

The implementation is on target when:

- every walker stereo decision is either a graph traversal action or a query
  against component assignment state;
- RDKit-specific behavior appears only in named RDKit writer layers;
- terminal support is produced online, with no finished-string cleanup pass;
- known witnesses pass through the same generic assignment/filter/obligation
  mechanism, not bespoke string or molecule cases;
- Z3 scripts remain useful for exploration but are not needed in CI or runtime.
