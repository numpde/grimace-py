# Frontier Support-State Checkpoint

Branch: `stereo-constraint-model`

Date: 2026-05-18

## Purpose

This is the North-Star checkpoint after the shared-carrier boundary and
visible-marker basis work. It synthesizes the minimal online
frontier/support-state model that should own stereo support from here, before
more runtime mutation.

The goal is not a new abstraction layer for its own sake. The goal is to make
support-shaping decisions come from explicit online facts and survivor queries,
not from walker-field repair, graph-topology shortcuts, or local token patches.

## Current Status

The branch now has three important pieces of evidence:

- shared-carrier boundary deltas are measured and pinned;
- visible-marker policy variants are diagnostic payload, not ad hoc local
  inspection;
- the old deferred-token topology guard has been replaced in runtime by a
  topology-free frontier predicate:
  `remaining non-selected visible-edge basis && remaining shared visible-edge
  basis`.

That means `deferred_token_legacy_topology_guard_applies` is no longer a
runtime support rule. It remains only as a legacy diagnostic comparison.

The larger remaining procedural surface is `resolved_selected_neighbors` and
the paths that recover selected carrier neighbors from mixed walker fields
instead of asking the support boundary for the currently surviving assignment
state.

## Source Of Truth

The source of truth should be a support-state query derived from the current
walker frontier. It should consume facts and return surviving model state.

Inputs:

- component carrier facts;
- token-phase facts;
- known committed token flips;
- inferred token observations;
- positive marker events;
- no-marker events;
- RDKit writer-policy traversal facts;
- pending deferred marker obligations.

Outputs:

- surviving carrier assignment ids by component;
- surviving token-phase assignment ids by component;
- surviving marker-row ids by component;
- selected carrier neighbor when it is forced by surviving assignments;
- legal token flips for the current deferred token;
- marker obligations still pending at the current frontier;
- trace facts needed to distinguish completion identities.

The important property is monotonicity: every emitted token adds facts and
shrinks or preserves survivor sets. It must not repair completed strings after
the fact.

## Ownership Boundaries

Semantic model owns:

- component membership;
- side domains;
- carrier assignments;
- token-phase assignments;
- marker-placement rows;
- parser-backed semantic constraints.

RDKit writer-policy layer owns:

- traversal observations;
- selected visible marker slots;
- no-marker observations;
- ring-closure and branch marker placement;
- deferred marker obligations;
- RDKit-specific suppression or preference among semantically valid marker
  spellings.

Walker owns:

- prefix text;
- action stack;
- frontier branching;
- carrying already emitted facts forward.

The walker should not own final interpretation of selected carrier neighbors
or deferred-token legality when those depend on coupled component state.

Diagnostics own:

- legacy comparisons;
- policy-variant deltas;
- row ids and survivor counts;
- current-red gap classification.

Diagnostics may mention legacy topology or field-derived selected-neighbor
state. Runtime support should not.

## Required Invariants

- Public runtime support remains exact RDKit writer-string support for the
  supported writer regime.
- Parser-backed semantic equivalence is useful evidence, but it is not a
  substitute for RDKit writer-string parity in public support tests.
- A visible marker basis is accepted only with a row-survivor witness.
- A shared visible marker is interpreted per component, then joined through the
  single emitted token.
- Known token flips override inferred observations without duplicating
  constraints.
- Inferred token observations must not enter runtime as committed
  `StereoTokenFlipFact`s.
- Completion identity must include all support-shaping facts consumed by the
  support boundary, including marker events.
- Legacy comparisons must be named as legacy or shadow diagnostics.

## Rejected Alternatives

Keep graph-topology guards
: Rejected for runtime. Topology correlated with the first witness, but it was
  not the invariant. The row/frontier predicate is more direct and now matches
  the legacy diagnostic set on pinned visible-marker fixtures.

Repair selected carriers after the walk
: Rejected as a final design. It can explain old behavior, but it creates dual
  truth between walker fields and model survivors.

Use parser equivalence as the public oracle
: Rejected for public support. It is a semantic layer, not RDKit writer parity.

Switch immediately to a native propagator
: Deferred. Rows are still the clearest executable specification, and current
  pinned row counts do not justify a rewrite.

Materialize every possible completed spelling and filter later
: Rejected. It violates the online support constraint and obscures token-level
  decoder semantics.

## Minimal Runtime Shape

The straight-line target is:

1. Extract a named support-state query that returns survivor ids and forced
   selected carrier neighbors from the current facts.
2. Make selected-neighbor lookup consume that query rather than
   `resolved_selected_neighbors_from_fields`.
3. Keep old field-derived selected neighbors only as a shadow assertion during
   migration.
4. Route deferred-token support and commit through the same support-state
   query.
5. Delete or demote any remaining support-shaping logic that reads raw walker
   fields after the support-state query has already consumed the same fact.

This is still online: each frontier state computes support from facts available
at that prefix only.

## Exit Criteria

`resolved_selected_neighbors` can stop being runtime support logic when:

- every runtime selected-neighbor decision is available from the support-state
  survivor query;
- field-derived selected neighbors are used only for shadow diagnostics or
  removed;
- shared-carrier delta fixtures show no lost terminal support;
- exact public invariants pass;
- pinned RDKit parity passes;
- known-gap fixtures do not show support broadening outside explicitly
  promoted cases.

`deferred_token_legacy_topology_guard_applies` can be removed when:

- diagnostics no longer need to compare the old topology-gated policy variant;
- the frontier predicate remains pinned against the visible-marker fixture
  corpus;
- no open Backlog or Decision row references the topology-gated variant as
  active evidence.

The next runtime mutation should therefore target selected-neighbor ownership,
not new marker-placement special cases.

## Next Slice

Create a support-state selected-neighbor query with three outputs:

- forced selected neighbor per side when the survivor set makes it unique;
- unresolved marker for sides still ambiguous under current facts;
- survivor-count diagnostics for carrier, token-phase, and marker rows.

Use it first as a shadow comparison against current
`resolved_selected_neighbors`. Only after pinned deltas are visible should it
replace runtime selected-neighbor lookup.
