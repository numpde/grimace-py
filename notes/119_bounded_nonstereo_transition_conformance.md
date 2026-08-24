# Bounded independent non-stereo transition conformance

## Objective

Before tetrahedral semantics changes the writer relation, qualify the complete
prepared non-stereo kernel against a test-only transition model that is
algorithmically and representationally independent of production.

For every bounded reachable state, compare production and oracle choices as:

```text
{ emitted text, normalized semantic successor }
```

The comparison establishes one-step soundness, completeness, successor
identity, equal-text branch preservation, source immutability, acceptance, and
bounded no-dead-end evidence. It does not use future-string support to validate
ordinary candidates.

## Shared primitive input

One test-only `FixtureSpec` contains only atom strings and raw fixed-endpoint
bond facts. Two independent builders consume it:

```text
FixtureSpec
    |-- production builder -> PreparedGraph/PreparedMolecule/PreparedNonStereo
    `-- oracle builder     -> dense adjacency/components/rendering/plan domains
```

The oracle never reads derived facts back from the production surface. In
particular, it does not inspect prepared components, constraint factors, role
partitions, residual partitions, frontiers, or solver state.

## Independent oracle

The oracle owns dense semantic state:

```text
visited atoms
per-bond progress and orientation
active atom and ordered branch-return atoms
oracle-owned bond-plan domains
ring-label ownership
typed pending lexical commitment
```

It stores no residual attachments. For any active or suspended atom it
recomputes connected components of the unvisited induced graph and groups raw
unrepresented incidences from scratch.

The oracle owns its own plan vocabulary:

```text
Traversal, Ring00, Ring10, Ring01, Ring11
```

After every restriction, it enumerates plan assignments independently within
each original molecular component. It retains exactly assignments whose
Traversal-valued bonds form a spanning tree, then projects the survivors back
to every current domain. This scope is specific to the present prepared
non-stereo model, which has no cross-component semantic factor.

## Transition law

The unrooted initial state exposes every atom as a distinct root. The first root
emits its atom. A later `.` choice has already selected, visited, and activated
its next root; only that root's atom text remains pending.

At an active atom, ring phase precedes child phase. Ring phase exists when a
ring closes locally or one residual attachment has multiple active incidences.
An opening may remove an incidence only when its plan supports Ring and another
incidence in that attachment remains Traversal-capable after exact projection.
Emit and omit choices are fixed-endpoint plan restrictions. A closure uses the
same fixed-endpoint projections and must resolve its plan to one value.

With no ring phase, every attachment has one incidence. Exactly one attachment
makes its sole child inline. With two or more attachments, every attachment is
independently selectable as a branch; no inline continuation is preselected.
No attachment closes a branch visibly with `)`, or silently completes a
top-level component. `FinishComponent` is never an observable action.

Branch selection timing is token-precise. `(` restricts and commits the chosen
bond and child but leaves the parent active and the return stack unchanged. An
explicit branch bond token does the same. Only the branch child atom consumes
the selected parent attachment, pushes the remaining parent frame, traverses
the bond, and activates the child. Inline bond tokens likewise leave the parent
active until the child atom is emitted.

Pending commitments retain the selected atom, bond, endpoint, and label. Only
their deterministic lexical chain may be followed during candidate
prevalidation, with a hard depth bound of two. The simulated continuation is
discarded: the compared published successor retains its pending token.
"Normalization" in this checker means only silent `FinishComponent`
consumption and never consumes pending visible syntax. Semantic branching is
never followed recursively.

## Immediate validity

Candidate validity is stated through local facts rather than asking whether a
successor has future choices:

- every domain is the nonempty exact assignment projection;
- traversed bonds are Traversal, open rings retain Ring support, and closed
  rings have one Ring plan;
- graph progress and traversal orientation agree with the primitive graph;
- every begun original component has one traversal root;
- labels correspond exactly to open rings or a pending explicit closure;
- each pending variant agrees with its already-committed structural object;
- declarative attachments have the required ring/child/completion shape;
- top-level completion has no open ring or pending syntax.

An ordinary empty choice list is valid only for acceptance. Bounded native
fixtures must not report backend, spelling, or writer-invariant failure.

## Production observation boundary

`cfg(test)` accessors copy raw production facts only:

```text
visited bits and bond progress
active and suspended stored frames
current bond-plan domains
label ownership
typed pending commitment
effective test spelling-label limit
```

They do not call choice generation, frontier derivation, residual
recomputation, normalization, or acceptance. The checker normalizes incidence
groups and recomputes acceptance itself. Persistent identity, residual
component IDs, queues, counters, and allocation layout are excluded.

If two production states share the declared semantic snapshot, their projected
choice sets must also agree. A disagreement means either the snapshot omitted a
semantic fact or production depends on nonsemantic state.

For each stack position, the production frame's stored attachments are compared
with attachments independently recomputed for the oracle's corresponding
suspended atom. Stack order is exact; incidence and group order is normalized.

## Bounded envelope

The exhaustive layer starts unrooted from all 76 labelled simple graphs with
zero through four atoms, distinct atom strings, and all-elided bonds. Tests are
split into through-three-atom, four-atom acyclic/disconnected, and four-atom
cyclic groups.

Curated explicit fixtures add:

- each prepared explicit token in a triangle and both fixed-endpoint orders;
- two simultaneously open explicit rings with distinct live labels;
- fused or bridged surfaces with multiple explicit ring plans and closure
  orders;
- disconnected explicit cyclic components with silent completion and label
  reuse.

Each fixture records state, transition, queue, open-ring, forced-chain, and
exact-assignment work. State, transition, queue, and per-projection assignment
limits are pinned with measured margins from the first green run, so accidental
transition or oracle-search growth fails explicitly.

## Forbidden dependencies

The oracle does not call production frontier or residual logic, either solver
backend, South Star 1, RDKit, the former action planner, normalization helpers,
continuation search, or terminal-support enumeration.

## Completion gate

The slice is complete when production and oracle choice sets agree exactly for
the bounded envelope, sources remain unchanged, identical duplicate choices
are absent, attachment/frame/pending/label/plan facts agree, acceptance agrees,
and simultaneous explicit rings are reached.

Production behavior changes only for an independently reproduced mismatch,
with the fix isolated from expansion of the checker envelope.
