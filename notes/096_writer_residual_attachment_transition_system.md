# Writer Residual Attachment Transition System

Date: 2026-05-31

Context:

- `notes/095_online_serializer_desiderata.txt`
- `python/grimace/_south_star1/SPEC.md`

## Purpose

Define the next mathematical direction for `WRITER_SHAPED` cyclic support.

The core object should not be a ring spine, cycle basis, spanning tree, or
precomputed render program. It should be a prefix state for a graph grammar:
a partial SMILES word together with live graph obligations, open writer frames,
ring-closure obligations, and residual stereo constraints.

The general solution is an online residual attachment transition system over
the molecule's block-cut structure, with ring closures represented as
first-class edge obligations and stereo represented as event-driven residual
factors.

## State Object

Let the prepared molecule graph be:

```text
G = (V, E)
```

A writer prefix state should contain at least:

```text
W  atoms whose syntax occurrence has been emitted
T  bonds emitted as tree/entry bonds
C  bonds completed as ring-closure bonds
P  pending half-emitted syntax actions
F  open writer frames: active atom and branch-return atoms
R  open ring endpoints, label allocator state, and closure obligations
S  residual stereo store: variables, assignments, pending factors, closed factors
Pi writer-policy state: component roots, token domains, label policy
```

`T` is retrospective state, not a preselected full traversal tree. A completed
string may imply a discovered tree and closure set, but that decomposition is
the result of transitions, not an input plan.

## Residual Attachments

At a prefix, let:

```text
U = V \ W
```

The live graph work is not a list of unvisited neighbor atoms. It is a set of
residual attachment components:

```text
Omega = a connected component of the not-yet-written graph,
        together with its boundary incidences into the written/open graph.
```

Boundary must be an incidence set, not only an atom set. A simple ring can have
two distinct boundary bonds to the same visited atom. A fused block can have
several boundary incidences and internal cyclic rank.

This model collapses correctly in the acyclic case: every residual attachment
component has one boundary incidence, so the writer sees ordinary child
obligations.

For cyclic systems, multiple boundary incidences into the same residual
component are one obligation class. They must not become arbitrary branch and
continuation children just because a spanning tree cut would separate them.

## Branch Semantics

Branches should range over residual obligation classes, not raw edges.

At an open writer atom `a`, group live boundary incidences by residual
attachment component. The writer policy acts on those quotient obligations:

```text
one residual component = one live obligation
multiple incidences into the same component do not become multiple branches
```

The acyclic unique-child rule is the rank-zero case:

```text
if exactly one residual obligation class remains at an open atom,
it is inline continuation, not a parenthesized branch.
```

The cyclic generalization is:

```text
if exactly one residual attachment component remains at an open atom,
the writer may enter or advance that component, but may not split it into
arbitrary branch children.
```

This removes same-ring branch overgeneration structurally. In a benzene entry,
the two same-ring directions belong to one residual component. The search space
never contains a choice to make one direction a side branch and the other an
inline continuation.

## Ring Closures

A ring closure is not an edge outside a preselected spanning tree. It is an edge
obligation whose two syntax endpoints are emitted at different times.

Each bond should be in exactly one semantic class:

```text
latent residual edge
boundary incidence from a residual component
pending post-bond entry
emitted tree-entry bond
open closure endpoint
closed closure bond
invalid or stale obligation
```

Transitions move atoms and bonds between these classes:

```text
enter child:
  boundary incidence -> pending entry -> tree-entry bond + visited atom

open ring endpoint:
  boundary incidence or visited-visited residual edge -> open closure endpoint

close ring endpoint:
  open closure endpoint + current endpoint -> closed closure bond

branch open:
  residual obligation class is suspended under a branch frame

branch close:
  return to an owner of remaining obligation classes

dot/component boundary:
  legal only when the current component has no residual obligations
```

No bond should be in two classes. No bond should be in zero classes.

## Open-Frame Ownership

SMILES syntax is local. A future token that must be attached to a visited atom
is legal only while that atom is still syntactically open, or while the required
endpoint has already been emitted.

Define:

```text
OpenAtoms(state) =
  {active atom}
  union {branch-stack return atoms}
  union {pending-entry parent}
  union {atoms with open closure endpoint obligations}
```

Every unfulfilled graph obligation must be owned by an open atom, an already
open closure endpoint, or a cyclic-block obligation whose boundary endpoints
are still syntactically representable.

Conversely, every open branch frame must still own unresolved work. A stale
branch frame is not valid merely because its return atom is reachable in the
written tree.

The current acyclic snapshot audit enforces this principle for child
obligations. Cyclic support should extend the same invariant to residual
attachment components and ring-closure endpoints.

## Block-Cut Metadata

Static graph decomposition is useful as metadata:

```text
bridges
articulation atoms
biconnected components
block-cut tree
edge membership in cyclic blocks
component cyclic rank
atom/bond incidence maps
```

It must not become writer semantics:

```text
chosen cycle basis
chosen ring cuts
all spanning trees
pre-labeled branch/continuation roles
precomputed render programs
```

The block-cut tree separates bridge obligations, articulation-side obligations,
cyclic-block obligations, and exocyclic substituent obligations. It does not
choose the serialization plan.

Inside a cyclic block, the dynamic object is an online residual attachment
decomposition. As atoms are emitted, the unwritten graph may split into smaller
attachment components. Those splits create real separate obligations. Branches
inside a cyclic block are legal only when the residual obligation structure has
actually split into distinct live obligations under the writer policy.

## Stereo Events

The graph-obligation system must emit semantic events, not infer chemistry from
rendered strings:

```text
atom occurrence emitted
bond occurrence emitted
branch opened
branch closed
ring endpoint emitted
ring endpoint paired
local order closed
component boundary emitted
EOS finalized
```

The residual stereo CSP consumes these events. Every stereo variable assignment
must be explained by writer events, and every event that creates stereo syntax
must be represented in residual state.

Delayed factors should follow a lifecycle:

```text
created when the first relevant occurrence is emitted
pending while the scope is incomplete
closed exactly once when the scope is complete
represented as an ordinary residual factor after closure
auditable in snapshots
```

Ring endpoint events are first-class stereo events. A ring digit may be one half
of a directional carrier or ring-pair factor, so cyclic support cannot add ring
closures as a late rendering pass.

## Canonical State Identity

Canonical writer state should include semantic continuation data:

```text
visited atoms
edge obligation partition
open writer frames
pending entry state
open and closed closure endpoints
ring-label allocator state
residual attachment component identity
block/cyclic obligation state
residual stereo store
delayed stereo factors
component/root policy state
output-token boundary state
```

It should exclude derivation history:

```text
chosen cycle basis
chosen spanning tree explanation
debug trace history
already-rendered suffix
temporary DFS choices that no longer affect future syntax
```

Start conservative. Merge exact semantic states first. Automorphism-aware
merging can come later. Overmerging cyclic states can silently corrupt counts;
undermerging retains proof-enumerator behavior under a different name.

## Count Semantics

The frontier remains determinized by emitted token text.

Keep separate:

```text
support_count     distinct final strings
completion_count  weighted completing witnesses behind a prefix or token
multiplicity      immediate or retained path/state weight
```

Cyclic support increases the risk of confusing these counts because several
closure orders, labels, or retrospective tree explanations may converge to the
same rendered continuation.

## Main Pitfalls

Avoid these regressions:

```text
branching over edges instead of residual components
using a cycle basis as writer semantics
forgetting token locality at frozen atoms
treating ring labels as cosmetic text
letting closure endpoints bypass residual stereo events
overmerging states with different open endpoints or label allocator state
undermerging by retaining arbitrary derivation history
confusing support count with witness count
allowing residual obligations attached to frozen atoms
special-casing simple rings as the long-term model
```

## Synthesis

`WRITER_SHAPED` should be a deterministic frontier construction over canonical
residual states.

Each state consists of:

```text
an emitted-prefix graph
an open writer-frame stack
a partition of graph edges into emitted, pending, open-closure, closed-closure,
  and residual classes
residual attachment components of the unvisited graph
cyclic/block metadata used only as structural constraints
ring-label allocator state
residual stereo CSP state
writer-policy state
```

Transitions consume one legal residual obligation and emit one token or
token-level event. Branches range over residual obligation classes. Rings are
represented by closure obligations. Stereo constraints are activated and closed
through writer events. Counts are computed over determinized emitted-token
frontiers, while witness multiplicity is carried separately by weighted state
cursors.

This is the conceptual center for cyclic `WRITER_SHAPED` work.
