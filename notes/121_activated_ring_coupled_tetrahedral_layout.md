# Activated ring-coupled tetrahedral layout

## Decision

The first ring-coupled tetrahedral implementation uses one prepared local role-
pattern variable and one latent layout factor per meaningful writer-entry
context. The factor is activated exactly when the atom token is emitted and
remains active thereafter.

This is the first concrete use of factor activation in South Star 2. The
lifecycle is deliberately one-way:

```text
prepared latent -> active forever
```

There is no runtime factor creation, retirement, extension registry, public
capability class, or serialized activation delta.

## Local semantic variables

Each prepared four-ligand center retains its existing 24-value complete-order
variable. Its explicit bond ligands, in reference-order position, define the
bits of one role-pattern variable:

```text
bit 0 = Traversal, bit 1 = Ring
```

Four bond ligands produce 16 pattern values. Three bond ligands plus one
virtual hydrogen produce eight. The virtual hydrogen has no role bit.

The atom event restricts the pattern variable to every role assignment admitted
by the prospective writer frame:

```text
entry bond, if any                    = Traversal
ring endpoints already waiting here  = Ring
each residual attachment             = exactly one Traversal incidence
                                       and every other incidence Ring
```

This restriction does not choose a traversal tree or ring subset. It retains
the Cartesian product of every attachment's valid traversal incidence.

## Context-specific latent factor

Preparation creates a root-context factor and one entry-context factor per
incident bond. Each factor contains:

```text
complete-order variable
role-pattern variable
incident bond-decision variables with their role partitions
precompiled supported order values for each role-pattern value
```

The fixed context prefix is:

```text
root                 []
root with H          [H]
entered              [entry bond]
entered with H       [entry bond, H]
```

An order-pattern pair is supported exactly when the order starts with the fixed
prefix, the entry bond is Traversal, and the remaining bond occurrences contain
all Ring roles before all Traversal roles. The factor also requires every
incident bond decision to belong to the role class selected by its pattern bit.
Values inside a bond role class remain indistinguishable to this factor, so
ring-endpoint placement alternatives are preserved.

The projector enumerates at most 24 by 16 order-pattern pairs. It projects exact
support onto the order variable, pattern variable, and each incident bond
decision without enumerating writer suffixes.

## Solver lifecycle contract

The solver-neutral transition is:

```rust
transitioned(restrictions, activate)
```

It represents exactly the source assignments satisfying all restrictions,
previously active factors, and newly activated factors. Contradiction is exact;
backend failure remains operational. Activation is idempotent. The persistent
active-factor set is semantic solver state and is observable only through the
private solver contract and test-only writer observations.

Initial consistency enforces only always-active factors. Propagation skips
latent factors, enqueues a newly activated factor, and enqueues only active
neighbors after a domain reduction.

## Exact routing

The native solver compiles immutable potential semantic components from
always-active binary factors, latent tetrahedral factors, and spanning-tree
projectors. Runtime exact search derives its core from the factors active in the
current state:

```text
always-active binary variables
+ order and pattern variables of active tetrahedral factors
```

Incident bond variables remain projected structural variables. Search branches
on the smallest unresolved active core domain. Inactive centers contribute no
core variable and receive no factor revision.

When every structural variable shared with active tetrahedral factors has a
singleton role class, local tetrahedral projection and spanning-tree projection
are already exact for those centers. Their unresolved local orders do not by
themselves trigger mixed search.

## Writer transition boundary

Atom-token preparation needs the frame produced by the structural event. The
candidate path therefore constructs a private traversal fork first, derives the
prospective frame and its pattern domain, and then performs one solver
transition containing all restrictions and at most one activation. The fork is
published only after consistency succeeds.

The same one-transition rule applies to ring, branch, and inline events. The
walker records one local occurrence stream:

```text
entry bond, if any
virtual H, if prepared
ring occurrences
traversal child occurrences
```

An explicit ring bond token commits its occurrence before the pending label;
the label never commits it again. Branch occurrence is committed at `(`, and
inline occurrence is committed at the explicit bond token or combined elided
child-atom token.

## Completion

Before discarding a tetrahedral frame, the writer requires:

```text
singleton complete-order domain equal to the procedural occurrence order
singleton role-pattern domain equal to procedural Ring/Traversal occurrences
incident bond roles equal to that pattern
the frame's root or entry context factor active
```

The active factor remains after frame completion with singleton core domains.

## Rejected alternatives

An always-active factor would connect unvisited centers to spanning-tree
components and create premature semantic search. Independent order and role
domains would lose the correlation needed for future feasibility. A ring-prefix
length cannot identify the one traversal incidence retained by each residual
attachment. Dynamic factor creation or retirement adds lifecycle machinery
without semantic need.

## Qualification boundary

Independent tests must cover the factor truth table, event-derived pattern
domains, native/exhaustive activation equivalence, inactive-factor locality,
mixed active centers, and curated one-step writer successors. The established
non-stereo transition oracle remains graph-general and must stay green. No
ordinary choice path may enumerate terminal strings or reachable writer states.
