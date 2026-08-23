# Ring-bond representation plan

## Scope

This slice replaces the fixed closure-only ring-bond spelling convention with
one static per-bond CSP decision:

```text
Traversal
Ring00
Ring10
Ring01
Ring11
```

The bits refer to immutable prepared endpoints A and B. They state where the
one prepared bond token is explicitly rendered; they do not encode opening
versus closure order. Placement is meaningless for traversal and therefore
belongs in the same dependent decision variable rather than in generic
endpoint variables or dynamic factors.

Elided prepared bonds admit `{Traversal, Ring00}`. Explicit prepared bonds
admit `{Traversal, Ring10, Ring01, Ring11}`. Chemical decisions about whether
a prepared token may be elided remain outside this slice.

## Constraint ownership

- A spanning-tree edge carries disjoint Traversal and Ring value masks and
  projects only through those masks.
- `PreparedNonStereo` compiles the writer-specific constraint model, one bond
  representation variable per bond, fixed endpoint identity, placement masks,
  and endpoint-relative rendering.
- `WriterState` consumes a prepared model plus bond-variable and role-mask
  mappings. Structural attempts may receive one source-local domain refinement
  so role and placement are restricted in one solver batch.
- The visible writer derives opening and closure choices by intersecting the
  surviving plan domain with fixed-endpoint emit/omit masks.

No binary relation is added for bond placement. Pure non-stereo preparation
therefore remains a pure spanning-tree model and creates no mixed exact-search
component.

## Transition law

Traversal restricts the plan to `{Traversal}`.

At a ring opening, each supported emit/omit projection is one candidate. Its
placement refinement implies Ring, is propagated atomically with the
structural opening, and leaves at least one compatible closure projection.
Label allocation occurs only after semantic success. An explicit opening emits
the endpoint-relative bond token followed by a pending ring label.

At closure, each supported emit/omit projection is one candidate. The selected
refinement makes the representation plan singleton, closes the structural
endpoint, and emits either the label or bond token followed by a pending label.

Component completion requires both clean visible labels and singleton plans for
every represented ring bond.

## Alternatives rejected

- Generic role/endpoint binary factors would unnecessarily pull every edge
  into the mixed semantic search core through the spanning-tree projector.
- Two sequential restrictions (`Ring`, then placement) would duplicate
  propagation and expose an intermediate state that is not a semantic choice.
- Dynamic factor or variable lifecycle adds no expressive power to a static,
  monotonically narrowed plan.
- Deriving arrow text from opening order would lose fixed-endpoint dative
  orientation.

## Acceptance gates

- role-partitioned spanning projection matches exhaustive assignment support
  on multi-valued bounded fixtures;
- placement restrictions force and reject structural roles exactly;
- pure prepared models compile no mixed exact descriptor;
- opening and closure choices match an independent mask oracle;
- explicit bonds admit opening-only, closure-only, and both-end placement;
- elided bonds admit only no-token placement;
- all prepared bond-token variants and both fixed endpoint directions are
  exercised;
- one candidate performs one solver restriction batch and preserves its source;
- open rings retain compatible closure support and closed rings have singleton
  plans;
- component boundaries retain no open label or unresolved ring plan;
- bounded visible exploration has no writer invariant failure.

## Deferred work

Atom rendering, chemical elision, directional stereo, tetrahedral parity,
dynamic factor lifecycle, public APIs, snapshots, counting, pending caches, and
persistent ring-label storage remain outside this slice.
