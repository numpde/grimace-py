# Traversal-resolved tetrahedral parity

## Objective

Add prepared tetrahedral atom-token alternatives for four-ligand centers whose
incident bond decisions are already singleton `Traversal` when the atom token
is emitted.

The token selects one parity class of future local ligand orders. Later branch
and inline commitments narrow one live 24-value order domain. Candidate
contradiction is filtered before publication, and frame completion requires the
domain to resolve to the procedurally committed order.

This slice does not model tetrahedral centers with unresolved ring-capable
incidences.

## Semantic split

Three kinds of fact remain separate:

```text
prepared atom fact
    four fixed ligands, reference order, two complete token strings

procedural walker fact
    entry bond and later bond occurrences in lexical commitment order

constraint fact
    surviving complete permutations of the four prepared ligands
```

The walker records occurrences, not parity. The CSP records compatible complete
orders, not traversal frames. Prepared spelling maps even and odd permutation
classes to complete atom tokens without performing chirality perception or
bracket construction.

## Prepared centers

A ligand is either:

```rust
Bond(BondId)
VirtualHydrogen
```

A center has exactly four reference ligands, three or four distinct incident
bonds, at most one virtual hydrogen, and two distinct nonempty token strings.
Every graph incidence at the center must occur once in the reference order.

The prepared constraint model contains one isolated variable per center with
values `0..24`. Each value denotes one permutation of the reference order.
Preparation owns masks for parity and ordered prefixes. No tetrahedral factor
is added in this slice, so the native exact-component plan does not classify
the variable as a semantic search core.

The constraint assembly boundary remains private and concrete. It builds bond
decision variables and spanning-tree factors, admits the isolated tetrahedral
variables requested by the visible surface, and returns their stable IDs. It
is not an extension registry or mutable model API.

## Order and parity

All 24 permutations are enumerated in one stable internal order. Permutation
parity is inversion parity relative to the prepared reference order.

Context contributes an initial prefix:

```text
root, no H       []
root, one H      [H]
entered, no H    [entry bond]
entered, one H   [entry bond, H]
```

Atom-token selection restricts the order variable to the intersection of the
context-prefix mask and token-parity mask. A later parent bond commitment
intersects it with the mask for the complete procedural prefix followed by that
bond. All restrictions are monotone.

For traversal-only centers every remaining attachment is a singleton and every
permutation of the remaining children is writer-realizable. The 24-value
domain is therefore exact future-feasibility state rather than a suffix-search
approximation.

## Procedural timing

Each live frame records:

```text
atom
entry bond, if entered from a parent
committed child bonds in lexical selection order
residual attachments
```

The branch bond is appended at `(`. An explicit inline bond is appended at its
bond token. An elided inline bond is appended in the same transition that emits
and enters the child atom. Child entry initializes the child frame with its
entry bond. Closing `)` only restores the already-narrowed parent.

The first root atom token selects parity while beginning the component. A later
`.` begins the selected component without selecting parity and leaves a pending
root atom frontier. Explicit branch and inline bond tokens likewise retain a
pending atom frontier.

## Atomic candidate restrictions

One private writer-state operation assembles all restrictions caused by one
semantic choice:

```text
mandatory bond role or endpoint placement
parent tetrahedral prefix, if a child bond is selected
child tetrahedral context and parity, if its atom is emitted now
```

Restrictions for the same variable are intersected. Empty intersections are
semantic contradiction. The solver receives one nonempty batch exactly once,
and procedural mutation occurs only after consistency succeeds.

This operation replaces the current one-bond specialization internally. It is
not a public event or constraint-delta framework.

## Pending atom frontiers

Pending syntax no longer implies one visible successor. Fixed atom stages have
one attempt; tetrahedral atom stages have up to two parity attempts. Backend
failure aborts the complete pending batch, while contradiction removes only the
affected parity attempt.

Prevalidation may traverse only the bounded lexical chain:

```text
dot -> atom frontier
open parenthesis -> optional bond -> atom frontier
inline bond -> atom frontier
```

It checks that at least one atom-token attempt succeeds and then stops. It does
not select parity or inspect future graph choices.

## Traversal-only boundary

Before generating tetrahedral atom-token alternatives, every incident bond's
projected role must be exactly `Traversal`. If any incident bond still supports
`Ring`, choice generation reports `ChoiceFailure::Incomplete` with a typed
`WriterIncompleteness::TetrahedralRingCoupling` reason for that
local atom event.

Preparation does not reject such centers or classify supported topology. The
following ring-coupled slice replaces this event-local tripwire with an exact
order/role relation.

## Completion

Before `CloseBranch` discards an active tetrahedral frame, silent
`FinishComponent` discards a top-level frame, or inline child entry replaces
its completed parent frame:

```text
entry bond, optional H, and committed bonds contain all four ligands
the order-variable domain is singleton
the singleton permutation equals that procedural order
```

This is a local frame check, not a whole-molecule transition scan.

## Serious alternatives rejected for this slice

### Static order/role coupling

Binary relations from every center order to incident bond decisions would make
the spanning-tree factor connect many centers into a mixed exact-search core.
That cost and lifecycle choice are deferred until ring-incidence semantics are
concrete and measurable.

### A 48-value root/entered variable

Root versus entry context changes only the required prefix. It does not change
the semantic object, which remains one permutation of four fixed ligands.

### Procedural parity repair

Storing a token bit and flipping it after child choices would scatter semantic
repair across traversal events and permit invalid candidates to be advertised.
The exact order domain keeps the delayed relation explicit.

### General atom-rendering or extension frameworks

The concrete prepared alternatives require only fixed text or two parity texts
and one isolated variable. Broader rendering, registries, factor lifecycle,
and public model mutation have no consumer here.

## Main regret risks

- The pending path changes from one forced attempt to a branch-preserving
  frontier; failure precedence must remain backend-first and category exact.
- Parent bond commitment and child atom entry happen at different visible
  tokens for explicit bonds; narrowing either center at the wrong token would
  corrupt local order.
- Completion checks must inspect the frame being discarded, not the parent
  restored afterward.
- The traversal-only tripwire must abort with implementation incompleteness rather
  than silently filter a candidate.
- Adding isolated variables must not create an exact-search descriptor or scan
  unrelated bond domains.

## Qualification

Tests independently enumerate all 24 permutations and cover four-bond root and
entered centers, virtual hydrogen in both contexts, suspended parent prefixes,
pending atom branching after `.`, `(`, explicit branch bonds, and explicit
inline bonds, adjacent centers, contradiction and backend failure semantics,
one solver batch per semantic choice, and frame-local completion.

The bounded non-stereo transition checker remains additive and green. No
ring-incidence tetrahedral center, directional bond stereo, chemical atom
rendering, public state identity, counting, or Python boundary is introduced.
