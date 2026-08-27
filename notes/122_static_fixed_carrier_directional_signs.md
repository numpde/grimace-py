# Static fixed-carrier directional signs

## Decision

The first directional-stereo slice prepares one binary canonical sign for each
physical carrier bond and compiles fixed-carrier double-bond relations into
always-active binary XOR constraints.

Directional sign is separate from bond representation:

```text
bond representation = Traversal or Ring(endpoint placement)
carrier sign        = SlashAtFixedA or BackslashAtFixedA
```

The sign remains meaningful while the carrier's structural role is unresolved.
No directional factor activation, component phase variable, new factor type, or
mixed exact-search core is introduced.

## Prepared input

A directional relation identifies one double bond, its fixed left and right
endpoints, the candidate carriers at each endpoint, and the required XOR of the
outward signs. This slice compiles relations with exactly one carrier on each
side. Multi-carrier sides remain admitted prepared facts but produce private
local incompleteness when a candidate carrier event is reached.

Preparation validates that:

- the configured double bond joins the fixed endpoints and has prepared token
  `Double`;
- every carrier is incident to its configured endpoint and is not the double
  bond;
- carrier entries are distinct and use an elided, explicit-single, or aromatic
  base token;
- one physical carrier reused by several relations owns one sign variable;
- each compiled parity component is consistent.

The canonical sign is relative to immutable `PreparedBond::a()` and
`PreparedBond::b()` endpoints:

```text
0 = SlashAtFixedA
1 = BackslashAtFixedA
```

At fixed endpoint A, the values render as `/` and `\`. At fixed endpoint B,
the glyphs reverse.

For carrier `c` viewed from double-bond endpoint `e`:

```text
endpoint_flip(c, e) = 0 when e == c.A, otherwise 1
outward_sign(c, e)  = canonical_sign(c) xor endpoint_flip(c, e)
```

For a prepared outward relation `left xor right = q`, preparation compiles:

```text
canonical_left xor canonical_right
    = q xor endpoint_flip(left, left_endpoint)
        xor endpoint_flip(right, right_endpoint)
```

This formula is the single source of truth for traversal reversal, shared
carrier reuse, and parity-component validation.

## Static component compilation

Carrier bonds are nodes in a parity graph and prepared relations are XOR edges.
Preparation checks each component, chooses its minimum `BondId` as root, and
computes every carrier's offset from that root. Each non-root sign variable is
joined to the root variable by one ordinary always-active binary relation.

The resulting factor graph is an acyclic star. Both global sign phases remain
available initially. Restricting one sign propagates the whole connected sign
component through ordinary binary revision; it causes neither binary exact
search nor mixed exact search.

If a candidate-sharing region includes a multi-carrier side, the whole region
is marked incomplete rather than partially compiled.

## Writer transitions

A compiled carrier represented as a traversal edge emits exactly one `/` or
`\` token. The directional token replaces the prepared elided, `-`, or `:`
traversal spelling.

For an inline carrier, one visible candidate atomically restricts the bond to
`Traversal`, restricts the canonical sign, applies any parent tetrahedral-order
restriction, commits the inline structural child, emits the endpoint-relative
directional glyph, and leaves the child atom pending.

For a branch carrier, `(` commits the traversal child and parent local order but
does not select a sign. Its pending frontier emits `/` or `\`, restricts the
sign, and then leaves the branch atom pending. The distinct glyph token is the
sign-selection event.

Selecting a shared carrier sign propagates other connected sign domains, but no
other carrier token is emitted before its own structural event.

Ring opening or closure on a directional carrier is private incompleteness for
this slice. Likewise, touching a candidate in an unresolved multi-carrier
region is private carrier-selection incompleteness. Either condition aborts the
complete choice batch; it is not candidate contradiction and does not force the
carrier to traversal.

## Completion law

At molecular-component completion, every compiled directional carrier in that
component must have been represented as traversal, its sign domain must be
singleton, and its directional token must have been committed exactly once.
The static XOR factors then guarantee the complete component relation.

This check belongs at top-level component completion, not on every transition.

## Qualification boundary

Qualification must separately establish:

- canonical/outward sign conversion and fixed-endpoint rendering;
- equivalence of each original parity graph and its root-relative binary star;
- native/exhaustive projected-domain equality with zero exact-search runs;
- independence of sign and bond-role domains before concrete writer events;
- inline, branch, reverse-traversal, shared-carrier, tetrahedral, disconnected,
  ring-incomplete, and multi-carrier-incomplete behavior;
- one solver restriction batch per sign choice and fail-fast batch semantics;
- focused one-step agreement with an independent directional oracle.

No carrier selection, directional ring endpoint placement, stereo perception,
suffix construction, factor retirement, or public state API belongs to this
slice.
