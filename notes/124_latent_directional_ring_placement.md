# Latent directional ring placement

## Decision

A compiled directional carrier keeps its physical mark independent of its
structural role:

```text
CarrierMark = Plain | SlashAtFixedA | BackslashAtFixedA
BondRole    = Traversal | Ring
```

When the carrier becomes a ring bond, one additional local plan records where
its visible bond token occurs relative to the immutable prepared endpoints:

```text
PlainNone | PlainAtA | PlainAtB | PlainAtBoth
SlashAtA  | SlashAtB
BackslashAtA | BackslashAtB
```

Marked both-end plans remain excluded until the selected parser dialect is
pinned to admit them. Plain plan availability follows the prepared base token:
elided carriers admit only `PlainNone`; explicit single and aromatic carriers
admit the three token-bearing Plain plans.

## Lifecycle and ownership

Each compiled physical carrier owns one ring-plan variable and one latent
mark/plan factor. Before a ring opening the factor is inactive and the plan has
no semantic effect. The first ring endpoint atomically:

```text
restricts BondRole = Ring
activates the mark/plan factor
restricts endpoint token placement
applies tetrahedral occurrence restrictions
opens the structural ring
allocates and validates the label
```

The factor remains active. Ring closure resolves the correlated mark and plan.
There is no factor retirement.

Ordinary bonds retain the existing combined traversal/ring-placement decision.
Compiled directional candidates use a role-only bond decision plus the separate
latent ring plan. This prevents an inactive future ring obligation from joining
the static directional mark component to the spanning-tree projector.

## Visible transitions

A label-only opening restricts only plans omitting a token at the current
endpoint. It does not select a mark. One successor may therefore retain Plain,
slash, and backslash alternatives whose distinguishing token is emitted only at
closure.

A token-emitting endpoint publishes one choice per visible ordinary, `/`, or
`\` token. The token restricts both endpoint placement and the physical mark;
the ring label remains pending. Label-only closure and token-emitting closure
apply the symmetric projection. Closure requires singleton mark and plan before
the bond is complete.

Fixed-endpoint rendering remains canonical and independent of opening order.

## Failure ordering

Every tentative ring alternative must complete all already-implemented checks
before it can be classified:

```text
CSP contradiction or invalid immediate writer successor -> discard candidate
ring-label exhaustion                              -> spelling rejection
backend failure                                    -> abort the batch
surviving missing behavior                         -> private incompleteness
```

The temporary directional-ring incompleteness probe must therefore perform
normal successor validation and label rendering. It is deleted once compiled
directional ring placement is implemented.

## Exactness and locality

The latent factor is a two-variable exact projector over mark and plan. The
structural role is already singleton Ring when the factor becomes active. The
plan variable appears nowhere else, so fixed-point propagation with the static
directional binary component is exact. The factor is excluded from binary and
mixed exact-search cores.

Inactive factors perform no revisions. Opening uses one transition and one
activation; closure uses one transition. Pending labels do not repeat semantic
or structural commitment.

## Boundaries

More than two carrier candidates per side remains the only directional private
incompleteness. Automatic carrier discovery, E/Z perception, marked both-end
ring syntax without pinned dialect evidence, chemical atom rendering, factor
retirement, snapshots, counting, and suffix support remain outside this slice.
