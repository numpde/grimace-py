# Selectable traversal directional sites

## Decision

Replace the fixed-carrier binary sign model with one uniform static finite-domain
model for directional sides containing one or two prepared carrier candidates.

Each physical candidate owns one variable independent of bond representation:

```text
CarrierMark = Plain | SlashAtFixedA | BackslashAtFixedA
```

Each configured alkene side owns one pattern variable. A pattern records every
candidate's mark and the resulting abstract side phase. Static binary relations
connect the pattern to each physical mark variable and connect the two side
patterns across the configured double bond. No carrier, phase, traversal tree,
or future token sequence is selected during preparation.

## Prepared semantics

A prepared carrier contains a physical bond and a local `side_flip`. For a
non-plain mark on carrier `c` at configured endpoint `e`:

```text
canonical_sign(c) = 0 for SlashAtFixedA, 1 for BackslashAtFixedA
outward(c, e)     = canonical_sign(c) xor endpoint_flip(c, e)
side_phase(c)     = outward(c, e) xor side_flip(c)
```

A valid side pattern has at least one non-plain candidate and every non-plain
candidate yields the same side phase. A relation retains exactly the left/right
pattern pairs satisfying `left_phase xor right_phase = side_phase_xor`.

One-candidate sides therefore have two patterns. Two-candidate sides normally
have six: either candidate marked alone, or both marked consistently, for each
phase. One physical bond shared by several sides reuses one mark variable.

Only existing always-active binary factors are used:

```text
side pattern -> each physical carrier mark
left side pattern -> right side pattern
```

Acyclic directional components require propagation only. Cyclic shared-site
components use the existing exact binary search. Bond-role, ring-placement, and
tetrahedral variables do not enter the directional binary component.

Sides with more than two candidates remain prepared private incompleteness.
Repeated relations for one physical double bond remain invalid input.

## Writer semantics

A traversal event enumerates the currently supported local marks:

```text
Plain              -> ordinary elided, `-`, or `:` spelling
SlashAtFixedA      -> `/` from fixed A, `\` from fixed B
BackslashAtFixedA  -> `\` from fixed A, `/` from fixed B
```

Marked glyphs replace ordinary bond spelling. Every visible alternative applies
one atomic transition containing the traversal-role decision, selected mark,
parent tetrahedral prefix, and any child atom/context restrictions.

For inline traversal, an elided plain alternative may emit the child atom in the
same transition. Explicit plain and marked alternatives emit a bond token and
leave the child atom pending.

For branch traversal, `(` commits the child and traversal role but not the mark.
Its pending traversal-emission frontier may contain the child atom directly,
ordinary explicit bond text, `/`, or `\`. The parenthesis is published only when
at least one such immediate alternative survives.

## Incompleteness ordering

Private incompleteness is reported only after every implemented restriction for
the candidate has survived. Mandatory role, available ring-plan refinement,
tetrahedral prefix, mark restriction, and current static factors are propagated
first. Contradictory candidates are discarded; backend failure aborts the batch;
only a surviving candidate that needs missing spelling aborts as incomplete.

For ring events, a supported `Plain` mark uses ordinary ring spelling and is
restricted atomically. Slash/backslash ring alternatives are tested through the
implemented semantic boundary and report `DirectionalRingEndpoint` only when at
least one survives. Thus contradictory missing alternatives do not suppress a
complete plain ring frontier.

The same ordering applies to sides with more than two candidates.

## Completion and qualification

At molecular-component completion, every represented candidate mark and every
side pattern in that component is singleton and agrees with the emitted spelling
and prepared relation. A ring-represented candidate cannot retain a marked value
in this slice.

Qualification must independently establish side-pattern truth tables, native
versus exhaustive projection, shared-candidate cyclic exactness, inline and
branch one-step choices, reversal, tetrahedral composition, ring/plain behavior,
failure precedence, and locality. The existing fixed-carrier behavior is the
one-candidate special case; the root-relative sign-star implementation is
removed rather than retained as a second semantics path.

Directional ring glyph placement, sides with more than two candidates,
automatic stereo/carrier perception, suffix support, dynamic factors, public
state identity, counting, and caching remain outside this slice.
