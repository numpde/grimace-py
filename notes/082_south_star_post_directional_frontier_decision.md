# South Star Post-Directional Frontier Decision

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 228: Choose next post-directional frontier`

## Purpose

Choose the next South Star frontier after stabilizing the directional
product/coupling vocabulary.

The decision criterion remains fixed: prefer the next slice that strengthens
principled, fixed-molecule semantic enumeration and the one-truth reference
spine. Do not choose a target merely because it is small, unsupported, or easy
to render.

## Current Position

The branch now has:

- 78 unified-reference-backed cases;
- proof-backed independent and coupled multi-tetrahedral composition;
- proof-backed mixed ring/tetrahedral composition;
- directional product/coupling tests that separate extractor-owned
  directional coupling from compositional mixed-obligation coupling.

The two tiny adversarial unsupported witnesses are still query and dative, but
those are not fixed-molecule South Star gaps.

## Alternatives

### Alternative 1: Pin Larger Polycyclic Ring/Tetra Branch

Witness: `F[C@H]1CC2CCC1C2[C@H](Cl)Br`

Pros:

- fixed-molecule semantics;
- already supported;
- same proof shape as the smaller mixed ring/tetra fixture;
- useful scale stress at 3160 outputs.

Cons:

- mostly larger evidence for an already admitted proof shape;
- large fixture cost;
- does not exercise the newly stabilized directional vocabulary.

Recommendation: defer. Keep it as a scale fixture after the next new proof
shape.

### Alternative 2: Polycyclic Ring/Tetra Plus Directional Composition

Representative witnesses:

- `F[C@H]1CC2CCC1C2/C=C/Cl`;
- `F[C@H]1CC2CCC1C2.F/C=C/Cl`.

Pros:

- fixed-molecule semantics;
- currently support-gated, so it is a real frontier;
- composes two already important proof families: polycyclic ring/tetrahedral
  obligations and directional marker obligations;
- directly tests whether the one-truth spine can combine independent and
  coupled components across ring closures, directional markers, and fragment
  order without RDKit writer baggage.

Cons:

- harder than pinning a supported fixture;
- needs a proof-model slice before any support-gate change;
- may expose missing traversal/rendering skeleton facts for polycyclic
  directional composition.

Recommendation: choose this as the next main frontier.

### Alternative 3: Aromatic Directional Overlays

Pros:

- now that directional vocabulary is clearer, this can be investigated more
  cleanly;
- relevant to marker-policy boundaries.

Cons:

- still partly a semantic-policy question;
- may not represent ordinary fixed-molecule stereochemistry;
- easy to confuse parser artifacts with semantic support.

Recommendation: keep as a policy/probe track after the next fixed-molecule
mixed proof-model slice.

### Alternative 4: Query Or Unspecified Bonds

Witness: `C~C`

Pros:

- small and explicitly unsupported;
- useful boundary test.

Cons:

- not one fixed molecule graph;
- requires a query-product contract.

Recommendation: do not choose now.

### Alternative 5: Dative Or Coordination Bonds

Witness: `N->[O]`

Pros:

- chemistry-facing unsupported surface;
- useful boundary test.

Cons:

- coordination semantics require charge, valence, and parser-contract work;
- known RDKit quirks make it a poor next semantic-spine target.

Recommendation: do not choose now.

## Decision

Choose **polycyclic ring/tetra plus directional composition** as the next
frontier.

The first implementation step should not be a support-gate expansion. It
should be a proof-model inventory: identify the exact facts and component
partition needed to combine polycyclic ring/tetrahedral obligations with
directional obligations, then decide whether the witnesses are independent
products, coupled components, or unsupported because a required fact is missing
from the current spine.

## Immediate Backlog

1. Inventory polycyclic ring/tetra plus directional witnesses.
2. Define the mixed polycyclic ring/tetra/directional proof model.
3. Add the first support-gate and fixture slice only if the proof model is
   explicit and generated outputs are proof-derived.
