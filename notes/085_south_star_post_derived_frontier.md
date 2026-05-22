# South Star Post-Derived Frontier

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 238: Recompute post-derived frontier roadmap`

## Purpose

Reassess the South Star roadmap after admitting mixed polycyclic
ring/tetrahedral plus directional coverage and adding compact derived-support
fixtures for large disconnected products.

The goal is still a principled semantic enumerator:

- one reference spine for graph facts, traversal skeletons, stereo facts,
  constraint solving, rendering, and semantic parse-back;
- RDKit writer parity kept out of the South Star support contract;
- large products represented by authorities, composition facts, counts,
  digests, and diagnostic runners rather than unchecked string dumps.

## Current State

Checked state after `South Star 237`:

- exact first-domain cases: `6`, with `102` expected outputs;
- expanded-support cases: `74`, with `11971` expected outputs;
- derived-support cases: `1`, with one `18816`-output product;
- unified-reference-backed cases in readiness: `80`;
- public API blocker case ids in readiness: none;
- supported feature areas: `35`;
- support-gate blocker categories: `16`.

The current derived-support case is
`disconnected_polycyclic_tetrahedral_directional_fragments`, with fragments
`polycyclic_ring_tetrahedral_bridged_center` (`784` outputs) and
`isolated_alkene_e` (`12` outputs). Its full product has `18816` outputs and is
pinned by digest plus sentinel outputs. Full all-output semantic parse-back is
available through:

`PYTHONPATH=python:. python3 -m unittest tests.run_south_star_derived_support_diagnostics -q`

The main South Star runner now treats the full parse-back as diagnostic-only,
not ordinary default-suite cost.

## What Changed Conceptually

The branch is no longer proving only small local syntax families. It now has a
working product story for:

- independent directional components;
- coupled directional components;
- tetrahedral atom stereo;
- ring/tetrahedral interactions;
- polycyclic ring/tetrahedral obligations;
- mixed connected ring/tetrahedral plus directional obligations;
- disconnected products composed from fragment authorities.

That is a meaningful step toward the South Star, because it exercises the same
one-truth spine across traversal, constraint, renderer, and parse-back layers.

The main remaining risk is not that one more witness is missing. The risk is
that new areas could be admitted as local renderer exceptions instead of
through the reference spine.

## Serious Alternatives

### Alternative 1: Promote Toward A Public `MolToSmilesEnumS`

Pros:

- readiness currently reports no public API blocker case ids;
- package-readiness evidence is much stronger than before;
- the current private API has broad fixed-molecule semantic coverage.

Cons:

- docs and release contract still need a separate pass;
- performance evidence is diagnostic, not release-grade;
- the South Star contract should be explicit about semantic support versus
  RDKit writer parity before export.

Recommendation: do not export next. Keep using readiness as a pressure test,
but finish the next semantic frontier first.

### Alternative 2: Add More Large Product Fixtures

Pros:

- validates scale;
- the compact derived-support representation is now available;
- easy to add more products from existing fragment authorities.

Cons:

- mostly repeats the product mechanism already proven;
- risks growing fixtures without increasing semantic breadth;
- can hide real frontier questions behind bigger numbers.

Recommendation: defer unless a large fixture introduces a genuinely new
coupling shape.

### Alternative 3: Probe Aromatic Directional Surfaces

Pros:

- explicitly present in support-gate blocker categories;
- naturally follows directional support and aromatic atom-text support;
- likely to clarify whether directional markers over aromatic surfaces are
  principled molecule semantics, parser artifacts, or policy-only syntax.

Cons:

- easy to confuse writer/parser behavior with semantic support;
- may produce a "do not admit yet" result rather than a fixture;
- needs careful separation between aromatic graph semantics and directional
  stereo semantics.

Recommendation: choose as the next investigation slice, but start with a probe
and policy note, not support-gate expansion.

### Alternative 4: Query Bonds And Query Atoms

Pros:

- still explicit blockers;
- important for a complete input-boundary story.

Cons:

- query inputs do not denote one fixed molecule;
- admitting them would require a query-product semantics, not the current
  fixed-molecule South Star contract.

Recommendation: keep as explicit fail-fast boundary work, not next admission.

### Alternative 5: Dative And Coordination Bonds

Pros:

- chemistry-facing surface;
- explicitly present in blockers;
- eventually important if the semantic contract broadens.

Cons:

- coordination semantics involve charge, valence, and parser-policy choices;
- prior RDKit exploration found suspicious serializer behavior near
  dative/metal/carbonyl regions;
- a premature implementation would blur principled semantics and
  RDKit-specific quirks.

Recommendation: keep as a later policy/investigation track.

## Recommended Next Queue

1. Probe aromatic directional surfaces.

   Determine whether representative aromatic directional inputs can be
   interpreted as fixed-molecule South Star semantics. The output should be a
   note and focused tests that either keep the current fail-fast boundary or
   define a principled admission path.

2. Strengthen fail-fast boundary tests for query and dative inputs.

   These are not next admission targets, but the current public story should
   make clear that they are intentionally outside fixed-molecule South Star
   semantics.

3. Audit package-readiness docs and command inventory.

   The code now has a named derived-support diagnostic runner and compact large
   product fixtures. Documentation should make the default-vs-diagnostic split
   explicit before any export discussion.

4. Only after those, revisit public API export.

   The export question should be evaluated after the next semantic-boundary
   probe, not immediately after a product-scale success.

