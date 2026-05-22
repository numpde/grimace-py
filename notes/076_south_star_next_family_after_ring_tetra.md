# South Star Next Family After Ring/Tetra

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 213: Choose next semantic family after ring/tetra`

## Purpose

Choose the next South Star semantic family after admitting representative
polycyclic ring/tetrahedral support.

The decision criterion is not "which unsupported string is smallest." The
criterion is which family best advances a principled, fixed-molecule,
mathematically anchored semantic SMILES enumerator without importing RDKit
writer baggage or a new product model accidentally.

## Current Frontier

After notes 071-075 and commits through `938d307`:

- the small adversarial corpus has only two unsupported witnesses:
  query/unspecified bond `C~C` and dative bond `N->[O]`;
- expanded fixtures now include two `polycyclic_ring_tetrahedral` witnesses;
- the ring/tetrahedral support gate is domain-level and cross-checked against
  non-fixture witnesses.

This means the next step should be chosen from a broader South Star roadmap,
not from the tiny adversarial unsupported list alone.

## Alternative 1: Query Or Unspecified Bonds

Representative witness: `C~C`.

Pros:

- smallest current adversarial unsupported witness;
- already fail-fast and easy to keep pinned as a boundary.

Cons:

- query bonds do not denote one fixed molecule graph;
- a principled implementation would need a query-SMILES product model, not just
  another bond renderer token;
- supporting it now risks weakening the South Star contract from fixed-molecule
  semantic enumeration to pattern enumeration.

Recommendation: do not choose next. Keep it as an explicit fail-fast boundary
until a query-product contract exists.

## Alternative 2: Dative Or Coordination Bonds

Representative witness: `N->[O]`.

Pros:

- also a current adversarial unsupported witness;
- real chemistry-facing surface rather than a synthetic test artifact.

Cons:

- the semantic object is not just a bond token: charge, valence,
  coordination interpretation, and parser behavior matter;
- prior RDKit exploration already found dative/metal edge behavior that is
  suspicious enough to isolate as writer behavior;
- easy to implement as a token special case without a clean semantic model.

Recommendation: do not choose next. Keep coordination as a separate future
track with its own semantic model.

## Alternative 3: Aromatic Directional Overlays

Representative surface: aromatic molecules with manually directional bond
facts.

Pros:

- directly touches marker placement policy;
- related to aromatic support, which has been expanded recently.

Cons:

- the natural fixed-molecule witness set is unclear;
- South Star first needs to decide whether directional markers on aromatic
  surfaces are meaningful semantic annotations, invalid input facts, or parser
  artifacts;
- current aromatic support is mostly atom text, traversal, and fused aromatic
  topology, not directional stereo semantics.

Recommendation: keep as a policy/probe track, not the next implementation
family.

## Alternative 4: Larger Mixed Ring/Stereo Composition

Representative surfaces:

- ring/tetrahedral plus exocyclic directional stereo;
- polycyclic ring/tetrahedral plus additional acyclic stereo;
- disconnected mixtures of ring, directional, and tetrahedral components.

Pros:

- fixed-molecule semantics;
- exercises decomposition into independent components plus coupled local
  components;
- directly tests the South Star premise that support is a product of typed
  semantic obligations rather than case-specific renderings.

Cons:

- if started from a large molecule, it can become fixture accumulation rather
  than proof-shape work;
- needs careful distinction between "new primitive family" and "composition of
  existing primitive families."

Recommendation: promising, but should be framed as compositional scaling rather
than as another case family.

## Alternative 5: Multi-Center Tetrahedral Scaling

Representative surfaces:

- two independent acyclic tetrahedral centers;
- ring-local tetrahedral center plus independent acyclic tetrahedral center;
- two tetrahedral centers in the same connected component with no direct
  symmetry coupling;
- a deliberately coupled/symmetric witness only after the independent case is
  clear.

Pros:

- fixed-molecule ordinary chemistry;
- mathematically central: independent components should combine by product,
  while coupled components need an explicit shared constraint representation;
- strong SSoT fit because expected support can be generated from the same
  molecule facts, traversal events, tetrahedral obligations, solver
  assignments, renderer, semantic parse-back, and deduplication spine;
- advances package readiness more than query or dative support.

Cons:

- may expose performance and product-size issues;
- requires explicit guardrails so tests prove factorization rather than just
  pinning another large expected set.

Recommendation: choose this as the next South Star family, under the broader
name **fixed-molecule compositional stereo scaling**.

## Decision

Choose fixed-molecule compositional stereo scaling as the next family.

This means the next implementation sequence should not start by admitting
query or dative bonds. It should first prove that the current one-truth spine
scales from single-obligation witnesses to multiple obligations:

1. independent tetrahedral centers;
2. ring/tetrahedral plus independent tetrahedral center;
3. ring/tetrahedral plus directional component;
4. disconnected products of the same components;
5. only then, coupled or symmetric multi-center cases.

The first proof authority should state whether a case is generated by:

- a direct unified-reference product of independent component obligations; or
- one connected constraint component with multiple coupled obligations.

That distinction is more important than the specific first molecule.

## Immediate Backlog

Open follow-up tasks:

1. Define the compositional-stereo proof contract and authority names.
2. Inventory small witnesses for independent, connected, disconnected, and
   coupled multi-obligation cases.
3. Add the first data-driven fixtures only after the proof contract says which
   cases are independent products and which are coupled components.

