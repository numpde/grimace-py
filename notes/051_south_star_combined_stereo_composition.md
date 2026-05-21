# South Star Combined Stereo Composition

Branch: `south-star`

Date: 2026-05-21

Task: `South Star 161: Expand combined stereo composition`

## Current Finding

The current graph-native path can already generate several mixed-stereo
surfaces:

- two independent directional-bond components in one connected acyclic graph;
- one directional-bond component plus one tetrahedral center in an acyclic
  graph;
- one directional-bond component in one fragment and one tetrahedral center in
  another fragment.

That is not enough to promote fixtures. The South Star standard is
proof-backed support through the shared spine, not "the current generator
returns strings."

## Probe Results

Representative local probes:

- `F/C=C/Cl`: supported; one directional component; 12 outputs.
- `F/C=C/C/C=C/Cl`: supported; two directional components; 48 outputs.
- `F/C=C/[C@H](Cl)Br`: supported; one directional component plus one
  tetrahedral center; 40 outputs.
- `F[C@H](Cl)/C=C/Br`: supported; tetrahedral center adjacent to a
  directional component; 40 outputs.
- `F/C=C\Cl.O[C@H](F)Cl`: supported; disconnected directional/tetrahedral
  composition; 288 outputs.
- `F[C@H]1CCCC(/C=C/Cl)C1`: still gated by `ring_molecule` and
  `ring_tetrahedral_interaction`.

These are capability probes only. They are not expected-support authority.

## Proof Gap

The current proof helpers can separately explain:

- directional-bond marker equations;
- tetrahedral ligand-order obligations;
- disconnected fragment composition;
- nonstereo and ring-stereo traversal obligations.

They do not yet expose one combined proof record saying:

1. each independent stereo component contributes a separate assignment factor;
2. coupled components share a constraint equation system;
3. tetrahedral renderer obligations compose with directional marker
   assignments without becoming a separate mini-oracle;
4. disconnected mixed-stereo fragments compose by fragment-order product plus
   per-fragment support product;
5. deduplication is still first-occurrence over rendered event/assignment
   pairs.

That is the required next layer before adding mixed-stereo fixtures as
unified-reference-backed cases.

## Split Plan

`South Star 161` is too broad as one implementation card. Split it into proof
families:

1. independent directional-component product proof;
2. acyclic directional-plus-tetrahedral composition proof;
3. disconnected mixed-stereo composition proof;
4. ring/tetrahedral/directional interaction gate audit.

Only after one of those proof records exists should its fixture cases be added
to `tests/fixtures/south_star_expanded_support/expanded_domain_v1.json`.

