# South Star Directional Product/Coupling Standard

Branch: `south-star`

Date: 2026-05-22

Task: `South Star 226: Define directional product/coupling proof standard`

## Purpose

Define how South Star should classify directional stereo obligations as
independent products or coupled components after the reconciliation in
`South Star 225`.

This note is about principled South Star semantic enumeration. It is not an
RDKit writer-parity rule and it does not decide aromatic directional overlays.

## Ownership Boundary

Directional-directional coupling is owned by
`extract_south_star_components(...)`.

That extractor builds `SouthStarSemanticStereoComponent` objects from source
double-bond stereo features and their eligible carrier edges. Its coupling
unit is shared directional marker support, especially shared carrier edges.

The compositional proof layer must not rediscover directional-directional
coupling by graph distance, atom proximity, or fixture labels. It consumes each
directional component as one directional obligation.

## Directional Product Standard

Two directional obligations are an independent product when
`extract_south_star_components(...)` emits them as separate components.

This remains true even if the two components are graph-near or share a carrier
endpoint atom. Graph-nearness alone is not coupling. The proof obligation is:

- component ids are distinct;
- component carrier-edge sets are disjoint;
- marker equations are component-local;
- solver assignments factor as the product of component-local assignments;
- rendered outputs match the generated proof outputs after deduplication.

Representative witness:

`F/C=C/C/C=C/Cl`

It is an independent product of `directional:component:0` and
`directional:component:1`.

## Directional Coupling Standard

Directional-directional obligations are coupled only when the directional
component extractor already grouped the source features into one component.

The usual reason is shared directional carrier support, represented by
`SouthStarComponentCoupling` and its `coupling_causes`.

The compositional proof helper should not create a second coupling vocabulary
for directional-directional pairs.

## Mixed Directional/Tetrahedral Standard

A directional obligation and a tetrahedral obligation are coupled when the
tetrahedral obligation shares atom support with the directional obligation.

This is not directional-directional coupling. It is a mixed-obligation
relationship: the directional marker placement and tetrahedral atom-token
orientation share a traversal/rendering support surface.

Representative witness:

`F/C=C/[C@H](Cl)Br`

It is a coupled component:

`tetrahedral:3+directional:component:0`

with coupling reason:

`shared_directional_obligation_atom`

## Adjacent-But-Factorable Cases

Adjacent-but-factorable is allowed.

The important example is directional-directional composition where two
components are close in the molecular graph but do not share carrier edges and
remain separate under `extract_south_star_components(...)`.

This category should be explicit in tests because otherwise future code may
reintroduce graph-distance coupling and collapse a valid product into a
coupled component.

## Test Ownership

Tests should enforce the boundary in three places:

1. `tests/south_star/test_compositional_stereo_proof.py` should assert the
   product/coupling classification exposed by
   `compositional_stereo_proof_report(...)`.
2. `tests/south_star/test_expanded_support_fixtures.py` should continue to
   check fixture equality against the relevant proof oracle.
3. Directional component tests should assert extractor-level coupling causes,
   so the compositional proof helper does not become the source of truth for
   directional-directional grouping.

## Non-Goals

This standard does not cover:

- aromatic directional overlays;
- dative or coordination bonds;
- query or unspecified bonds;
- RDKit-specific directional marker minimization;
- parser-equivalence substitution for exact South Star support.

Those need separate contracts if they are admitted later.
