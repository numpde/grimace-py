# South Star Package API Contract

Branch: `south-star`

Date: 2026-05-20

Task: `South Star 22: Draft package API contract`

## Status

`MolToSmilesEnumS` is a provisional name for semantic SMILES enumeration. It is
not a public package API yet.

The current public runtime remains `MolToSmilesEnum`, whose contract is RDKit
writer parity for the supported `canonical=False, doRandom=True` regime.
`MolToSmilesEnumS` must remain separate until the private graph-native
implementation covers the declared traversal support, its conformance oracle is
documented, and its unsupported-feature gates are explicit.

## Intended Contract

`MolToSmilesEnumS(mol, ...)` should enumerate SMILES strings that preserve the
input molecule's graph and intended stereo semantics under a documented
South Star annotation policy.

The target is semantic correctness, not RDKit writer-string equality:

- every output must parse;
- every output must preserve the non-isomeric graph;
- every output must preserve the intended isomeric stereo assignment;
- output annotation policy must be named and swappable;
- RDKit writer parity is comparison metadata, not the pass/fail authority.

## First Supported Molecule Class

The first package-facing class should stay narrow:

- one connected acyclic molecule;
- explicit directional double-bond stereo represented by slash/backslash
  carriers;
- acyclic alkene-style carriers;
- hetero imine or oxime-style carriers using the same directional semantics;
- independent components and explicitly named coupled components, including
  shared carrier edges;
- same-side alternate carrier edges under maximal carrier annotation.

This is not yet support for all RDKit stereo surfaces.

## Unsupported Fail-Fast Cases

Unsupported cases should fail before enumeration and report explicit reasons.
Initial unsupported categories include:

- query atoms or query bonds;
- tetrahedral atom stereo;
- unsupported bond types;
- dative or metal-containing surfaces;
- disconnected molecules;
- ring stereo and ring-closure carrier bases;
- aromatic directional surfaces;
- any component whose marker equations cannot be stated locally.

No unsupported case should silently fall back to `MolToSmilesEnum` or emit a
partial semantic support set.

## Annotation Policy

The initial policy is maximal eligible-carrier annotation:

- if a carrier can express a semantic stereo feature, require a directional
  marker on that emitted carrier;
- expose every marker direction that preserves at least one surviving local
  semantic assignment;
- keep policy decisions outside component extraction and marker equations.

Future policies may be minimal, canonical, or writer-like. They should share
the same component facts and support-state boundary.

## Complexity Contract

The implementation should be component-factorized:

- independent semantic stereo components are independent support-state factors;
- coupling is allowed only through named coupling causes;
- online token checks inspect affected components, not unrelated global rows;
- diagnostics expose component counts, local assignment counts, affected
  component counts, and estimated product size.

The final support size may multiply across independent components. The online
query path should not repeatedly scan or filter unrelated components.

## Conformance Oracle

The South Star conformance oracle has separate checks:

- RDKit parseability, used as parser evidence;
- non-isomeric canonical graph equivalence;
- isomeric canonical stereo equivalence;
- RDKit-independent marker-placement conformance for the current South Star
  directional-marker subset.

RDKit parser behavior remains evidence, not the definition of South Star
validity. The current marker-placement check is deliberately narrower than a
full OpenSMILES parser: it covers the atom, branch, double-bond, and
slash/backslash forms emitted by the current South Star seed enumerator. A
stronger OpenSMILES-style grammar criterion can replace or augment that check
later without changing the semantic graph/stereo checks.

## Difference From `MolToSmilesEnum`

`MolToSmilesEnum` answers: "What strings does Grimace believe RDKit's writer
can emit in the supported parity regime?"

`MolToSmilesEnumS` should answer: "What strings are semantically valid under
the chosen South Star annotation policy?"

Expected comparison categories:

- `intersection`: accepted by both surfaces;
- `SouthStarOnly`: semantically valid but not in RDKit parity support;
- `RDKitParityOnly`: emitted by RDKit parity support but not accepted by the
  current South Star policy or graph-native seed enumerator.

These categories are diagnostics. Equality is not the South Star goal.

## Current Implementation Boundary

Current branch code has a private graph-native implementation boundary plus
fixture-backed comparison support.

- `tests.helpers.south_star_enum_s.mol_to_smiles_enum_s_prototype_for_case`
  remains fixture-backed comparison/test support.
- `grimace._south_star` owns the private graph-native implementation. Tests and
  diagnostics import that internal boundary directly; the public `grimace`
  package does not export it.
- `grimace._south_star.enum_s.mol_to_smiles_enum_s_graph_native_for_case`
  derives outputs from the molecule graph, component marker assignments,
  traversal events, marker-slot equations, and solved marker assignments.
- The graph-native seed does not use fixture-positive strings or
  `MolToSmilesEnum` outputs as generation input.
- The correctness harness and parity comparison diagnostics now consume the
  graph-native implementation by default.

This is still not package-ready. The graph-native implementation now enumerates
roots and child/main-branch orders for the current connected acyclic tree
subset, but it still excludes rings, disconnected molecules, tetrahedral atom
stereo, aromatic output, and broader atom/bond syntax. Before package exposure,
the public contract must either implement those surfaces or explicitly define a
narrower semantic language.

## Naming

`MolToSmilesEnumS` is acceptable as a provisional internal name. Public naming
should remain undecided until the contract is implemented and reviewed. The
name must communicate that this is semantic enumeration, not RDKit writer
parity.
